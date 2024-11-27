import os
import pytz
import tsdf
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta
from collections import Counter
from scipy.stats import gaussian_kde

from paradigma.constants import DataColumns
from paradigma.tremor.tremor_analysis_config import TremorFeatureExtractionConfig, TremorDetectionConfig, TremorQuantificationConfig
from paradigma.tremor.feature_extraction import extract_spectral_domain_features
from paradigma.windowing import tabulate_windows
from paradigma.util import get_end_iso8601, write_df_data, read_metadata


def extract_tremor_features(df: pd.DataFrame, config: TremorFeatureExtractionConfig) -> pd.DataFrame:
    # group sequences of timestamps into windows
    df_windowed = tabulate_windows(config,df)

    # transform the signals from the temporal domain to the spectral domain using the fast fourier transform
    # and extract spectral features
    df_windowed = extract_spectral_domain_features(config, df_windowed)

    return df_windowed

def extract_tremor_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: TremorFeatureExtractionConfig) -> None:
    # Load data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Extract tremor features
    df_windowed = extract_tremor_features(df, config)

    # Store data
    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_windowed[config.time_colname][-1:].values[0] + config.window_length_s))

    metadata_samples.end_iso8601 = end_iso8601
    metadata_samples.file_name = 'tremor_values.bin'
    metadata_time.end_iso8601 = end_iso8601
    metadata_time.file_name = 'tremor_time.bin'

    metadata_samples.channels = list(config.d_channels_values.keys())
    metadata_samples.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'tremor_meta.json', df_windowed)


def detect_tremor(df: pd.DataFrame, config: TremorDetectionConfig, path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:
   
    # Initialize the classifier
    coefficients = np.loadtxt(os.path.join(path_to_classifier_input, config.coefficients_file_name))
    threshold = np.loadtxt(os.path.join(path_to_classifier_input, config.thresholds_file_name))

    # Scale the mfcc's
    mean_scaling = np.loadtxt(os.path.join(path_to_classifier_input, config.mean_scaling_file_name))
    std_scaling = np.loadtxt(os.path.join(path_to_classifier_input, config.std_scaling_file_name))
    mfcc = df.loc[:, df.columns.str.startswith('mfcc')]
    mfcc_scaled = (mfcc-mean_scaling)/std_scaling

    # Create a logistic regression model with pre-defined coefficients
    log_reg = LogisticRegression(penalty = None)
    log_reg.classes_ = np.array([0, 1]) 
    log_reg.intercept_ = coefficients[0]
    log_reg.coef_ = coefficients[1:].reshape(1, -1)
    log_reg.n_features_in_ = int(mfcc.shape[1])
    log_reg.feature_names_in_ = mfcc.columns

    # Get the tremor probability 
    df[DataColumns.PRED_TREMOR_PROBA] = log_reg.predict_proba(mfcc_scaled)[:, 1] 

    # Make prediction based on pre-defined threshold
    df[DataColumns.PRED_TREMOR_LOGREG] = (df[DataColumns.PRED_TREMOR_PROBA] >= threshold).astype(int)

    # Perform extra checks for rest tremor 
    peak_check = (df['freq_peak'] >= config.fmin_peak) & (df['freq_peak']<=config.fmax_peak) # peak within 3-7 Hz
    movement_check = df['low_freq_power'] <= config.movement_threshold # little non-tremor arm movement
    df[DataColumns.PRED_TREMOR_CHECKED] = ((df[DataColumns.PRED_TREMOR_LOGREG]==1) & (peak_check==True) & (movement_check == True)).astype(int)
    
    return df


def detect_tremor_io(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: TremorDetectionConfig) -> None:
    
    # Load the data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    df = detect_tremor(df, config, path_to_classifier_input)

    # Prepare the metadata
    metadata_samples.file_name = 'tremor_values.bin'
    metadata_time.file_name = 'tremor_time.bin'

    metadata_samples.channels = list(config.d_channels_values.keys())
    metadata_samples.units = list(config.d_channels_values.values())

    metadata_time.channels = [config.time_colname]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'tremor_meta.json', df)


def aggregate_tremor_power(tremor_power: pd.Series, config: TremorQuantificationConfig):

    tremor_power = np.log10(tremor_power+1) # convert to log scale
    
    # calculate median and 90th percentile of tremor power
    tremor_power_median = tremor_power.median()
    tremor_power_90th_perc = tremor_power.quantile(config.percentile_tremor_power)
    
    # calculate modal tremor power
    bin_edges = np.linspace(0, 6, 301)
    kde = gaussian_kde(tremor_power)
    kde_values = kde(bin_edges)
    max_index = np.argmax(kde_values)
    tremor_power_mode = bin_edges[max_index]

    return tremor_power_median, tremor_power_90th_perc, tremor_power_mode


def quantify_tremor(df: pd.DataFrame, config: TremorQuantificationConfig, start_time: datetime):

    # Create time array in local time zone
    utc_start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
    local_start_time = utc_start_time.astimezone(pytz.timezone("Europe/Amsterdam"))
    time = [local_start_time + timedelta(seconds=x) for x in df.time]

    # determine valid days
    daytime_hours = range(config.daytime_hours_lower_bound,config.daytime_hours_upper_bound)
    nr_windows_per_day = Counter([dt.date() for dt in time if dt.hour in daytime_hours])
    valid_days = [day for day, count in nr_windows_per_day.items() 
                  if count * config.window_length_s / 3600 >= config.valid_day_threshold_hr]
    nr_valid_days = len(valid_days)

    # remove windows during non-valid days, non-daytime hours and non-tremor arm movement
    df_filtered = df.loc[[dt.date() in valid_days for dt in time]]
    df_filtered = df_filtered.loc[[dt.hour in daytime_hours for dt in time]]
    nr_windows_daytime = df_filtered.shape[0] # number of windows during daytime on valid days
    df_filtered = df_filtered.loc[df_filtered.low_freq_power <= config.movement_threshold]
    nr_windows_daytime_rest = df_filtered.shape[0] # number of windows during daytime on valid days without non-tremor arm movement

    # calculate weekly tremor time
    tremor_time= np.sum(df_filtered['pred_tremor_checked']) / df_filtered.shape[0] * 100

    # calculate weekly tremor power measures
    tremor_power = df_filtered.loc[df_filtered['pred_tremor_checked'] == 1, 'tremor_power']
    tremor_power_median, tremor_power_90th_perc, tremor_power_mode = aggregate_tremor_power(tremor_power,config)
    
    # store aggregates in json format
    d_aggregates = {
        'metadata': {
            'nr_valid_days': nr_valid_days,
            'nr_windows_daytime': nr_windows_daytime,
            'nr_windows_daytime_rest': nr_windows_daytime_rest
        },
        'weekly_tremor_measures': {
            'tremor_time': tremor_time,
            'tremor_power_median': tremor_power_median,
            'tremor_power_mode': tremor_power_mode,
            'tremor_power_90th_perc': tremor_power_90th_perc
        }
    }

    return d_aggregates


def quantify_tremor_io(path_to_feature_input: Union[str, Path], path_to_prediction_input: Union[str, Path], output_path: Union[str, Path], config: TremorQuantificationConfig) -> None:
    
    # Load the features & predictions
    metadata_time, metadata_samples = read_metadata(path_to_feature_input, config.meta_filename, config.time_filename, config.values_filename)
    df_features = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    metadata_dict = tsdf.load_metadata_from_path(os.path.join(path_to_prediction_input, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_samples = metadata_dict[config.values_filename]
    df_predictions = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Validate
    # Dataframes have same length
    assert df_features.shape[0] == df_predictions.shape[0]

    # Dataframes have same time column
    assert df_features[DataColumns.TIME].equals(df_predictions[DataColumns.TIME])

    # Subset features
    df_features = df_features[['tremor_power', 'low_freq_power']]

    # Concatenate predictions and tremor power
    df = pd.concat([df_predictions, df_features], axis=1)

    # Compute weekly aggregated tremor measures
    d_aggregates = quantify_tremor(df, config, start_time = metadata_time.start_iso8601)

    # Save output
    with open(os.path.join(output_path,"weekly_tremor.json"), 'w') as json_file:
        json.dump(d_aggregates, json_file, indent=4)
