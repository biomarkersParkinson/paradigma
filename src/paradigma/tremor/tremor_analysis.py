import os
import pytz
import tsdf
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from sklearn.linear_model import LogisticRegression
from scipy.stats import gaussian_kde

from paradigma.constants import DataColumns
from paradigma.config import TremorFeatureExtractionConfig, TremorDetectionConfig, TremorQuantificationConfig
from paradigma.tremor.feature_extraction import extract_spectral_domain_features
from paradigma.segmenting import tabulate_windows
from paradigma.util import get_end_iso8601, write_df_data, read_metadata, WindowedDataExtractor


def extract_tremor_features(df: pd.DataFrame, config: TremorFeatureExtractionConfig) -> pd.DataFrame:
    """
    This function groups sequences of timestamps into windows and subsequently extracts 
    tremor features from windowed gyroscope data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing sensor data, which includes time and gyroscope data. The data should be
        structured with the necessary columns as specified in the `config`.

    config : TremorFeatureExtractionConfig
        Configuration object containing parameters for feature extraction, including column names for time, gyroscope data,
        as well as settings for windowing, and feature computation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing extracted tremor features and a column corresponding to time.
    
    Notes
    -----
    - This function groups the data into windows based on timestamps.
    - The input DataFrame must include columns as specified in the `config` object for proper feature extraction.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the required columns as specified in the configuration or if any step in the feature extraction fails.
    """
    # group sequences of timestamps into windows
    windowed_cols = [DataColumns.TIME] + config.gyroscope_cols
    windowed_data = tabulate_windows(df, windowed_cols, config.window_length_s, config.window_step_length_s, config.sampling_frequency)

    extractor = WindowedDataExtractor(windowed_cols)

    # Extract the start time and gyroscope data from the windowed data
    idx_time = extractor.get_index(DataColumns.TIME)
    idx_gyro = extractor.get_slice(config.gyroscope_cols)

    # Extract data
    start_time = np.min(windowed_data[:, :, idx_time], axis=1)
    windowed_gyro = windowed_data[:, :, idx_gyro]

    df_features = pd.DataFrame(start_time, columns=[DataColumns.TIME])
    
    # transform the signals from the temporal domain to the spectral domain and extract tremor features
    df_spectral_features = extract_spectral_domain_features(config, windowed_gyro)

    # Combine spectral features with the start time
    df_features= pd.concat([df_features, df_spectral_features], axis=1)

    return df_features

def extract_tremor_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: TremorFeatureExtractionConfig) -> None:
    # Load data
    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Extract tremor features
    df_windowed = extract_tremor_features(df, config)

    # Store data
    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_windowed[DataColumns.TIME][-1:].values[0] + config.window_length_s))

    metadata_values.end_iso8601 = end_iso8601
    metadata_values.file_name = 'tremor_values.bin'
    metadata_time.end_iso8601 = end_iso8601
    metadata_time.file_name = 'tremor_time.bin'

    metadata_values.channels = list(config.d_channels_values.keys())
    metadata_values.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_values, output_path, 'tremor_meta.json', df_windowed)


def detect_tremor(df: pd.DataFrame, config: TremorDetectionConfig, path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:
    """
    Detects tremor in the input DataFrame using a pre-trained classifier and applies a threshold to the predicted probabilities.

    This function performs the following steps:
    1. Loads the pre-trained classifier and scaling parameters from the provided directory.
    2. Scales the relevant features in the input DataFrame (`df`) using the loaded scaling parameters.
    3. Makes predictions using the classifier to estimate the probability of tremor.
    4. Applies a threshold to the predicted probabilities to classify whether tremor is detected or not.
    5. Adds the predicted probabilities and the classification result to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing extracted tremor features. The DataFrame must include
        the necessary columns as specified in the classifier's feature names.

    config : TremorDetectionConfig
        Configuration object containing the classifier file name, threshold file name, and other settings for gait detection.

    path_to_classifier_input : Union[str, Path]
        The path to the directory containing the classifier file, threshold value, scaler parameters, and other necessary input
        files for tremor detection.

    Returns
    -------
    pd.DataFrame
        The input DataFrame (`df`) with two additional columns:
        - `PRED_TREMOR_PROBA`: Predicted probability of tremor based on the classifier.
        - `PRED_TREMOR_LOGREG`: Binary classification result (True for tremor, False for no tremor), based on the threshold applied to `PRED_TREMOR_PROBA`.
        - `PRED_TREMOR_CHECKED`: Binary classification result (True for tremor, False for no tremor), after performing extra checks for rest tremor on `PRED_TREMOR_LOGREG`.
        
    Notes
    -----
    - The threshold used to classify tremor is loaded from a file and applied to the predicted probabilities.

    Raises
    ------
    FileNotFoundError
        If the classifier, scaler, or threshold files are not found at the specified paths.
    ValueError
        If the DataFrame does not contain the expected features for prediction or if the prediction fails.
    """

    # Initialize the classifier
    coefficients = np.loadtxt(os.path.join(path_to_classifier_input, config.coefficients_file_name))
    threshold = np.loadtxt(os.path.join(path_to_classifier_input, config.thresholds_file_name))

    # Scale the MFCC's
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
    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    df = detect_tremor(df, config, path_to_classifier_input)

    # Prepare the metadata
    metadata_values.file_name = 'tremor_values.bin'
    metadata_time.file_name = 'tremor_time.bin'

    metadata_values.channels = list(config.d_channels_values.keys())
    metadata_values.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_values, output_path, 'tremor_meta.json', df)


def aggregate_tremor_power(tremor_power: pd.Series, config: TremorQuantificationConfig):
    
    """
    Converts the tremor power to the log scale and subsequentyly computes three aggregates of tremor power:
    the median, mode and perdentile specified in TremorQuantificationConfig.     
    """

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


def quantify_tremor(df: pd.DataFrame, config: TremorQuantificationConfig):
    """
    Quantifies the amount of tremor time and tremor power, aggregated over all windows in the input dataframe.
    Tremor time is calculated as the number of the detected tremor windows, as percentage of the number of windows 
    without significant non-tremor movement (at rest). For tremor power the following aggregates are derived:
    the median, mode and percentile of tremor power specified in the configuration object. 
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing extracted tremor features. The DataFrame must include
        the necessary columns as specified in the classifier's feature names.

    config : TremorQuantificationConfig
        Configuration object containing the percentile for aggregating tremor power.

    Returns
    -------
    A json file with the aggregated tremor time and tremor power measures, as well as the total number of windows
    available in the input dataframe, and the number of windows at rest.

    Notes
    -----
    - Tremor power is converted to log scale, after adding a constant of 1, so that zero tremor power
    corresponds to a value of 0 in log scale.
    - The modal tremor power is computed based on gaussian kernel density estimation.
  
    """

    nr_windows_total = df.shape[0] # number of windows in the input dataframe

    # remove windows with detected non-tremor movements to control for the amount of arm activities performed
    df_filtered = df.loc[df.low_freq_power <= config.movement_threshold]
    nr_windows_rest = df_filtered.shape[0] # number of windows without non-tremor arm movement

    # calculate weekly tremor time
    tremor_time= np.sum(df_filtered['pred_tremor_checked']) / nr_windows_rest * 100 # as percentage of total measured time without non-tremor arm movement

    # calculate weekly tremor power measures
    tremor_power = df_filtered.loc[df_filtered['pred_tremor_checked'] == 1, 'tremor_power']
    tremor_power_median, tremor_power_90th_perc, tremor_power_mode = aggregate_tremor_power(tremor_power,config)
    
    # store aggregates in json format
    d_aggregates = {
        'metadata': {
            'nr_windows_total': nr_windows_total,
            'nr_windows_rest': nr_windows_rest
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
    metadata_time, metadata_values = read_metadata(path_to_feature_input, config.meta_filename, config.time_filename, config.values_filename)
    df_features = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    metadata_dict = tsdf.load_metadata_from_path(os.path.join(path_to_prediction_input, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_values = metadata_dict[config.values_filename]
    df_predictions = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Subset features
    df_features = df_features[['tremor_power', 'low_freq_power']]

    # Concatenate predictions and tremor power
    df = pd.concat([df_predictions, df_features], axis=1)

    # Compute weekly aggregated tremor measures
    d_aggregates = quantify_tremor(df, config)

    # Save output
    with open(os.path.join(output_path,"weekly_tremor.json"), 'w') as json_file:
        json.dump(d_aggregates, json_file, indent=4)
