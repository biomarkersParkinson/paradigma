import os
import tsdf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from sklearn.linear_model import LogisticRegression

from paradigma.constants import DataColumns
from paradigma.tremor.tremor_analysis_config import TremorFeatureExtractionConfig, TremorDetectionConfig
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
    df[DataColumns.PRED_TREMOR_LOGREG] = df[DataColumns.PRED_TREMOR_PROBA] >= threshold
    df[DataColumns.PRED_TREMOR_LOGREG] = df[DataColumns.PRED_TREMOR_LOGREG].astype(int) # save as int

    # Perform extra checks for rest tremor 
    peak_check = (df['freq_peak'] >= config.fmin_peak) & (df['freq_peak']<=config.fmax_peak) # peak within 3-7 Hz
    movement_check = df['low_freq_power'] <= config.movement_treshold # little non-tremor arm movement
    df[DataColumns.PRED_TREMOR_CHECKED] = (df[DataColumns.PRED_TREMOR_LOGREG]==1) & (peak_check==True) & (movement_check == True)
    df[DataColumns.PRED_TREMOR_CHECKED] = df[DataColumns.PRED_TREMOR_CHECKED].astype(int) # save as int
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