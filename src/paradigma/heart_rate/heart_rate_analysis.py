from typing import Union
from pathlib import Path
import pandas as pd
import os
import numpy as np

import tsdf

from paradigma.constants import DataColumns
from paradigma.config import SignalQualityFeatureExtractionConfig, SignalQualityClassificationConfig, \
    HeartRateExtractionConfig, HeartRateExtractionConfig
from paradigma.heart_rate.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features
from paradigma.heart_rate.heart_rate_estimation import assign_sqa_label, extract_hr_segments, extract_hr_from_segment
from paradigma.segmenting import tabulate_windows

from paradigma.util import read_metadata

def extract_signal_quality_features(df: pd.DataFrame, config: SignalQualityFeatureExtractionConfig) -> pd.DataFrame:
    # Group sequences of timestamps into windows
    ppg_windowed = tabulate_windows(
        df=df, 
        columns=[config.ppg_colname],
        window_length_s=config.window_length_s,
        window_step_length_s=config.window_step_length_s,
        fs=config.sampling_frequency
    )[0]

    # Compute statistics of the temporal domain signals
    df_temporal_features = extract_temporal_domain_features(config, ppg_windowed, quality_stats=['var', 'mean', 'median', 'kurtosis', 'skewness'])
    
    # Compute statistics of the spectral domain signals
    df_spectral_features = extract_spectral_domain_features(config, ppg_windowed)

    return pd.concat([df_temporal_features, df_spectral_features], axis=1)


def extract_signal_quality_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: SignalQualityFeatureExtractionConfig) -> None:
    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)
    
    # Extract signal quality features
    # Extract signal quality features
    df_windowed = extract_signal_quality_features(df, config)
    return df_windowed


def signal_quality_classification(df: pd.DataFrame, config: SignalQualityClassificationConfig, path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:
    """
    Classify the signal quality of the PPG signal using a logistic regression classifier.
    The classifier is trained on features extracted from the PPG signal.
    The features are extracted using the extract_signal_quality_features function.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the PPG signal features.
    config : SignalQualityClassificationConfig
        The configuration for the signal quality classification.
    path_to_classifier_input : Union[str, Path]
        The path to the directory containing the classifier.

    Returns
    -------
    df_sqa pd.DataFrame
        The DataFrame containing the PPG signal quality predictions.
    """
    
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', config.classifier_file_name))
    lr_clf = clf['model']  # Load the logistic regression classifier
    mu = clf['mu']  # load the mean, 2D array
    sigma = clf['sigma'] # load the standard deviation, 2D array

    # Assign feature names to the classifier
    lr_clf.feature_names_in_ = ['var', 'mean', 'median', 'kurtosis', 'skewness', 'f_dom', 'rel_power', 'spectral_entropy', 'signal_to_noise', 'auto_corr']

    # Normalize features using mu and sigma
    X_normalized = (df[lr_clf.feature_names_in_] - mu.ravel()) / sigma.ravel()  # Use .ravel() to convert the 2D arrays (mu and sigma) to 1D arrays

    # Make predictions for PPG signal quality assessment, and assign the probabilities to the DataFrame and drop the features
    df[DataColumns.PRED_SQA_PROBA] = lr_clf.predict_proba(X_normalized)[:, 0]
    df_sqa = df[[DataColumns.PRED_SQA_PROBA]] # Return DataFrame with only the predicted probabilities
    
    return df_sqa   


def signal_quality_classification_io(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: SignalQualityClassificationConfig) -> None:
    
    # Load the data
    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df_windowed = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    df_sqa = signal_quality_classification(df_windowed, config, path_to_classifier_input)


def estimate_heart_rate(df_sqa: pd.DataFrame, df_ppg_preprocessed: pd.DataFrame, config: HeartRateExtractionConfig) -> pd.DataFrame:  
    """
    Estimate the heart rate from the PPG signal using the time-frequency domain method.

    Parameters
    ----------
    df_sqa : pd.DataFrame
        The DataFrame containing the signal quality assessment predictions.
    df_ppg_preprocessed : pd.DataFrame
        The DataFrame containing the preprocessed PPG signal.
    config : HeartRateExtractionConfig
        The configuration for the heart rate estimation.

    Returns
    -------
    df_hr : pd.DataFrame
        The DataFrame containing the heart rate estimations.
    """

    # Extract NumPy arrays for faster operations
    ppg_post_prob = df_sqa[DataColumns.PRED_SQA_PROBA].to_numpy()
    #acc_label = df_sqa.loc[:, DataColumns.ACCELEROMETER_LABEL].to_numpy() # Adjust later in data columns to get the correct label, should be first intergrated in feature extraction and classification
    ppg_preprocessed = df_ppg_preprocessed.values 
    time_idx = df_ppg_preprocessed.columns.get_loc(DataColumns.TIME) # Get the index of the time column
    ppg_idx = df_ppg_preprocessed.columns.get_loc(DataColumns.PPG) # Get the index of the PPG column
    
    # Assign window-level probabilities to individual samples
    sqa_label = assign_sqa_label(ppg_post_prob, config) # assigns a signal quality label to every individual data point
    v_start_idx, v_end_idx = extract_hr_segments(sqa_label, config.min_hr_samples) # extracts heart rate segments based on the SQA label
    
    v_hr_rel = np.array([])
    t_hr_rel = np.array([])

    edge_add = 2 * config.sampling_frequency  # Add 2s on both sides of the segment for HR estimation
    step_size = config.hr_est_samples  # Step size for HR estimation

    # Estimate the maximum size for preallocation
    valid_segments = (v_start_idx >= edge_add) & (v_end_idx <= len(ppg_preprocessed) - edge_add) # check if the segments are valid, e.g. not too close to the edges (2s)
    valid_start_idx = v_start_idx[valid_segments]   # get the valid start indices
    valid_end_idx = v_end_idx[valid_segments]    # get the valid end indices
    max_size = np.sum((valid_end_idx - valid_start_idx) // step_size) # maximum size for preallocation
  
    # Preallocate arrays
    v_hr_rel = np.empty(max_size, dtype=float) 
    t_hr_rel = np.empty(max_size, dtype=float)  

    # Track current position
    hr_pos = 0

    for start_idx, end_idx in zip(valid_start_idx, valid_end_idx):
        # Extract extended PPG segment
        extended_ppg_segment = ppg_preprocessed[start_idx - edge_add : end_idx + edge_add, ppg_idx]

        # Estimate heart rate
        hr_est = extract_hr_from_segment(
            extended_ppg_segment,
            config.tfd_length,
            config.sampling_frequency,
            config.kern_type,
            config.kern_params,
        )
        n_hr = len(hr_est)  # Number of heart rate estimates
        end_idx_time = n_hr * step_size + start_idx  # Calculate end index for time, different from end_idx since it is always a multiple of step_size, while end_idx is not

        # Extract relative time for HR estimates
        hr_time = ppg_preprocessed[start_idx : end_idx_time : step_size, time_idx]

        # Insert into preallocated arrays
        v_hr_rel[hr_pos:hr_pos + n_hr] = hr_est
        t_hr_rel[hr_pos:hr_pos + n_hr] = hr_time
        hr_pos += n_hr

    df_hr = pd.DataFrame({"rel_time": t_hr_rel, "heart_rate": v_hr_rel})

    return df_hr
