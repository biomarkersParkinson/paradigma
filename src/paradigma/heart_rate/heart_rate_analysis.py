from typing import Union
from pathlib import Path

import pandas as pd
import os
import numpy as np
import numpy as np

import tsdf
import tsdf.constants 
from paradigma.config import SignalQualityFeatureExtractionConfig, SignalQualityClassificationConfig, HeartRateExtractionConfig, HeartRateExtractionConfig
from paradigma.util import read_metadata
from paradigma.segmenting import tabulate_windows, tabulate_windows_legacy
from paradigma.heart_rate.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features, extract_temporal_domain_features_numpy, extract_spectral_domain_features_numpy
from paradigma.heart_rate.heart_rate_estimation import assign_sqa_label, extract_hr_segments, extract_hr_from_segment
from paradigma.constants import DataColumns

def extract_signal_quality_features(df: pd.DataFrame, config: SignalQualityFeatureExtractionConfig) -> pd.DataFrame:
    # Group sequences of timestamps into windows
    df_windowed = tabulate_windows_legacy(config, df)

    # Compute statistics of the temporal domain signals
    df_windowed = extract_temporal_domain_features(config, df_windowed, quality_stats=['var', 'mean', 'median', 'kurtosis', 'skewness'])
    
    # Compute statistics of the spectral domain signals
    df_windowed = extract_spectral_domain_features(config, df_windowed)

    df_windowed = df_windowed.drop(columns = ['green'])  # Drop the values channel since it is no longer needed
    return df_windowed

def extract_signal_quality_features_numpy(df: pd.DataFrame, config: SignalQualityFeatureExtractionConfig) -> pd.DataFrame:
    # Group sequences of timestamps into windows
    ppg_windowed = tabulate_windows(config, df, columns=[config.ppg_colname])[0]

    # Compute statistics of the temporal domain signals
    df_temporal_features = extract_temporal_domain_features_numpy(config, ppg_windowed, quality_stats=['var', 'mean', 'median', 'kurtosis', 'skewness'])
    
    # Compute statistics of the spectral domain signals
    df_spectral_features = extract_spectral_domain_features_numpy(config, ppg_windowed)

    return pd.concat([df_temporal_features, df_spectral_features], axis=1)


def extract_signal_quality_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: SignalQualityFeatureExtractionConfig) -> None:
    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)
    
    # Extract signal quality features
    # Extract signal quality features
    df_windowed = extract_signal_quality_features(df, config)
    return df_windowed


def signal_quality_classification(df_windowed: pd.DataFrame, config: SignalQualityClassificationConfig, path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:
    """
    Classify the signal quality of the PPG signal using a logistic regression classifier.
    The classifier is trained on features extracted from the PPG signal.
    The features are extracted using the extract_signal_quality_features function.

    Parameters
    ----------
    df_windowed_windowed : pd.DataFrame
        The DataFrame containing the PPG signal features features.
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

    # Prepare the data
    lr_clf.feature_names_in_ = ['var', 'mean', 'median', 'kurtosis', 'skewness', 'f_dom', 'rel_power', 'spectral_entropy', 'signal_to_noise', 'auto_corr']
    X = df_windowed.loc[:, lr_clf.feature_names_in_]

    # Normalize features using mu and sigma
    X_normalized = (X[lr_clf.feature_names_in_] - mu.ravel()) / sigma.ravel()  # Use .ravel() to convert the 2D arrays (mu and sigma) to 1D arrays

    # Make predictions for PPG signal quality assessment, and assign the probabilities to the DataFrame and drop the features
    df_windowed[DataColumns.PRED_SQA_PROBA] = lr_clf.predict_proba(X_normalized)[:, 0]
    df_sqa = df_windowed.drop(columns=lr_clf.feature_names_in_)
    
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
       
    # Assign window-level probabilities to individual samples
    ppg_post_prob = df_sqa.loc[:, DataColumns.PRED_SQA_PROBA].to_numpy()
    #acc_label = df_sqa.loc[:, DataColumns.ACCELEROMETER_LABEL].to_numpy() # Adjust later in data columns to get the correct label, should be first intergrated in feature extraction and classification

    sqa_label = assign_sqa_label(ppg_post_prob, config) # assigns a signal quality label to every individual data point
    v_start_idx, v_end_idx = extract_hr_segments(sqa_label, config.min_hr_samples) # extracts heart rate segments based on the SQA label
    
    fs = config.sampling_frequency

    v_hr_rel = np.array([])
    t_hr_rel = np.array([])

    edge_add = 2 * fs  # Add 2s on both sides of the segment for HR estimation

    # Estimate the maximum size for preallocation
    valid_segments = (v_start_idx >= edge_add) & (v_end_idx <= len(df_ppg_preprocessed) - edge_add) # check if the segments are valid, e.g. not too close to the edges (2s)
    valid_start_idx = v_start_idx[valid_segments]   # get the valid start indices
    valid_end_idx = v_end_idx[valid_segments]    # get the valid end indices
    max_size = np.sum((valid_end_idx - valid_start_idx) // config.hr_est_samples) # maximum size for preallocation
  
    # Preallocate arrays
    v_hr_rel = np.empty(max_size, dtype=float) 
    t_hr_rel = np.empty(max_size, dtype=float)  

    # Track current position
    hr_pos = 0

    for start_idx, end_idx in zip(valid_start_idx, valid_end_idx):        
        # Extract the extended PPG segment for HR estimation
        extended_ppg_segment = df_ppg_preprocessed[DataColumns.PPG][start_idx - edge_add : end_idx + edge_add]

        # Perform HR estimation
        hr_est = extract_hr_from_segment(extended_ppg_segment, config.tfd_length, fs, config.kern_type, config.kern_params)
        n_hr = len(hr_est)  # Number of HR estimates
        # Generate HR estimation time array
        rel_segment_time = df_ppg_preprocessed[DataColumns.TIME][start_idx:end_idx].values # relative time in seconds after the start of the segment of each sample
        # Extract relative time for each HR estimation point, taking every `hr_est_samples`-th sample
        hr_time = rel_segment_time[:n_hr * config.hr_est_samples : config.hr_est_samples] # relative time extracted for the HR estimation points, every `hr_est_samples`-th sample


        # Insert results into preallocated arrays
        v_hr_rel[hr_pos:hr_pos + n_hr] = hr_est
        t_hr_rel[hr_pos:hr_pos + n_hr] = hr_time  # Use hr_pos to track both arrays simultaneously
        
        # Update current position for the next iteration
        hr_pos += n_hr

    df_hr = pd.DataFrame({"rel_time": t_hr_rel, "heart_rate": v_hr_rel})

    return df_hr
