from typing import Union
from pathlib import Path
import pandas as pd
import os
import numpy as np

import tsdf

from paradigma.constants import DataColumns
from paradigma.config import SignalQualityFeatureExtractionConfig, SignalQualityFeatureExtractionConfigAcc, SignalQualityClassificationConfig, \
    HeartRateExtractionConfig, HeartRateExtractionConfig
from paradigma.heart_rate.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features, extract_accelerometer_feature
from paradigma.heart_rate.heart_rate_estimation import assign_sqa_label, extract_hr_segments, extract_hr_from_segment
from paradigma.segmenting import tabulate_windows

from paradigma.util import read_metadata, WindowedDataExtractor

def extract_signal_quality_features(config_ppg: SignalQualityFeatureExtractionConfig, df_ppg: pd.DataFrame, config_acc: SignalQualityFeatureExtractionConfigAcc, df_acc: pd.DataFrame) -> pd.DataFrame:
    """	
    Extract signal quality features from the PPG signal.
    The features are extracted from the temporal and spectral domain of the PPG signal.
    The temporal domain features include variance, mean, median, kurtosis, skewness, signal-to-noise ratio, and autocorrelation.
    The spectral domain features include the dominant frequency, relative power, spectral entropy.

    Parameters
    ----------
    df_ppg : pd.DataFrame
        The DataFrame containing the PPG signal.
    df_acc : pd.DataFrame
        The DataFrame containing the accelerometer signal.
    config_ppg: SignalQualityFeatureExtractionConfig
        The configuration for the signal quality feature extraction of the ppg signal.
    config_acc: SignalQualityFeatureExtractionConfigAcc
        The configuration for the signal quality feature extraction of the accelerometer signal.

    Returns
    -------
    df_features : pd.DataFrame
        The DataFrame containing the extracted signal quality features.
    
    """
    # Group sequences of timestamps into windows
    ppg_windowed = tabulate_windows(config_ppg, df_ppg, columns=[config_ppg.ppg_colname])[0]
    acc_windowed_cols = [DataColumns.TIME] + config_acc.accelerometer_cols
    acc_windowed = tabulate_windows(config_acc, df_acc, acc_windowed_cols)

    extractor = WindowedDataExtractor(acc_windowed_cols)
    idx_time = extractor.get_index(DataColumns.TIME)
    idx_acc = extractor.get_slice(config_acc.accelerometer_cols)

    # Extract data
    start_time = np.min(acc_windowed[:, :, idx_time], axis=1)
    acc_values_windowed = acc_windowed[:, :, idx_acc]

    df_features = pd.DataFrame(start_time, columns=[DataColumns.TIME])
    # Compute features of the temporal domain of the PPG signal
    df_temporal_features = extract_temporal_domain_features(config_ppg, ppg_windowed, quality_stats=['var', 'mean', 'median', 'kurtosis', 'skewness'])
    
    # Combine temporal features with the start time
    df_features= pd.concat([df_features, df_temporal_features], axis=1)

    # Compute features of the spectral domain of the PPG signal
    df_spectral_features = extract_spectral_domain_features(config_ppg, ppg_windowed)

    # Combine the spectral features with the previously computed temporal features
    df_features = pd.concat([df_features, df_spectral_features], axis=1)
    
    # Compute periodicity feature of the accelerometer signal
    df_accelerometer_feature = extract_accelerometer_feature(config_acc, acc_values_windowed, ppg_windowed)

    # Combine the accelerometer feature with the previously computed features
    df_features = pd.concat([df_features, df_accelerometer_feature], axis=1)

    return df_features


def extract_signal_quality_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config_ppg: SignalQualityFeatureExtractionConfig, config_acc: SignalQualityFeatureExtractionConfigAcc) -> None:
    """
    Extract signal quality features from the PPG signal and save them to a file.

    Parameters
    ----------
    input_path : Union[str, Path]
        The path to the directory containing the preprocessed PPG and accelerometer data.
    output_path : Union[str, Path]
        The path to the directory where the extracted features will be saved.
    config_ppg: SignalQualityFeatureExtractionConfig
        The configuration for the signal quality feature extraction of the ppg signal.
    config_acc: SignalQualityFeatureExtractionConfigAcc
        The configuration for the signal quality feature extraction of the accelerometer signal.

    Returns
    -------
    df_windowed : pd.DataFrame
        The DataFrame containing the extracted signal quality features.

    """	
    # Load PPG data
    metadata_time, metadata_values = read_metadata(input_path, config_ppg.meta_filename, config_ppg.time_filename, config_ppg.values_filename)
    df_ppg = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)
    
    # Load IMU data
    metadata_time, metadata_values = read_metadata(input_path, config_acc.meta_filename, config_acc.time_filename, config_acc.values_filename)
    df_acc = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Extract signal quality features
    df_windowed = extract_signal_quality_features(config_ppg, df_ppg, config_acc, df_acc)
    
    # Save the extracted features
    #TO BE ADDED
    return df_windowed


def signal_quality_classification(df: pd.DataFrame, config: SignalQualityClassificationConfig, path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:
    """
    Classify the signal quality of the PPG signal using a logistic regression classifier. A probability close to 1 indicates a high-quality signal, while a probability close to 0 indicates a low-quality signal.
    The classifier is trained on features extracted from the PPG signal. The features are extracted using the extract_signal_quality_features function.
    The accelerometer signal is used to determine the signal quality based on the power ratio of the accelerometer signal and returns a binary label based on a threshold.
    A value of 1 on the indicates no/minor periodic motion influence of the accelerometer on the PPG signal, 0 indicates major periodic motion influence.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the PPG features and the accelerometer feature for signal quality classification.
    config : SignalQualityClassificationConfig
        The configuration for the signal quality classification.
    path_to_classifier_input : Union[str, Path]
        The path to the directory containing the classifier.

    Returns
    -------
    df_sqa pd.DataFrame
        The DataFrame containing the PPG signal quality predictions (both probabilities of the PPG signal quality classification and the accelerometer label based on the threshold).
    """
    
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', config.classifier_file_name))
    lr_clf = clf['model']  # Load the logistic regression classifier
    mu = clf['mu']  # load the mean, 2D array
    sigma = clf['sigma'] # load the standard deviation, 2D array

    with open(os.path.join(path_to_classifier_input, 'thresholds', config.threshold_file_name), 'r') as file:
        acc_threshold = float(file.read())  # Load the accelerometer threshold from the file

    # Assign feature names to the classifier
    lr_clf.feature_names_in_ = ['var', 'mean', 'median', 'kurtosis', 'skewness', 'f_dom', 'rel_power', 'spectral_entropy', 'signal_to_noise', 'auto_corr']

    # Normalize features using mu and sigma
    X_normalized = (df[lr_clf.feature_names_in_] - mu.ravel()) / sigma.ravel()  # Use .ravel() to convert the 2D arrays (mu and sigma) to 1D arrays

    # Make predictions for PPG signal quality assessment, and assign the probabilities to the DataFrame and drop the features
    df[DataColumns.PRED_SQA_PROBA] = lr_clf.predict_proba(X_normalized)[:, 0]
    df[DataColumns.PRED_SQA_ACC_LABEL] = (df[DataColumns.ACC_POWER_RATIO] < acc_threshold).astype(int)  # Assign accelerometer label to the DataFrame based on the threshold
    df_sqa = df[[DataColumns.PRED_SQA_PROBA, DataColumns.PRED_SQA_ACC_LABEL]]  # Select the relevant columns, namely the predicted probabilities for the PPG signal quality and the accelerometer label
    
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
    acc_label = df_sqa.loc[:, DataColumns.PRED_SQA_ACC_LABEL].to_numpy() # Adjust later in data columns to get the correct label, should be first intergrated in feature extraction and classification
    ppg_preprocessed = df_ppg_preprocessed.values 
    time_idx = df_ppg_preprocessed.columns.get_loc(DataColumns.TIME) # Get the index of the time column
    ppg_idx = df_ppg_preprocessed.columns.get_loc(DataColumns.PPG) # Get the index of the PPG column
    
    # Assign window-level probabilities to individual samples
    sqa_label = assign_sqa_label(ppg_post_prob, config, acc_label) # assigns a signal quality label to every individual data point
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
