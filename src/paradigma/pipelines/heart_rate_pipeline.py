import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.signal import welch
from scipy.signal.windows import hamming, hann
import tsdf
from typing import List

from paradigma.constants import DataColumns
from paradigma.config import SignalQualityFeatureExtractionConfig, SignalQualityFeatureExtractionAccConfig, \
    SignalQualityClassificationConfig, HeartRateExtractionConfig
from paradigma.feature_extraction import compute_auto_correlation, compute_dominant_frequency, compute_relative_power, \
    compute_spectral_entropy, compute_statistics, compute_signal_to_noise_ratio
from paradigma.pipelines.heart_rate_utils import assign_sqa_label, extract_hr_segments, extract_hr_from_segment
from paradigma.segmenting import tabulate_windows, WindowedDataExtractor
from paradigma.util import read_metadata, aggregate_parameter


def extract_signal_quality_features(
        config_ppg: SignalQualityFeatureExtractionConfig, 
        df_ppg: pd.DataFrame, 
        config_acc: SignalQualityFeatureExtractionAccConfig, 
        df_acc: pd.DataFrame
    ) -> pd.DataFrame:
    """	
    Extract signal quality features from the PPG signal.
    The features are extracted from the temporal and spectral domain of the PPG signal.
    The temporal domain features include variance, mean, median, kurtosis, skewness, signal-to-noise ratio, and autocorrelation.
    The spectral domain features include the dominant frequency, relative power, spectral entropy.

    Parameters
    ----------
    config_ppg: SignalQualityFeatureExtractionConfig
        The configuration for the signal quality feature extraction of the ppg signal.
    df_ppg : pd.DataFrame
        The DataFrame containing the PPG signal.
    config_acc: SignalQualityFeatureExtractionAccConfig
        The configuration for the signal quality feature extraction of the accelerometer signal.
    df_acc : pd.DataFrame
        The DataFrame containing the accelerometer signal.

    Returns
    -------
    df_features : pd.DataFrame
        The DataFrame containing the extracted signal quality features.
    
    """
    # Group sequences of timestamps into windows
    ppg_windowed = tabulate_windows(
        df=df_ppg, 
        columns=[config_ppg.ppg_colname],
        window_length_s=config_ppg.window_length_s,
        window_step_length_s=config_ppg.window_step_length_s,
        fs=config_ppg.sampling_frequency
    )[0]

    acc_windowed_cols = [DataColumns.TIME] + config_acc.accelerometer_cols
    acc_windowed = tabulate_windows(
        df=df_acc,
        columns=acc_windowed_cols,
        window_length_s=config_acc.window_length_s,
        window_step_length_s=config_acc.window_step_length_s,
        fs=config_acc.sampling_frequency
    )

    extractor = WindowedDataExtractor(acc_windowed_cols)
    idx_time = extractor.get_index(DataColumns.TIME)
    idx_acc = extractor.get_slice(config_acc.accelerometer_cols)

    # Extract data
    start_time = np.min(acc_windowed[:, :, idx_time], axis=1)
    acc_values_windowed = acc_windowed[:, :, idx_acc]

    df_features = pd.DataFrame(start_time, columns=[DataColumns.TIME])
    # Compute features of the temporal domain of the PPG signal
    df_temporal_features = extract_temporal_domain_features(ppg_windowed, config_ppg, quality_stats=['var', 'mean', 'median', 'kurtosis', 'skewness'])
    
    # Combine temporal features with the start time
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # Compute features of the spectral domain of the PPG signal
    df_spectral_features = extract_spectral_domain_features(ppg_windowed, config_ppg)

    # Combine the spectral features with the previously computed temporal features
    df_features = pd.concat([df_features, df_spectral_features], axis=1)
    
    # Compute periodicity feature of the accelerometer signal
    df_accelerometer_feature = extract_accelerometer_feature(acc_values_windowed, ppg_windowed, config_acc)

    # Combine the accelerometer feature with the previously computed features
    df_features = pd.concat([df_features, df_accelerometer_feature], axis=1)

    return df_features


def extract_signal_quality_features_io(input_path: str | Path, output_path: str | Path, config_ppg: SignalQualityFeatureExtractionConfig, config_acc: SignalQualityFeatureExtractionAccConfig) -> pd.DataFrame:
    """
    Extract signal quality features from the PPG signal and save them to a file.

    Parameters
    ----------
    input_path : str | Path
        The path to the directory containing the preprocessed PPG and accelerometer data.
    output_path : str | Path
        The path to the directory where the extracted features will be saved.
    config_ppg: SignalQualityFeatureExtractionConfig
        The configuration for the signal quality feature extraction of the ppg signal.
    config_acc: SignalQualityFeatureExtractionAccConfig
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


def signal_quality_classification(
        df: pd.DataFrame, 
        config: SignalQualityClassificationConfig,
        path_to_classifier_input: str | Path
    ) -> pd.DataFrame:
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
    path_to_classifier_input : str | Path
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
    
    return df[[DataColumns.PRED_SQA_PROBA, DataColumns.PRED_SQA_ACC_LABEL]]  # Return only the relevant columns, namely the predicted probabilities for the PPG signal quality and the accelerometer label


def signal_quality_classification_io(
        input_path: str | Path, 
        output_path: str | Path, 
        path_to_classifier_input: str | Path, 
        config: SignalQualityClassificationConfig
    ) -> None:
    
    # Load the data
    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df_windowed = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    df_sqa = signal_quality_classification(df_windowed, config, path_to_classifier_input)


def estimate_heart_rate(
        df_sqa: pd.DataFrame, 
        df_ppg_preprocessed: pd.DataFrame, 
        config: HeartRateExtractionConfig
    ) -> pd.DataFrame:  
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

    df_hr = pd.DataFrame({"time": t_hr_rel, "heart_rate": v_hr_rel})

    return df_hr


def aggregate_heart_rate(hr_values: np.ndarray, aggregates: List[str] = ['mode', '99p']) -> dict:
    """
    Aggregate the heart rate estimates using the specified aggregation methods.

    Parameters
    ----------
    hr_values : np.ndarray
        The array containing the heart rate estimates
    aggregates : List[str]
        The list of aggregation methods to be used for the heart rate estimates. The default is ['mode', '99p'].

    Returns
    -------
    aggregated_results : dict
        The dictionary containing the aggregated results of the heart rate estimates.
    """
    # Initialize the dictionary for the aggregated results
    aggregated_results = {}

    # Initialize the dictionary for the aggregated results with the metadata
    aggregated_results = {
    'metadata': {
        'nr_hr_est': len(hr_values)
    },
    'hr_aggregates': {}
}
    for aggregate in aggregates:
        aggregated_results['hr_aggregates'][f'{aggregate}_{DataColumns.HEART_RATE}'] = aggregate_parameter(hr_values, aggregate)

    return aggregated_results


def aggregate_heart_rate_io(
        full_path_to_input: str | Path, 
        full_path_to_output: str | Path, 
        aggregates: List[str] = ['mode', '99p']
    ) -> None:
    """
    Extract heart rate from the PPG signal and save the aggregated heart rate estimates to a file.

    Parameters
    ----------
    input_path : str | Path
        The path to the directory containing the heart rate estimates.
    output_path : str | Path
        The path to the directory where the aggregated heart rate estimates will be saved.
    aggregates : List[str]
        The list of aggregation methods to be used for the heart rate estimates. The default is ['mode', '99p'].
    """

    # Load the heart rate estimates
    with open(full_path_to_input, 'r') as f:
        df_hr = json.load(f)
    
    # Aggregate the heart rate estimates
    hr_values = df_hr['heart_rate'].values
    df_hr_aggregates = aggregate_heart_rate(hr_values, aggregates)

    # Save the aggregated heart rate estimates
    with open(full_path_to_output, 'w') as json_file:
        json.dump(df_hr_aggregates, json_file, indent=4)


def extract_temporal_domain_features(
        ppg_windowed: np.ndarray, 
        config: SignalQualityFeatureExtractionConfig, 
        quality_stats: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
    """
    Compute temporal domain features for the ppg signal. The features are added to the dataframe. Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------
    ppg_windowed: np.ndarray
        The dataframe containing the windowed accelerometer signal

    config: SignalQualityFeatureExtractionConfig
        The configuration object containing the parameters for the feature extraction

    quality_stats: list, optional
        The statistics to be computed for the gravity component of the accelerometer signal (default: ['mean', 'std'])
    
    Returns
    -------
    pd.DataFrame
        The dataframe with the added temporal domain features.
    """
    
    feature_dict = {}
    for stat in quality_stats:
        feature_dict[stat] = compute_statistics(ppg_windowed, stat)
    
    feature_dict['signal_to_noise'] = compute_signal_to_noise_ratio(ppg_windowed)  
    feature_dict['auto_corr'] = compute_auto_correlation(ppg_windowed, config.sampling_frequency)
    return pd.DataFrame(feature_dict)


def extract_spectral_domain_features(
        ppg_windowed: np.ndarray,
        config: SignalQualityFeatureExtractionConfig, 
    ) -> pd.DataFrame:
    """
    Calculate the spectral features (dominant frequency, relative power, and spectral entropy)
    for each segment of a PPG signal using a single Welch's method computation. The features are added to the dataframe. 
    Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------
    ppg_windowed: np.ndarray
        The dataframe containing the windowed ppg signal

    config: SignalQualityFeatureExtractionConfig
        The configuration object containing the parameters for the feature extraction

    Returns
    -------
    pd.DataFrame
        The dataframe with the added spectral domain features.
    """
    d_features = {}

    window = hamming(config.window_length_welch, sym = True)

    freqs, psd = welch(
        ppg_windowed,
        fs=config.sampling_frequency,
        window=window,
        noverlap=config.overlap_welch_window,
        nfft=max(256, 2 ** int(np.log2(ppg_windowed.shape[1]))),
        detrend=False,
        axis=1
    )

    n_samples_window = ppg_windowed.shape[1]

    # Calculate each feature using the computed PSD and frequency array
    d_features['f_dom'] = compute_dominant_frequency(freqs, psd)
    d_features['rel_power'] = compute_relative_power(freqs, psd, config)
    d_features['spectral_entropy'] = compute_spectral_entropy(psd, n_samples_window) 

    return pd.DataFrame(d_features)


def extract_acc_power_feature(
        f1: np.ndarray, 
        PSD_acc: np.ndarray, 
        f2: np.ndarray, 
        PSD_ppg: np.ndarray
    ) -> np.ndarray:
    """
    Extract the accelerometer power feature in the PPG frequency range.

    Parameters
    ----------
    f1: np.ndarray
        The frequency bins of the accelerometer signal.
    PSD_acc: np.ndarray
        The power spectral density of the accelerometer signal.
    f2: np.ndarray
        The frequency bins of the PPG signal.
    PSD_ppg: np.ndarray
        The power spectral density of the PPG signal.

    Returns
    -------
    np.ndarray
        The accelerometer power feature in the PPG frequency range
    """
    
    # Find the index of the maximum PSD value in the PPG signal
    max_PPG_psd_idx = np.argmax(PSD_ppg, axis=1)
    max_PPG_freq_psd = f2[max_PPG_psd_idx]
    
    # Find the neighboring indices of the maximum PSD value in the PPG signal
    df_idx = np.column_stack((max_PPG_psd_idx - 1, max_PPG_psd_idx, max_PPG_psd_idx + 1))    
    
    # Find the index of the closest frequency in the accelerometer signal to the first harmonic of the PPG frequency
    corr_acc_psd_fh_idx = np.argmin(np.abs(f1[:, None] - max_PPG_freq_psd*2), axis=0)
    fh_idx = np.column_stack((corr_acc_psd_fh_idx - 1, corr_acc_psd_fh_idx, corr_acc_psd_fh_idx + 1))   
    
    # Compute the power in the ranges corresponding to the PPG frequency
    acc_power_PPG_range = (
        np.trapz(PSD_acc[np.arange(PSD_acc.shape[0])[:, None], df_idx], f1[df_idx], axis=1) +
        np.trapz(PSD_acc[np.arange(PSD_acc.shape[0])[:, None], fh_idx], f1[fh_idx], axis=1)
    )

    # Compute the total power across the entire frequency range
    acc_power_total = np.trapz(PSD_acc, f1)
    
    # Compute the power ratio of the accelerometer signal in the PPG frequency range
    acc_power_ratio = acc_power_PPG_range / acc_power_total
    
    return acc_power_ratio

def extract_accelerometer_feature(
        acc_windowed: np.ndarray, 
        ppg_windowed: np.ndarray,
        config: SignalQualityFeatureExtractionAccConfig
    ) -> pd.DataFrame:
    """
    Extract accelerometer features from the accelerometer signal in the PPG frequency range.
    
    Parameters
    ----------    
    acc_windowed: np.ndarray
        The dataframe containing the windowed accelerometer signal

    ppg_windowed: np.ndarray
        The dataframe containing the corresponding windowed ppg signal
    
    config: SignalQualityFeatureExtractionAccConfig
        The configuration object containing the parameters for the feature extraction

    Returns
    -------
    pd.DataFrame
        The dataframe with the relative power accelerometer feature.
    """
    
    d_acc_feature = {}

    window_acc = hann(config.window_length_welch_acc, sym = True)
    window_ppg = hann(config.window_length_welch_ppg, sym = True)

    freqs_acc, psd_acc = welch(
        acc_windowed,
        fs=config.sampling_frequency,
        window=window_acc,
        noverlap=config.overlap_welch_window_acc,
        nfft=config.nfft_acc,
        detrend=False,
        axis=1
    )

    psd_acc = np.sum(psd_acc, axis=2)  # Sum the PSDs of the three axes

    freqs_ppg, psd_ppg = welch(
        ppg_windowed,
        fs=config.sampling_frequency_ppg,
        window=window_ppg,
        noverlap=config.overlap_welch_window_ppg,
        nfft=config.nfft_ppg,
        detrend=False,
        axis=1
    )

    d_acc_feature['acc_power_ratio'] = extract_acc_power_feature(freqs_acc, psd_acc, freqs_ppg, psd_ppg)

    return pd.DataFrame(d_acc_feature)


