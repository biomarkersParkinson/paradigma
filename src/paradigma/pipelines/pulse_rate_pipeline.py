import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.signal import welch
from scipy.signal.windows import hamming, hann
import tsdf
from typing import List

from paradigma.classification import ClassifierPackage
from paradigma.constants import DataColumns
from paradigma.config import PulseRateConfig
from paradigma.feature_extraction import compute_statistics, compute_signal_to_noise_ratio, compute_auto_correlation, \
    compute_dominant_frequency, compute_relative_power, compute_spectral_entropy
from paradigma.pipelines.pulse_rate_utils import assign_sqa_label, extract_pr_segments, extract_pr_from_segment
from paradigma.segmenting import tabulate_windows, WindowedDataExtractor
from paradigma.util import aggregate_parameter

def extract_signal_quality_features(df_ppg: pd.DataFrame, df_acc: pd.DataFrame, ppg_config: PulseRateConfig, acc_config: PulseRateConfig) -> pd.DataFrame:
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
    ppg_config: PulseRateConfig
        The configuration for the signal quality feature extraction of the PPG signal.
    acc_config: PulseRateConfig
        The configuration for the signal quality feature extraction of the accelerometer signal.

    Returns
    -------
    df_features : pd.DataFrame
        The DataFrame containing the extracted signal quality features.
    
    """
    # Group sequences of timestamps into windows
    ppg_windowed_cols = [DataColumns.TIME, ppg_config.ppg_colname]
    ppg_windowed = tabulate_windows(
        df=df_ppg, 
        columns=ppg_windowed_cols,
        window_length_s=ppg_config.window_length_s,
        window_step_length_s=ppg_config.window_step_length_s,
        fs=ppg_config.sampling_frequency
    )

    # Extract data from the windowed PPG signal
    extractor = WindowedDataExtractor(ppg_windowed_cols)
    idx_time = extractor.get_index(DataColumns.TIME)
    idx_ppg = extractor.get_index(ppg_config.ppg_colname)
    start_time_ppg = np.min(ppg_windowed[:, :, idx_time], axis=1) # Start time of the window is relative to the first datapoint in the PPG data
    ppg_values_windowed = ppg_windowed[:, :, idx_ppg]

    acc_windowed_cols = [DataColumns.TIME] + acc_config.accelerometer_cols
    acc_windowed = tabulate_windows(
        df=df_acc,
        columns=acc_windowed_cols,
        window_length_s=acc_config.window_length_s,
        window_step_length_s=acc_config.window_step_length_s,
        fs=acc_config.sampling_frequency
    )

    # Extract data from the windowed accelerometer signal
    extractor = WindowedDataExtractor(acc_windowed_cols)
    idx_acc = extractor.get_slice(acc_config.accelerometer_cols)
    acc_values_windowed = acc_windowed[:, :, idx_acc]

    df_features = pd.DataFrame(start_time_ppg, columns=[DataColumns.TIME])
    # Compute features of the temporal domain of the PPG signal
    df_temporal_features = extract_temporal_domain_features(ppg_values_windowed, ppg_config, quality_stats=['var', 'mean', 'median', 'kurtosis', 'skewness'])
    
    # Combine temporal features with the start time
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # Compute features of the spectral domain of the PPG signal
    df_spectral_features = extract_spectral_domain_features(ppg_values_windowed, ppg_config)

    # Combine the spectral features with the previously computed temporal features
    df_features = pd.concat([df_features, df_spectral_features], axis=1)
    
    # Compute periodicity feature of the accelerometer signal
    df_accelerometer_feature = extract_accelerometer_feature(acc_values_windowed, ppg_values_windowed, acc_config)

    # Combine the accelerometer feature with the previously computed features
    df_features = pd.concat([df_features, df_accelerometer_feature], axis=1)

    return df_features


def signal_quality_classification(df: pd.DataFrame, config: PulseRateConfig, full_path_to_classifier_package: str | Path) -> pd.DataFrame:
    """
    Classify the signal quality of the PPG signal using a logistic regression classifier. A probability close to 1 indicates a high-quality signal, while a probability close to 0 indicates a low-quality signal.
    The classifier is trained on features extracted from the PPG signal. The features are extracted using the extract_signal_quality_features function.
    The accelerometer signal is used to determine the signal quality based on the power ratio of the accelerometer signal and returns a binary label based on a threshold.
    A value of 1 on the indicates no/minor periodic motion influence of the accelerometer on the PPG signal, 0 indicates major periodic motion influence.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the PPG features and the accelerometer feature for signal quality classification.
    config : PulseRateConfig
        The configuration for the signal quality classification.
    full_path_to_classifier_package : str | Path
        The path to the directory containing the classifier.

    Returns
    -------
    df_sqa pd.DataFrame
        The DataFrame containing the PPG signal quality predictions (both probabilities of the PPG signal quality classification and the accelerometer label based on the threshold).
    """
    clf_package = ClassifierPackage.load(full_path_to_classifier_package)  # Load the classifier package
    clf = clf_package.classifier  # Load the logistic regression classifier

    # Apply scaling to relevant columns
    scaled_features = clf_package.transform_features(df.loc[:, clf.feature_names_in]) # Apply scaling to the features 

    # Make predictions for PPG signal quality assessment, and assign the probabilities to the DataFrame and drop the features
    df[DataColumns.PRED_SQA_PROBA] = clf.predict_proba(scaled_features)[:, 0]
    df[DataColumns.PRED_SQA_ACC_LABEL] = (df[DataColumns.ACC_POWER_RATIO] < config.threshold_sqa_accelerometer).astype(int)  # Assign accelerometer label to the DataFrame based on the threshold
    
    return df[[DataColumns.TIME, DataColumns.PRED_SQA_PROBA, DataColumns.PRED_SQA_ACC_LABEL]]  # Return only the relevant columns, namely the predicted probabilities for the PPG signal quality and the accelerometer label


def estimate_pulse_rate(df_sqa: pd.DataFrame, df_ppg_preprocessed: pd.DataFrame, config: PulseRateConfig) -> pd.DataFrame:  
    """
    Estimate the pulse rate from the PPG signal using the time-frequency domain method.

    Parameters
    ----------
    df_sqa : pd.DataFrame
        The DataFrame containing the signal quality assessment predictions.
    df_ppg_preprocessed : pd.DataFrame
        The DataFrame containing the preprocessed PPG signal.
    config : PulseRateConfig
        The configuration for the pulse rate estimation.

    Returns
    -------
    df_pr : pd.DataFrame
        The DataFrame containing the pulse rate estimations.
    """

    # Extract NumPy arrays for faster operations
    ppg_post_prob = df_sqa[DataColumns.PRED_SQA_PROBA].to_numpy()
    acc_label = df_sqa.loc[:, DataColumns.PRED_SQA_ACC_LABEL].to_numpy() # Adjust later in data columns to get the correct label, should be first intergrated in feature extraction and classification
    ppg_preprocessed = df_ppg_preprocessed.values 
    time_idx = df_ppg_preprocessed.columns.get_loc(DataColumns.TIME) # Get the index of the time column
    ppg_idx = df_ppg_preprocessed.columns.get_loc(DataColumns.PPG) # Get the index of the PPG column
    
    # Assign window-level probabilities to individual samples
    sqa_label = assign_sqa_label(ppg_post_prob, config, acc_label) # assigns a signal quality label to every individual data point
    v_start_idx, v_end_idx = extract_pr_segments(sqa_label, config.min_pr_samples) # extracts pulse rate segments based on the SQA label
    
    v_pr_rel = np.array([])
    t_pr_rel = np.array([])

    edge_add = 2 * config.sampling_frequency  # Add 2s on both sides of the segment for PR estimation
    step_size = config.pr_est_samples  # Step size for PR estimation

    # Estimate the maximum size for preallocation
    valid_segments = (v_start_idx >= edge_add) & (v_end_idx <= len(ppg_preprocessed) - edge_add) # check if the segments are valid, e.g. not too close to the edges (2s)
    valid_start_idx = v_start_idx[valid_segments]   # get the valid start indices
    valid_end_idx = v_end_idx[valid_segments]    # get the valid end indices
    max_size = np.sum((valid_end_idx - valid_start_idx) // step_size) # maximum size for preallocation
  
    # Preallocate arrays
    v_pr_rel = np.empty(max_size, dtype=float) 
    t_pr_rel = np.empty(max_size, dtype=float)  

    # Track current position
    pr_pos = 0

    for start_idx, end_idx in zip(valid_start_idx, valid_end_idx):
        # Extract extended PPG segment
        extended_ppg_segment = ppg_preprocessed[start_idx - edge_add : end_idx + edge_add, ppg_idx]

        # Estimate pulse rate
        pr_est = extract_pr_from_segment(
            extended_ppg_segment,
            config.tfd_length,
            config.sampling_frequency,
            config.kern_type,
            config.kern_params,
        )
        n_pr = len(pr_est)  # Number of pulse rate estimates
        end_idx_time = n_pr * step_size + start_idx  # Calculate end index for time, different from end_idx since it is always a multiple of step_size, while end_idx is not

        # Extract relative time for PR estimates
        pr_time = ppg_preprocessed[start_idx : end_idx_time : step_size, time_idx]

        # Insert into preallocated arrays
        v_pr_rel[pr_pos:pr_pos + n_pr] = pr_est
        t_pr_rel[pr_pos:pr_pos + n_pr] = pr_time
        pr_pos += n_pr

    df_pr = pd.DataFrame({"time": t_pr_rel, "pulse_rate": v_pr_rel})

    return df_pr


def aggregate_pulse_rate(pr_values: np.ndarray, aggregates: List[str] = ['mode', '99p']) -> dict:
    """
    Aggregate the pulse rate estimates using the specified aggregation methods.

    Parameters
    ----------
    pr_values : np.ndarray
        The array containing the pulse rate estimates
    aggregates : List[str]
        The list of aggregation methods to be used for the pulse rate estimates. The default is ['mode', '99p'].

    Returns
    -------
    aggregated_results : dict
        The dictionary containing the aggregated results of the pulse rate estimates.
    """
    # Initialize the dictionary for the aggregated results
    aggregated_results = {}

    # Initialize the dictionary for the aggregated results with the metadata
    aggregated_results = {
    'metadata': {
        'nr_pr_est': len(pr_values)
    },
    'pr_aggregates': {}
}
    for aggregate in aggregates:
        aggregated_results['pr_aggregates'][f'{aggregate}_{DataColumns.PULSE_RATE}'] = aggregate_parameter(pr_values, aggregate)

    return aggregated_results


def extract_temporal_domain_features(
        ppg_windowed: np.ndarray, 
        config: PulseRateConfig, 
        quality_stats: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
    """
    Compute temporal domain features for the ppg signal. The features are added to the dataframe. Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------
    ppg_windowed: np.ndarray
        The dataframe containing the windowed accelerometer signal

    config: PulseRateConfig
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
        feature_dict[stat] = compute_statistics(ppg_windowed, stat, abs_stats=True)
    
    feature_dict['signal_to_noise'] = compute_signal_to_noise_ratio(ppg_windowed)  
    feature_dict['auto_corr'] = compute_auto_correlation(ppg_windowed, config.sampling_frequency)
    return pd.DataFrame(feature_dict)


def extract_spectral_domain_features(
        ppg_windowed: np.ndarray,
        config: PulseRateConfig, 
    ) -> pd.DataFrame:
    """
    Calculate the spectral features (dominant frequency, relative power, and spectral entropy)
    for each segment of a PPG signal using a single Welch's method computation. The features are added to the dataframe. 
    Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------
    ppg_windowed: np.ndarray
        The dataframe containing the windowed ppg signal

    config: PulseRateConfig
        The configuration object containing the parameters for the feature extraction

    Returns
    -------
    pd.DataFrame
        The dataframe with the added spectral domain features.
    """
    d_features = {}

    window = hamming(config.window_length_welch, sym = True)

    n_samples_window = ppg_windowed.shape[1]

    freqs, psd = welch(
        ppg_windowed,
        fs=config.sampling_frequency,
        window=window,
        noverlap=config.overlap_welch_window,
        nfft=max(256, 2 ** int(np.log2(n_samples_window))),
        detrend=False,
        axis=1
    )

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
        config: PulseRateConfig
    ) -> pd.DataFrame:
    """
    Extract accelerometer features from the accelerometer signal in the PPG frequency range.
    
    Parameters
    ----------    
    acc_windowed: np.ndarray
        The dataframe containing the windowed accelerometer signal

    ppg_windowed: np.ndarray
        The dataframe containing the corresponding windowed ppg signal
    
    config: PulseRateConfig
        The configuration object containing the parameters for the feature extraction

    Returns
    -------
    pd.DataFrame
        The dataframe with the relative power accelerometer feature.
    """
    
    if config.sensor not in ['imu', 'ppg']:
        raise ValueError("Sensor not recognized.")
    
    d_freq = {}
    d_psd = {}
    for sensor in ['imu', 'ppg']:
        config.set_sensor(sensor)

        if sensor == 'imu':
            windows = acc_windowed
        else:
            windows = ppg_windowed

        window_type = hann(config.window_length_welch, sym = True)
        d_freq[sensor], d_psd[sensor] = welch(
            windows,
            fs=config.sampling_frequency,
            window=window_type,
            noverlap=config.overlap_welch_window,
            nfft=config.nfft,
            detrend=False,
            axis=1
        )

    d_psd['imu'] = np.sum(d_psd['imu'], axis=2)  # Sum the PSDs of the three axes

    acc_power_ratio = extract_acc_power_feature(d_freq['imu'], d_psd['imu'], d_freq['ppg'], d_psd['ppg'])

    return pd.DataFrame(acc_power_ratio, columns=['acc_power_ratio'])


