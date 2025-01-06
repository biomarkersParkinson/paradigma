from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.signal.windows import hamming, hann
from scipy.stats import kurtosis, skew
from paradigma.config import SignalQualityFeatureExtractionConfig, SignalQualityFeatureExtractionConfigAcc

def generate_statistics(
        data: np.ndarray,
        statistic: str
    ) -> np.ndarray:
    """Generate statistics for a single sensor and axis. The function is used with the apply function in pandas.
    
    Parameters
    ----------
    data: pd.np.ndarray
        The sensor column to be aggregated (e.g. x-axis of accelerometer)
    statistic: str
        The statistic to be computed ['mean', 'var', 'median', 'kurtosis', 'skewness']
        
    Returns
    -------
    list
        the statistic for the sensor segments
    """
    if statistic == 'mean':
        return np.mean(np.abs(data), axis=1)
    elif statistic == 'var':
        return np.var(data, ddof=1, axis=1)  # ddof=1 for unbiased variance is used, same as matlab
    elif statistic == 'median':
        return np.median(np.abs(data), axis=1)
    elif statistic == 'kurtosis':
        return kurtosis(data, fisher=False, axis=1) # Fisher's definition of kurtosis is used, same as matlab
    elif statistic == 'skewness':
        return skew(data, axis=1)
    else:
        raise ValueError("Statistic not recognized.")

def compute_signal_to_noise_ratio(
        ppg_windowed: np.ndarray
    ) -> list:
    """
    Compute the signal to noise ratio of the PPG signal.
    
    Args:
    ppg_segments: PPG signal of shape ...
    
    Returns:
    list: Signal to noise ratio of the PPG windows.
    """
    
    arr_signal = np.var(ppg_windowed, axis=1)
    arr_noise = np.var(np.abs(ppg_windowed), axis=1)
    signal_to_noise_ratio = arr_signal / arr_noise
    
    return signal_to_noise_ratio

def compute_auto_correlation(
        ppg_windowed: np.ndarray, 
        fs: int
    ) -> np.ndarray:
    """
    Compute the autocorrelation of the PPG signal.
    
    Args:
        ppg_segments: 2D array where each row is a segment of the PPG signal.
        fs (int): Sampling frequency of the PPG signal.
    
    
    Returns:
        list: Autocorrelation of the PPG segments.
    """

    auto_correlations = biased_autocorrelation(ppg_windowed, fs*3)
    peaks = [find_peaks(x, height=0.01)[0] for x in auto_correlations]
    sorted_peak_values = [np.sort(auto_correlations[i, indices])[::-1] for i, indices in enumerate(peaks)]
    auto_correlations = [x[0] if len(x) > 0 else 0 for x in sorted_peak_values]

    return np.asarray(auto_correlations)

def biased_autocorrelation(
        ppg_windowed: np.ndarray, 
        max_lag: int
    ) -> np.ndarray:
    """
    Compute the biased autocorrelation of a signal (similar to matlabs autocorr function), where the normalization factor 
    is the length of the original signal, and boundary effects are considered.
    
    Args:
        x: Input signal (1D array).
        max_lag (int): Maximum lag to compute autocorrelation.
    
    Returns:
        np.ndarray: Biased autocorrelation values for lags 0 to max_lag.
    """
    zero_mean_ppg = ppg_windowed - np.mean(ppg_windowed, axis=1, keepdims=True) # Remove the mean of the signal to make it zero-mean
    N = zero_mean_ppg.shape[1]
    autocorr_values = np.zeros((zero_mean_ppg.shape[0], max_lag + 1))
    
    for lag in range(max_lag + 1):
        # Compute autocorrelation for current lag
        overlapping_points = zero_mean_ppg[:, :N-lag] * zero_mean_ppg[:, lag:]
        autocorr_values[:, lag] = np.sum(overlapping_points, axis=1) / N  # Divide by N (biased normalization)
    
    return autocorr_values/autocorr_values[:, 0, np.newaxis] # Normalize the autocorrelation values

def compute_dominant_frequency(
        freqs: np.ndarray, 
        psd: np.ndarray
    ) -> float:
    """
    Identify the dominant frequency (peak frequency) in the power spectral density.
    """
    peak_idx = np.argmax(psd, axis=1)
    return freqs[peak_idx]

def compute_relative_power(
        freqs: np.ndarray, 
        psd: np.ndarray, 
        config
    ) -> float:
    """
    Calculate relative power within the dominant frequency band in the physiological range (0.75 - 3 Hz).
    """
    hr_range_mask = (freqs >= config.freq_band_physio[0]) & (freqs <= config.freq_band_physio[1])
    hr_range_idx = np.where(hr_range_mask)[0]
    peak_idx = np.argmax(psd[:, hr_range_idx], axis=1)
    peak_freqs = freqs[hr_range_idx[peak_idx]]

    dom_band_idx = [np.where((freqs >= peak_freq - config.bandwidth) & (freqs <= peak_freq + config.bandwidth))[0] for peak_freq in peak_freqs]
    rel_power = [np.trapz(psd[j, idx], freqs[idx]) / np.trapz(psd[j, :], freqs) for j, idx in enumerate(dom_band_idx)]
    return rel_power

def compute_spectral_entropy(
        psd: np.ndarray, 
        n_samples: int
    ) -> float:
    """
    Calculate the spectral entropy from the normalized power spectral density.
    """
    psd_norm = psd / np.sum(psd, axis=1, keepdims=True)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm), axis=1) / np.log2(n_samples)
    return spectral_entropy

def extract_temporal_domain_features(
        config: SignalQualityFeatureExtractionConfig, 
        ppg_windowed: np.ndarray, 
        quality_stats: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
    """
    Compute temporal domain features for the ppg signal. The features are added to the dataframe. Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------

    config: SignalQualityFeatureExtractionConfig
        The configuration object containing the parameters for the feature extraction
    
    ppg_windowed: pd.DataFrame
        The dataframe containing the windowed accelerometer signal

    quality_stats: list, optional
        The statistics to be computed for the gravity component of the accelerometer signal (default: ['mean', 'std'])
    
    Returns
    -------
    pd.DataFrame
        The dataframe with the added temporal domain features.
    """
    
    feature_dict = {}
    for stat in quality_stats:
        feature_dict[stat] = generate_statistics(ppg_windowed, stat)
    
    feature_dict['signal_to_noise'] = compute_signal_to_noise_ratio(ppg_windowed)  # feature 9
    feature_dict['auto_corr'] = compute_auto_correlation(ppg_windowed, config.sampling_frequency) # feature 10
    return pd.DataFrame(feature_dict)

def extract_spectral_domain_features(
        config: SignalQualityFeatureExtractionConfig, 
        ppg_windowed: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculate the spectral features (dominant frequency, relative power, and spectral entropy)
    for each segment of a PPG signal using a single Welch's method computation. The features are added to the dataframe. 
    Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------
    config: SignalQualityFeatureExtractionConfig
        The configuration object containing the parameters for the feature extraction

    ppg_windowed: np.ndarray
        The dataframe containing the windowed ppg signal

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

    # Calculate each feature using the computed PSD and frequency array
    d_features['f_dom'] = compute_dominant_frequency(freqs, psd)
    d_features['rel_power'] = compute_relative_power(freqs, psd, config)
    d_features['spectral_entropy'] = compute_spectral_entropy(psd, ppg_windowed.shape[1])   # ppg_windowed.shape[1] is the number of samples in the window

    return pd.DataFrame(d_features)

def extract_acc_power_feature(
        f1: np.ndarray, 
        PSD_acc: np.ndarray, 
        f2: np.ndarray, 
        PSD_ppg: np.ndarray
    ) -> float:
    """
    TO BE ADJUSTED
    Calculates the power ratio of the accelerometer signal in the PPG frequency range.
    
    Args:
    f1 (numpy.ndarray): Frequency bins for the accelerometer signal.
    PSD_acc (numpy.ndarray): Power Spectral Density of the accelerometer signal.
    f2 (numpy.ndarray): Frequency bins for the PPG signal.
    PSD_ppg (numpy.ndarray): Power Spectral Density of the PPG signal.
    
    Returns:
    float: The power ratio of the accelerometer signal in the PPG frequency range.
    """
    
    # Find the index of the maximum PSD value in the PPG signal
    max_PPG_psd_idx = np.argmax(PSD_ppg)
    max_PPG_freq_psd = f2[max_PPG_psd_idx]
    
    # Find the index of the closest frequency in the accelerometer signal to the dominant PPG frequency
    corr_acc_psd_df_idx = np.argmin(np.abs(max_PPG_freq_psd - f1))
    
    df_idx = np.arange(corr_acc_psd_df_idx-1, corr_acc_psd_df_idx+2)
    
    # Find the index of the closest frequency in the accelerometer signal to the first harmonic of the PPG frequency
    corr_acc_psd_fh_idx = np.argmin(np.abs(max_PPG_freq_psd*2 - f1))
    fh_idx = np.arange(corr_acc_psd_fh_idx-1, corr_acc_psd_fh_idx+2)
    
    # Calculate the power ratio
    acc_power_PPG_range = np.trapz(PSD_acc[df_idx], f1[df_idx]) + np.trapz(PSD_acc[fh_idx], f1[fh_idx])
    acc_power_total = np.trapz(PSD_acc, f1)
    
    acc_power_ratio = acc_power_PPG_range / acc_power_total
    
    return acc_power_ratio

def extract_accelerometer_feature(config: SignalQualityFeatureExtractionConfigAcc, acc_windowed: pd.DataFrame, ppg_windowed: pd.DataFrame) -> pd.DataFrame:
    """
    Extract accelerometer features from the accelerometer signal in the PPG frequency range.
    
    Parameters
    ----------
    config: SignalQualityFeatureExtractionConfigAcc
        The configuration object containing the parameters for the feature extraction
    
    acc_windowed: pd.DataFrame
        The dataframe containing the windowed accelerometer signal

    ppg_windowed: pd.DataFrame
        The dataframe containing the corresponding windowed ppg signal
    
    Returns
    -------
    pd.DataFrame
        The dataframe with the relative power accelerometer feature.
    """
    
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

    psd_acc = np.sum(psd_acc, axis=1)

    freqs_ppg, psd_ppg = welch(
        ppg_windowed,
        fs=config.sampling_frequency_ppg,
        window=window_ppg,
        noverlap=config.overlap_welch_window_ppg,
        nfft=config.nfft_ppg,
        detrend=False,
        axis=1
    )

    acc_power_ratio = extract_acc_power_feature(freqs_acc, psd_acc, freqs_ppg, psd_ppg)

    return None


