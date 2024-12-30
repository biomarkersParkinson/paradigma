from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.signal.windows import hamming
from scipy.stats import kurtosis, skew
from paradigma.config import SignalQualityFeatureExtractionConfig

def generate_statistics(
        sensor_col: pd.Series,
        statistic: str
    ) -> list:
    """Generate statistics for a single sensor and axis. The function is used with the apply function in pandas.
    
    Parameters
    ----------
    sensor_col: pd.Series
        The sensor column to be aggregated (e.g. x-axis of accelerometer)
    statistic: str
        The statistic to be computed ['mean', 'var', 'median', 'kurtosis', 'skewness']
        
    Returns
    -------
    list
        the statistic for the sensor segments
    """
    if statistic == 'mean':
        return [np.mean(np.abs(x)) for x in sensor_col]
    elif statistic == 'var':
        return [np.var(x, ddof=1) for x in sensor_col]  # ddof=1 for unbiased variance is used, same as matlab
    elif statistic == 'median':
        return [np.median(np.abs(x)) for x in sensor_col]
    elif statistic == 'kurtosis':
        return [kurtosis(x, fisher = False) for x in sensor_col] # Fisher's definition of kurtosis is used, same as matlab
    elif statistic == 'skewness':
        return [skew(x) for x in sensor_col]
    else:
        raise ValueError("Statistic not recognized.")
    
def generate_statistics_numpy(
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
        ppg_segments: np.ndarray
    ) -> list:
    """
    Compute the signal to noise ratio of the PPG signal.
    
    Args:
    ppg_segments: PPG signal of shape ...
    
    Returns:
    list: Signal to noise ratio of the PPG windows.
    """
    signal_to_noise_ratios = []
    for segment in ppg_segments:
        arr_signal = np.var(segment)
        arr_noise = np.var(np.abs(segment))
        signal_to_noise_ratio = arr_signal / arr_noise
        signal_to_noise_ratios.append(signal_to_noise_ratio)
    
    return signal_to_noise_ratios

def compute_signal_to_noise_ratio_numpy(
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
        ppg_segments: np.ndarray, 
        fs: int
    ) -> list:
    """
    Compute the autocorrelation of the PPG signal.
    
    Args:
        ppg_segments: 2D array where each row is a segment of the PPG signal.
        fs (int): Sampling frequency of the PPG signal.
    
    
    Returns:
        list: Autocorrelation of the PPG segments.
    """
    auto_correlations = []

    for segment in ppg_segments:
        autocorrelations = biased_autocorrelation(segment, fs*3)
        peaks, _ = find_peaks(autocorrelations, height=0.01)
        peak_values = autocorrelations[peaks]
        sorted_peaks = np.sort(peak_values)[::-1]
        if len(sorted_peaks) > 0:
            auto_corr = sorted_peaks[0]
        else:
            auto_corr = 0
        auto_correlations.append(auto_corr)

    return auto_correlations

def compute_auto_correlation_numpy(
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

    auto_correlations = biased_autocorrelation_numpy(ppg_windowed, fs*3)
    peaks = [find_peaks(x, height=0.01)[0] for x in auto_correlations]
    sorted_peak_values = [np.sort(auto_correlations[i, indices])[::-1] for i, indices in enumerate(peaks)]
    auto_correlations = [x[0] if len(x) > 0 else 0 for x in sorted_peak_values]

    return np.asarray(auto_correlations)

def biased_autocorrelation(
        x: np.ndarray, 
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
    x = x - np.mean(x) # Remove the mean of the signal to make it zero-mean
    N = len(x)
    autocorr_values = np.zeros(max_lag + 1)
    
    for lag in range(max_lag + 1):
        # Compute autocorrelation for current lag
        overlapping_points = x[:N-lag] * x[lag:]
        autocorr_values[lag] = np.sum(overlapping_points) / N  # Divide by N (biased normalization)
    
    return autocorr_values/autocorr_values[0] # Normalize the autocorrelation values

def biased_autocorrelation_numpy(
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
    
    return autocorr_values/autocorr_values[0] # Normalize the autocorrelation values

def compute_dominant_frequency(
        freqs: np.ndarray, 
        psd: np.ndarray
    ) -> float:
    """
    Identify the dominant frequency (peak frequency) in the power spectral density.
    """
    peak_idx = np.argmax(psd)
    return freqs[peak_idx]

def compute_dominant_frequency_numpy(
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
    hr_range_idx = np.where((freqs >= config.freq_band_physio[0]) & (freqs <= config.freq_band_physio[1]))[0]
    peak_idx = np.argmax(psd[hr_range_idx])
    peak_freq = freqs[hr_range_idx[peak_idx]]
    
    dom_band_idx = np.where((freqs >= peak_freq - config.bandwidth) & (freqs <= peak_freq + config.bandwidth))[0]
    rel_power = np.trapz(psd[dom_band_idx], freqs[dom_band_idx]) / np.trapz(psd, freqs)
    return rel_power

def compute_relative_power_numpy(
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
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm)) / np.log2(n_samples)
    return spectral_entropy

def compute_spectral_entropy_numpy(
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
        df_windowed: pd.DataFrame, 
        quality_stats: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
    """
    Compute temporal domain features for the ppg signal. The features are added to the dataframe. Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------

    config: SignalQualityFeatureExtractionConfig
        The configuration object containing the parameters for the feature extraction
    
    df_windowed: pd.DataFrame
        The dataframe containing the windowed accelerometer signal

    quality_stats: list, optional
        The statistics to be computed for the gravity component of the accelerometer signal (default: ['mean', 'std'])
    
    Returns
    -------
    pd.DataFrame
        The dataframe with the added temporal domain features.
    """
    
    for stat in quality_stats:
        df_windowed[f'{stat}'] = generate_statistics(
            sensor_col=df_windowed[config.ppg_colname],
            statistic=stat
            )
    
    df_windowed['signal_to_noise'] = compute_signal_to_noise_ratio(df_windowed[config.ppg_colname])  # feature 9
    df_windowed['auto_corr'] = compute_auto_correlation(df_windowed[config.ppg_colname], config.sampling_frequency) # feature 10
    return df_windowed

def extract_temporal_domain_features_numpy(
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
        feature_dict[stat] = generate_statistics_numpy(ppg_windowed, stat)
    
    feature_dict['signal_to_noise'] = compute_signal_to_noise_ratio_numpy(ppg_windowed)  # feature 9
    feature_dict['auto_corr'] = compute_auto_correlation_numpy(ppg_windowed, config.sampling_frequency) # feature 10
    return pd.DataFrame(feature_dict)

def extract_spectral_domain_features(
        config: SignalQualityFeatureExtractionConfig, 
        df_windowed: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculate the spectral features (dominant frequency, relative power, and spectral entropy)
    for each segment of a PPG signal using a single Welch's method computation. The features are added to the dataframe. 
    Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------
    config: SignalQualityFeatureExtractionConfig
        The configuration object containing the parameters for the feature extraction

    df_windowed: pd.DataFrame
        The dataframe containing the windowed ppg signal

    Returns
    -------
    pd.DataFrame
        The dataframe with the added spectral domain features.
    """
    fs = config.sampling_frequency
    ppg_segments = df_windowed[config.ppg_colname]

    dominant_frequencies = []
    relative_powers = []
    spectral_entropies = []
    window = hamming(config.window_length_welch, sym = True)

    for segment in ppg_segments:
        # Compute power spectral density (PSD) once using Welch's method
        freqs, psd = welch(
            segment,
            fs=fs,
            window=window,
            noverlap=config.overlap_welch_window,
            nfft=max(256, 2 ** int(np.log2(len(segment)))),
            detrend=False
        )

        # Calculate each feature using the computed PSD and frequency array
        dominant_frequencies.append(compute_dominant_frequency(freqs, psd))
        relative_powers.append(compute_relative_power(freqs, psd, config))
        spectral_entropies.append(compute_spectral_entropy(psd, len(segment)))

    df_windowed['f_dom'] =dominant_frequencies
    df_windowed['rel_power'] = relative_powers
    df_windowed['spectral_entropy'] = spectral_entropies

    return df_windowed

def extract_spectral_domain_features_numpy(
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
    d_features['f_dom'] = compute_dominant_frequency_numpy(freqs, psd)
    d_features['rel_power'] = compute_relative_power_numpy(freqs, psd, config)
    d_features['spectral_entropy'] = compute_spectral_entropy_numpy(psd, ppg_windowed.shape[0])

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
