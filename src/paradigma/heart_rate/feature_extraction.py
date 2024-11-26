from typing import List, Tuple, Union
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.signal.windows import hamming
from scipy.stats import kurtosis, skew
from paradigma.heart_rate.heart_rate_analysis_config import PPGconfig

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

def compute_signal_to_noise_ratio(
        ppg_segments: np.ndarray
    ) -> list:
    """
    Compute the signal to noise ratio of the PPG signal.
    
    Args:
    arr_ppg (numpy.ndarray): PPG signal.
    
    Returns:
    list: Signal to noise ratio of the PPG windows.
    """
    l_signal_to_noise_ratios = []
    for segment in ppg_segments:
        arr_signal = np.var(segment)
        arr_noise = np.var(np.abs(segment))
        signal_to_noise_ratio = arr_signal / arr_noise
        l_signal_to_noise_ratios.append(signal_to_noise_ratio)
    
    return l_signal_to_noise_ratios

def compute_auto_correlation(
        ppg_segments: np.ndarray, 
        fs: int
    ) -> list:
    """
    Compute the autocorrelation of the PPG signal.
    
    Args:
        ppg_signal (np.ndarray): 2D array where each row is a segment of the PPG signal.
        fs (int): Sampling frequency of the PPG signal.
    
    
    Returns:
        list: Autocorrelation of the PPG segments.
    """
    l_auto_correlations = []

    for segment in ppg_segments:
        autocorrelations = biased_autocorrelation(segment, fs*3)
        peaks, _ = find_peaks(autocorrelations, height=0.01)
        peak_values = autocorrelations[peaks]
        sorted_peaks = np.sort(peak_values)[::-1]
        if len(sorted_peaks) > 0:
            auto_corr = sorted_peaks[0]
        else:
            auto_corr = 0
        l_auto_correlations.append(auto_corr)

    return l_auto_correlations

def biased_autocorrelation(
        x: np.ndarray, 
        max_lag: int
    ) -> np.ndarray:
    """
    Compute the biased autocorrelation of a signal (similar to matlabs autocorr function), where the normalization factor 
    is the length of the original signal, and boundary effects are considered.
    
    Args:
        x (np.ndarray): Input signal (1D array).
        max_lag (int): Maximum lag to compute autocorrelation.
    
    Returns:
        np.ndarray: Biased autocorrelation values for lags 0 to max_lag.
    """
    x = np.array(x) # Ensure x is a numpy array instead of a list
    x = x - np.mean(x) # Remove the mean of the signal to make it zero-mean
    N = len(x)
    autocorr_values = np.zeros(max_lag + 1)
    
    for lag in range(max_lag + 1):
        # Compute autocorrelation for current lag
        overlapping_points = x[:N-lag] * x[lag:]
        autocorr_values[lag] = np.sum(overlapping_points) / N  # Divide by N (biased normalization)
    
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

def compute_relative_power(
        freqs: np.ndarray, 
        psd: np.ndarray
    ) -> float:
    """
    Calculate relative power within the dominant frequency band in the physiological range (0.75 - 3 Hz).
    """
    hr_range_idx = np.where((freqs >= 0.75) & (freqs <= 3))[0]
    peak_idx = np.argmax(psd[hr_range_idx])
    peak_freq = freqs[hr_range_idx[peak_idx]]
    
    dom_band_idx = np.where((freqs >= peak_freq - 0.2) & (freqs <= peak_freq + 0.2))[0]
    rel_power = np.trapz(psd[dom_band_idx], freqs[dom_band_idx]) / np.trapz(psd, freqs)
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

def extract_temporal_domain_features(
        config: PPGconfig, 
        df_windowed:pd.DataFrame, 
        l_quality_stats=['mean', 'std']
    ) -> pd.DataFrame:
    """
    Compute temporal domain features for the ppg signal. The features are added to the dataframe. Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------

    config: GaitFeatureExtractionConfig
        The configuration object containing the parameters for the feature extraction
    
    df_windowed: pd.DataFrame
        The dataframe containing the windowed accelerometer signal

    l_gravity_stats: list, optional
        The statistics to be computed for the gravity component of the accelerometer signal (default: ['mean', 'std'])
    
    Returns
    -------
    pd.DataFrame
        The dataframe with the added temporal domain features.
    """
    
    
    for stat in l_quality_stats:
        df_windowed[f'{stat}'] = generate_statistics(
            sensor_col=df_windowed[config.ppg_colname],
            statistic=stat
            )
    

    df_windowed[f'signal_to_noise'] = compute_signal_to_noise_ratio(df_windowed[config.ppg_colname])  # feature 9
    df_windowed[f'auto_corr'] = compute_auto_correlation(df_windowed[config.ppg_colname], config.sampling_frequency) # feature 10

    return df_windowed

def extract_spectral_domain_features(
        config: PPGconfig, 
        df_windowed:pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculate the spectral features (dominant frequency, relative power, and spectral entropy)
    for each segment of a PPG signal using a single Welch's method computation. The features are added to the dataframe. 
    Therefore the original dataframe is modified, and the modified dataframe is returned.

    Parameters
    ----------
    config: PPGconfig
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
    win_len = 3 * fs  # 3-second window for Welch's method
    window = hamming(win_len, sym=True)  # Hamming window for spectral estimation
    overlap = win_len // 2  # 50% overlap

    l_dominant_frequencies = []
    l_relative_powers = []
    l_spectral_entropies = []

    for segment in ppg_segments:
        # Compute power spectral density (PSD) once using Welch's method
        freqs, psd = welch(
            segment,
            fs=fs,
            window=window,
            noverlap=overlap,
            nfft=max(256, 2 ** int(np.log2(len(segment)))),
            detrend=False
        )

        # Calculate each feature using the computed PSD and frequency array
        l_dominant_frequencies.append(compute_dominant_frequency(freqs, psd))
        l_relative_powers.append(compute_relative_power(freqs, psd))
        l_spectral_entropies.append(compute_spectral_entropy(psd, len(segment)))

    df_windowed[f'f_dom'] = l_dominant_frequencies
    df_windowed[f'rel_power'] = l_relative_powers
    df_windowed[f'spectral_entropy'] = l_spectral_entropies

    return df_windowed


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
