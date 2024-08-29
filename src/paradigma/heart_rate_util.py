from typing import List, Tuple, Union
import pickle
import pandas as pd
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import kurtosis, skew


def extract_ppg_features(arr_ppg: np.ndarray, sampling_frequency: int) -> np.ndarray:
    # Number of features
    feature_count = 10
    
    # Initialize features array
    features_ppg = np.zeros(feature_count)
    
    # Time-domain features
    absPPG = np.abs(arr_ppg)
    features_ppg[0] = np.var(arr_ppg)  # Feature 1: variance
    features_ppg[1] = np.mean(absPPG)  # Feature 2: mean
    features_ppg[2] = np.median(absPPG)  # Feature 3: median
    features_ppg[3] = kurtosis(arr_ppg)  # Feature 4: kurtosis
    features_ppg[4] = skew(arr_ppg)  # Feature 5: skewness
    
    window = 3 * sampling_frequency  # 90 samples for Welch's method => fr = 2/3 = 0.67 Hz --> not an issue with a clear distinct frequency
    overlap = int(0.5 * window)  # 45 samples overlap for Welch's Method
    
    f, P = welch(arr_ppg, sampling_frequency, nperseg=window, noverlap=overlap)
    
    # Find the dominant frequency
    maxIndex = np.argmax(P)
    features_ppg[5] = f[maxIndex]  # Feature 6: dominant frequency
    
    # Find indices of f in relevant physiological heart range 45-180 bpm (0.75 - 3 Hz)
    ph_idx = np.where((f >= 0.75) & (f <= 3))[0]
    maxIndex_ph = np.argmax(P[ph_idx])
    dominantFrequency_ph = f[ph_idx[maxIndex_ph]]
    f_dom_band = np.where((f >= dominantFrequency_ph - 0.2) & (f <= dominantFrequency_ph + 0.2))[0]
    features_ppg[6] = np.trapz(P[f_dom_band]) / np.trapz(P)  # Feature 7: relative power
    
    # Normalize the power spectrum
    pxx_norm = P / np.sum(P)
    
    # Compute spectral entropy
    features_ppg[7] = -np.sum(pxx_norm * np.log2(pxx_norm)) / np.log2(len(arr_ppg))  # Feature 8: spectral entropy
    
    # Signal to noise ratio
    arr_signal = np.var(arr_ppg)
    arr_noise = np.var(absPPG)
    features_ppg[8] = arr_signal / arr_noise  # Feature 9: surrogate of signal to noise ratio
    
    # Autocorrelation features
    ppg_series = pd.Series(arr_ppg)
    autocorrelations = [ppg_series.autocorr(lag=i) for i in range(sampling_frequency*3)]
    
    # Finding peaks in autocorrelation
    peaks, _ = peakdet(np.array(autocorrelations), delta=0.01)
    sorted_peaks = np.sort(peaks)
    #TODO: double check if this is correct
    print(sorted_peaks)
    
    if len(sorted_peaks) > 1:
        features_ppg[9] = sorted_peaks[1]  # Feature 10: the second highest peak
    else:
        features_ppg[9] = 0  # Set to 0 if there is no clear second peak
    
    return features_ppg

# Example usage:
# PPG = np.random.randn(300)  # Example PPG signal, replace with actual data
# fs = 50  # Example sampling frequency, replace with actual sampling rate
# features_df = ppg_features(PPG, fs)
# print(features_df)


def peakdet(v: np.ndarray, delta, x: Union[np.ndarray, None]=None) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Detect peaks in a vector.
    
    Args:
    v (numpy array): Input vector.
    delta (float): Minimum difference between a peak and its surrounding values.
    x (numpy array, optional): Indices corresponding to the values in v. If not provided, indices are generated.

    Returns:
    maxtab (list of tuples): Local maxima as (index, value) pairs.
    mintab (list of tuples): Local minima as (index, value) pairs.
    """
    
    if x is None:
        x = np.arange(len(v))
    else:
        if len(v) != len(x):
            raise ValueError("Input vectors v and x must have the same length")

    # Detect maxima
    max_indices, _ = find_peaks(v, height=delta)
    maxtab = [(x[idx], v[idx]) for idx in max_indices]
    
    # Detect minima by inverting the signal
    min_indices, _ = find_peaks(-v, height=delta)
    mintab = [(x[idx], v[idx]) for idx in min_indices]
    
    return maxtab, mintab

# Example usage:
# v = [0, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0]
# delta = 1
# maxtab, mintab = peakdet(v, delta)
# print("Maxima:", maxtab)
# print("Minima:", mintab)


def calculate_power_ratio(f1: np.ndarray, PSD_acc: np.ndarray, f2: np.ndarray, PSD_ppg: np.ndarray) -> float:
    """
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

# Example usage:
# f1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# PSD_acc = np.array([1, 2, 3, 2, 1])
# f2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# PSD_ppg = np.array([1, 3, 2, 1, 0.5])
# result = acc_feature(f1, PSD_acc, f2, PSD_ppg)
# print(result)

def read_PPG_quality_classifier(classifier_path: str):
    """
    Read the PPG quality classifier from a file.

    Parameters
    ----------
    classifier_path : str
        The path to the classifier file.

    Returns
    -------
    dict
        The classifier dictionary.
    """
    with open(classifier_path, 'rb') as f:
        clf = pickle.load(f)
    return clf

