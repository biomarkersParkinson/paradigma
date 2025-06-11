import numpy as np
import pandas as pd
from typing import List, Tuple

from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks, windows
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA

from paradigma.config import PulseRateConfig


def compute_statistics(data: np.ndarray, statistic: str, abs_stats: bool=False) -> np.ndarray:
    """
    Compute a specific statistical measure along the timestamps of a 2D or 3D array.

    Parameters
    ----------
    data : np.ndarray
        A 2D or 3D NumPy array where statistics are computed.
    statistic : str
        The statistic to compute. Supported values are:
        - 'mean': Mean.
        - 'median': Median.
        - 'var': Variance.
        - 'std': Standard deviation.
        - 'max': Maximum.
        - 'min': Minimum.
        - 'kurtosis': Kurtosis.
        - 'skewness': Skewness.
    abs_stats : bool, optional
        Whether to compute the statistics on the absolute values of the data for 
        the mean and median (default: False).

    Returns
    -------
    np.ndarray
        A 1D or 2D array containing the computed statistic for each row (2D)
        or the entire array (1D).

    Raises
    ------
    ValueError
        If the specified `statistic` is not supported or if the input data has an invalid shape.
    """
    if statistic not in ['mean', 'median', 'var', 'std', 'max', 'min', 'kurtosis', 'skewness']:
        raise ValueError(f"Statistic '{statistic}' is not supported.")
    
    if data.ndim > 3 or data.ndim < 2:
        raise ValueError("Input data must be a 1D, 2D or 3D array.")

    if statistic == 'mean':
        if abs_stats:
            return np.mean(np.abs(data), axis=1)
        else:
            return np.mean(data, axis=1)
    elif statistic == 'median':
        if abs_stats:
            return np.median(np.abs(data), axis=1)
        else:
            return np.median(data, axis=1)
    elif statistic == 'var':
        return np.var(data, ddof=1, axis=1)
    elif statistic == 'std':
        return np.std(data, axis=1)
    elif statistic == 'max':
        return np.max(data, axis=1)
    elif statistic == 'min':
        return np.min(data, axis=1)
    elif statistic == 'kurtosis':
        return kurtosis(data, fisher=False, axis=1)
    elif statistic == 'skewness':
        return skew(data, axis=1)
    else:
        raise ValueError(f"Statistic '{statistic}' is not supported.")


def compute_std_euclidean_norm(data: np.ndarray) -> np.ndarray:
    """
    Compute the standard deviation of the Euclidean norm for each window of sensor data.

    The function calculates the Euclidean norm (L2 norm) across sensor axes for each 
    timestamp within a window, and then computes the standard deviation of these norms 
    for each window.

    Parameters
    ----------
    data : np.ndarray
        A 3D NumPy array of shape (n_windows, n_timestamps, n_axes), where:
        - `n_windows` is the number of windows.
        - `n_timestamps` is the number of time steps per window.
        - `n_axes` is the number of sensor axes (e.g., 3 for x, y, z).

    Returns
    -------
    np.ndarray
        A 1D array of shape (n_windows,) containing the standard deviation of the 
        Euclidean norm for each window.
    """
    norms = np.linalg.norm(data, axis=2)  # Norm along the sensor axes (norm per timestamp, per window)
    return np.std(norms, axis=1)  # Standard deviation per window


def compute_power_in_bandwidth(
        freqs: np.ndarray,
        psd: np.ndarray, 
        fmin: float,
        fmax: float,
        include_max: bool = True,
        spectral_resolution: float = 1,
        cumulative_sum_method: str = 'trapz'
    ) -> np.ndarray:
    """
    Compute the logarithmic power within specified frequency bands for each sensor axis.

    This function integrates the power spectral density (PSD) over user-defined frequency 
    bands and computes the logarithm of the resulting power for each axis of the sensor.

    Parameters
    ----------
    freqs : np.ndarray
        A 1D array of shape (n_frequencies,) containing the frequencies corresponding 
        to the PSD values.
    psd : np.ndarray
        A 2D array of shape (n_windows, n_frequencies) or 3D array of shape (n_windows, n_frequencies, n_axes)
        representing the power spectral density (PSD) of the sensor data.
    fmin : float
        The lower bound of the frequency band in Hz.
    fmax : float
        The upper bound of the frequency band in Hz.
    include_max : bool, optional
        Whether to include the maximum frequency in the search range (default: True).
    spectral_resolution : float, optional
        The spectral resolution of the PSD in Hz (default: 1).
    cumulative_sum_method : str, optional
        The method used to integrate the PSD over the frequency band. Supported values are: 
        - 'trapz': Trapezoidal rule.
        - 'sum': Simple summation (default: 'trapz').

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_windows, n_axes) containing the power within
        the specified frequency band for each window and each sensor axis.
    """
    # Create a mask for frequencies within the current band range (low, high)
    if include_max:
        band_mask = (freqs >= fmin) & (freqs <= fmax)
    else:
        band_mask = (freqs >= fmin) & (freqs < fmax)
    
    # Integrate PSD over the selected frequency band using the band mask
    if psd.ndim == 2:
        masked_psd = psd[:, band_mask]
    elif psd.ndim == 3:
        masked_psd = psd[:, band_mask, :]

    if cumulative_sum_method == 'trapz':
        band_power = spectral_resolution * np.trapz(masked_psd, freqs[band_mask], axis=1)
    elif cumulative_sum_method == 'sum':
        band_power = spectral_resolution * np.sum(masked_psd, axis=1)
    else:
        raise ValueError("cumulative_sum_method must be 'trapz' or 'sum'.")

    return band_power


def compute_total_power(psd: np.ndarray) -> np.ndarray:
    """
    Compute the total power by summing the power spectral density (PSD) across frequency bins.

    This function calculates the total power for each window and each sensor axis by 
    summing the PSD values across all frequency bins.

    Parameters
    ----------
    psd : np.ndarray
        A 3D array of shape (n_windows, n_frequencies, n_axes) representing the 
        power spectral density (PSD) of the sensor data.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_windows, n_axes) containing the total power for each 
        window and each sensor axis.
    """
    return np.sum(psd, axis=-1)  # Sum across frequency bins


def extract_tremor_power(
    freqs: np.ndarray, 
    total_psd: np.ndarray,
    fmin: float = 3,
    fmax: float = 7,
    spectral_resolution: float = 0.25
    ) -> np.ndarray:

    """Computes the tremor power (1.25 Hz around the peak within the tremor frequency band)
    
    Parameters
    ----------
    total_psd: np.ndarray
        The power spectral density of the gyroscope signal summed over the three axes
    freqs: np.ndarray
        Frequency vector corresponding to the power spectral density
    fmin: float
        The lower bound of the tremor frequency band in Hz (default: 3)
    fmax: float
        The upper bound of the tremor frequency band in Hz (default: 7)
    spectral_resolution: float
        The spectral resolution of the PSD in Hz (default: 0.25)
        
    Returns
    -------
    pd.Series
        The tremor power across windows
    """
    
    freq_idx = (freqs >= fmin) & (freqs <= fmax)
    peak_idx = np.argmax(total_psd[:, freq_idx], axis=1) + np.min(np.where(freq_idx)[0])
    left_idx =  np.maximum((peak_idx - 0.5 / spectral_resolution).astype(int), 0)
    right_idx = (peak_idx + 0.5 / spectral_resolution).astype(int)

    row_indices = np.arange(total_psd.shape[1])
    row_indices = np.tile(row_indices, (total_psd.shape[0], 1))
    left_idx = left_idx[:, None]
    right_idx = right_idx[:, None]

    mask = (row_indices >= left_idx) & (row_indices <= right_idx)

    tremor_power = spectral_resolution * np.sum(total_psd * mask, axis=1)

    return tremor_power


def compute_dominant_frequency(
        freqs: np.ndarray, 
        psd: np.ndarray, 
        fmin: float | None = None, 
        fmax: float | None = None
    ) -> np.ndarray:
    """
    Compute the dominant frequency within a specified frequency range for each window and sensor axis.

    The dominant frequency is defined as the frequency corresponding to the maximum power in the 
    power spectral density (PSD) within the specified range.

    Parameters
    ----------
    freqs : np.ndarray
        A 1D array of shape (n_frequencies,) containing the frequencies corresponding 
        to the PSD values.
    psd : np.ndarray
        A 2D array of shape (n_windows, n_frequencies) or a 3D array of shape 
        (n_windows, n_frequencies, n_axes) representing the power spectral density.
    fmin : float
        The lower bound of the frequency range (inclusive).
    fmax : float
        The upper bound of the frequency range (exclusive).

    Returns
    -------
    np.ndarray
        - If `psd` is 2D: A 1D array of shape (n_windows,) containing the dominant frequency 
          for each window.
        - If `psd` is 3D: A 2D array of shape (n_windows, n_axes) containing the dominant 
          frequency for each window and each axis.

    Raises
    ------
    ValueError
        If `fmin` or `fmax` is outside the bounds of the `freqs` array.
        If `psd` is not a 2D or 3D array.
    """
    # Set default values for fmin and fmax to the minimum and maximum frequencies if not provided
    if fmin is None:
        fmin = freqs[0]
    if fmax is None:
        fmax = freqs[-1]

    # Validate the frequency range
    if fmin < freqs[0] or fmax > freqs[-1]:
        raise ValueError(f"fmin {fmin} or fmax {fmax} are out of bounds of the frequency array.")
    
    # Find the indices corresponding to fmin and fmax
    min_index = np.searchsorted(freqs, fmin)
    max_index = np.searchsorted(freqs, fmax)

    # Slice the PSD and frequency array to the desired range
    psd_filtered = psd[:, min_index:max_index] if psd.ndim == 2 else psd[:, min_index:max_index, :]
    freqs_filtered = freqs[min_index:max_index]

    # Compute dominant frequency
    if psd.ndim == 3:
        # 3D: Compute for each axis
        return np.array([
            freqs_filtered[np.argmax(psd_filtered[:, :, i], axis=1)]
            for i in range(psd.shape[-1])
        ]).T
    elif psd.ndim == 2:
        # 2D: Compute for each window
        return freqs_filtered[np.argmax(psd_filtered, axis=1)]
    else:
        raise ValueError("PSD array must be 2D or 3D.")
    

def extract_frequency_peak(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float | None = None,
    fmax: float | None = None,
    include_max: bool = True
    ) -> pd.Series:

    """Extract the frequency of the peak in the power spectral density within the specified frequency band.
    
    Parameters
    ----------
    freqs: pd.Series
        Frequency vector corresponding to the power spectral density
    psd: pd.Series
        The total power spectral density of the gyroscope signal
    fmin: float
        The lower bound of the frequency band in Hz (default: None). If not provided, the minimum frequency is used.
    fmax: float
        The upper bound of the frequency band in Hz (default: None). If not provided, the maximum frequency is used.
    include_max: bool
        Whether to include the maximum frequency in the search range (default: True)
        
    Returns
    -------
    pd.Series
        The frequency of the peak across windows
    """    
    # Set fmin and fmax to maximum range if not provided
    if fmin is None:
        fmin = freqs[0]
    if fmax is None:
        fmax = freqs[-1]

    # Find the indices corresponding to fmin and fmax
    if include_max:
        freq_idx = np.where((freqs>=fmin) & (freqs<=fmax))[0]
    else:
        freq_idx = np.where((freqs>=fmin) & (freqs<fmax))[0]

    peak_idx = np.argmax(psd[:, freq_idx], axis=1)
    frequency_peak = freqs[freq_idx][peak_idx]

    return frequency_peak


def compute_relative_power(
        freqs: np.ndarray, 
        psd: np.ndarray, 
        config: PulseRateConfig
    ) -> list:
    """
    Calculate relative power within the dominant frequency band in the physiological range (0.75 - 3 Hz).

    Parameters
    ----------
    freqs: np.ndarray
        The frequency bins of the power spectral density.
    psd: np.ndarray
        The power spectral density of the signal.
    config: PulseRateConfig
        The configuration object containing the parameters for the feature extraction. The following
        attributes are used:
        - freq_band_physio: tuple
            The frequency band for physiological pulse rate (default: (0.75, 3)).
        - bandwidth: float
            The bandwidth around the peak frequency to consider for relative power calculation (default: 0.5).

    Returns
    -------
    list
        The relative power within the dominant frequency band in the physiological range (0.75 - 3 Hz). 
    
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
    ) -> np.ndarray:
    """
    Calculate the spectral entropy from the normalized power spectral density.

    Parameters
    ----------
    psd: np.ndarray
        The power spectral density of the signal.   
    n_samples: int
        The number of samples in the window.

    Returns
    -------
    np.ndarray
        The spectral entropy of the power spectral density.
    """
    psd_norm = psd / np.sum(psd, axis=1, keepdims=True)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm), axis=1) / np.log2(n_samples)
    
    return spectral_entropy


def compute_mfccs(
        total_power_array: np.ndarray, 
        config, 
        total_power_type: str = 'psd',
        mel_scale: bool = True,
        multiplication_factor: float = 1,
        rounding_method: str = 'floor'       
    ) -> np.ndarray:
    """
    Generate Mel Frequency Cepstral Coefficients (MFCCs) from the total power spectral density or spectrogram of the signal.

    MFCCs are commonly used features in signal processing for tasks like audio and 
    vibration analysis. In this version, we adjusted the MFFCs to the human activity
    range according to: https://www.sciencedirect.com/science/article/abs/pii/S016516841500331X#f0050.
    This function calculates MFCCs by applying a filterbank 
    (in either the mel scale or linear scale) to the total power of the signal, 
    followed by a Discrete Cosine Transform (DCT) to obtain coefficients.

    Parameters
    ----------
    total_power_array : np.ndarray
        2D array of shape (n_windows, n_frequencies) containing the total power 
        of the signal for each window.
        OR
        3D array of shape (n_windows, n_frequencies, n_segments) containing the total spectrogram
        of the signal for each window.
    config : object
        Configuration object containing the following attributes:
        - window_length_s : int
            Duration of each analysis window in seconds.
        - sampling_frequency : int
            Sampling frequency of the data in Hz (default: 100).
        - mfcc_low_frequency : float
            Lower bound of the frequency band in Hz (default: 0).
        - mfcc_high_frequency : float
            Upper bound of the frequency band in Hz (default: 25).
        - mfcc_n_dct_filters : int
            Number of triangular filters in the filterbank (default: 20).
        - mfcc_n_coefficients : int
            Number of coefficients to extract (default: 12).
    total_power_type : str, optional
        The type of the total power array. Supported values are 'psd' and 'spectrogram' (default: 'psd').
    mel_scale : bool, optional
        Whether to use the mel scale for the filterbank (default: True).
    multiplication_factor : float, optional
        Multiplication factor for the Mel scale conversion (default: 1). For tremor, the recommended
        value is 1. For gait, this is 4.
    rounding_method : str, optional
        The method used to round the filter points. Supported values are 'round' and 'floor' (default: 'floor').

    Returns
    -------
    np.ndarray
        2D array of MFCCs with shape `(n_windows, n_coefficients)`, where each row
        contains the MFCCs for a corresponding window.
    ...

    Notes
    -----
    - The function includes filterbank normalization to ensure proper scaling.
    - DCT filters are constructed to minimize spectral leakage.
    """
    
    # Check if total_power_type is either 'psd' or 'spectrogram'
    if total_power_type not in ['psd', 'spectrogram']:
        raise ValueError("total_power_type should be set to either 'psd' or 'spectrogram'")

    # Compute window length in samples
    window_length = config.window_length_s * config.sampling_frequency
    
    # Determine the length of subwindows used in the spectrogram computation
    if total_power_type == 'spectrogram':
        nr_subwindows = total_power_array.shape[2]
        window_length = int(window_length/(nr_subwindows - (nr_subwindows - 1) * config.overlap_fraction))

    # Generate filter points
    if mel_scale:
        freqs = np.linspace(
            melscale(config.mfcc_low_frequency, multiplication_factor), 
            melscale(config.mfcc_high_frequency, multiplication_factor), 
            num=config.mfcc_n_dct_filters + 2
        )
        freqs = inverse_melscale(freqs, multiplication_factor)
    else:
        freqs = np.linspace(
            config.mfcc_low_frequency, 
            config.mfcc_high_frequency, 
            num=config.mfcc_n_dct_filters + 2
        )
    
    if rounding_method == 'round':
        filter_points = np.round(
            window_length / config.sampling_frequency * freqs
        ).astype(int)  + 1

    elif rounding_method == 'floor':
        filter_points = np.floor(
            window_length / config.sampling_frequency * freqs
        ).astype(int) + 1

    # Construct triangular filterbank
    filters = np.zeros((len(filter_points) - 2, int(window_length / 2 + 1)))
    for j in range(len(filter_points) - 2):
        filters[j, filter_points[j] : filter_points[j + 2]] = windows.triang(
            filter_points[j + 2] - filter_points[j]
        ) 
        # Normalize filter coefficients
        filters[j, :] /= (
            config.sampling_frequency/window_length * np.sum(filters[j,:])
        ) 

    # Apply filterbank to total power
    if total_power_type == 'spectrogram':
        power_filtered = np.tensordot(total_power_array, filters.T, axes=(1,0))
    elif total_power_type == 'psd':
        power_filtered = np.dot(total_power_array, filters.T)
        
    # Convert power to logarithmic scale
    log_power_filtered = np.log10(power_filtered + 1e-10)

    # Generate DCT filters
    dct_filters = np.empty((config.mfcc_n_coefficients, config.mfcc_n_dct_filters))
    dct_filters[0, :] = 1.0 / np.sqrt(config.mfcc_n_dct_filters)

    samples = (
        np.arange(1, 2 * config.mfcc_n_dct_filters, 2) * np.pi / (2.0 * config.mfcc_n_dct_filters)
    )

    for i in range(1, config.mfcc_n_coefficients):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / config.mfcc_n_dct_filters)

    # Compute MFCCs
    mfccs = np.dot(log_power_filtered, dct_filters.T) 

    if total_power_type == 'spectrogram':
        mfccs = np.mean(mfccs, axis=1)

    return mfccs


def melscale(x: np.ndarray, multiplication_factor: float = 1) -> np.ndarray:
    """
    Maps linear frequency values to the Mel scale.

    Parameters
    ----------
    x : np.ndarray
        Linear frequency values to be converted to the Mel scale.
    multiplication_factor : float, optional
        Multiplication factor for the Mel scale conversion (default: 1). For tremor, the recommended
        value is 1. For gait, this is 4.

    Returns
    -------
    np.ndarray
        Frequency values mapped to the Mel scale.
    """
    return (64.875 / multiplication_factor) * np.log10(1 + x / (17.5 / multiplication_factor))


def inverse_melscale(x: np.ndarray, multiplication_factor: float = 1) -> np.ndarray:
    """
    Maps values from the Mel scale back to linear frequencies.

    This function performs the inverse transformation of the Mel scale,
    converting perceptual frequency values to their corresponding linear frequency values.

    Parameters
    ----------
    x : np.ndarray
        Frequency values on the Mel scale to be converted back to linear frequencies.

    Returns
    -------
    np.ndarray
        Linear frequency values corresponding to the given Mel scale values.
    """
    return (17.5 / multiplication_factor) * (10 ** (x / (64.875 / multiplication_factor)) - 1)


def pca_transform_gyroscope(
        df: pd.DataFrame,
        y_gyro_colname: str,
        z_gyro_colname: str,
) -> np.ndarray:
    """
    Perform principal component analysis (PCA) on gyroscope data to estimate velocity.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the gyroscope data.
    y_gyro_colname : str
        The column name for the y-axis gyroscope data.
    z_gyro_colname : str
        The column name for the z-axis gyroscope data.
        
    Returns
    -------
    np.ndarray
        The estimated velocity based on the principal component of the gyroscope data.
    """
    # Convert gyroscope columns to NumPy arrays
    y_gyro_array = df[y_gyro_colname].to_numpy()
    z_gyro_array = df[z_gyro_colname].to_numpy()

    # Fit PCA
    fit_data = np.column_stack((y_gyro_array, z_gyro_array))
    full_data = fit_data

    pca = PCA(n_components=2, svd_solver='auto', random_state=22)
    pca.fit(fit_data)
    velocity = pca.transform(full_data)[:, 0]  # First principal component

    return np.asarray(velocity)


def compute_angle(time_array: np.ndarray, velocity_array: np.ndarray) -> np.ndarray:
    """
    Compute the angle from the angular velocity using cumulative trapezoidal integration.
    
    Parameters
    ----------
    time_array : np.ndarray
        The time array corresponding to the angular velocity data.
    velocity_array : np.ndarray
        The angular velocity data to integrate.
        
    Returns
    -------
    np.ndarray
        The estimated angle based on the cumulative trapezoidal integration of the angular velocity.
    """
    # Perform integration and apply absolute value
    angle_array = cumulative_trapezoid(
        y=velocity_array, 
        x=time_array, 
        initial=0
    )
    return np.abs(angle_array)


def remove_moving_average_angle(angle_array: np.ndarray, fs: float) -> pd.Series:
    """
    Remove the moving average from the angle to correct for drift.

    Parameters
    ----------
    angle_array : np.ndarray
        The angle array to remove the moving average from.
    fs : float
        The sampling frequency of the data.
    
    Returns
    -------
    pd.Series
        The angle array with the moving average removed.
    """
    window_size = int(2 * (fs * 0.5) + 1)
    angle_ma = np.array(pd.Series(angle_array).rolling(
        window=window_size, 
        min_periods=1, 
        center=True, 
        closed='both'
    ).mean())
    
    return angle_array - angle_ma


def extract_angle_extremes(
        angle_array: np.ndarray,
        sampling_frequency: float,
        max_frequency_activity: float = 1.75,
    ) -> tuple[List[int], List[int], List[int]]:
    """
    Extract extrema (minima and maxima) indices from the angle array.
    
    Parameters
    ----------
    angle_array : np.ndarray
        The angle array to extract extrema from.
    sampling_frequency : float
        The sampling frequency of the data.
    max_frequency_activity : float, optional
        The maximum frequency of human activity in Hz (default: 1.75).
    
    Returns
    -------
    tuple
        A tuple containing the indices of the angle extrema, minima, and maxima.
    """
    distance = sampling_frequency / max_frequency_activity
    prominence = 2  

    # Find minima and maxima indices for each window
    minima_indices = find_peaks(
        x=-angle_array, 
        distance=distance, 
        prominence=prominence
    )[0]
    maxima_indices = find_peaks(
        x=angle_array, 
        distance=distance, 
        prominence=prominence
    )[0]

    minima_indices = np.array(minima_indices, dtype=object)
    maxima_indices = np.array(maxima_indices, dtype=object)

    i_pks = 0
    if minima_indices.size > 0 and maxima_indices.size > 0:
        if maxima_indices[0] > minima_indices[0]:
            # Start with a minimum
            while i_pks < minima_indices.size - 1 and i_pks < maxima_indices.size:
                if minima_indices[i_pks + 1] < maxima_indices[i_pks]:
                    if angle_array[minima_indices[i_pks + 1]] < angle_array[minima_indices[i_pks]]:
                        minima_indices = np.delete(minima_indices, i_pks)
                    else:
                        minima_indices = np.delete(minima_indices, i_pks + 1)
                    i_pks -= 1

                if i_pks >= 0 and minima_indices[i_pks] > maxima_indices[i_pks]:
                    if angle_array[maxima_indices[i_pks]] < angle_array[maxima_indices[i_pks - 1]]:
                        maxima_indices = np.delete(maxima_indices, i_pks)
                    else:
                        maxima_indices = np.delete(maxima_indices, i_pks - 1)
                    i_pks -= 1
                i_pks += 1

        elif maxima_indices[0] < minima_indices[0]:
            # Start with a maximum
            while i_pks < maxima_indices.size - 1 and i_pks < minima_indices.size:
                if maxima_indices[i_pks + 1] < minima_indices[i_pks]:
                    if angle_array[maxima_indices[i_pks + 1]] < angle_array[maxima_indices[i_pks]]:
                        maxima_indices = np.delete(maxima_indices, i_pks + 1)
                    else:
                        maxima_indices = np.delete(maxima_indices, i_pks)
                    i_pks -= 1

                if i_pks >= 0 and maxima_indices[i_pks] > minima_indices[i_pks]:
                    if angle_array[minima_indices[i_pks]] < angle_array[minima_indices[i_pks - 1]]:
                        minima_indices = np.delete(minima_indices, i_pks - 1)
                    else:
                        minima_indices = np.delete(minima_indices, i_pks)
                    i_pks -= 1
                i_pks += 1

    # Combine remaining extrema and compute range of motion
    angle_extrema_indices = np.sort(np.concatenate([minima_indices, maxima_indices]))

    return list(angle_extrema_indices), list(minima_indices), list(maxima_indices)


def compute_range_of_motion(angle_array: np.ndarray, extrema_indices: List[int]) -> np.ndarray:
    """
    Compute the range of motion of a time series based on the angle extrema.
    
    Parameters
    ----------
    angle_array : np.ndarray
        The angle array to compute the range of motion from.
    extrema_indices : List[int]
        The indices of the angle extrema.
    
    Returns
    -------
    np.ndarray
        The range of motion of the time series.
    """
    # Ensure extrema_indices is a NumPy array of integers
    if not isinstance(extrema_indices, list):
        raise TypeError("extrema_indices must be a list of integers.")

    # Check bounds
    if np.any(np.array(extrema_indices) < 0) or np.any(np.array(extrema_indices) >= len(angle_array)):
        raise ValueError("extrema_indices contains out-of-bounds indices.")
    
    # Extract angle amplitudes (minima and maxima values)
    angle_extremas = angle_array[extrema_indices]

    # Compute the differences (range of motion) across all windows at once using np.diff
    range_of_motion = np.abs(np.diff(angle_extremas))

    return range_of_motion


def compute_peak_angular_velocity(
    velocity_array: np.ndarray,
    angle_extrema_indices: List[int],
) -> np.ndarray:
    """
    Compute the peak angular velocity of a time series based on the angle extrema.

    Parameters
    ----------
    velocity_array : np.ndarray
        The angular velocity array to compute the peak angular velocity from.
    angle_extrema_indices : List[int]
        The indices of the angle extrema.
    
    Returns
    -------
    np.ndarray
        The peak angular velocities of the time series.
    """
    if np.any(np.array(angle_extrema_indices) < 0) or np.any(np.array(angle_extrema_indices) >= len(velocity_array)):
        raise ValueError("angle_extrema_indices contains out-of-bounds indices.")
    
    if len(angle_extrema_indices) < 2:
        raise ValueError("angle_extrema_indices must contain at least two indices.")
    
    # Initialize a list to store the peak velocities 
    pav = []

    # Compute peak angular velocities
    for i in range(len(angle_extrema_indices) - 1):
        # Get the current and next extrema index
        current_peak_idx = angle_extrema_indices[i]
        next_peak_idx = angle_extrema_indices[i + 1]
        segment = velocity_array[current_peak_idx:next_peak_idx]

        pav.append(np.max(np.abs(segment)))

    return np.array(pav)


def compute_forward_backward_peak_angular_velocity(
    velocity_array: np.ndarray,
    angle_extrema_indices: List[int],
    minima_indices: List[int],
    maxima_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the peak angular velocity of a time series based on the angle extrema.

    Parameters
    ----------
    velocity_array : np.ndarray
        The angular velocity array to compute the peak angular velocity from.
    angle_extrema_indices : List[int]
        The indices of the angle extrema.
    minima_indices : List[int]
        The indices of the minima.
    maxima_indices : List[int]
        The indices of the maxima.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the forward and backward peak angular velocities for minima and maxima.
    """
    if np.any(np.array(angle_extrema_indices) < 0) or np.any(np.array(angle_extrema_indices) >= len(velocity_array)):
        raise ValueError("angle_extrema_indices contains out-of-bounds indices.")
    
    if len(angle_extrema_indices) < 2:
        raise ValueError("angle_extrema_indices must contain at least two indices.")
    
    if len(minima_indices) == 0:
        raise ValueError("No minima indices found.")
    
    if len(maxima_indices) == 0:
        raise ValueError("No maxima indices found.")

    # Initialize lists to store the peak velocities
    forward_pav = []
    backward_pav = []

    # Compute peak angular velocities
    for i in range(len(angle_extrema_indices) - 1):
        # Get the current and next extrema index
        current_peak_idx = angle_extrema_indices[i]
        next_peak_idx = angle_extrema_indices[i + 1]
        segment = velocity_array[current_peak_idx:next_peak_idx]

        # Check if the current peak is a minimum or maximum and calculate peak velocity accordingly
        if current_peak_idx in minima_indices:
            forward_pav.append(np.max(np.abs(segment)))
        elif current_peak_idx in maxima_indices:
            backward_pav.append(np.max(np.abs(segment)))

    # Convert lists to numpy arrays
    forward_pav = np.array(forward_pav)
    backward_pav = np.array(backward_pav)

    return forward_pav, backward_pav


def compute_signal_to_noise_ratio(
        ppg_windowed: np.ndarray
    ) -> np.ndarray:
    """
    Compute the signal to noise ratio of the PPG signal.
    
    Parameters
    ----------
    ppg_windowed: np.ndarray
        The windowed PPG signal.

    Returns
    -------
    np.ndarray
        The signal to noise ratio of the PPG signal.
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
    Compute the biased autocorrelation of the PPG signal. The autocorrelation is computed up to 3 seconds. The highest peak value is selected as the autocorrelation value. If no peaks are found, the value is set to 0.
    The biased autocorrelation is computed using the biased_autocorrelation function. It differs from the unbiased autocorrelation in that the normalization factor is the length of the original signal, and boundary effects are considered. This results in a smoother autocorrelation function.
    
    Parameters
    ----------
    ppg_windowed: np.ndarray
        The windowed PPG signal.
    fs: int
        The sampling frequency of the PPG signal.

    Returns
    -------
    np.ndarray
        The autocorrelation of the PPG signal.
    """

    auto_correlations = biased_autocorrelation(ppg_windowed, fs*3) # compute the biased autocorrelation of the PPG signal up to 3 seconds
    peaks = [find_peaks(x, height=0.01)[0] for x in auto_correlations] # find the peaks of the autocorrelation
    sorted_peak_values = [np.sort(auto_correlations[i, indices])[::-1] for i, indices in enumerate(peaks)] # sort the peak values in descending order
    auto_correlations = [x[0] if len(x) > 0 else 0 for x in sorted_peak_values] # get the highest peak value if there are any peaks, otherwise set to 0

    return np.asarray(auto_correlations)

def biased_autocorrelation(
        ppg_windowed: np.ndarray, 
        max_lag: int
    ) -> np.ndarray:
    """
    Compute the biased autocorrelation of a signal (similar to matlabs autocorr function), where the normalization factor 
    is the length of the original signal, and boundary effects are considered.
    
    Parameters
    ----------
    ppg_windowed: np.ndarray
        The windowed PPG signal.
    max_lag: int
        The maximum lag for the autocorrelation.

    Returns
    -------
    np.ndarray
        The biased autocorrelation of the PPG signal.

    """
    zero_mean_ppg = ppg_windowed - np.mean(ppg_windowed, axis=1, keepdims=True) # Remove the mean of the signal to make it zero-mean
    N = zero_mean_ppg.shape[1]
    autocorr_values = np.zeros((zero_mean_ppg.shape[0], max_lag + 1))
    
    for lag in range(max_lag + 1):
        # Compute autocorrelation for current lag
        overlapping_points = zero_mean_ppg[:, :N-lag] * zero_mean_ppg[:, lag:]
        autocorr_values[:, lag] = np.sum(overlapping_points, axis=1) / N  # Divide by N (biased normalization)
    
    return autocorr_values/autocorr_values[:, 0, np.newaxis] # Normalize the autocorrelation values
