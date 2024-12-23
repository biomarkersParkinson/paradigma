from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scipy import signal
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks, periodogram

from paradigma.constants import DataColumns


def compute_statistics(data: np.ndarray, statistic: str) -> np.ndarray:
    """
    Compute a specific statistical measure along the rows of a 2D array.

    Parameters
    ----------
    data : np.ndarray
        A 2D NumPy array where statistics are computed along rows.
    statistic : str
        The statistic to compute. Supported values are:
        - 'mean': Compute the mean.
        - 'std': Compute the standard deviation.
        - 'max': Compute the maximum.
        - 'min': Compute the minimum.

    Returns
    -------
    np.ndarray
        A 1D array containing the computed statistic for each row.

    Raises
    ------
    ValueError
        If the specified `statistic` is not supported.
    """
    if statistic == 'mean':
        return np.mean(data, axis=1)
    elif statistic == 'std':
        return np.std(data, axis=1)
    elif statistic == 'max':
        return np.max(data, axis=1)
    elif statistic == 'min':
        return np.min(data, axis=1)
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


def compute_power_in_bandwidth(config, psd: np.ndarray, freqs: np.ndarray) -> dict:
    """
    Compute the logarithmic power within specified frequency bands for each sensor axis.

    This function integrates the power spectral density (PSD) over user-defined frequency 
    bands and computes the logarithm of the resulting power for each axis of the sensor.

    Parameters
    ----------
    config : object
        A configuration object with the following attributes:
        - `d_frequency_bandwidths` (dict): A dictionary mapping band names (str) to 
          tuples of frequency ranges (low, high) in Hz.
        - `sensor` (str): The name of the sensor used for prefixing column names.
    psd : np.ndarray
        A 3D array of shape (n_windows, n_frequencies, n_axes) representing the 
        power spectral density (PSD) of the sensor data.
    freqs : np.ndarray
        A 1D array of shape (n_frequencies,) containing the frequencies corresponding 
        to the PSD values.

    Returns
    -------
    dict
        A dictionary where each key represents the logarithmic power for a specific 
        frequency band and sensor axis. The keys are formatted as 
        `"{sensor}_x_{band_name}"`, `"{sensor}_y_{band_name}"`, and 
        `"{sensor}_z_{band_name}"`.

    Notes
    -----
    - The logarithmic transformation is applied to prevent numerical instability with very 
      small values by adding a small constant (1e-10) before taking the logarithm.
    - Integration is performed using the trapezoidal rule (`np.trapz`).
    """
    band_powers = {}
    for band_name, (low, high) in config.d_frequency_bandwidths.items():
        # Create a mask for frequencies within the current band range (low, high)
        band_mask = (freqs >= low) & (freqs < high)
        
        # Integrate PSD over the selected frequency band using the band mask
        band_power = np.log10(np.trapz(psd[:, band_mask, :], freqs[band_mask], axis=1) + 1e-10)
        
        band_powers.update({
            f'{config.sensor}_x_{band_name}': band_power[:, 0],
            f'{config.sensor}_y_{band_name}': band_power[:, 1],
            f'{config.sensor}_z_{band_name}': band_power[:, 2],
        })

    return band_powers


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


def compute_dominant_frequency(
        psd: np.ndarray, 
        freqs: np.ndarray, 
        fmin: float, 
        fmax: float
    ) -> np.ndarray:
    """
    Compute the dominant frequency within a specified frequency range for each window and sensor axis.

    The dominant frequency is defined as the frequency corresponding to the maximum power in the 
    power spectral density (PSD) within the specified range.

    Parameters
    ----------
    psd : np.ndarray
        A 2D array of shape (n_windows, n_frequencies) or a 3D array of shape 
        (n_windows, n_frequencies, n_axes) representing the power spectral density.
    freqs : np.ndarray
        A 1D array of shape (n_frequencies,) containing the frequencies corresponding 
        to the PSD values.
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


def compute_mfccs(
        config,
        total_power_array: np.ndarray,
        mel_scale: bool = True,
        ) -> np.ndarray:
    """
    Generate Mel Frequency Cepstral Coefficients (MFCCs) from the total power of the signal.

    MFCCs are commonly used features in signal processing for tasks like audio and 
    vibration analysis. In this version, we adjusted the MFFCs to the human activity
    range according to: https://www.sciencedirect.com/science/article/abs/pii/S016516841500331X#f0050.
    This function calculates MFCCs by applying a filterbank 
    (in either the mel scale or linear scale) to the total power of the signal, 
    followed by a Discrete Cosine Transform (DCT) to obtain coefficients.

    Parameters
    ----------
    config : object
        Configuration object containing the following attributes:
        - window_length_s : int
            Duration of each analysis window in seconds.
        - sampling_frequency : int
            Sampling frequency of the data (default: 100 Hz).
        - mfcc_low_frequency : float
            Lower bound of the frequency band (default: 0 Hz).
        - mfcc_high_frequency : float
            Upper bound of the frequency band (default: 25 Hz).
        - mfcc_n_dct_filters : int
            Number of triangular filters in the filterbank (default: 20).
        - mfcc_n_coefficients : int
            Number of coefficients to extract (default: 12).
    total_power_array : np.ndarray
        2D array of shape (n_windows, n_frequencies) containing the total power 
        of the signal for each window.
    mel_scale : bool, optional
        Whether to use the mel scale for the filterbank (default: True).

    Returns
    -------
    np.ndarray
        2D array of MFCCs with shape `(n_windows, n_coefficients)`, where each row
        contains the MFCCs for a corresponding window.
    ...

    Raises
    ------
    ValueError
        If the filter points cannot be constructed due to incompatible dimensions.

    Notes
    -----
    - The function includes filterbank normalization to ensure proper scaling.
    - DCT filters are constructed to minimize spectral leakage.
    """
    # Compute window length in samples
    window_length = config.window_length_s * config.sampling_frequency
    
    # Generate filter points
    if mel_scale:
        freqs = np.linspace(
            melscale(config.mfcc_low_frequency), 
            melscale(config.mfcc_high_frequency), 
            num=config.mfcc_n_dct_filters + 2
        )
        freqs = inverse_melscale(freqs)
    else:
        freqs = np.linspace(
            config.mfcc_low_frequency, 
            config.mfcc_high_frequency, 
            num=config.mfcc_n_dct_filters + 2
        )

    filter_points = np.floor(
        (window_length + 1) / config.sampling_frequency * freqs
    ).astype(int)  

    # Construct triangular filterbank
    filters = np.zeros((len(filter_points) - 2, int(window_length / 2 + 1)))
    for j in range(len(filter_points) - 2):
        filters[j, filter_points[j] : filter_points[j + 2]] = signal.windows.triang(
            filter_points[j + 2] - filter_points[j]
        ) 
        # Normalize filter coefficients
        filters[j, :] /= (
            config.sampling_frequency/window_length * np.sum(filters[j,:])
        ) 

    # Apply filterbank to total power
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

    return mfccs


def melscale(x: np.ndarray) -> np.ndarray:
    """
    Maps linear frequency values to the Mel scale.

    Parameters
    ----------
    x : np.ndarray
        Linear frequency values to be converted to the Mel scale.

    Returns
    -------
    np.ndarray
        Frequency values mapped to the Mel scale.
    """
    return 16.22 * np.log10(1 + x / 4.375)


def inverse_melscale(x: np.ndarray) -> np.ndarray:
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
    return 4.375 * (10 ** (x / 16.22) - 1)


def pca_transform_gyroscope(
        config,
        df: pd.DataFrame,
) -> np.ndarray:
    """
    Apply Principal Component Analysis (PCA) to the y-axis and z-axis of the gyroscope signal 
    to extract the velocity in the arm swing direction. PCA is applied only to the data corresponding 
    to predicted gait timestamps to maximize the similarity to the velocity in the arm swing direction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the raw gyroscope data and the predicted gait labels. The gyroscope data should include
        columns for the y-axis and z-axis of the gyroscope, and the predicted gait column (boolean).
        
    config : object
        Configuration object containing the column name for the predicted gait (`pred_gait_colname`).

    Returns
    -------
    np.ndarray
        1D array of the first principal component corresponding to the angular velocity in the arm swing direction. 
        This represents the projected gyroscope signal along the primary axis of variation identified by PCA.
    """
    # Convert gyroscope columns to NumPy arrays
    y_gyro_array = df[DataColumns.GYROSCOPE_Y].to_numpy()
    z_gyro_array = df[DataColumns.GYROSCOPE_Z].to_numpy()

    # Filter data based on predicted gait
    gait_mask = df[DataColumns.PRED_GAIT] == 1
    y_gyro_gait_array = y_gyro_array[gait_mask]
    z_gyro_gait_array = z_gyro_array[gait_mask]

    # Combine columns for PCA
    gait_data = np.column_stack((y_gyro_gait_array, z_gyro_gait_array))
    full_data = np.column_stack((y_gyro_array, z_gyro_array))

    pca = PCA(n_components=2, svd_solver='auto', random_state=22)
    pca.fit(gait_data)
    velocity = pca.transform(full_data)[:, 0]  # First principal component

    return np.asarray(velocity)


def compute_angle(
        config,
        df: pd.DataFrame,
    ) -> np.ndarray:
    """
    Apply cumulative trapezoidal integration to extract the angle from the angular velocity (gyroscope signal).
    The integration is performed over the given time values to obtain the angle estimation.

    Parameters
    ----------
    config : object
        Configuration object containing the column names for velocity (`velocity_colname`) 
        and time (`time_colname`).
        
    df : pd.DataFrame
        DataFrame containing the velocity (angular velocity) and time data.

    Returns
    -------
    np.ndarray
        1D array of the estimated angle, derived from integrating the angular velocity over time.
        The output represents the angle, which is always non-negative due to the use of the absolute value.
    """
    # Ensure input is a NumPy array
    velocity_array = np.asarray(df[DataColumns.VELOCITY])
    time_array = np.asarray(df[DataColumns.TIME])

    # Perform integration and apply absolute value
    angle_array = cumulative_trapezoid(velocity_array, time_array, initial=0)
    return np.abs(angle_array)


def remove_moving_average_angle(
        config,
        df: pd.DataFrame,
    ) -> pd.Series:
    """
    Remove the moving average from the angle to account for potential drift in the signal.
    This method subtracts a centered moving average from the angle signal to remove low-frequency drift.

    Parameters
    ----------
    config : object
        Configuration object containing the angle column name (`angle_colname`) 
        and sampling frequency (`sampling_frequency`).
        
    df : pd.DataFrame
        DataFrame containing the angle data.

    Returns
    -------
    pd.Series
        The estimated angle after removing the moving average, 
        which accounts for potential drift in the signal.
    """
    window_size = int(2 * (config.sampling_frequency * 0.5) + 1)
    angle_ma = df[DataColumns.ANGLE].rolling(window=window_size, min_periods=1, center=True, closed='both').mean()
    
    return df[DataColumns.ANGLE] - angle_ma


def compute_angle_and_velocity_from_gyro(
        config,
        df: pd.DataFrame, 
    ) -> pd.DataFrame:
    """
    Compute both the angle and velocity from the raw gyroscope signal using principal component 
    analysis (PCA) for velocity estimation, and cumulative trapezoidal integration for angle estimation.

    This function processes the raw gyroscope signal to obtain two key outputs:
    1. The velocity, which is extracted from the principal component of the gyroscope's y- and z-axes.
    2. The angle, which is obtained by integrating the angular velocity and removing any drift using a moving average.

    Parameters
    ----------
    config : object
        Configuration object containing necessary column names (`velocity_colname`, `angle_colname`) 
        and other relevant parameters for processing.

    df : pd.DataFrame
        The DataFrame containing raw gyroscope data with at least the y- and z-axes of the gyroscope.

    Returns
    -------
    pd.Series, pd.Series
        Two `pd.Series` objects:
        - The first `pd.Series` corresponds to the estimated angle after integration and drift removal.
        - The second `pd.Series` corresponds to the velocity computed from the gyroscope signal using PCA.
    """
    # Compute the velocity using PCA
    df[DataColumns.VELOCITY] = pca_transform_gyroscope(
        config=config,
        df=df,
    )

    # Integrate angular velocity to obtain the angle
    df[DataColumns.ANGLE] = compute_angle(
        config=config,
        df=df
    )

    # Remove moving average from the angle to correct for drift
    df[DataColumns.ANGLE] = remove_moving_average_angle(
        config=config,
        df=df
    )

    return df[DataColumns.ANGLE], df[DataColumns.VELOCITY]


def extract_angle_extremes(
        config,
        windowed_angle: np.ndarray,
        dominant_frequencies: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Extract angle extrema (minima and maxima) from smoothed angle signals,
    adhering to specific criteria. This function removes consecutive
    identical extrema, alternates between minima and maxima, and computes
    the range of motion for each window of data.

    Parameters
    ----------
    config : object
        Configuration object containing relevant parameters, such as the
        sampling frequency.
        
    windowed_angle : np.ndarray
        A 2D numpy array of shape (N, M) where N is the number of windows,
        and M is the number of samples per window. Each row represents a
        smoothed angle signal for a window.

    Returns
    -------
    angle_extrema_indices : list of np.ndarray
        A list of N numpy arrays, where each array contains the sorted
        indices of the remaining angle extrema (minima and maxima) for the
        corresponding window.
        
    minima_indices : np.ndarray
        A 1D numpy array of objects, where each element is a numpy array
        containing the indices of the minima extrema for the corresponding
        window after processing.
        
    maxima_indices : np.ndarray
        A 1D numpy array of objects, where each element is a numpy array
        containing the indices of the maxima extrema for the corresponding
        window after processing.
    """
    distances = config.sampling_frequency * 0.6 / dominant_frequencies
    prominence = 2  
    n_windows = windowed_angle.shape[0]

    # Find minima and maxima indices for each window
    minima_indices = [
        find_peaks(-windowed_angle[i], distance=distances[i], prominence=prominence)[0]
        for i in range(n_windows)
    ]
    maxima_indices = [
        find_peaks(windowed_angle[i], distance=distances[i], prominence=prominence)[0] 
        for i in range(n_windows)
    ]

    minima_indices = np.array(minima_indices, dtype=object)
    maxima_indices = np.array(maxima_indices, dtype=object)

    # Process each window to remove consecutive identical extrema and ensure alternation
    for window_idx in range(n_windows):
        i_pks = 0
        if minima_indices[window_idx].size > 0 and maxima_indices[window_idx].size > 0:
            if maxima_indices[window_idx][0] > minima_indices[window_idx][0]:
                # Start with a minimum
                while i_pks < minima_indices[window_idx].size - 1 and i_pks < maxima_indices[window_idx].size:
                    if minima_indices[window_idx][i_pks + 1] < maxima_indices[window_idx][i_pks]:
                        if windowed_angle[window_idx][minima_indices[window_idx][i_pks + 1]] < windowed_angle[window_idx][minima_indices[window_idx][i_pks]]:
                            minima_indices[window_idx] = np.delete(minima_indices[window_idx], i_pks)
                        else:
                            minima_indices[window_idx] = np.delete(minima_indices[window_idx], i_pks + 1)
                        i_pks -= 1

                    if i_pks >= 0 and minima_indices[window_idx][i_pks] > maxima_indices[window_idx][i_pks]:
                        if windowed_angle[window_idx][maxima_indices[window_idx][i_pks]] < windowed_angle[window_idx][maxima_indices[window_idx][i_pks - 1]]:
                            maxima_indices[window_idx] = np.delete(maxima_indices[window_idx], i_pks)
                        else:
                            maxima_indices[window_idx] = np.delete(maxima_indices[window_idx], i_pks - 1)
                        i_pks -= 1
                    i_pks += 1

            elif maxima_indices[window_idx][0] < minima_indices[window_idx][0]:
                # Start with a maximum
                while i_pks < maxima_indices[window_idx].size - 1 and i_pks < minima_indices[window_idx].size:
                    if maxima_indices[window_idx][i_pks + 1] < minima_indices[window_idx][i_pks]:
                        if windowed_angle[window_idx][maxima_indices[window_idx][i_pks + 1]] < windowed_angle[window_idx][maxima_indices[window_idx][i_pks]]:
                            maxima_indices[window_idx] = np.delete(maxima_indices[window_idx], i_pks + 1)
                        else:
                            maxima_indices[window_idx] = np.delete(maxima_indices[window_idx], i_pks)
                        i_pks -= 1

                    if i_pks >= 0 and maxima_indices[window_idx][i_pks] > minima_indices[window_idx][i_pks]:
                        if windowed_angle[window_idx][minima_indices[window_idx][i_pks]] < windowed_angle[window_idx][minima_indices[window_idx][i_pks - 1]]:
                            minima_indices[window_idx] = np.delete(minima_indices[window_idx], i_pks - 1)
                        else:
                            minima_indices[window_idx] = np.delete(minima_indices[window_idx], i_pks)
                        i_pks -= 1
                    i_pks += 1

    # Combine remaining extrema and compute range of motion
    angle_extrema_indices = [
        np.sort(np.concatenate([minima_indices[window_idx], maxima_indices[window_idx]])) 
        for window_idx in range(n_windows)
    ]

    return angle_extrema_indices, minima_indices, maxima_indices


def compute_range_of_motion(
        windowed_angle,
        windowed_extrema_indices,
) -> np.ndarray:
    """
    Compute the range of motion (RoM) for each window based on angle extrema.

    Parameters
    ----------
    windowed_angle : np.ndarray
        A 2D numpy array where each row represents a window of angle values.

    windowed_extrema_indices : list of np.ndarray
        A list where each element contains the indices of the extrema (minima and maxima)
        for the corresponding window.

    Returns
    -------
    np.ndarray
        A 1D numpy array where each element is the mean range of motion for the corresponding window.
    """    
    # Extract angle amplitudes (minima and maxima values)
    angle_amplitudes = [
        windowed_angle[window_idx][windowed_extrema_indices[window_idx]] for window_idx in range(windowed_angle.shape[0])
    ]

    # Compute the differences (range of motion) across all windows at once using np.diff
    range_of_motion = np.asarray([np.abs(np.diff(window)) for window in angle_amplitudes], dtype=object)

    # Compute the mean range of motion for each window (if no extrema found, the result will be 0)
    mean_rom = np.array([np.mean(window) if len(window) > 0 else 0 for window in range_of_motion])

    return mean_rom


def compute_peak_angular_velocity(
    velocity_window: np.ndarray,
    angle_extrema_indices: List[np.ndarray],
    minima_indices: np.ndarray,
    maxima_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the forward and backward peak angular velocities for each window.

    The forward peak velocity is the maximum velocity between a maximum peak 
    and the next minimum peak, while the backward peak velocity is the maximum 
    velocity between a minimum peak and the next maximum peak.

    Parameters
    ----------
    velocity_window : numpy.ndarray
        A 2D array of shape (N, M), where N is the number of windows and M is the 
        number of velocity values per window. Each row represents a window containing 
        velocity data.
        
    angle_extrema_indices : list of np.ndarray
        A list of N lists, where each list contains the indices of the extrema 
        (peaks) in the velocity data for the corresponding window.
        
    minima_indices : np.ndarray
       A 1D numpy array of objects, where each element is a numpy array
        containing the indices of the minima extrema for the corresponding
        window after processing.
        
    maxima_indices : np.ndarray
        A 1D numpy array of objects, where each element is a numpy array
        containing the indices of the maxima extrema for the corresponding
        window after processing.

    Returns
    -------
    forward_pav_mean : numpy.ndarray
        A 1D array containing the mean forward peak velocities for each window.
        
    backward_pav_mean : numpy.ndarray
        A 1D array containing the mean backward peak velocities for each window.
        
    forward_pav_std : numpy.ndarray
        A 1D array containing the standard deviation of forward peak velocities for each window.
        
    backward_pav_std : numpy.ndarray
        A 1D array containing the standard deviation of backward peak velocities for each window.
    """
    # Initialize lists to store the peak velocities for each window
    forward_pav = []
    backward_pav = []

    for window_idx in range(velocity_window.shape[0]):
        # Initialize lists for forward and backward peak velocities for the current window
        window_forward_pav = []  
        window_backward_pav = [] 

        if len(minima_indices[window_idx]) > 0 and len(maxima_indices[window_idx]) > 0:
            # Extract the relevant data for the current window
            extrema_indices = angle_extrema_indices[window_idx]
            velocity_array = velocity_window[window_idx]

            for i in range(len(extrema_indices) - 1):
                # Get the current and next extrema index
                current_peak_idx = extrema_indices[i]
                next_peak_idx = extrema_indices[i + 1]
                segment = velocity_array[current_peak_idx:next_peak_idx]

                # Check if the current peak is a minimum or maximum and calculate peak velocity accordingly
                if current_peak_idx in minima_indices[window_idx]:
                    window_forward_pav.append(np.max(np.abs(segment)))
                elif current_peak_idx in maxima_indices[window_idx]:
                    window_backward_pav.append(np.max(np.abs(segment)))

        # Append results of this window to the main lists
        forward_pav.append(window_forward_pav)
        backward_pav.append(window_backward_pav)

    # Convert lists to numpy arrays
    forward_pav = np.array(forward_pav, dtype=object)
    backward_pav = np.array(backward_pav, dtype=object)

    # Calculate the mean and standard deviation for each window
    forward_pav_mean = np.array([np.mean(window) if len(window) > 0 else 0 for window in forward_pav])
    backward_pav_mean = np.array([np.mean(window) if len(window) > 0 else 0 for window in backward_pav])
    forward_pav_std = np.array([np.std(window) if len(window) > 0 else 0 for window in forward_pav])
    backward_pav_std = np.array([np.std(window) if len(window) > 0 else 0 for window in backward_pav])

    return forward_pav_mean, backward_pav_mean, forward_pav_std, backward_pav_std


def extract_temporal_domain_features(
        config, 
        windowed_acc: np.ndarray, 
        windowed_grav: np.ndarray, 
        grav_stats: List[str] = ['mean']
        ) -> pd.DataFrame:
    """
    Compute temporal domain features for the accelerometer signal.

    This function calculates various statistical features for the gravity signal 
    and computes the standard deviation of the accelerometer's Euclidean norm.

    Parameters
    ----------
    config : object
        Configuration object containing the accelerometer and gravity column names.
    windowed_acc : numpy.ndarray
        A 2D numpy array of shape (N, M) where N is the number of windows and M is 
        the number of accelerometer values per window.
    windowed_grav : numpy.ndarray
        A 2D numpy array of shape (N, M) where N is the number of windows and M is 
        the number of gravity signal values per window.
    grav_stats : list of str, optional
        A list of statistics to compute for the gravity signal (default is ['mean']).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the computed features, with each row corresponding 
        to a window and each column representing a specific feature.
    """
    # Compute gravity statistics (e.g., mean, std, etc.)
    feature_dict = {}
    for stat in grav_stats:
        stats_result = compute_statistics(windowed_grav, statistic=stat)
        for i, col in enumerate(config.gravity_cols):
            feature_dict[f'{col}_{stat}'] = stats_result[:, i]

    # Compute standard deviation of the Euclidean norm of the accelerometer signal
    feature_dict['accelerometer_std_norm'] = compute_std_euclidean_norm(windowed_acc)

    return pd.DataFrame(feature_dict)


def extract_spectral_domain_features(
        config, 
        sensor: str, 
        windowed_data: np.ndarray
    ) -> pd.DataFrame:
    """
    Compute spectral domain features for a sensor's data.

    This function computes the periodogram, extracts power in specific frequency bands, 
    calculates the dominant frequency, and computes Mel-frequency cepstral coefficients (MFCCs) 
    for a given sensor's windowed data.

    Parameters
    ----------
    config : object
        Configuration object containing settings such as sampling frequency, window type, 
        frequency bands, and MFCC parameters.
    sensor : str
        The name of the sensor (e.g., 'accelerometer', 'gyroscope').
    windowed_data : numpy.ndarray
        A 2D numpy array where each row corresponds to a window of sensor data.

    Returns
    -------
    dict
        The updated feature dictionary containing the extracted spectral features, including 
        power in frequency bands, dominant frequencies, and MFCCs for each window.
    """
    config.sensor = sensor

    # Initialize a dictionary to hold the results
    feature_dict = {}

    # Compute periodogram (power spectral density)
    freqs, psd = periodogram(windowed_data, fs=config.sampling_frequency, 
                             window=config.window_type, axis=1)

    # Compute power in specified frequency bands
    band_powers  = compute_power_in_bandwidth(config, psd, freqs)

    # Add power band features to the feature_dict
    feature_dict.update(band_powers)

    # Compute dominant frequency for each axis
    dominant_frequencies = compute_dominant_frequency(
        psd=psd, 
        freqs=freqs, 
        fmin=config.spectrum_low_frequency, 
        fmax=config.spectrum_high_frequency
    )

    # Add dominant frequency features to the feature_dict
    for axis, freq in zip(config.axes, dominant_frequencies.T):
        feature_dict[f'{sensor}_{axis}_dominant_frequency'] = freq

    # Compute total power in the PSD
    total_power_psd = compute_total_power(psd)

    # Compute MFCCs
    mfccs = compute_mfccs(
        config,
        total_power_array=total_power_psd,
    )

    # Combine the MFCCs into the features DataFrame
    mfcc_colnames = [f'{sensor}_mfcc_{x}' for x in range(1, config.mfcc_n_coefficients + 1)]
    for i, colname in enumerate(mfcc_colnames):
        feature_dict[colname] = mfccs[:, i]

    return pd.DataFrame(feature_dict)


def extract_angle_features(
        config,
        windowed_angle: np.ndarray,
        windowed_velocity: np.ndarray,
    ) -> pd.DataFrame:
    """
    Extract angle-related features from windowed angle and velocity data.

    This function calculates spectral and temporal features for the angle signal, including:
    - Dominant frequency of the angle signal in specific frequency ranges.
    - Range of motion based on consecutive extrema in the angle signal.
    - Forward and backward peak angular velocities from the velocity signal.

    Parameters
    ----------
    config : object
        Configuration object containing parameters such as sampling frequency, frequency ranges, etc.
    windowed_angle : np.ndarray
        Array of windowed angle data.
    windowed_velocity : np.ndarray
        Array of windowed velocity data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted features from angle and velocity data.

    Notes
    -----
    - The function computes spectral features using the periodogram (power spectral density) of the angle signal.
    - Temporal features like range of motion and angular velocities are calculated from the extrema (minima and maxima) in the angle signal.
    - The extracted features are returned as a DataFrame for easier integration with other data.

    """
    # Initialize an empty dictionary to hold the features
    feature_dict = {}

    # Compute the periodogram (power spectral density) of the angle signal
    freqs, psd = periodogram(windowed_angle, fs=config.sampling_frequency, window=config.window_type, axis=1)

    # Compute dominant frequencies in the angle signal,
    # which is used when detecting peaks
    dominant_freqs_angle_narrow = compute_dominant_frequency(
        psd=psd, 
        freqs=freqs, 
        fmin=config.angle_fmin, 
        fmax=config.angle_fmax
    )

    # Compute dominant frequencies in the angle signal for a broader frequency range
    dominant_freqs_angle_broad = compute_dominant_frequency(
        psd=psd,
        freqs=freqs,
        fmin=config.spectrum_low_frequency,
        fmax=config.spectrum_high_frequency
    )
    feature_dict[f'{DataColumns.ANGLE}_dominant_frequency'] = dominant_freqs_angle_broad

    # Extract extrema (minima and maxima) indices for the angle signal
    angle_extrema_indices, minima_indices, maxima_indices = extract_angle_extremes(
        config=config,
        windowed_angle=windowed_angle,
        dominant_frequencies=dominant_freqs_angle_narrow,
    )

    # Calculate range of motion based on extrema indices
    feature_dict['range_of_motion'] = compute_range_of_motion(
        windowed_angle=windowed_angle,
        windowed_extrema_indices=angle_extrema_indices,
    )

    # Compute the forward and backward peak angular velocities
    forward_peak_velocity_mean, backward_peak_velocity_mean, forward_peak_velocity_std, backward_peak_velocity_std = compute_peak_angular_velocity(
        velocity_window=windowed_velocity,
        angle_extrema_indices=angle_extrema_indices,
        minima_indices=minima_indices,
        maxima_indices=maxima_indices,
    )

    # Add the angular velocity features to the dictionary
    feature_dict['forward_peak_velocity_mean'] = forward_peak_velocity_mean
    feature_dict['backward_peak_velocity_mean'] = backward_peak_velocity_mean
    feature_dict['forward_peak_velocity_std'] = forward_peak_velocity_std
    feature_dict['backward_peak_velocity_std'] = backward_peak_velocity_std
    
    return pd.DataFrame(feature_dict)