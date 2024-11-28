from typing import List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scipy import signal, fft
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks, periodogram

from paradigma.constants import DataColumns
from paradigma.segmenting import tabulate_windows, create_segments, discard_segments


def compute_statistics(data: np.ndarray, statistic: str) -> np.ndarray:
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
    Generate the standard deviation of the Euclidean norm of sensor axes.

    Parameters
    ----------
    data: np.ndarray
        3D array where each entry corresponds to a window (rows) and each axis (columns).

    Returns
    -------
    np.ndarray
        1D array of standard deviations for each window's norm.
    """
    # Compute the norm for each timestamp in each window
    norms = np.linalg.norm(data, axis=2)  # Norm along axis 2 (sensor axes)
    # Compute standard deviation of the norm per window
    return np.std(norms, axis=1)


def compute_power_in_bandwidth(config, psd, freqs):
    band_powers = {}
    for band_name, (low, high) in config.d_frequency_bandwidths.items():
        band_mask = (freqs >= low) & (freqs < high)
        
        band_power = np.log10(np.trapz(psd[:, band_mask, :], freqs[band_mask], axis=1) + 1e-10)
        
        band_powers.update({
            f'{config.sensor}_x_{band_name}': band_power[:, 0],
            f'{config.sensor}_y_{band_name}': band_power[:, 1],
            f'{config.sensor}_z_{band_name}': band_power[:, 2],
        })

    return pd.DataFrame(band_powers)


def compute_total_power(psd):
    return np.sum(psd, axis=-1) # Sum across frequency bins


def compute_dominant_frequency(psd, freqs, fmin, fmax):
    # Ensure fmin and fmax are within the bounds of the frequency array
    if fmin < freqs[0] or fmax > freqs[-1]:
        raise ValueError(f"fmin {fmin} or fmax {fmax} are out of bounds of the frequency array.")
    
    # Get the indices corresponding to fmin and fmax
    min_index = np.searchsorted(freqs, fmin)
    max_index = np.searchsorted(freqs, fmax)

    # Slice the PSD and frequency array to the desired range
    psd_filtered = psd[:, min_index:max_index]
    freqs_filtered = freqs[min_index:max_index]

    if psd.ndim == 3:
        return np.array([
            freqs_filtered[np.argmax(psd_filtered[:, :, i], axis=1)]
            for i in range(psd.shape[-1])
        ]).T
    elif psd.ndim == 2:
        return freqs_filtered[np.argmax(psd_filtered, axis=1)]
    else:
        raise ValueError("PSD array must be 2D or 3D.")


def compute_mfccs(
        config,
        total_power_array: np.ndarray,
        mel_scale: bool = True,
        ) -> pd.DataFrame:
    """Generate Mel Frequency Cepstral Coefficients from the total power of the signal.
    
    Parameters
    ----------
    total_power_array: np.ndarray
        The total power of the signal, extracted using compute_power
    window_length_s: int
        The number of seconds a window constitutes
    sampling_frequency: int
        The sampling frequency of the data (default: 100)
    low_frequency: int
        The lower bound of the frequency band (default: 0)
    high_frequency: int
        The upper bound of the frequency band (default: 25)
    n_filters: int
        The number of DCT filters (default: 20)
    n_coefficients: int
        The number of coefficients to extract (default: 12)
    
    Returns
    -------
    pd.DataFrame
        A dataframe with a single column corresponding to a single MFCC
    """
    window_length = config.window_length_s * config.sampling_frequency
    
    # compute filter points
    if mel_scale:
        freqs = np.linspace(melscale(config.mfcc_low_frequency), melscale(config.mfcc_high_frequency), num=config.mfcc_n_dct_filters + 2)
        freqs = inverse_melscale(freqs)
    else:
        freqs = np.linspace(config.mfcc_low_frequency, config.mfcc_high_frequency, num=config.mfcc_n_dct_filters + 2)

    filter_points = np.floor((window_length + 1) / config.sampling_frequency * freqs).astype(int)  

    # construct filterbank
    filters = np.zeros((len(filter_points) - 2, int(window_length / 2 + 1)))
    for j in range(len(filter_points) - 2):
        filters[j, filter_points[j] : filter_points[j + 2]] = signal.windows.triang(filter_points[j + 2] - filter_points[j]) # triangular filters based on edges
        filters[j, :] /= (config.sampling_frequency/window_length * np.sum(filters[j,:])) # normalization of the filter coefficients

    # filter signal
    power_filtered = np.dot(total_power_array, filters.T) 
    
    # convert to log scale
    log_power_filtered = np.log10(power_filtered)

    # generate cepstral coefficients
    dct_filters = np.empty((config.mfcc_n_coefficients, config.mfcc_n_dct_filters))
    dct_filters[0, :] = 1.0 / np.sqrt(config.mfcc_n_dct_filters)

    samples = np.arange(1, 2 * config.mfcc_n_dct_filters, 2) * np.pi / (2.0 * config.mfcc_n_dct_filters)

    for i in range(1, config.mfcc_n_coefficients):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / config.mfcc_n_dct_filters)

    mfccs = np.dot(log_power_filtered, dct_filters.T) 

    return mfccs


def melscale(x):
    "Maps values of x to the melscale"
    return 16.22 * np.log10(1 + x / 4.375)

def inverse_melscale(x):
    "Inverse of the melscale"
    return 4.375 * (10 ** (x / 16.22) - 1)


def pca_transform_gyroscope(
        config,
        df: pd.DataFrame,
) -> np.ndarray:
    """Apply principal component analysis (PCA) on the y-axis and z-axis of the raw gyroscope signal
    to extract the velocity. PCA is applied to the predicted gait timestamps only to maximize the similarity
    to the velocity in the arm swing direction. 
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the gyroscope data
    y_gyro_colname: str
        The column name of the y-axis of the gyroscope
    z_gyro_colname: str
        The column name of the z-axis of the gyroscope
    pred_gait_colname: str
        The column name of the predicted gait boolean

    Returns
    -------
    pd.Series
        The first principal component corresponding to the angular velocity in the arm swing direction
    """
    # Convert gyroscope columns to NumPy arrays
    y_gyro_array = df[DataColumns.GYROSCOPE_Y].to_numpy()
    z_gyro_array = df[DataColumns.GYROSCOPE_Z].to_numpy()

    # Filter data based on predicted gait
    gait_mask = df[config.pred_gait_colname] == 1
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
    """Apply cumulative trapezoidal integration to extract the angle from the velocity.
    
    Parameters
    ----------
    velocity_col: pd.Series
        The angular velocity (gyroscope) column to be integrated
    time_col: pd.Series
        The time column corresponding to the angular velocity
        
    Returns
    -------
    pd.Series
        An estimation of the angle extracted from the angular velocity
    """
    # Ensure input is a NumPy array
    velocity_array = np.asarray(df[config.velocity_colname])
    time_array = np.asarray(df[config.time_colname])

    # Perform integration and apply absolute value
    angle_array = cumulative_trapezoid(velocity_array, time_array, initial=0)
    return np.abs(angle_array)


def remove_moving_average_angle(
        config,
        df: pd.DataFrame,
    ) -> pd.Series:
    """Remove the moving average from the angle to account for potential drift in the signal.
    
    Parameters
    ----------
    angle_col: pd.Series
        The angle column to be processed, obtained using compute_angle
    sampling_frequency: int
        The sampling frequency of the data (default: 100)
        
    Returns
    -------
    pd.Series
        The estimated angle without potential drift
    """
    window_size = int(2*(config.sampling_frequency*0.5)+1)
    angle_ma = df[config.angle_colname].rolling(window=window_size, min_periods=1, center=True, closed='both').mean()
    
    return df[config.angle_colname] - angle_ma


def compute_angle_and_velocity_from_gyro(
        config,
        df: pd.DataFrame, 
        ) -> pd.DataFrame:
    df[config.velocity_colname] = pca_transform_gyroscope(
        config=config,
        df=df,
    )

    # integrate the angular velocity to obtain an estimation of the angle
    df[config.angle_colname] = compute_angle(
        velocity_col=df[config.velocity_colname],
        time_col=df[config.time_colname]
    )

    # remove the moving average from the angle to account for possible drift caused by the integration
    # of noise in the angular velocity
    df[config.angle_colname] = remove_moving_average_angle(
        config=config,
        df=df
    )

    return df[config.angle_colname], df[config.velocity_colname]


def extract_angle_extremes(
        config,
        windowed_angle: np.ndarray,
        dominant_frequencies: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
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
    mean_rom : np.ndarray
        A 1D numpy array of shape (N,) where each element is the mean range
        of motion for the corresponding window. If no extrema are found in
        a window, the mean range of motion is 0.
        
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
    
    angle_amplitudes = [
        windowed_angle[window_idx][windowed_extrema_indices[window_idx]] for window_idx in range(windowed_angle.shape[0])
    ]

    range_of_motion = np.asarray([np.abs(np.diff(window)) for window in angle_amplitudes], dtype=object)
    mean_rom = np.array([np.mean(window) if len(window) > 0 else 0 for window in range_of_motion])

    return mean_rom


def compute_peak_angular_velocity(
        velocity_window,
        angle_extrema_indices,
        minima_indices,
        maxima_indices,
) -> np.ndarray:
    """
    Calculate the forward and backward peak velocities for each window of data.

    The forward peak velocity is the maximum value of the velocity between 
    a maximum peak and the next minimum peak, while the backward peak velocity
    is the maximum value of the velocity between a minimum peak and the next 
    maximum peak.

    Parameters
    ----------
    velocity_window : list of numpy arrays
        A list of N windows, each containing a 1D numpy array of velocity values.
        
    angle_extrema_indices : list of lists of integers
        A list of N lists, where each list contains the indices of the extrema 
        (peaks) in the velocity data for the corresponding window.
        
    minima_indices : list of lists of integers
        A list of N lists, where each list contains the indices of the minimum 
        extrema for the corresponding window.
        
    maxima_indices : list of lists of integers
        A list of N lists, where each list contains the indices of the maximum 
        extrema for the corresponding window.

    Returns
    -------
    forward_pav : numpy array
        A 2D numpy array where each row contains the forward peak velocities 
        for the corresponding window.
        
    backward_pav : numpy array
        A 2D numpy array where each row contains the backward peak velocities 
        for the corresponding window.
    """
    # Initialize lists to store the forward and backward peak velocities
    forward_pav = []
    backward_pav = []

    for window_idx in range(velocity_window.shape[0]):
        if len(minima_indices[window_idx]) > 0 and len(maxima_indices[window_idx]) > 0:
            # Prepare arrays to store forward and backward peak velocities for this window
            window_forward_pav = []  
            window_backward_pav = [] 

            # Get the extrema indices for this window
            extrema_indices = angle_extrema_indices[window_idx]
            # Convert the velocity window to an array if it's not already
            velocity_array = velocity_window[window_idx]

            for i in range(len(extrema_indices) - 1):
                current_peak_idx = extrema_indices[i]
                next_peak_idx = extrema_indices[i + 1]
                segment = velocity_array[current_peak_idx:next_peak_idx]

                # Check if current peak is a minimum
                if current_peak_idx in minima_indices[window_idx]:
                    # Calculate forward peak velocity between this maximum and the next minimum
                    window_forward_pav.append(np.max(np.abs(segment)))

                elif current_peak_idx in maxima_indices[window_idx]:
                    # Calculate backward peak velocity between this minimum and the next maximum
                    window_backward_pav.append(np.max(np.abs(segment)))

            # Append results of this window to the main lists
            forward_pav.append(window_forward_pav)
            backward_pav.append(window_backward_pav)

        else:
            forward_pav.append([])
            backward_pav.append([])

    # Convert results to numpy arrays
    forward_pav = np.array(forward_pav, dtype=object)
    backward_pav = np.array(backward_pav, dtype=object)

    forward_pav_mean = np.array([np.mean(window) if len(window) > 0 else 0 for window in forward_pav])
    backward_pav_mean = np.array([np.mean(window) if len(window) > 0 else 0 for window in backward_pav])
    forward_pav_std = np.array([np.std(window) if len(window) > 0 else 0 for window in forward_pav])
    backward_pav_std = np.array([np.std(window) if len(window) > 0 else 0 for window in backward_pav])

    return forward_pav_mean, backward_pav_mean, forward_pav_std, backward_pav_std


def extract_temporal_domain_features(
        config, 
        windowed_acc: np.ndarray, 
        windowed_grav: np.ndarray, 
        l_grav_stats: List[str] = ['mean']
        ) -> pd.DataFrame:
    """
    Compute temporal domain features for the accelerometer signal.
    """
    df_features = pd.DataFrame()

    # Compute statistics for gravity columns
    for stat in l_grav_stats:
        stats_result = compute_statistics(windowed_grav, statistic=stat)
        for i, col in enumerate(config.l_gravity_cols):
            df_features[f'{col}_{stat}'] = stats_result[:, i]

    # Compute standard deviation of the norm
    df_features['std_norm'] = compute_std_euclidean_norm(windowed_acc)

    return df_features


def extract_spectral_domain_features(
        config, 
        sensor: str, 
        windowed_data: np.ndarray
        ) -> pd.DataFrame:
    
    config.sensor = sensor

    freqs, psd = periodogram(windowed_data, fs=config.sampling_frequency, window=config.window_type, axis=1)

    df_features = compute_power_in_bandwidth(config, psd, freqs)

    freq_colnames = [f'{sensor}_{axis}_dominant_frequency' for axis in config.l_axes]
    df_features[freq_colnames] = compute_dominant_frequency(
        psd=psd, 
        freqs=freqs, 
        fmin=config.spectrum_low_frequency, 
        fmax=config.spectrum_high_frequency
    )

    total_power_psd = compute_total_power(psd)

    mfccs = compute_mfccs(
        config,
        total_power_array=total_power_psd,
    )

    df_features = pd.concat([df_features, pd.DataFrame(mfccs, columns=[f'{sensor}_mfcc_{x}' for x in range(1, mfccs.shape[1]+1)])], axis=1)

    return df_features


def extract_angle_features(
        config,
        windowed_angle: np.ndarray,
        windowed_velocity: np.ndarray,
    ) -> pd.DataFrame:

    df_features = pd.DataFrame()

    freqs, psd = periodogram(windowed_angle, fs=config.sampling_frequency, window=config.window_type, axis=1)

    dominant_freqs_angle = compute_dominant_frequency(
        psd=psd, 
        freqs=freqs, 
        fmin=config.angle_fmin, 
        fmax=config.angle_fmax
    )

    df_features['dominant_frequency'] = compute_dominant_frequency(
        psd=psd,
        freqs=freqs,
        fmin=config.spectrum_low_frequency,
        fmax=config.spectrum_high_frequency
    )

    # determine the extrema (minima and maxima) of the angle signal
    angle_extrema_indices, minima_indices, maxima_indices = extract_angle_extremes(
        config=config,
        windowed_angle=windowed_angle,
        dominant_frequencies=dominant_freqs_angle,
    )

    # calculate the change in angle between consecutive extrema (minima and maxima) of the angle signal inside the window
    df_features['range_of_motion'] = compute_range_of_motion(
        windowed_angle=windowed_angle,
        angle_extrema_indices=angle_extrema_indices,
    )

    # compute the forward and backward peak angular velocity using the extrema of the angular velocity
    df_features['forward_peak_velocity_mean'], df_features['backward_peak_velocity_mean'], df_features['forward_peak_velocity_std'], df_features['backward_peak_velocity_std'] = compute_peak_angular_velocity(
        velocity_window=windowed_velocity,
        angle_extrema_indices=angle_extrema_indices,
        minima_indices=minima_indices,
        maxima_indices=maxima_indices,
    )
    
    return df_features