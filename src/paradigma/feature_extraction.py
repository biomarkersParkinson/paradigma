from typing import List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scipy import signal, fft
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks

from paradigma.constants import DataColumns
from paradigma.gait_analysis_config import IMUConfig


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
        The statistic to be computed [mean, std, max, min]
        
    Returns
    -------
    list
        The aggregated statistics
    """
    if statistic == 'mean':
        return [np.mean(x) for x in sensor_col]
    elif statistic == 'std':
        return [np.std(x) for x in sensor_col]
    elif statistic == 'max':
        return [np.max(x) for x in sensor_col]
    elif statistic == 'min':
        return [np.min(x) for x in sensor_col]
    else:
        raise ValueError("Statistic not recognized.")


def generate_std_norm(
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.Series:
    """Generate the standard deviation of the norm of the accelerometer axes.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the accelerometer axes
    cols: List[str]
        The names of the columns containing the accelerometer axes
        
    Returns
    -------
    pd.Series
        The standard deviation of the norm of the accelerometer axes
    """
    return df[cols].apply(
        lambda row: np.std(np.sqrt(np.sum(row**2))),
        axis=1
    )
    

def compute_fft(
        values: list,
        window_type: str = 'hann',
        sampling_frequency: int = 100,
    ) -> tuple:
    """Compute the Fast Fourier Transform (FFT) of a signal.
    
    Parameters
    ----------
    values: list
        The values of the signal (e.g., accelerometer data) of a single window.
    window_type: str
        The type of window to be used for the FFT (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
        
    Returns
    -------
    tuple
        The FFT values and the corresponding frequencies
    """
    w = signal.get_window(window_type, len(values), fftbins=False)
    yf = 2*fft.fft(values*w)[:int(len(values)/2+1)]
    xf = fft.fftfreq(len(values), 1/sampling_frequency)[:int(len(values)/2+1)]

    return yf, xf
    

def signal_to_ffts(
        sensor_col: pd.Series,
        window_type: str = 'hann',
        sampling_frequency: int = 100,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute the Fast Fourier Transform (FFT) of a signal per window (can probably be combined with compute_fft and simplified).

    Parameters
    ----------
    sensor_col: pd.Series
        The sensor column to be transformed (e.g. x-axis of accelerometer)
    window_type: str
        The type of window to be used for the FFT (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    
    Returns
    -------
    tuple
        Two lists: (frequencies, FFT values), which can be concatenated as columns to the DataFrame.
    """
    # Use list comprehensions to compute FFT for each window
    l_freqs_total = [compute_fft(values=row, window_type=window_type, sampling_frequency=sampling_frequency)[1]
                     for row in sensor_col]
    l_values_total = [compute_fft(values=row, window_type=window_type, sampling_frequency=sampling_frequency)[0]
                      for row in sensor_col]

    return l_freqs_total, l_values_total
    

def compute_power_in_bandwidth(
        sensor_col: pd.Series,
        fmin: float,
        fmax: float,
        sampling_frequency: int = 100,
        window_type: str = 'hann',
    ) -> pd.Series:
    """Computes the power in a specific frequency band for a specified sensor and axis.
    
    Parameters
    ----------
    ssensor_col: pd.Series
        A pandas Series where each entry is a list of sensor values for one window.
    fmin: float
        The lower bound of the frequency band
    fmax: float
        The upper bound of the frequency band
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    window_type: str
        The type of window to be used for the FFT (default: 'hann')

    Returns
    -------
    float
        The power in the specified frequency band, or NaN if no valid frequencies are found.
    """
    # Function to compute power for a single window
    def compute_single_window(window: list) -> float:
        # Compute the power spectral density (PSD) using periodogram
        fxx, pxx = signal.periodogram(window, fs=sampling_frequency, window=window_type)
        
        # Ensure fmin and fmax are within the frequency range
        if fmin < fxx[0] or fmax > fxx[-1]:
            return np.nan
        
        # Get indices corresponding to fmin and fmax
        ind_min = np.argmax(fxx >= fmin)
        ind_max = np.argmax(fxx >= fmax)
        
        # Compute power in the specified frequency band
        band_power = np.trapz(pxx[ind_min:ind_max], fxx[ind_min:ind_max])
        
        # Handle log calculation (avoid log(0) or negative values)
        return np.log10(band_power) if band_power > 0 else np.nan
    
    # Apply the function to each window in the series
    return sensor_col.apply(compute_single_window)


def compute_perc_power(
        sensor_col: pd.Series,
        fmin_band: float,
        fmax_band: float,
        fmin_total: float = 0,
        fmax_total: float = 100,
        sampling_frequency: int = 100,
        window_type: str = 'hann'
    ) -> float:
    """
    Computes the percentage of power in a specific frequency band for each window in the sensor_col.
    
    Parameters
    ----------
    sensor_col: pd.Series
        A pandas Series, where each entry is a list of sensor values for one window.
    fmin_band: float
        The lower bound of the frequency band
    fmax_band: float
        The upper bound of the frequency band
    fmin_total: float
        The lower bound of the frequency spectrum (default: 0)
    fmax_total: float
        The upper bound of the frequency spectrum (default: 100)
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    window_type: str
        The type of window to be used for the FFT (default: 'hann')
    
    Returns
    -------
    pd.Series
        A pandas Series containing the percentage of power in the specified frequency band for each window.
    """
    # Function to compute power percentage for a single window
    def compute_single_window(window: list) -> float:
        angle_power_band = compute_power_in_bandwidth(
            sensor_col=window,
            fmin=fmin_band,
            fmax=fmax_band,
            sampling_frequency=sampling_frequency,
            window_type=window_type
        )
        
        angle_power_total = compute_power_in_bandwidth(
            sensor_col=window,
            fmin=fmin_total,
            fmax=fmax_total,
            sampling_frequency=sampling_frequency,
            window_type=window_type
        )
        
        if angle_power_total > 0 and not np.isnan(angle_power_total):
            return angle_power_band / angle_power_total
        else:
            return np.nan
    
    # Apply the function to each window in the series
    return sensor_col.apply(compute_single_window)


def get_dominant_frequency(
        signal_ffts: list,
        signal_freqs: list,
        fmin: float,
        fmax: float
        ) -> float:
    """Note: signal_ffts and signal_freqs are single cells (which corresponds to a single window) of signal_ffts and signal_freqs, as it is used with apply function.
    
    Computes the dominant frequency in a specific frequency band.
    
    Parameters
    ----------
    signal_ffts: list
        The FFT values of the signal of a single window
    signal_freqs: list
        The corresponding frequencies of the FFT values
    fmin: int
        The lower bound of the frequency band
    fmax: int
        The upper bound of the frequency band
    
    Returns
    -------
    float
        The dominant frequency in the specified frequency band, or NaN if no valid frequency is found.
    """
    # Find the indices of frequencies within the specified band
    valid_indices = np.where((signal_freqs > fmin) & (signal_freqs < fmax))[0]
    
    # If no valid indices are found, return NaN
    if len(valid_indices) == 0:
        return np.nan
    
    # Extract the corresponding FFT values and frequencies
    signal_ffts_band = np.abs(signal_ffts[valid_indices])
    signal_freqs_band = signal_freqs[valid_indices]
    
    # Find the index of the maximum FFT magnitude in the band
    dominant_index = np.argmax(signal_ffts_band)
    
    # Return the corresponding frequency as the dominant frequency
    return signal_freqs_band[dominant_index]
    

def compute_power(
        df: pd.DataFrame,
        fft_cols: list
    ) -> pd.Series:
    """Compute the power of the FFT values.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the FFT values
    fft_cols: list
        The names of the columns containing the FFT values
    
    Returns
    -------
    pd.Series
        The power of the FFT values
    """
    # Compute the power for each FFT column and sum them up directly
    total_power = df[fft_cols].apply(
        lambda row: sum(np.square(np.abs(row))), axis=1
    )

    return total_power
    

def generate_cepstral_coefficients(
        total_power_col: pd.Series,
        window_length_s: int,
        sampling_frequency: int = 100,
        low_frequency: int = 0,
        high_frequency: int = 25,
        n_filters: int = 20,
        n_coefficients: int = 12,
        ) -> pd.DataFrame:
    """Generate cepstral coefficients from the total power of the signal.
    
    Parameters
    ----------
    total_power_col: pd.Series
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
        A dataframe with a single column corresponding to a single cepstral coefficient
    """
    # Determine window length in samples
    window_length = window_length_s * sampling_frequency
    
    # Compute the filter points based on frequency band limits
    freqs = np.linspace(low_frequency, high_frequency, n_filters + 2)
    filter_points = np.floor((window_length + 1) / sampling_frequency * freqs).astype(int)
    
    # Construct the filterbank
    filters = np.zeros((n_filters, window_length // 2 + 1))
    for j in range(n_filters):
        filters[j, filter_points[j]:filter_points[j+1]] = np.linspace(0, 1, filter_points[j+1] - filter_points[j])
        filters[j, filter_points[j+1]:filter_points[j+2]] = np.linspace(1, 0, filter_points[j+2] - filter_points[j+1])
    
    # Apply filterbank to power spectrum
    power_filtered = np.dot(np.vstack(total_power_col.values), filters.T)
    
    # Take the log of the filtered power (log-mel filtering step)
    log_power_filtered = 10 * np.log10(np.maximum(power_filtered, 1e-10))  # Avoid log(0) with a small epsilon
    
    # Create DCT filters (for the cepstral coefficients)
    samples = np.arange(1, 2 * n_filters, 2) * np.pi / (2.0 * n_filters)
    dct_filters = np.sqrt(2.0 / n_filters) * np.cos(np.outer(np.arange(n_coefficients), samples))
    dct_filters[0, :] /= np.sqrt(2)  # First row adjustment
    
    # Compute cepstral coefficients
    cepstral_coefs = np.dot(log_power_filtered, dct_filters.T)
    
    # Return as a DataFrame with named columns
    return pd.DataFrame(cepstral_coefs, columns=[f'cc_{i+1}' for i in range(n_coefficients)])


def pca_transform_gyroscope(
        df: pd.DataFrame,
        y_gyro_colname: str,
        z_gyro_colname: str,
        pred_gait_colname: str,
) -> pd.Series:
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
    pca = PCA(n_components=2, svd_solver='auto', random_state=22)
    pca.fit([(i,j) for i,j in zip(df.loc[df[pred_gait_colname]==1, y_gyro_colname], df.loc[df[pred_gait_colname]==1, z_gyro_colname])])
    yz_gyros = pca.transform([(i,j) for i,j in zip(df[y_gyro_colname], df[z_gyro_colname])])

    velocity = [x[0] for x in yz_gyros]

    return pd.Series(velocity)


def compute_angle(
        velocity_col: pd.Series,
        time_col: pd.Series,
    ) -> pd.Series:
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
    angle_col = cumulative_trapezoid(velocity_col, time_col, initial=0)
    return pd.Series([x*-1 if x<0 else x for x in angle_col])


def remove_moving_average_angle(
        angle_col: pd.Series,
        sampling_frequency: int = 100,
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
    angle_ma = angle_col.rolling(window=int(2*(sampling_frequency*0.5)+1), min_periods=1, center=True, closed='both').mean()
    
    return pd.Series(angle_col - angle_ma)


def extract_angle_extremes(
        df: pd.DataFrame,
        angle_colname: str,
        dominant_frequency_colname: str,
        sampling_frequency: int = 100,
    ) -> pd.Series:
    """Extract the peaks of the angle (minima and maxima) from the smoothed angle signal that adhere to a set of specific requirements.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the angle signal
    angle_colname: str
        The name of the column containing the smoothed angle signal
    dominant_frequency_colname: str
        The name of the column containing the dominant frequency
    sampling_frequency: int
        The sampling frequency of the data (default: 100)

    Returns
    -------
    pd.Series
        The extracted angle extremes (peaks)
    """

    # Determine gap between peaks based on dominant frequency
    distances = sampling_frequency * 0.6 / df[dominant_frequency_colname]

    # Find peaks (maxima) and troughs (minima)
    df['angle_maxima'] = df[angle_colname].apply(lambda x: find_peaks(x, distance=distances, prominence=2)[0])
    df['angle_minima'] = df[angle_colname].apply(lambda x: find_peaks(-x, distance=distances, prominence=2)[0])

    # Create new columns for extrema that adhere to criteria
    df['angle_new_minima'] = df['angle_minima']
    df['angle_new_maxima'] = df['angle_maxima']

    for index, row in df.iterrows():
        i_pks = 0  # iterable to keep track of consecutive min-min and max-max versus min-max
        minima = row['angle_new_minima']
        maxima = row['angle_new_maxima']

        # Ensure we have both minima and maxima to compare
        while i_pks < min(len(minima), len(maxima)):
            # Check for consecutive minima or maxima and remove the less extreme ones
            if minima[i_pks] < maxima[i_pks]:  # Start with a minimum
                # If consecutive minima, keep the smaller one
                if i_pks + 1 < len(minima) and minima[i_pks + 1] < maxima[i_pks]:
                    if row[angle_colname][minima[i_pks + 1]] < row[angle_colname][minima[i_pks]]:
                        minima = np.delete(minima, i_pks)
                    else:
                        minima = np.delete(minima, i_pks + 1)
                else:
                    i_pks += 1
            else:  # Start with a maximum
                if i_pks + 1 < len(maxima) and maxima[i_pks + 1] < minima[i_pks]:
                    if row[angle_colname][maxima[i_pks + 1]] > row[angle_colname][maxima[i_pks]]:
                        maxima = np.delete(maxima, i_pks)
                    else:
                        maxima = np.delete(maxima, i_pks + 1)
                else:
                    i_pks += 1
        
        # Store the updated lists
        df.at[index, 'angle_new_minima'] = minima
        df.at[index, 'angle_new_maxima'] = maxima

    # Handle any potential scalar/vector issues
    df['angle_extrema_values'] = df.apply(
        lambda x: [x[angle_colname][i] for i in sorted(np.concatenate([x['angle_new_minima'], x['angle_new_maxima']]))], axis=1
    )

    return df['angle_extrema_values']


def extract_range_of_motion(
        angle_extrema_values_col: pd.Series,
    ) -> pd.Series:
    """Extract the range of motion from the angle extrema values.
    
    Parameters
    ----------
    angle_extrema_values_col: pd.Series
        The column containing the angle extrema values
    
    Returns
    -------
    pd.Series
        The range of motion
    """
    def compute_amplitude(extrema_values):
        # Calculate amplitude for consecutive extrema
        amplitudes = []
        for i in range(len(extrema_values) - 1):
            value, next_value = extrema_values[i], extrema_values[i + 1]
            if (value > 0 and next_value < 0) or (value < 0 and next_value > 0):
                # Opposite sign: amplitude is sum of absolute values
                amplitudes.append(abs(value) + abs(next_value))
            else:
                # Same sign: amplitude is the absolute difference
                amplitudes.append(abs(next_value) - abs(value))
        return amplitudes
    
    # Apply amplitude computation to each row of extrema values
    angle_amplitudes = angle_extrema_values_col.apply(compute_amplitude)
    
    # Flatten the list of amplitudes for each window into a single list
    return pd.Series([amplitude for sublist in angle_amplitudes for amplitude in sublist])


def extract_peak_angular_velocity(
        df: pd.DataFrame,
        velocity_colname: str,
        angle_minima_colname: str,
        angle_maxima_colname: str,
) -> pd.DataFrame:
    """Extract the forward and backward peak angular velocity from the angular velocity.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the angular velocity
    velocity_colname: str
        The column name of the angular velocity
    angle_minima_colname: str
        The column name of the column containing the angle minima
    angle_maxima_colname: str
        The column name of the column containing the angle maxima
        
    Returns
    -------
    pd.DataFrame
        The dataframe with the forward and backward peak angular velocity
    """
    def compute_peak_velocity(extrema_indices, velocities, minima, maxima):
        """Helper function to compute forward and backward peak velocities."""
        forward_peaks, backward_peaks = [], []
        
        for i in range(len(extrema_indices) - 1):
            peak_index, next_index = extrema_indices[i], extrema_indices[i + 1]
            
            # Forward peak: between maxima and next extremum
            if peak_index in maxima:
                forward_peaks.append(np.abs(np.min(velocities[peak_index:next_index])))
            
            # Backward peak: between minima and next extremum
            if peak_index in minima:
                backward_peaks.append(np.abs(np.max(velocities[peak_index:next_index])))
        
        return forward_peaks, backward_peaks

    # Apply the peak velocity extraction for each row in the DataFrame
    result = df.apply(lambda row: compute_peak_velocity(
        np.sort(np.concatenate([row[angle_minima_colname], row[angle_maxima_colname]])),
        row[velocity_colname],
        row[angle_minima_colname],
        row[angle_maxima_colname]
    ), axis=1)
    
    # Split the results into two separate columns
    df['forward_peak_ang_vel'], df['backward_peak_ang_vel'] = zip(*result)
    
    return df


def extract_temporal_domain_features(config: IMUConfig, df_windowed:pd.DataFrame, l_gravity_stats=['mean', 'std']) -> pd.DataFrame:
    """
    Compute temporal domain features for the accelerometer signal. The features are added to the dataframe. Therefore the original dataframe is modified, and the modified dataframe is returned.

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
    
    # compute the mean and standard deviation of the gravity component of the acceleration signal for each axis
    for col in config.l_gravity_cols:
        for stat in l_gravity_stats:
            df_windowed[f'{col}_{stat}'] = generate_statistics(
                sensor_col=df_windowed[col],
                statistic=stat
                )

    # compute the standard deviation of the Euclidean norm of the three axes
    df_windowed['std_norm_acc'] = generate_std_norm(
        df=df_windowed,
        cols=config.l_accelerometer_cols
        )
    
    df_windowed = df_windowed.drop(columns=config.l_gravity_cols)
    
    return df_windowed


def extract_spectral_domain_features(config, df_windowed, sensor, l_sensor_colnames):

    for col in l_sensor_colnames:

        # transform the temporal signal to the spectral domain using the fast fourier transform
        df_windowed[f'{col}_freqs'], df_windowed[f'{col}_fft'] = signal_to_ffts(
            sensor_col=df_windowed[col],
            window_type=config.window_type,
            sampling_frequency=config.sampling_frequency
            )

        # compute the power in distinct frequency bandwidths
        for bandwidth, frequencies in config.d_frequency_bandwidths.items():
            df_windowed[col + '_' + bandwidth] = compute_power_in_bandwidth(
                sensor_col=df_windowed[col],
                fmin=frequencies[0],
                fmax=frequencies[1],
                sampling_frequency=config.sampling_frequency,
                window_type=config.window_type,
            )
            

        # compute the dominant frequency, i.e., the frequency with the highest power
        df_windowed[col+'_dominant_frequency'] = df_windowed.apply(lambda x: get_dominant_frequency(
            signal_ffts=x[col+'_fft'], 
            signal_freqs=x[col+'_freqs'],
            fmin=config.spectrum_low_frequency,
            fmax=config.spectrum_high_frequency
            ), axis=1
        )

    # compute the power summed over the individual frequency bandwidths to obtain the total power
    df_windowed['total_power'] = compute_power(
        df=df_windowed,
        fft_cols=[f'{col}_fft' for col in l_sensor_colnames])

    # compute the cepstral coefficients of the total power signal
    cc_cols = generate_cepstral_coefficients(
        total_power_col=df_windowed['total_power'],
        window_length_s=config.window_length_s,
        sampling_frequency=config.sampling_frequency,
        low_frequency=config.spectrum_low_frequency,
        high_frequency=config.spectrum_high_frequency,
        n_filters=config.n_dct_filters_cc,
        n_coefficients=config.n_coefficients_cc
        )
    
    df_windowed = df_windowed.drop(columns=
                                   ['total_power'] + \
                                   [f'{x}_fft' for x in l_sensor_colnames] + \
                                   [f'{x}_freqs' for x in l_sensor_colnames] + \
                                   l_sensor_colnames)

    df_windowed = pd.concat([df_windowed, cc_cols], axis=1)

    df_windowed = df_windowed.rename(columns={f'cc_{cc_nr}': f'cc_{cc_nr}_{sensor}' for cc_nr in range(1,config.n_coefficients_cc+1)}).rename(columns={'window_start': 'time'})

    return df_windowed
