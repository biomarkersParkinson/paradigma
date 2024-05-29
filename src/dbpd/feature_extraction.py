import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scipy import signal, fft

from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks


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
        cols: list,
    ) -> pd.Series:
    """Generate the standard deviation of the norm of the accelerometer axes.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the accelerometer axes
    cols: list
        The names of the columns containing the accelerometer axes
        
    Returns
    -------
    pd.Series
        The standard deviation of the norm of the accelerometer axes
    """
    return df.apply(
        lambda x: np.std(np.sqrt(sum(
            [np.array([y**2 for y in x[col]]) for col in cols]
        ))), axis=1)
    

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
    ) -> tuple:
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
        Lists of FFT values and corresponding frequencies which can be concatenated as column to the dataframe
    """
    l_values_total = []
    l_freqs_total = []
    for row in sensor_col:
        l_values, l_freqs = compute_fft(
            values=row,
            window_type=window_type,
            sampling_frequency=sampling_frequency)
        l_values_total.append(l_values)
        l_freqs_total.append(l_freqs)

    return l_freqs_total, l_values_total
    

def compute_power_in_bandwidth(
        sensor_col: list,
        fmin: int,
        fmax: int,
        sampling_frequency: int = 100,
        window_type: str = 'hann',
    ) -> float:
    """Note: sensor_col is a single cell (which corresponds to a single window) of sensor_col, as it is used with apply function. 
    Probably we want a smarter way of doing this.
    
    Computes the power in a specific frequency band for a specified sensor and axis.
    
    Parameters
    ----------
    sensor_col: list
        The sensor column to be transformed (e.g. x-axis of accelerometer). This corresponds to a single window, which is a single row of the dataframe, 
        and contains values of individual timestamps composing the window.
    fmin: int
        The lower bound of the frequency band
    fmax: int
        The upper bound of the frequency band
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    window_type: str
        The type of window to be used for the FFT (default: 'hann')

    Returns
    -------
    float
        The power in the specified frequency band    
    """
    fxx, pxx = signal.periodogram(sensor_col, fs=sampling_frequency, window=window_type)
    ind_min = np.argmax(fxx > fmin) - 1
    ind_max = np.argmax(fxx > fmax) - 1
    return np.log10(np.trapz(pxx[ind_min:ind_max], fxx[ind_min:ind_max]))


def compute_perc_power(
        sensor_col: list,
        fmin_band: int,
        fmax_band: int,
        fmin_total: int = 0,
        fmax_total: int = 100,
        sampling_frequency: int = 100,
        window_type: str = 'hann'
    ) -> float:
    """Note: sensor_col is a single cell (which corresponds to a single window) of sensor_col, as it is used with apply function.

    Computes the percentage of power in a specific frequency band for a specified sensor and axis.
    
    Parameters
    ----------
    sensor_col: list
        The sensor column to be transformed (e.g. x-axis of accelerometer). This corresponds to a single window, which is a single row of the dataframe
    fmin_band: int
        The lower bound of the frequency band
    fmax_band: int
        The upper bound of the frequency band
    fmin_total: int
        The lower bound of the frequency spectrum (default: 0)
    fmax_total: int
        The upper bound of the frequency spectrum (default: 100)
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    window_type: str
        The type of window to be used for the FFT (default: 'hann')
    
    Returns
    -------
    float
        The percentage of power in the specified frequency band
    """
    angle_power_band = compute_power_in_bandwidth(
        sensor_col=sensor_col,
        fmin=fmin_band,
        fmax=fmax_band,
        sampling_frequency=sampling_frequency,
        window_type=window_type
        )
    
    angle_power_total = compute_power_in_bandwidth(
        sensor_col=sensor_col,
        fmin=fmin_total,
        fmax=fmax_total,
        sampling_frequency=sampling_frequency,
        window_type=window_type
        )
    
    return angle_power_band / angle_power_total


def get_dominant_frequency(
        signal_ffts: list,
        signal_freqs: list,
        fmin: int,
        fmax: int
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
        The dominant frequency in the specified frequency band
    """
    valid_indices = np.where((signal_freqs>fmin) & (signal_freqs<fmax))
    signal_freqs_adjusted = signal_freqs[valid_indices]
    signal_ffts_adjusted = signal_ffts[valid_indices]

    idx = np.argmax(np.abs(signal_ffts_adjusted))
    return np.abs(signal_freqs_adjusted[idx])
    

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
    for col in fft_cols:
        df['{}_power'.format(col)] = df[col].apply(lambda x: np.square(np.abs(x)))

    return df.apply(lambda x: sum([np.array([y for y in x[col+'_power']]) for col in fft_cols]), axis=1)
    

def generate_cepstral_coefficients(
        total_power_col: pd.Series,
        window_length_s: int,
        sampling_frequency: int = 100,
        low_frequency: int = 0,
        high_frequency: int = 50,
        filter_length: int = 16,
        n_dct_filters: int = 16,
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
        The upper bound of the frequency band (default: 50)
    filter_length: int
        The length of the DCT filter (default: 16)
    n_dct_filters: int
        The number of DCT filters (default: 16)
    
    Returns
    -------
    pd.DataFrame
        A dataframe with a single column corresponding to a single cepstral coefficient
    """
    window_length = window_length_s * sampling_frequency
    
    # compute filter points
    freqs = np.linspace(low_frequency, high_frequency, num=filter_length+2)
    filter_points = np.floor((window_length + 1) / sampling_frequency * freqs).astype(int)  

    # construct filterbank
    filters = np.zeros((len(filter_points)-2, int(window_length/2+1)))
    for j in range(len(filter_points)-2):
        filters[j, filter_points[j] : filter_points[j+1]] = np.linspace(0, 1, filter_points[j+1] - filter_points[j])
        filters[j, filter_points[j+1] : filter_points[j+2]] = np.linspace(1, 0, filter_points[j+2] - filter_points[j+1])

    # filter signal
    power_filtered = [np.dot(filters, x) for x in total_power_col]
    log_power_filtered = [10.0 * np.log10(x) for x in power_filtered]

    # generate cepstral coefficients
    dct_filters = np.empty((n_dct_filters, filter_length))
    dct_filters[0, :] = 1.0 / np.sqrt(filter_length)

    samples = np.arange(1, 2 * filter_length, 2) * np.pi / (2.0 * filter_length)

    for i in range(1, n_dct_filters):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_length)

    cepstral_coefs = [np.dot(dct_filters, x) for x in log_power_filtered]

    return pd.DataFrame(np.vstack(cepstral_coefs), columns=['cc_{}'.format(j+1) for j in range(n_dct_filters)])


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
    # determine peaks
    df['angle_maxima'] = df.apply(lambda x: find_peaks(x[angle_colname], distance=sampling_frequency * 0.6 / x[dominant_frequency_colname], prominence=2)[0], axis=1)
    df['angle_minima'] = df.apply(lambda x: find_peaks([-x for x in x[angle_colname]], distance=sampling_frequency * 0.6 / x[dominant_frequency_colname], prominence=2)[0], axis=1) 

    df['angle_new_minima'] = df['angle_minima'].copy()
    df['angle_new_maxima'] = df['angle_maxima'].copy()

    for index, _ in df.iterrows():
        i_pks = 0                                       # iterable to keep track of consecutive min-min and max-max versus min-max
        n_min = df.loc[index, 'angle_new_minima'].size  # number of minima in window
        n_max = df.loc[index, 'angle_new_maxima'].size  # number of maxima in window

        if n_min > 0 and n_max > 0: 
            if df.loc[index, 'angle_new_maxima'][0] > df.loc[index, 'angle_new_minima'][0]: # if first minimum occurs before first maximum, start with minimum
                while i_pks < df.loc[index, 'angle_new_minima'].size - 1 and i_pks < df.loc[index, 'angle_new_maxima'].size: # only continue if there's enough minima and maxima to perform operations
                    if df.loc[index, 'angle_new_minima'][i_pks+1] < df.loc[index, 'angle_new_maxima'][i_pks]: # if angle of next minimum comes before the current maxima, we have two minima in a row
                        if df.loc[index, angle_colname][df.loc[index, 'angle_new_minima'][i_pks+1]] < df.loc[index, angle_colname][df.loc[index, 'angle_new_minima'][i_pks]]: # if second minimum if smaller than first, keep second
                            df.at[index, 'angle_new_minima'] = np.delete(df.loc[index, 'angle_new_minima'], i_pks)
                        else: # otherwise keep the first
                            df.at[index, 'angle_new_minima'] = np.delete(df.loc[index, 'angle_new_minima'], i_pks+1)
                        i_pks -= 1
                    if i_pks >= 0 and df.loc[index, 'angle_new_minima'][i_pks] > df.loc[index, 'angle_new_maxima'][i_pks]:
                        if df.loc[index, angle_colname][df.loc[index, 'angle_new_maxima'][i_pks]] < df.loc[index, angle_colname][df.loc[index, 'angle_new_maxima'][i_pks-1]]:
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks) 
                        else:
                            df.loc[index, 'angle_maxima_deleted'].append(df.loc[index, 'angle_new_maxima'][i_pks-1])
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks-1) 
                        i_pks -= 1
                    i_pks += 1

            elif df.loc[index, 'angle_new_maxima'][0] < df.loc[index, 'angle_new_minima'][0]: # if the first maximum occurs before the first minimum, start with the maximum
                while i_pks < df.loc[index, 'angle_new_minima'].size and i_pks < df.loc[index, 'angle_new_maxima'].size-1:
                    if df.loc[index, 'angle_new_minima'][i_pks] > df.loc[index, 'angle_new_maxima'][i_pks+1]:
                        if df.loc[index, angle_colname][df.loc[index, 'angle_new_maxima'][i_pks+1]] > df.loc[index, angle_colname][df.loc[index, 'angle_new_maxima'][i_pks]]:
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks) 
                        else:
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks+1) 
                        i_pks -= 1
                    if i_pks > 0 and df.loc[index, 'angle_new_minima'][i_pks] < df.loc[index, 'angle_new_maxima'][i_pks]:
                        if df.loc[index, angle_colname][df.loc[index, 'angle_new_minima'][i_pks]] < df.loc[index, angle_colname][df.loc[index, 'angle_new_minima'][i_pks-1]]:
                            df.at[index, 'angle_new_minima'] = np.delete(df.loc[index, 'angle_new_minima'], i_pks-1) 
                        
                        else:
                            df.at[index, 'angle_new_minima'] = np.delete(df.loc[index, 'angle_new_minima'], i_pks) 
                        i_pks -= 1
                    i_pks += 1

    # for some peculiar reason, if a single item remains in the row for angle_new_minima or
    # angle_new_maxima, it could be either a scalar or a vector.
    for col in ['angle_new_minima', 'angle_new_maxima']:
        df.loc[df.apply(lambda x: type(x[col].tolist())==int, axis=1), col] = df.loc[df.apply(lambda x: type(x[col].tolist())==int, axis=1), col].apply(lambda x: [x])

    # extract amplitude
    df['angle_extrema_values'] = df.apply(lambda x: [x[angle_colname][i] for i in np.concatenate([x['angle_new_minima'], x['angle_new_maxima']])], axis=1) 

    return


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
    angle_amplitudes = np.empty((len(angle_extrema_values_col), 0)).tolist()

    for i, extrema_values in enumerate(angle_extrema_values_col):
        l_amplitudes = []
        for j, value in enumerate(extrema_values):
            if j < len(extrema_values)-1:
                # if extrema are on different sides of 0
                if (value > 0 and extrema_values[j+1] < 0) or (value < 0 and extrema_values[j+1] > 0):
                    l_amplitudes.append(np.sum(np.abs(value) + np.abs(extrema_values[j+1])))
                # elif extrema are both positive or both negative, and current extreme is closer to 0
                elif np.abs(value) < np.abs(extrema_values[j+1]):
                    l_amplitudes.append(np.subtract(np.abs(extrema_values[j+1]), np.abs(value)))
                # or if extrema are both positive and negative, and current extreme is further waay from 0
                else:
                    l_amplitudes.append(np.subtract(np.abs(value), np.abs(extrema_values[j+1])))

        angle_amplitudes[i].append([x for x in l_amplitudes])

    return [y for item in angle_amplitudes for y in item]


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
    df['forward_peak_ang_vel'] = np.empty((len(df), 0)).tolist()
    df['backward_peak_ang_vel'] = np.empty((len(df), 0)).tolist()

    for index, row in df.iterrows():
        if len(row[angle_minima_colname]) > 0 and len(row[angle_maxima_colname]) > 0:
            l_extrema_indices = np.sort(np.concatenate((row[angle_minima_colname], row[angle_maxima_colname])))
            for j, peak_index in enumerate(l_extrema_indices):
                if peak_index in row[angle_maxima_colname] and j < len(l_extrema_indices) - 1:
                    df.loc[index, 'forward_peak_ang_vel'].append(np.abs(min(row[velocity_colname][l_extrema_indices[j]:l_extrema_indices[j+1]])))
                elif peak_index in row[angle_minima_colname] and j < len(l_extrema_indices) - 1:
                    df.loc[index, 'backward_peak_ang_vel'].append(np.abs(max(row[velocity_colname][l_extrema_indices[j]:l_extrema_indices[j+1]])))
    
    return