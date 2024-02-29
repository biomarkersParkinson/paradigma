import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA

from scipy import signal, fft

from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks


def create_window(
        df: pd.DataFrame,
        window_nr: int,
        lower_index: int,
        upper_index: int,
        data_point_level_cols: list
    ) -> list:
    """Transforms (a subset of) a dataframe into a single row

    Parameters
    ----------
    df: pd.DataFrame
        The original dataframe to be windowed
    window_nr: int
        The identification of the window
    lower_index: int
        The dataframe index of the first sample to be windowed
    upper_index: int
        The dataframe index of the final sample to be windowed
    data_point_level_cols: list
        The columns in sensor_df that are to be kept as individual datapoints in a list instead of aggregates

    Returns
    -------
    l_subset_squeezed: list
        Rows corresponding to single windows
    """
    df_subset = df.loc[lower_index:upper_index, data_point_level_cols].copy()
    l_subset_squeezed = [window_nr+1, lower_index, upper_index] + df_subset.values.T.tolist()

    return l_subset_squeezed
    

def tabulate_windows(
        config,
        df: pd.DataFrame,
        data_point_level_cols: list,
    ) -> pd.DataFrame:
    """Compiles multiple windows into a single dataframe

    Parameters
    ----------
    df: pd.DataFrame
        The original dataframe to be windowed
    window_length: int
        The number of samples a window constitutes
    window_step_size: int
        The number of samples between the start of the previous and the start of the next window
    data_point_level_cols: list
        The columns in sensor_df that are to be kept as individual datapoints in a list instead of aggregates

    Returns
    -------
    df_windowed: pd.DataFrame
        Dataframe with each row corresponding to an individual window
    """

    df = df.reset_index(drop=True)

    if config.window_step_size <= 0:
        raise Exception("Step size should be larger than 0.")
    if config.window_length > df.shape[0]:
        return 

    l_windows = []
    n_windows = math.floor(
        (df.shape[0] - config.window_length) / 
         config.window_step_size
        ) + 1

    for window_nr in range(n_windows):
        lower = window_nr * config.window_step_size
        upper = window_nr * config.window_step_size + config.window_length - 1
        l_windows.append(create_window(df, window_nr, lower, upper, data_point_level_cols))

    df_windows = pd.DataFrame(l_windows, columns=['window_nr', 'window_start', 'window_end'] + data_point_level_cols)
            
    return df_windows.reset_index(drop=True)


def generate_statistics(
        df: pd.DataFrame,
        sensor_col: str,
        statistic: str
    ):
    if statistic == 'mean':
        return df.apply(lambda x: np.mean(x[sensor_col]), axis=1)
    elif statistic == 'std':
        return df.apply(lambda x: np.std(x[sensor_col]), axis=1)
    elif statistic == 'max':
        return df.apply(lambda x: np.max(x[sensor_col]), axis=1)
    elif statistic == 'min':
        return df.apply(lambda x: np.min(x[sensor_col]), axis=1)

def generate_std_norm(
        df: pd.DataFrame,
        cols: list,
    ):
    return df.apply(
        lambda x: np.std(np.sqrt(sum(
            [np.array([y**2 for y in x[col]]) for col in cols]
        ))), axis=1)
    

def compute_fft(
        config,
        values: list,
    ):

    w = signal.get_window(config.window_type, len(values), fftbins=False)
    yf = 2*fft.fft(values*w)[:int(len(values)/2+1)]
    xf = fft.fftfreq(len(values), 1/config.resampling_frequency)[:int(len(values)/2+1)]

    return yf, xf
    

def signal_to_ffts(
        config,
        df: pd.DataFrame,
        sensor_col: str,
    ):
    l_values_total = []
    l_freqs_total = []
    for _, row in df.iterrows():
        l_values, l_freqs = compute_fft(
            values=row[sensor_col],
            window_type=config.window_type)
        l_values_total.append(l_values)
        l_freqs_total.append(l_freqs)

    return l_freqs_total, l_values_total
    

def compute_power_in_bandwidth(
        config,
        sensor_col,
        fmin: int,
        fmax: int,
    ):
    fxx, pxx = signal.periodogram(sensor_col, fs=config.resampling_frequency, window=config.window_type)
    ind_min = np.argmax(fxx > fmin) - 1
    ind_max = np.argmax(fxx > fmax) - 1
    return np.log10(np.trapz(pxx[ind_min:ind_max], fxx[ind_min:ind_max]))


def get_dominant_frequency(
        signal_ffts: pd.Series,
        signal_freqs: pd.Series,
        fmin: int,
        fmax: int
        ):
    
    valid_indices = np.where((signal_freqs>fmin) & (signal_freqs<fmax))
    signal_freqs_adjusted = signal_freqs[valid_indices]
    signal_ffts_adjusted = signal_ffts[valid_indices]

    idx = np.argmax(np.abs(signal_ffts_adjusted))
    return np.abs(signal_freqs_adjusted[idx])
    

def compute_power(
        df: pd.DataFrame,
        fft_cols: list
    ):
    for col in fft_cols:
        df['{}_power'.format(col)] = df[col].apply(lambda x: np.square(np.abs(x)))

    return df.apply(lambda x: sum([np.array([y for y in x[col+'_power']]) for col in fft_cols]), axis=1)
    

def generate_cepstral_coefficients(
        config,
        df: pd.DataFrame,
        total_power_col: str,
        low_frequency: int,
        high_frequency: int,
        filter_length: int,
        n_dct_filters: int,
        ):
    
    # compute filter points
    freqs = np.linspace(low_frequency, high_frequency, num=filter_length+2)
    filter_points = np.floor((config.window_length + 1) / config.resampling_frequency * freqs).astype(int)  

    # construct filterbank
    filters = np.zeros((len(filter_points)-2, int(config.window_length/2+1)))
    for j in range(len(filter_points)-2):
        filters[j, filter_points[j] : filter_points[j+1]] = np.linspace(0, 1, filter_points[j+1] - filter_points[j])
        filters[j, filter_points[j+1] : filter_points[j+2]] = np.linspace(1, 0, filter_points[j+2] - filter_points[j+1])

    # filter signal
    power_filtered = df[total_power_col].apply(lambda x: np.dot(filters, x))
    log_power_filtered = power_filtered.apply(lambda x: 10.0 * np.log10(x))

    # generate cepstral coefficients
    dct_filters = np.empty((n_dct_filters, filter_length))
    dct_filters[0, :] = 1.0 / np.sqrt(filter_length)

    samples = np.arange(1, 2 * filter_length, 2) * np.pi / (2.0 * filter_length)

    for i in range(1, n_dct_filters):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_length)

    cepstral_coefs = log_power_filtered.apply(lambda x: np.dot(dct_filters, x))

    return pd.DataFrame(np.vstack(cepstral_coefs), columns=['cc_{}'.format(j+1) for j in range(n_dct_filters)])


def pca_transform_gyroscope(
        df: pd.DataFrame,
        y_gyro_column: str,
        z_gyro_column: str,
        pred_gait_column: str,
) -> pd.Series:
    """Apply principal component analysis (PCA) on the y-axis and z-axis of the raw gyroscope signal
    to extract the velocity. PCA is applied to the predicted gait timestamps only to maximize the similarity
    to the velocity in the arm swing direction. """
    pca = PCA(n_components=2, svd_solver='auto', random_state=22)
    pca.fit([(i,j) for i,j in zip(df.loc[df[pred_gait_column]==1, y_gyro_column], df.loc[df[pred_gait_column]==1, z_gyro_column])])
    yz_gyros = pca.transform([(i,j) for i,j in zip(df[y_gyro_column], df[z_gyro_column])])

    df['velocity'] = [x[0] for x in yz_gyros]

    return df['velocity']


def compute_angle(
        config,
        df: pd.DataFrame,
        velocity_column: str,
) -> pd.Series:
    """Apply cumulative trapezoidal integration to extract the angle from the velocity."""
    df['angle'] = cumulative_trapezoid(y=df[velocity_column], x=df[config.time_column], initial=0)
    return df['angle'].apply(lambda x: x*-1 if x<0 else x)


def remove_moving_average_angle(
        config,
        df: pd.DataFrame,
        angle_column: str,
) -> pd.Series:
    df['angle_ma'] = df[angle_column].rolling(window=int(2*(config.fs*0.5)+1), min_periods=1, center=True, closed='both').mean()
    
    return df[angle_column] - df['angle_ma']


def create_segments(
        config,
        df: pd.DataFrame,
        pred_gait_column: str,
) -> pd.DataFrame:
    array_new_segments = np.where((df[config.time_column] - df[config.time_column].shift() > 1.5/config.fs) | (df[pred_gait_column].ne(df[pred_gait_column].shift())), 1, 0)
    df['new_segment_cumsum'] = array_new_segments.cumsum()
    df_segments = pd.DataFrame(df.groupby(['new_segment_cumsum', pred_gait_column])[config.time_column].count()).reset_index()
    df_segments.columns = ['segment_nr', pred_gait_column, 'count']

    cols_to_append = ['segment_nr', 'count']

    for col in cols_to_append:
        df[col] = 0

    index_start = 0
    for _, row in df_segments.iterrows():
        len_segment = row['count']

        for col in cols_to_append:
            df.loc[index_start:index_start+len_segment-1, col] = row[col]

        index_start += len_segment

    df['length_segment_s'] = df['count'] / config.fs

    df = df.drop(columns=['count'])

    # subset gait
    df = df.loc[df[pred_gait_column]==1].reset_index(drop=True)
    
    # discard segments smaller than window length
    segment_length_bool = df.groupby(['segment_nr']).size() > config.window_length * config.fs

    df = df.loc[df['segment_nr'].isin(segment_length_bool.loc[segment_length_bool.values].index)]

    # reorder the segments - starting at 1
    for segment_nr in df['segment_nr'].unique():
        df.loc[df['segment_nr']==segment_nr, 'segment_nr_ordered'] = np.where(df['segment_nr'].unique()==segment_nr)[0][0] + 1

    df['segment_nr_ordered'] = df['segment_nr_ordered'].astype(int)

    df = df.drop(columns=['segment_nr'])
    df = df.rename(columns={'segment_nr_ordered': 'segment_nr'})

    return df


def extract_angle_extremes(
        config,
        df: pd.DataFrame,
        smooth_angle_column: str,
        dominant_frequency_column: str,
) -> pd.DataFrame:
    # determine peaks
    df['angle_maxima'] = df.apply(lambda x: find_peaks(x[smooth_angle_column], distance=config.fs * 0.6 / x[dominant_frequency_column], prominence=2)[0], axis=1)
    df['angle_minima'] = df.apply(lambda x: find_peaks([-x for x in x[smooth_angle_column]], distance=config.fs * 0.6 / x[dominant_frequency_column], prominence=2)[0], axis=1) 

    df['angle_new_minima'] = df['angle_minima'].copy()
    df['angle_new_maxima'] = df['angle_maxima'].copy()

    # to keep track of peaks deleted due to constraints
    df['angle_minima_deleted'] = np.empty((len(df), 0)).tolist()
    df['angle_maxima_deleted'] = np.empty((len(df), 0)).tolist()

    for index, row in df.iterrows():
        i_pks = 0                                       # iterable to keep track of consecutive min-min and max-max versus min-max
        n_min = len(df.loc[index, 'angle_new_minima'])  # number of minima in window
        n_max = len(df.loc[index, 'angle_new_maxima'])  # number of maxima in window

        if n_min > 0 and n_max > 0: 
            if df.loc[index, 'angle_new_maxima'][0] > df.loc[index, 'angle_new_minima'][0]: # if first minimum occurs before first maximum, start with minimum
                while i_pks < len(df.loc[index, 'angle_new_minima']) - 1 and i_pks < len(df.loc[index, 'angle_new_maxima']): # only continue if there's enough minima and maxima to perform operations
                    if df.loc[index, 'angle_new_minima'][i_pks+1] < df.loc[index, 'angle_new_maxima'][i_pks]: # if angle of next minimum comes before the current maxima, we have two minima in a row
                        if df.loc[index, smooth_angle_column][df.loc[index, 'angle_new_minima'][i_pks+1]] < df.loc[index, smooth_angle_column][df.loc[index, 'angle_new_minima'][i_pks]]: # if second minimum if smaller than first, keep second
                            df.loc[index, 'angle_minima_deleted'].append(df.loc[index, 'angle_new_minima'][i_pks])
                            df.at[index, 'angle_new_minima'] = np.delete(df.loc[index, 'angle_new_minima'], i_pks)
                        else: # otherwise keep the first
                            df.loc[index, 'angle_minima_deleted'].append(df.loc[index, 'angle_new_minima'][i_pks+1])
                            df.at[index, 'angle_new_minima'] = np.delete(df.loc[index, 'angle_new_minima'], i_pks+1)
                        i_pks -= 1

                    if i_pks >= 0 and df.loc[index, 'angle_new_minima'][i_pks] > df.loc[index, 'angle_new_maxima'][i_pks]:
                        if df.loc[index, smooth_angle_column][df.loc[index, 'angle_new_maxima'][i_pks]] < df.loc[index, smooth_angle_column][df.loc[index, 'angle_new_maxima'][i_pks-1]]:
                            df.loc[index, 'angle_maxima_deleted'].append(df.loc[index, 'angle_new_maxima'][i_pks])
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks) 
                        else:
                            df.loc[index, 'angle_maxima_deleted'].append(df.loc[index, 'angle_new_maxima'][i_pks-1])
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks-1) 
                        i_pks -= 1
                    i_pks += 1

            elif df.loc[index, 'angle_new_maxima'][0] < df.loc[index, 'angle_new_minima'][0]: # if the first maximum occurs before the first minimum, start with the maximum
                while i_pks < len(df.loc[index, 'angle_new_minima']) and i_pks < len(df.loc[index, 'angle_new_maxima'])-1:
                    if df.loc[index, 'angle_new_minima'][i_pks] > df.loc[index, 'angle_new_maxima'][i_pks+1]:
                        if df.loc[index, smooth_angle_column][df.loc[index, 'angle_new_maxima'][i_pks+1]] > df.loc[index, smooth_angle_column][df.loc[index, 'angle_new_maxima'][i_pks]]:
                            df.loc[index, 'angle_maxima_deleted'].append(df.loc[index, 'angle_new_maxima'][i_pks])
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks) 
                        else:
                            df.loc[index, 'angle_maxima_deleted'].append(df.loc[index, 'angle_new_maxima'][i_pks+1])
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks+1) 
                        i_pks -= 1
                    if i_pks > 0 and df.loc[index, 'angle_new_minima'][i_pks] < df.loc[index, 'angle_new_maxima'][i_pks]:
                        if df.loc[index, smooth_angle_column][df.loc[index, 'angle_new_minima'][i_pks]] < df.loc[index, smooth_angle_column][df.loc[index, 'angle_new_minima'][i_pks-1]]:
                            df.loc[index, 'angle_maxima_deleted'].append(df.loc[index, 'angle_new_maxima'][i_pks-1])
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks-1) 
                        
                        else:
                            df.loc[index, 'angle_maxima_deleted'].append(df.loc[index, 'angle_new_maxima'][i_pks])
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks) 
                        i_pks -= 1
                    i_pks += 1

    # extract amplitude
    df['angle_extrema_values'] = df.apply(lambda x: [x[smooth_angle_column][i] for i in np.concatenate((x['angle_new_minima'], x['angle_new_maxima']))] if len(x['angle_new_minima']) > 0 and len(x['angle_new_maxima']) > 0 else [], axis=1) 

    return df['angle_extrema_values']


def extract_range_of_motion(
        df: pd.DataFrame,
        angle_extrema_values_column: str,
):
    df['angle_amplitudes'] = np.empty((len(df), 0)).tolist()

    for i, extrema_values in enumerate(df[angle_extrema_values_column]):
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

        df.loc[i, 'angle_amplitudes'].append([x for x in l_amplitudes])

    return df['angle_amplitudes'].apply(lambda x: [y for item in x for y in item])


def extract_peak_angular_velocity(
        df: pd.DataFrame,
        velocity_column: str,
        angle_minima_column: str,
        angle_maxima_column: str,
) -> pd.DataFrame:
    df['forward_peak_ang_vel'] = np.empty((len(df), 0)).tolist()
    df['backward_peak_ang_vel'] = np.empty((len(df), 0)).tolist()

    for index, row in df.iterrows():
        if len(row[angle_minima_column]) > 0 and len(row[angle_maxima_column]) > 0:
            l_extrema_indices = np.sort(np.concatenate((row[angle_minima_column], row[angle_maxima_column])))
            for j, peak_index in enumerate(l_extrema_indices):
                if peak_index in row[angle_maxima_column] and j < len(l_extrema_indices) - 1:
                    df.loc[index, 'forward_peak_ang_vel'].append(np.abs(min(row[velocity_column][l_extrema_indices[j]:l_extrema_indices[j+1]])))
                elif peak_index in row[angle_minima_column] and j < len(l_extrema_indices) - 1:
                    df.loc[index, 'backward_peak_ang_vel'].append(np.abs(max(row[velocity_column][l_extrema_indices[j]:l_extrema_indices[j+1]])))

    return df