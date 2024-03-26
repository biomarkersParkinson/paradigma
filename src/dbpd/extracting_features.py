import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA

from scipy import signal, fft

from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks


def create_window(
        df: pd.DataFrame,
        time_column_name: str,
        window_nr: int,
        lower_index: int,
        upper_index: int,
        data_point_level_cols: list,
        segment_nr: int,
        sampling_frequency: int
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
    t_start_window = df.loc[lower_index, time_column_name]

    df_subset = df.loc[lower_index:upper_index, data_point_level_cols].copy()
    t_start = t_start_window
    t_end = upper_index/sampling_frequency + t_start_window

    if segment_nr is None:
        l_subset_squeezed = [window_nr+1, t_start, t_end] + df_subset.values.T.tolist()
    else:
        l_subset_squeezed = [segment_nr, window_nr+1, t_start, t_end] + df_subset.values.T.tolist()

    return l_subset_squeezed
    

def tabulate_windows(
        df: pd.DataFrame,
        time_column_name: str,
        data_point_level_cols: list,
        window_length_s: int,
        window_step_size_s: int,
        sampling_frequency: int,
        segment_nr_colname: str = None,
        segment_nr: int = None,
    ) -> pd.DataFrame:
    """Compiles multiple windows into a single dataframe

    Parameters
    ----------
    df: pd.DataFrame
        The original dataframe to be windowed
    time_column_name: str
        The name of the time column
    data_point_level_cols: list
        The names of the columns that are to be kept as individual datapoints in a list instead of aggregates
    window_length_s: int
        The number of seconds a window constitutes
    window_step_size_s: int
        The number of seconds between the end of the previous and the start of the next window
    sampling_frequency: int
        The sampling frequency of the data
    segment_nr_colname: str
        The name of the column that identifies the segment; set to None if not applicable
    segment_nr: int
        The identification of the segment; set to None if not applicable
    

    Returns
    -------
    df_windowed: pd.DataFrame
        Dataframe with each row corresponding to an individual window
    """

    window_length = sampling_frequency * window_length_s - 1
    window_step_size = sampling_frequency * window_step_size_s

    df = df.reset_index(drop=True)

    if window_step_size <= 0:
        raise Exception("Step size should be larger than 0.")
    if window_length > df.shape[0]:
        return 

    l_windows = []
    n_windows = math.floor(
        (df.shape[0] - window_length) / 
         window_step_size
        ) + 1

    for window_nr in range(n_windows):
        lower = window_nr * window_step_size
        upper = window_nr * window_step_size + window_length
        l_windows.append(
            create_window(
                df=df,
                time_column_name=time_column_name,
                window_nr=window_nr,
                lower_index=lower,
                upper_index=upper,
                data_point_level_cols=data_point_level_cols,
                segment_nr=segment_nr,
                sampling_frequency=sampling_frequency
            )
        )

    if segment_nr is None:
        df_windows = pd.DataFrame(l_windows, columns=['window_nr', 'window_start', 'window_end'] + data_point_level_cols)
    else:
        df_windows = pd.DataFrame(l_windows, columns=[segment_nr_colname, 'window_nr', 'window_start', 'window_end'] + data_point_level_cols)
            
    return df_windows.reset_index(drop=True)


def generate_statistics(
        sensor_col: pd.Series,
        statistic: str
    ):
    if statistic == 'mean':
        return [np.mean(x) for x in sensor_col]
    elif statistic == 'std':
        return [np.std(x) for x in sensor_col]
    elif statistic == 'max':
        return [np.max(x) for x in sensor_col]
    elif statistic == 'min':
        return [np.min(x) for x in sensor_col]

def generate_std_norm(
        df: pd.DataFrame,
        cols: list,
    ):
    return df.apply(
        lambda x: np.std(np.sqrt(sum(
            [np.array([y**2 for y in x[col]]) for col in cols]
        ))), axis=1)
    

def compute_fft(
        values: list,
        window_type: str,
        sampling_frequency: int,
    ):

    w = signal.get_window(window_type, len(values), fftbins=False)
    yf = 2*fft.fft(values*w)[:int(len(values)/2+1)]
    xf = fft.fftfreq(len(values), 1/sampling_frequency)[:int(len(values)/2+1)]

    return yf, xf
    

def signal_to_ffts(
        sensor_col: pd.Series,
        window_type: str,
        sampling_frequency: int,
    ):
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
        sampling_frequency: int,
        window_type: str,
    ):
    """Note: sensor_col is a single cell of sensor_col, as it is used with apply function. 
    Probably we want a smarter way of doing this."""
    fxx, pxx = signal.periodogram(sensor_col, fs=sampling_frequency, window=window_type)
    ind_min = np.argmax(fxx > fmin) - 1
    ind_max = np.argmax(fxx > fmax) - 1
    return np.log10(np.trapz(pxx[ind_min:ind_max], fxx[ind_min:ind_max]))


def compute_perc_power(
        sensor_col: list,
        fmin_band: int,
        fmax_band: int,
        fmin_total: int,
        fmax_total: int,
        sampling_frequency: int,
        window_type: str,
):
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
        total_power_col: pd.Series,
        window_length_s: int,
        sampling_frequency: int,
        low_frequency: int,
        high_frequency: int,
        filter_length: int,
        n_dct_filters: int,
        ) -> pd.DataFrame:
    
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
    to the velocity in the arm swing direction. """
    pca = PCA(n_components=2, svd_solver='auto', random_state=22)
    pca.fit([(i,j) for i,j in zip(df.loc[df[pred_gait_colname]==1, y_gyro_colname], df.loc[df[pred_gait_colname]==1, z_gyro_colname])])
    yz_gyros = pca.transform([(i,j) for i,j in zip(df[y_gyro_colname], df[z_gyro_colname])])

    velocity = [x[0] for x in yz_gyros]

    return pd.Series(velocity)


def compute_angle(
        velocity_col: pd.Series,
        time_col: pd.Series,
) -> pd.Series:
    """Apply cumulative trapezoidal integration to extract the angle from the velocity."""
    angle_col = cumulative_trapezoid(velocity_col, time_col, initial=0)
    return pd.Series([x*-1 if x<0 else x for x in angle_col])


def remove_moving_average_angle(
        angle_col: pd.Series,
        sampling_frequency: int,
) -> pd.Series:
    angle_ma = angle_col.rolling(window=int(2*(sampling_frequency*0.5)+1), min_periods=1, center=True, closed='both').mean()
    
    return pd.Series(angle_col - angle_ma)


def create_segments(
        df: pd.DataFrame,
        time_colname: str,
        segment_nr_colname: str,
        minimum_gap_s: int,
) -> pd.DataFrame:
    """Create segments based on the time column of the dataframe. Segments are defined as continuous time periods.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to be segmented
    time_colname: str
        The name of the time column
    minimum_gap_s: int
        The minimum gap in seconds to split up the time periods into segments
    """
    array_new_segments = np.where((df[time_colname] - df[time_colname].shift() > minimum_gap_s), 1, 0)
    df['new_segment_cumsum'] = array_new_segments.cumsum()
    df_segments = pd.DataFrame(df.groupby('new_segment_cumsum')[time_colname].count()).reset_index()
    df_segments.columns = [segment_nr_colname, 'length_segment_s']
    df_segments[segment_nr_colname] += 1

    df = df.drop(columns=['new_segment_cumsum'])

    cols_to_append = [segment_nr_colname, 'length_segment_s']

    for col in cols_to_append:
        df[col] = 0

    index_start = 0
    for _, row in df_segments.iterrows():
        len_segment = row['length_segment_s']

        for col in cols_to_append:
            df.loc[index_start:index_start+len_segment-1, col] = row[col]

        index_start += len_segment

    return df


def discard_segments(
        df: pd.DataFrame,
        time_colname: str,
        segment_nr_colname: str,
        minimum_segment_length_s: int,
):
    segment_length_bool = df.groupby(segment_nr_colname)[time_colname].apply(lambda x: x.max() - x.min()) > minimum_segment_length_s

    df = df.loc[df[segment_nr_colname].isin(segment_length_bool.loc[segment_length_bool.values].index)]

    # reorder the segments - starting at 1
    for segment_nr in df[segment_nr_colname].unique():
        df.loc[df[segment_nr_colname]==segment_nr, f'{segment_nr_colname}_ordered'] = np.where(df[segment_nr_colname].unique()==segment_nr)[0][0] + 1

    df[f'{segment_nr_colname}_ordered'] = df[f'{segment_nr_colname}_ordered'].astype(int)

    df = df.drop(columns=[segment_nr_colname])
    df = df.rename(columns={f'{segment_nr_colname}_ordered': segment_nr_colname})

    return df


def extract_angle_extremes(
        df: pd.DataFrame,
        smooth_angle_colname: str,
        dominant_frequency_colname: str,
        sampling_frequency: int,
) -> pd.Series:
    # determine peaks
    df['angle_maxima'] = df.apply(lambda x: find_peaks(x[smooth_angle_colname], distance=sampling_frequency * 0.6 / x[dominant_frequency_colname], prominence=2)[0], axis=1)
    df['angle_minima'] = df.apply(lambda x: find_peaks([-x for x in x[smooth_angle_colname]], distance=sampling_frequency * 0.6 / x[dominant_frequency_colname], prominence=2)[0], axis=1) 

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
                        if df.loc[index, smooth_angle_colname][df.loc[index, 'angle_new_minima'][i_pks+1]] < df.loc[index, smooth_angle_colname][df.loc[index, 'angle_new_minima'][i_pks]]: # if second minimum if smaller than first, keep second
                            df.at[index, 'angle_new_minima'] = np.delete(df.loc[index, 'angle_new_minima'], i_pks)
                        else: # otherwise keep the first
                            df.at[index, 'angle_new_minima'] = np.delete(df.loc[index, 'angle_new_minima'], i_pks+1)
                        i_pks -= 1
                    if i_pks >= 0 and df.loc[index, 'angle_new_minima'][i_pks] > df.loc[index, 'angle_new_maxima'][i_pks]:
                        if df.loc[index, smooth_angle_colname][df.loc[index, 'angle_new_maxima'][i_pks]] < df.loc[index, smooth_angle_colname][df.loc[index, 'angle_new_maxima'][i_pks-1]]:
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks) 
                        else:
                            df.loc[index, 'angle_maxima_deleted'].append(df.loc[index, 'angle_new_maxima'][i_pks-1])
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks-1) 
                        i_pks -= 1
                    i_pks += 1

            elif df.loc[index, 'angle_new_maxima'][0] < df.loc[index, 'angle_new_minima'][0]: # if the first maximum occurs before the first minimum, start with the maximum
                while i_pks < df.loc[index, 'angle_new_minima'].size and i_pks < df.loc[index, 'angle_new_maxima'].size-1:
                    if df.loc[index, 'angle_new_minima'][i_pks] > df.loc[index, 'angle_new_maxima'][i_pks+1]:
                        if df.loc[index, smooth_angle_colname][df.loc[index, 'angle_new_maxima'][i_pks+1]] > df.loc[index, smooth_angle_colname][df.loc[index, 'angle_new_maxima'][i_pks]]:
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks) 
                        else:
                            df.at[index, 'angle_new_maxima'] = np.delete(df.loc[index, 'angle_new_maxima'], i_pks+1) 
                        i_pks -= 1
                    if i_pks > 0 and df.loc[index, 'angle_new_minima'][i_pks] < df.loc[index, 'angle_new_maxima'][i_pks]:
                        if df.loc[index, smooth_angle_colname][df.loc[index, 'angle_new_minima'][i_pks]] < df.loc[index, smooth_angle_colname][df.loc[index, 'angle_new_minima'][i_pks-1]]:
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
    df['angle_extrema_values'] = df.apply(lambda x: [x[smooth_angle_colname][i] for i in np.concatenate([x['angle_new_minima'], x['angle_new_maxima']])], axis=1) 

    return


def extract_range_of_motion(
        angle_extrema_values_col: pd.Series,
) -> pd.Series:
    
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