import math
import numpy as np
import pandas as pd

from datetime import datetime
from scipy import signal, fft
from scipy.interpolate import CubicSpline

from dbpd.constants import DataColumns


class PreprocessingPipelineConfig:
    """Object used to configure and execute data preprocessing steps."""

    def __init__(
        self,
        time_column: str,
        sampling_frequency: int,
        resampling_frequency: int,
        window_length: int,
        window_step_size: int,
        window_type: str,
        verbose: int,
    ):
        self.verbose = verbose
        self.time_column = time_column
        self.sampling_frequency = sampling_frequency
        self.resampling_frequency = resampling_frequency
        self.window_length = window_length
        self.window_step_size = window_step_size
        self.window_type = window_type



def transform_time_array(
    time_array: np.ndarray,
    scale_factor: float,
    data_in_delta_time: bool,
) -> np.ndarray:
    """
    Transforms the time array to relative time (when defined in delta time) and scales the values.

    Parameters
    ----------
    time_array : np.ndarray
        The time array in milliseconds to transform.
    scale_factor : float
        The scale factor to apply to the time array.
    data_in_delta_time : bool - true if data is in delta time, and therefore needs to be converted to relative time.

    Returns
    -------
    array_like
        The transformed time array in seconds.
    """
    if data_in_delta_time:
        return np.cumsum(np.double(time_array)) / scale_factor
    return time_array


def resample_data(
    config,
    time_abs_array: np.ndarray,
    values_unscaled: np.ndarray,
    scale_factors: list,
) -> pd.DataFrame:
    """
    Resamples the IMU data to the resampling frequency. The data is scaled before resampling.

    Parameters
    ----------
    time_abs_array : np.ndarray
        The absolute time array.
    values_unscaled : np.ndarray
        The values to resample.
    scale_factors : list
        The scale factors to apply to the values.

    Returns
    -------
    pd.DataFrame
        The resampled data.
    """

    # scale data
    scaled_values = values_unscaled * scale_factors

    # resample
    t_resampled = np.arange(0, time_abs_array[-1], 1000 / config.resampling_frequency)

    # create dataframe
    df = pd.DataFrame(t_resampled, columns=[config.time_column])

    # interpolate IMU - maybe a separate method?
    for j, sensor_col in enumerate(
        [
            DataColumns.ACCELERATION_X,
            DataColumns.ACCELERATION_Y,
            DataColumns.ACCELERATION_Z,
            DataColumns.ROTATION_X,
            DataColumns.ROTATION_Y,
            DataColumns.ROTATION_Z,
        ]
    ):
        if not np.all(np.diff(time_abs_array) > 0):
            raise ValueError("time_abs_array is not strictly increasing")

        cs = CubicSpline(time_abs_array, scaled_values.T[j])
        df[sensor_col] = cs(df[config.time_column])

    return df


def butterworth_filter(
    config,
    single_sensor_col: np.ndarray,
    order: int,
    cutoff_frequency: float,
    passband: str,
):
    """Applies the Butterworth filter to a single sensor column

    Parameters
    ----------
    sensor_column: pd.Series
        A single column containing sensor data in float format
    frequency: int
        The sampling frequency of sensor_column in Hz
    order: int
        The exponential order of the filter
    cutoff_frequency: float
        The frequency at which the gain drops to 1/sqrt(2) that of the passband
    passband: str
        Type of passband: ['hp' or 'lp']
    verbose: bool
        The verbosity of the output

    Returns
    -------
    sensor_column_filtered: pd.Series
        The origin sensor column filtered applying a Butterworth filter"""

    sos = signal.butter(
        N=order,
        Wn=cutoff_frequency,
        btype=passband,
        analog=False,
        fs=config.resampling_frequency,
        output="sos",
    )
    return signal.sosfilt(sos, single_sensor_col)    


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
    power_filtered = config.df_windows[total_power_col].apply(lambda x: np.dot(filters, x))
    log_power_filtered = power_filtered.apply(lambda x: 10.0 * np.log10(x))

    # generate cepstral coefficients
    dct_filters = np.empty((n_dct_filters, filter_length))
    dct_filters[0, :] = 1.0 / np.sqrt(filter_length)

    samples = np.arange(1, 2 * filter_length, 2) * np.pi / (2.0 * filter_length)

    for i in range(1, n_dct_filters):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_length)

    cepstral_coefs = log_power_filtered.apply(lambda x: np.dot(dct_filters, x))

    return pd.DataFrame(np.vstack(cepstral_coefs), columns=['cc_{}'.format(j+1) for j in range(n_dct_filters)])