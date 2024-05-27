import numpy as np
import pandas as pd

from scipy import signal
from scipy.interpolate import CubicSpline

from dbpd.constants import DataColumns


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
        The transformed time array in milliseconds.
    """
    if data_in_delta_time:
        return np.cumsum(np.double(time_array)) / scale_factor
    return time_array


def resample_data(
    time_abs_array: np.ndarray,
    values_unscaled: np.ndarray,
    scale_factors: list,
    resampling_frequency: int,
    time_column: str,
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
    resampling_frequency : int
        The frequency to resample the data to.
    time_column : str
        The name of the time column.

    Returns
    -------
    pd.DataFrame
        The resampled data.
    """

    # scale data
    scaled_values = values_unscaled * scale_factors

    # resample
    t_resampled = np.arange(0, time_abs_array[-1], 1 / resampling_frequency)

    # create dataframe
    df = pd.DataFrame(t_resampled, columns=[time_column])

    # interpolate IMU - maybe a separate method?
    for j, sensor_col in enumerate(
        [
            DataColumns.ACCELEROMETER_X,
            DataColumns.ACCELEROMETER_Y,
            DataColumns.ACCELEROMETER_Z,
            DataColumns.GYROSCOPE_X,
            DataColumns.GYROSCOPE_Y,
            DataColumns.GYROSCOPE_Z,
        ]
    ):
        if not np.all(np.diff(time_abs_array) > 0):
            raise ValueError("time_abs_array is not strictly increasing")

        cs = CubicSpline(time_abs_array, scaled_values.T[j])
        df[sensor_col] = cs(df[time_column])

    return df


def butterworth_filter(
    single_sensor_col: np.ndarray,
    order: int,
    cutoff_frequency: float,
    passband: str,
    sampling_frequency: int,
):
    """
    Applies the Butterworth filter to a single sensor column

    Parameters
    ----------
    single_sensor_column: pd.Series
        A single column containing sensor data in float format
    order: int
        The exponential order of the filter
    cutoff_frequency: float
        The frequency at which the gain drops to 1/sqrt(2) that of the passband
    passband: str
        Type of passband: ['hp' or 'lp']
    sampling_frequency: int
        The sampling frequency of the sensor data

    Returns
    -------
    sensor_column_filtered: pd.Series
        The origin sensor column filtered applying a Butterworth filter
    """

    sos = signal.butter(
        N=order,
        Wn=cutoff_frequency,
        btype=passband,
        analog=False,
        fs=sampling_frequency,
        output="sos",
    )
    return signal.sosfilt(sos, single_sensor_col)
    