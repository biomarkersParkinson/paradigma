from typing import List
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline

from dbpd.constants import DataColumns, TimeUnit

def transform_time_array(
    time_array: np.ndarray,
    scale_factor: float,
    input_unit_type: TimeUnit,
    output_unit_type: TimeUnit,
    start_time: float = 0.0,
) -> np.ndarray:
    """
    Transforms the time array to relative time (when defined in delta time) and scales the values.

    Parameters
    ----------
    time_array : np.ndarray
        The time array in milliseconds to transform.
    scale_factor : float
        The scale factor to apply to the time array.
    input_unit_type : TimeUnit
        The time unit type of the input time array. Raw PPP data was in `TimeUnit.difference_ms`.
    output_unit_type : TimeUnit
        The time unit type of the output time array. The processing is often done in `TimeUnit.relative_ms`.
    start_time : float, optional
        The start time of the time array in UNIX milliseconds (default is 0.0)

    Returns
    -------
    time_array
        The transformed time array in milliseconds, with the specified time unit type.
    """
    # Scale time array and transform to relative time (`TimeUnit.relative_ms`) 
    if input_unit_type == TimeUnit.difference_ms:
    # Convert a series of differences into cumulative sum to reconstruct original time series.
        time_array = np.cumsum(np.double(time_array)) / scale_factor
    elif input_unit_type == TimeUnit.absolute_ms:
        # Set the start time if not provided.
        if start_time == 0.0:
            start_time = time_array[0]
        # Convert absolute time stamps into a time series relative to start_time.
        time_array = (time_array - start_time) / scale_factor
    elif input_unit_type == TimeUnit.relative_ms:
        # Scale the relative time series as per the scale_factor.
        time_array = time_array / scale_factor

    # Transform the time array from `TimeUnit.relative_ms` to the specified time unit type
    if output_unit_type == TimeUnit.absolute_ms:
        # Converts time array to absolute time by adding the start time to each element.
        time_array = time_array + start_time
    elif output_unit_type == TimeUnit.difference_ms:
        # Creates a new array starting with 0, followed by the differences between consecutive elements.
        time_array = np.diff(np.insert(time_array, 0, start_time))
    elif output_unit_type == TimeUnit.relative_ms:
        # The array is already in relative format, do nothing.
        pass
    return time_array


def resample_data(
    df: pd.DataFrame,
    time_column : DataColumns,
    unscaled_column_names : list,
    resampling_frequency: int,
    scale_factors: list = [],
    start_time : float = 0.0,
) -> pd.DataFrame:
    """
    Resamples the IMU data to the resampling frequency. The data is scaled before resampling.

    Parameters
    ----------
    time_abs_array : np.ndarray
        The absolute time array.
    values_unscaled : np.ndarray
        The values to resample.
    resampling_frequency : int
        The frequency to resample the data to.
    time_column : str
        The name of the time column.
    scale_factors : list, optional
        The scale factors to apply to the values before resampling (default is []).

    Returns
    -------
    pd.DataFrame
        The resampled data.
    """
    print("Type of unscaled_column_names: ", type(unscaled_column_names))
    time_abs_array=np.array(df[time_column])
    
    values_unscaled=np.array(df[unscaled_column_names])[0]


    # scale data
    if len(scale_factors) != 0 and scale_factors is not None:
        scaled_values = values_unscaled * scale_factors

    # resample
    t_resampled = np.arange(float, time_abs_array[-1], 1 / resampling_frequency)

    # create dataframe
    df = pd.DataFrame(t_resampled, columns=[time_column])

    # interpolate IMU - maybe a separate method?
    for j, sensor_col in enumerate(unscaled_column_names):
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
    