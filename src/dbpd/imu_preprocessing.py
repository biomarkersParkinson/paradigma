from typing import List
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline

import tsdf
from dbpd.constants import DataColumns, TimeUnit
from dbpd.util import write_data, read_metadata


class PreprocessingConfig:

    def __init__(self) -> None:
        self.meta_filename = 'IMU_meta.json'
        self.values_filename = 'IMU_samples.bin'
        self.time_filename = 'IMU_time.bin'

        self.acceleration_units = 'm/s^2'
        self.rotation_units = 'deg/s'

        self.l_acceleration_cols = [DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]
        self.time_colname = DataColumns.TIME

        self.d_channels_units = {
            DataColumns.ACCELEROMETER_X: self.acceleration_units,
            DataColumns.ACCELEROMETER_Y: self.acceleration_units,
            DataColumns.ACCELEROMETER_Z: self.acceleration_units,
            DataColumns.GYROSCOPE_X: self.rotation_units,
            DataColumns.GYROSCOPE_Y: self.rotation_units,
            DataColumns.GYROSCOPE_Z: self.rotation_units,
        }

        # participant information
        self.side_watch = 'right'

        # filtering
        self.sampling_frequency = 100
        self.lower_cutoff_frequency = 0.2
        self.filter_order = 4


def preprocess_imu_data(input_path: str, output_path: str, config: PreprocessingConfig) -> None:

    # Load data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Rename columns
    df = df.rename(columns={f'rotation_{a}': f'gyroscope_{a}' for a in ['x', 'y', 'z']})
    df = df.rename(columns={f'acceleration_{a}': f'accelerometer_{a}' for a in ['x', 'y', 'z']})

    # convert to relative seconds from delta milliseconds
    df[config.time_colname] = transform_time_array(
        time_array=df[config.time_colname],
        scale_factor=1000, 
        input_unit_type = TimeUnit.difference_ms,
        output_unit_type = TimeUnit.relative_ms)
    

    df = resample_data(
        df=df,
        time_column=config.time_colname,
        time_unit_type=TimeUnit.relative_ms,
        unscaled_column_names = list(config.d_channels_units.keys()),
        scale_factors=metadata_samples.scale_factors,
        resampling_frequency=config.sampling_frequency)
    
    if config.side_watch == 'left':
        df[DataColumns.ACCELEROMETER_X] *= -1

    for col in config.l_acceleration_cols:

        # change to correct units [g]
        if config.acceleration_units == 'm/s^2':
            df[col] /= 9.81

        for result, side_pass in zip(['filt', 'grav'], ['hp', 'lp']):
            df[f'{result}_{col}'] = butterworth_filter(
                single_sensor_col=np.array(df[col]),
                order=config.filter_order,
                cutoff_frequency=config.lower_cutoff_frequency,
                passband=side_pass,
                sampling_frequency=config.sampling_frequency,
                )
            
        df = df.drop(columns=[col])
        df = df.rename(columns={f'filt_{col}': col})

    # Store data
    for sensor, units in zip(['accelerometer', 'gyroscope'], ['g', config.rotation_units]):
        df_sensor = df[[config.time_colname] + [x for x in df.columns if sensor in x]]

        metadata_samples.channels = [x for x in df.columns if sensor in x]
        metadata_samples.units = list(np.repeat(units, len(metadata_samples.channels)))
        metadata_samples.file_name = f'{sensor}_samples.bin'

        metadata_time.file_name = f'{sensor}_time.bin'
        metadata_time.units = ['time_relative_ms']

        write_data(metadata_time, metadata_samples, output_path, f'{sensor}_meta.json', df_sensor)

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
        if np.isclose(start_time, 0.0, rtol=1e-09, atol=1e-09):
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
    time_unit_type: TimeUnit,
    unscaled_column_names : list,
    resampling_frequency: int,
    scale_factors: list = []
) -> pd.DataFrame:
    """
    Resamples the IMU data to the resampling frequency. The data is scaled before resampling.
    TODO: This method does not work on the PPG data because it is in the absolute time format. I added `time_unit_type` as a parameter to the method, but it is not used yet.
    Parameters
    ----------
    time_abs_array : np.ndarray
        The absolute time array.
    time_unit_type : TimeUnit
        The time unit type of the time array. The method currently works only for `TimeUnit.relative_ms`.
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
    values_unscaled=np.array(df[unscaled_column_names])

    # scale data
    if len(scale_factors) != 0 and scale_factors is not None:
        scaled_values = values_unscaled * scale_factors

    # resample
    t_resampled = np.arange(0, time_abs_array[-1], 1 / resampling_frequency)

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
    