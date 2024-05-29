import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline

import tsdf
from dbpd.constants import DataColumns
from dbpd.util import get_end_iso8601, write_data


class PreprocessingConfig:

    def __init__(self) -> None:
        self.meta_filename = 'IMU_meta.json'
        self.values_filename = 'IMU_samples.bin'
        self.time_filename = 'IMU_time.bin'

        self.acceleration_units = 'm/s^2'
        self.rotation_units = 'deg/s'

        self.d_channels_units = {
            DataColumns.ACCELEROMETER_X: self.acceleration_units,
            DataColumns.ACCELEROMETER_Y: self.acceleration_units,
            DataColumns.ACCELEROMETER_Z: self.acceleration_units,
            DataColumns.GYROSCOPE_X: self.rotation_units,
            DataColumns.GYROSCOPE_Y: self.rotation_units,
            DataColumns.GYROSCOPE_Z: self.rotation_units,
        }

        # filtering
        self.sampling_frequency = 100
        self.lower_cutoff_frequency = 0.3
        self.filter_order = 4


def preprocess_imu_data(input_path: str, output_path: str, config: PreprocessingConfig) -> None:

    # Load data
    metadata_dict = tsdf.load_metadata_from_path(os.path.join(input_path, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_samples = metadata_dict[config.values_filename]
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Rename columns
    df = df.rename(columns={f'rotation_{a}': f'gyroscope_{a}' for a in ['x', 'y', 'z']})
    df = df.rename(columns={f'acceleration_{a}': f'accelerometer_{a}' for a in ['x', 'y', 'z']})

    # convert to relative seconds from delta milliseconds
    df['time'] = transform_time_array(
        time_array=df['time'],
        scale_factor=1000, 
        data_in_delta_time=True)

    df = resample_data(
        time_abs_array=np.array(df['time']),
        values_unscaled=np.array(df[list(config.d_channels_units.keys())]),
        scale_factors=metadata_samples.scale_factors,
        resampling_frequency=config.sampling_frequency,
        time_column='time')

    # TODO: @Erik, please fix:
    # correct for sensor orientation - this subject has watch on right-hand side
    side_watch = 'left'
    df[DataColumns.ACCELEROMETER_Z] *= -1
    if side_watch == 'right':
        df[DataColumns.ACCELEROMETER_X] *= -1

    for col in [x for x in config.d_channels_units.keys() if 'accelerometer' in x]:

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
        meta_filename_store = f'{sensor}_meta.json'
        df_sensor = df[['time'] + [x for x in df.columns if sensor in x]]

        metadata_samples.__setattr__('channels', [x for x in df.columns if sensor in x])
        metadata_samples.__setattr__('units', list(np.repeat(units, len(metadata_samples.channels))))
        metadata_samples.__setattr__('meta_filename', meta_filename_store)
        metadata_samples.__setattr__('file_name', meta_filename_store.replace('_meta.json', '_samples.bin'))
        metadata_samples.__setattr__('file_dir_path', output_path)  

        metadata_time.__setattr__('file_dir_path', output_path)
        metadata_time.__setattr__('meta_filename', meta_filename_store)
        metadata_time.__setattr__('file_name', meta_filename_store.replace('_meta.json', '_time.bin'))
        metadata_time.__setattr__('units', ['time_relative_ms'])

        write_data(metadata_time, metadata_samples, output_path, meta_filename_store, df_sensor)


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
    