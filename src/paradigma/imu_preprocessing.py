from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

import tsdf
from paradigma.constants import DataColumns, TimeUnit
from paradigma.util import write_df_data, read_metadata
from paradigma.preprocessing_config import IMUPreprocessingConfig


def preprocess_imu_data(df: pd.DataFrame, config: IMUPreprocessingConfig, scale_factors: list) -> pd.DataFrame:

    # Rename columns
    df = df.rename(columns={f'rotation_{a}': f'gyroscope_{a}' for a in ['x', 'y', 'z']})
    df = df.rename(columns={f'acceleration_{a}': f'accelerometer_{a}' for a in ['x', 'y', 'z']})

    # convert to relative seconds from delta milliseconds
    df[config.time_colname] = transform_time_array(
        time_array=df[config.time_colname],
        scale_factor=1000, 
        input_unit_type = TimeUnit.DIFFERENCE_MS,
        output_unit_type = TimeUnit.RELATIVE_MS)
    

    df = resample_data(
        df=df,
        time_column=config.time_colname,
        time_unit_type=TimeUnit.RELATIVE_MS,
        unscaled_column_names = list(config.d_channels_imu.keys()),
        scale_factors=scale_factors,
        resampling_frequency=config.sampling_frequency)
    
    if config.side_watch == 'right':
        df[DataColumns.ACCELEROMETER_X] *= -1
        df[DataColumns.GYROSCOPE_Y] *= -1
        df[DataColumns.GYROSCOPE_Z] *= -1

    # Convert accelerometer data to correct units if necessary
    if config.acceleration_units == 'm/s^2':
        df[config.accelerometer_cols] /= 9.81
        
    # Extract accelerometer data
    accel_data = df[config.accelerometer_cols].values

    filter_renaming_configs = {
        "hp": {"result_columns": config.accelerometer_cols, "replace_original": True},
        "lp": {"result_columns": [f'{col}_grav' for col in config.accelerometer_cols], "replace_original": False},
    }

    # Apply filters in a loop
    for passband, filter_config in filter_renaming_configs.items():
        filtered_data = butterworth_filter(
            data=accel_data,
            order=config.filter_order,
            cutoff_frequency=config.lower_cutoff_frequency,
            passband=passband,
            sampling_frequency=config.sampling_frequency,
        )

        # Replace or add new columns based on configuration
        df[filter_config["result_columns"]] = filtered_data

    return df


def preprocess_imu_data_io(input_path: Union[str, Path], output_path: Union[str, Path], config: IMUPreprocessingConfig) -> None:

    # Load data
    metadata_time, metadata_samples = read_metadata(str(input_path), str(config.meta_filename),
                                                    str(config.time_filename), str(config.values_filename))
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Preprocess data
    df = preprocess_imu_data(df=df, config=config, scale_factors=metadata_samples.scale_factors)

    # Store data
    for sensor, units in zip(['accelerometer', 'gyroscope'], ['g', config.rotation_units]):
        df_sensor = df[[config.time_colname] + [x for x in df.columns if sensor in x]]

        metadata_samples.channels = [x for x in df.columns if sensor in x]
        metadata_samples.units = list(np.repeat(units, len(metadata_samples.channels)))
        metadata_samples.scale_factors = []
        metadata_samples.file_name = f'{sensor}_samples.bin'

        metadata_time.file_name = f'{sensor}_time.bin'
        metadata_time.units = ['time_relative_ms']

        write_df_data(metadata_time, metadata_samples, output_path, f'{sensor}_meta.json', df_sensor)


def transform_time_array(
    time_array: pd.Series,
    scale_factor: float,
    input_unit_type: str,
    output_unit_type: str,
    start_time: float = 0.0,
) -> np.ndarray:
    """
    Transforms the time array to relative time (when defined in delta time) and scales the values.

    Parameters
    ----------
    time_array : pd.Series
        The time array in milliseconds to transform.
    scale_factor : float
        The scale factor to apply to the time array.
    input_unit_type : str
        The time unit type of the input time array. Raw PPP data was in `TimeUnit.DIFFERENCE_MS`.
    output_unit_type : str
        The time unit type of the output time array. The processing is often done in `TimeUnit.RELATIVE_MS`.
    start_time : float, optional
        The start time of the time array in UNIX milliseconds (default is 0.0)

    Returns
    -------
    time_array
        The transformed time array in milliseconds, with the specified time unit type.
    """
    # Scale time array and transform to relative time (`TimeUnit.RELATIVE_MS`) 
    if input_unit_type == TimeUnit.DIFFERENCE_MS:
    # Convert a series of differences into cumulative sum to reconstruct original time series.
        time_array = np.cumsum(np.double(time_array)) / scale_factor
    elif input_unit_type == TimeUnit.ABSOLUTE_MS:
        # Set the start time if not provided.
        if np.isclose(start_time, 0.0, rtol=1e-09, atol=1e-09):
            start_time = time_array[0]
        # Convert absolute time stamps into a time series relative to start_time.
        time_array = (time_array - start_time) / scale_factor
    elif input_unit_type == TimeUnit.RELATIVE_MS:
        # Scale the relative time series as per the scale_factor.
        time_array = time_array / scale_factor

    # Transform the time array from `TimeUnit.RELATIVE_MS` to the specified time unit type
    if output_unit_type == TimeUnit.ABSOLUTE_MS:
        # Converts time array to absolute time by adding the start time to each element.
        time_array = time_array + start_time
    elif output_unit_type == TimeUnit.DIFFERENCE_MS:
        # Creates a new array starting with 0, followed by the differences between consecutive elements.
        time_array = np.diff(np.insert(time_array, 0, start_time))
    elif output_unit_type == TimeUnit.RELATIVE_MS:
        # The array is already in relative format, do nothing.
        pass
    return time_array


def resample_data(
    df: pd.DataFrame,
    time_column : str,
    time_unit_type: str,
    unscaled_column_names: List[str],
    resampling_frequency: int,
    scale_factors: List[float] = [],
    start_time: float = 0.0,
) -> pd.DataFrame:
    """
    Resamples the IMU data to the resampling frequency. The data is scaled before resampling.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to resample.
    time_column : str
        The name of the time column.
    time_unit_type : str
        The time unit type of the time array. The method currently works only for `TimeUnit.RELATIVE_MS`.
    unscaled_column_names : List[str]
        The names of the columns to resample.
    resampling_frequency : int
        The frequency to resample the data to.
    scale_factors : list, optional
        The scale factors to apply to the values before resampling (default is []).
    start_time : float, optional
        The start time of the time array, which is required if it is in absolute format (default is 0.0).

    Returns
    -------
    pd.DataFrame
        The resampled data.
    """
    # We need a start_time if the time is in absolute time format
    if time_unit_type == TimeUnit.ABSOLUTE_MS and start_time == 0.0:
        raise ValueError("start_time is required for absolute time format")

    # get time and values
    time_abs_array = np.array(df[time_column])
    values_array = np.array(df[unscaled_column_names])

    # Ensure the time array is strictly increasing
    if not np.all(np.diff(time_abs_array) > 0):
        raise ValueError("time_abs_array is not strictly increasing")

    # scale data
    if scale_factors:
        values_array = values_array * scale_factors

    # resample
    t_resampled = np.arange(start_time, time_abs_array[-1], 1 / resampling_frequency)
    
    # Perform vectorized interpolation
    interpolator = interp1d(time_abs_array, values_array, axis=0, kind="cubic")
    resampled_values = interpolator(t_resampled)

    # Create the output DataFrame
    df_resampled = pd.DataFrame(resampled_values, columns=unscaled_column_names)
    df_resampled[time_column] = t_resampled

    # Return the reordered DataFrame
    return df_resampled[[time_column] + unscaled_column_names]


def butterworth_filter(
    data: np.ndarray,
    order: int,
    cutoff_frequency: Union[float, List[float]],
    passband: str,
    sampling_frequency: int,
):
    """
    Applies the Butterworth filter to 2D sensor data

    Parameters
    ----------
    data : np.ndarray
        Array containing sensor data. Can be 1D (e.g., a single signal) or 2D (e.g., multi-axis sensor data).
    order: int
        The exponential order of the filter
    cutoff_frequency: float or List[float]
        The frequency at which the gain drops to 1/sqrt(2) that of the passband. If passband is 'band', then cutoff_frequency should be a list of two floats.
    passband: str
        Type of passband: ['hp', 'lp' or 'band']
    sampling_frequency: int
        The sampling frequency of the sensor data

    Returns
    -------
    np.ndarray
        The filtered sensor data after applying the Butterworth filter.
    """

    sos = signal.butter(
        N=order,
        Wn=cutoff_frequency,
        btype=passband,
        analog=False,
        fs=sampling_frequency,
        output="sos",
    )

    # Apply the filter, handling both 1D and 2D data
    if data.ndim == 1:  # 1D case
        return signal.sosfiltfilt(sos, data)
    elif data.ndim == 2:  # 2D case
        return signal.sosfiltfilt(sos, data, axis=0)
    else:
        raise ValueError("Data must be either 1D or 2D.")
    