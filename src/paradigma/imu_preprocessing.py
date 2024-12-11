from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

import tsdf
from paradigma.constants import DataColumns, TimeUnit
from paradigma.util import write_df_data, read_metadata
from paradigma.config import IMUConfig


def preprocess_imu_data(df: pd.DataFrame, config: IMUConfig, sensor: str) -> pd.DataFrame:
    """
    Preprocesses IMU data by renaming columns, transforming time units, resampling, and applying filters.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing raw accelerometer and/or gyroscope data.
    config : IMUConfig
        Configuration object containing various settings, such as time column name, accelerometer and/or gyroscope columns,
        filter settings, and sampling frequency.
    sensor: str
        Name of the sensor data to be preprocessed. Must be one of:
        - "accelerometer": Preprocess accelerometer data only.
        - "gyroscope": Preprocess gyroscope data only.
        - "both": Preprocess both accelerometer and gyroscope data.

    Returns
    -------
    pd.DataFrame
        The preprocessed accelerometer and or gyroscope data with the following transformations:
        - Renamed columns for accelerometer and gyroscope data.
        - Transformed time column to relative time in milliseconds.
        - Resampled data at the specified frequency.
        - Adjustments based on the specified `side_watch` (left/right).
        - Accelerometer data converted to the correct units, if necessary.
        - Filtered accelerometer data with high-pass and low-pass filtering applied.
    
    Notes
    -----
    - The function applies Butterworth filters to accelerometer data, both high-pass and low-pass.
    - The time column is converted from delta milliseconds to relative milliseconds.
    - Adjustments for the right-hand side watch are made by flipping the signs of specific columns.
    - If the accelerometer data is in 'm/s^2', it will be converted from 'g' to 'm/s^2' using gravity's constant (9.81 m/s^2).
    """

    # Extract sensor column
    if sensor == 'accelerometer':
        values_colnames = config.accelerometer_cols
    elif sensor == 'gyroscope':
        values_colnames = config.gyroscope_cols
    elif sensor == 'both':
        values_colnames = config.accelerometer_cols + config.gyroscope_cols
    else:
        raise('Sensor should be either accelerometer, gyroscope, or both')
    
    # Convert to relative seconds from delta milliseconds
    df[DataColumns.TIME] = transform_time_array(
        time_array=df[DataColumns.TIME],
        scale_factor=1000, 
        input_unit_type = TimeUnit.DIFFERENCE_MS,
        output_unit_type = TimeUnit.RELATIVE_MS)
        
    # Resample the data to the specified frequency
    df = resample_data(
        df=df,
        time_column=DataColumns.TIME,
        time_unit_type=TimeUnit.RELATIVE_MS,
        values_column_names = values_colnames,
        resampling_frequency=config.sampling_frequency)
    
    if sensor in ['gyroscope', 'both']:
        # Flip signs for right-side watch
        if config.side_watch == 'right':
            df[DataColumns.GYROSCOPE_Y] *= -1
            df[DataColumns.GYROSCOPE_Z] *= -1

    if sensor in ['accelerometer', 'both']:

        # Flip signs for right-side watch
        if config.side_watch == 'right':
            df[DataColumns.ACCELEROMETER_X] *= -1

        # Convert accelerometer data to correct units if necessary
        if config.acceleration_units == 'm/s^2':
            df[config.accelerometer_cols] /= 9.81
        
        # Extract accelerometer data for filtering
        accel_data = df[config.accelerometer_cols].values

        # Define filter configurations for high-pass and low-pass
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

        values_colnames += config.gravity_cols

    df = df[[config.time_colname, *values_colnames]]

    return df


def preprocess_imu_data_io(input_path: Union[str, Path], output_path: Union[str, Path], config: IMUConfig, sensor: str) -> None:

    # Load data
    metadata_time, metadata_values = read_metadata(str(input_path), str(config.meta_filename),
                                                    str(config.time_filename), str(config.values_filename))
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Rename columns
    df = df.rename(columns={f'rotation_{a}': f'gyroscope_{a}' for a in ['x', 'y', 'z']})
    df = df.rename(columns={f'acceleration_{a}': f'accelerometer_{a}' for a in ['x', 'y', 'z']})

    # Apply scale factors 
    df[list(config.d_channels_imu.keys())] *= metadata_values.scale_factors

    # Preprocess data
    df = preprocess_imu_data(df=df, config=config, sensor=sensor)

    # Store data
    for sensor, units in zip(['accelerometer', 'gyroscope'], ['g', config.rotation_units]):
        if any(sensor in col for col in df.columns):
            df_sensor = df[[DataColumns.TIME] + [x for x in df.columns if sensor in x]]

            metadata_values.channels = [x for x in df.columns if sensor in x]
            metadata_values.units = list(np.repeat(units, len(metadata_values.channels)))
            metadata_values.scale_factors = []
            metadata_values.file_name = f'{sensor}_values.bin'

            metadata_time.file_name = f'{sensor}_time.bin'
            metadata_time.units = ['time_relative_ms']

            write_df_data(metadata_time, metadata_values, output_path, f'{sensor}_meta.json', df_sensor)


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
    np.ndarray
        The transformed time array in milliseconds, with the specified time unit type.

    Notes
    -----
    - The function handles different time units (`TimeUnit.DIFFERENCE_MS`, `TimeUnit.ABSOLUTE_MS`, `TimeUnit.RELATIVE_MS`).
    - The transformation allows for scaling of the time array, converting between time unit types (e.g., relative, absolute, or difference).
    - When converting to `TimeUnit.RELATIVE_MS`, the function calculates the relative time starting from the provided or default start time.
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
    values_column_names: List[str],
    resampling_frequency: int,
    start_time: float = 0.0,
) -> pd.DataFrame:
    """
    Resamples IMU data to the specified frequency, scaling values before resampling.

    This function takes in sensor data, scales the data if scale factors are provided,
    and resamples the data to a specified frequency using cubic interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the sensor data.
    time_column : str
        The name of the column containing the time data.
    time_unit_type : str
        The time unit type of the time array. This should be 'relative_ms' or 'absolute_ms'.
    values_column_names : List[str]
        A list of column names that should be resampled.
    resampling_frequency : int
        The frequency to which the data should be resampled (in Hz).
    start_time : float, optional
        The start time of the time array, used for absolute time formats. Default is 0.0.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the resampled data, where each column contains resampled values.
        The time column will reflect the new resampling frequency.

    Raises
    ------
    ValueError
        If the time array is not strictly increasing.
        If the start_time is missing when using absolute time format.

    Notes
    -----
    The function uses cubic interpolation to resample the data to the specified frequency.
    It requires the input time array to be strictly increasing.
    """
    # Validate that start_time is provided if time_unit_type is 'absolute_ms'
    if time_unit_type == TimeUnit.ABSOLUTE_MS and start_time == 0.0:
        raise ValueError("start_time is required for absolute time format")

    # Extract time and values from DataFrame
    time_abs_array = np.array(df[time_column])
    values_array = np.array(df[values_column_names])

    # Ensure the time array is strictly increasing
    if not np.all(np.diff(time_abs_array) > 0):
        raise ValueError("time_abs_array is not strictly increasing")

    # Resample the time data using the specified frequency
    t_resampled = np.arange(start_time, time_abs_array[-1], 1 / resampling_frequency)
    
    # Interpolate the data using cubic interpolation
    interpolator = interp1d(time_abs_array, values_array, axis=0, kind="cubic")
    resampled_values = interpolator(t_resampled)

    # Create a DataFrame with the resampled data
    df_resampled = pd.DataFrame(resampled_values, columns=values_column_names)
    df_resampled[time_column] = t_resampled

    # Return the DataFrame with columns in the correct order
    return df_resampled[[time_column] + values_column_names]


def butterworth_filter(
    data: np.ndarray,
    order: int,
    cutoff_frequency: Union[float, List[float]],
    passband: str,
    sampling_frequency: int,
):
    """
    Applies a Butterworth filter to 1D or 2D sensor data.

    This function applies a low-pass, high-pass, or band-pass Butterworth filter to the 
    input data. The filter is designed using the specified order, cutoff frequency, 
    and passband type. The function can handle both 1D and 2D data arrays.

    Parameters
    ----------
    data : np.ndarray
        The sensor data to be filtered. Can be 1D (e.g., a single signal) or 2D 
        (e.g., multi-axis sensor data).
    order : int
        The order of the Butterworth filter. Higher values result in a steeper roll-off.
    cutoff_frequency : float or List[float]
        The cutoff frequency (or frequencies) for the filter. For a low-pass or high-pass filter, 
        this is a single float. For a band-pass filter, this should be a list of two floats, 
        specifying the lower and upper cutoff frequencies.
    passband : str
        The type of passband to apply. Options are:
        - 'hp' : high-pass filter
        - 'lp' : low-pass filter
        - 'band' : band-pass filter
    sampling_frequency : int
        The sampling frequency of the data in Hz. This is used to normalize the cutoff frequency.

    Returns
    -------
    np.ndarray
        The filtered sensor data. The shape of the output is the same as the input data.

    Raises
    ------
    ValueError
        If the input data has more than two dimensions, or if an invalid passband is specified.

    Notes
    -----
    The function uses `scipy.signal.butter` to design the filter and `scipy.signal.sosfiltfilt`
    to apply it using second-order sections (SOS) to improve numerical stability.
    """
    # Design the filter using second-order sections (SOS)
    sos = signal.butter(
        N=order,
        Wn=cutoff_frequency,
        btype=passband,
        analog=False,
        fs=sampling_frequency,
        output="sos",
    )

    # Apply the filter to the data
    if data.ndim == 1:  # 1D data case
        return signal.sosfiltfilt(sos, data)
    elif data.ndim == 2:  # 2D data case
        return signal.sosfiltfilt(sos, data, axis=0)
    else:
        raise ValueError("Data must be either 1D or 2D.")
    