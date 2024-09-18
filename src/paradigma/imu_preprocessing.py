from pathlib import Path
from typing import List, Union, Optional, Literal
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline

import tsdf
from paradigma.constants import DataColumns, TimeUnit
from paradigma.util import write_data, read_metadata
from paradigma.preprocessing_config import IMUPreprocessingConfig


def preprocess_imu_data(input_path: Union[str, Path], output_path: Union[str, Path], config: IMUPreprocessingConfig) -> None:

    # Load data
    metadata_time, metadata_samples = read_metadata(str(input_path), str(config.meta_filename),
                                                    str(config.time_filename), str(config.values_filename))
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

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
        scale_factors=metadata_samples.scale_factors,
        resampling_frequency=config.sampling_frequency)
    
    if config.side_watch == 'left':
        df[DataColumns.ACCELEROMETER_X] *= -1

    # Extract relevant columns for accelerometer data
    accel_cols = list(config.d_channels_accelerometer.keys())

    # Change to correct units [g]
    df[accel_cols] = df[accel_cols] / 9.81 if config.acceleration_units == 'm/s^2' else df[accel_cols]

    # Extract the accelerometer data as a 2D array
    accel_data = df[accel_cols].values

    # Define filtering passbands
    passbands = ['hp', 'lp'] 
    filtered_data = {}

    # Apply Butterworth filter for each passband and result type
    for result, passband in zip(['filt', 'grav'], passbands):
        filtered_data[result] = butterworth_filter(
            sensor_data=accel_data,
            order=config.filter_order,
            cutoff_frequency=config.lower_cutoff_frequency,
            passband=passband,
            sampling_frequency=config.sampling_frequency
        )

    # Create DataFrames from filtered data
    filtered_dfs = {f'{result}_{col}': pd.Series(data[:, i]) for i, col in enumerate(accel_cols) for result, data in filtered_data.items()}

    # Combine filtered columns into DataFrame
    filtered_df = pd.DataFrame(filtered_dfs)

    # Drop original accelerometer columns and append filtered results
    df = df.drop(columns=accel_cols).join(filtered_df).rename(columns={col: col.replace('filt_', '') for col in filtered_df.columns})

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
    scale_factors: Optional[List[float]] = None,
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
        The time unit type of the time array. Should be 'absolute_ms' or 'relative_ms'.
    unscaled_column_names : List[str]
        The names of the columns to resample.
    resampling_frequency : int
        The frequency to resample the data to (Hz).
    scale_factors : List[float], optional
        The scale factors to apply to the values before resampling (default is None).
    start_time : float, optional
        The start time of the time array, which is required if time_unit_type is 'absolute_ms' (default is 0.0).

    Returns
    -------
    pd.DataFrame
        The resampled data.
    """
    # Validate input
    if time_unit_type not in [TimeUnit.ABSOLUTE_MS, TimeUnit.RELATIVE_MS]:
        raise ValueError("Invalid time_unit_type. Choose 'absolute_ms' or 'relative_ms'.")
    
    if time_unit_type == TimeUnit.ABSOLUTE_MS and start_time == 0.0:
        raise ValueError("start_time is required for absolute time format.")

    if scale_factors is None:
        scale_factors = [1.0] * len(unscaled_column_names)
    
    if len(scale_factors) != len(unscaled_column_names):
        raise ValueError("The length of scale_factors must match the number of unscaled columns.")

    # Extract time and values
    time_abs_array = df[time_column].values
    values_unscaled = df[unscaled_column_names].values

    # Scale data
    scaled_values = values_unscaled * scale_factors

    # Determine the resampling intervals
    if time_unit_type == TimeUnit.ABSOLUTE_MS:
        t_resampled = np.arange(start_time, time_abs_array[-1], 1000 / resampling_frequency)
    else:  # RELATIVE_MS
        t_resampled = np.arange(0, time_abs_array[-1], 1000 / resampling_frequency)

    # Prepare DataFrame for resampled data
    df_resampled = pd.DataFrame({time_column: t_resampled})

    # Interpolate each sensor column
    if not np.all(np.diff(time_abs_array) > 0):
        raise ValueError("Time column is not strictly increasing.")
    
    for i, sensor_col in enumerate(unscaled_column_names):
        cs = CubicSpline(time_abs_array, scaled_values[:, i], bc_type='natural')
        df_resampled[sensor_col] = cs(df_resampled[time_column])

    return df_resampled


def butterworth_filter(
    sensor_data: np.ndarray,
    order: int,
    cutoff_frequency: Union[float, List[float]],
    passband: str,
    sampling_frequency: int,
) -> np.ndarray:
    """
    Applies the Butterworth filter to a single sensor column

    Parameters
    ----------
    sensor_data: np.ndarray
        A 2D array where each column contains sensor data.
    order: int
        The order of the filter
    cutoff_frequency: float or List[float]
        The cutoff frequency for the filter. For 'band' type, this should be a list of two floats.
    passband: str
        Type of passband filter: ['hp', 'lp', 'band', 'bs]
    sampling_frequency: int
        The sampling frequency of the sensor data

    Returns
    -------
    np.ndarray
        The filtered sensor data
    """

    # Validate passband type
    if passband not in ['hp', 'lp', 'band', 'bs']:
        raise ValueError("Invalid passband type. Choose from ['hp', 'lp', 'band', 'bs']")
    
    # Validate cutoff_frequency for bandpass and bandstop
    if passband in ['band', 'bs']:
        if not isinstance(cutoff_frequency, list) or len(cutoff_frequency) != 2:
            raise ValueError("For 'band' and 'bs' passbands, cutoff_frequency must be a list of two frequencies.")
    else:
        if isinstance(cutoff_frequency, list):
            raise ValueError("For 'hp' and 'lp' passbands, cutoff_frequency must be a single float.")

    sos = signal.butter(
        N=order,
        Wn=cutoff_frequency,
        btype=passband,
        analog=False,
        fs=sampling_frequency,
        output="sos",
    )
    
    return signal.sosfiltfilt(sos, sensor_data, axis=0)
    
    