import json
import numpy as np
import pandas as pd
import tsdf
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d
from typing import List, Tuple, Union
from datetime import datetime

from paradigma.constants import TimeUnit, DataColumns
from paradigma.config import PPGConfig, IMUConfig
from paradigma.util import write_df_data, read_metadata, invert_watch_side


def resample_data(
    df: pd.DataFrame,
    time_column : str,
    values_column_names: List[str],
    sampling_frequency: int,
    resampling_frequency: int,
    tolerance: float | None = None
) -> pd.DataFrame:
    """
    Resamples sensor data to a specified frequency using cubic interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the sensor data.
    time_column : str
        The name of the column containing the time data.
    values_column_names : List[str]
        A list of column names that should be resampled.
    sampling_frequency : int
        The original sampling frequency of the data (in Hz).
    resampling_frequency : int
        The frequency to which the data should be resampled (in Hz).
    tolerance : float, optional
        The tolerance added to the expected difference when checking 
        for contiguous timestamps. If not provided, it defaults to
        twice the expected interval.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the resampled data, where each column contains resampled values.
        The time column will reflect the new resampling frequency.

    Raises
    ------
    ValueError
        If the time array is not strictly increasing.

    Notes
    -----
    - Uses cubic interpolation for smooth resampling if there are enough points.
    - If only two timestamps are available, it falls back to linear interpolation.
    """
    # Set default tolerance if not provided to twice the expected interval
    if tolerance is None:
        tolerance = 2 * 1 / sampling_frequency

    # Extract time and values
    time_abs_array = np.array(df[time_column])
    values_array = np.array(df[values_column_names])

    # Ensure the time array is strictly increasing
    if not np.all(np.diff(time_abs_array) > 0):
        raise ValueError("Time array is not strictly increasing")
    
    # Ensure the time array is contiguous
    expected_interval = 1 / sampling_frequency
    timestamp_diffs = np.diff(time_abs_array)
    if np.any(np.abs(timestamp_diffs - expected_interval) > tolerance):
        raise ValueError("Time array is not contiguous")

    # Resample the time data using the specified frequency
    t_resampled = np.arange(time_abs_array[0], time_abs_array[-1], 1 / resampling_frequency)
    
    # Choose interpolation method
    interpolation_kind = "cubic" if len(time_abs_array) > 3 else "linear"
    interpolator = interp1d(time_abs_array, values_array, axis=0, kind=interpolation_kind, fill_value="extrapolate")
    
    # Interpolate
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

def preprocess_imu_data(df: pd.DataFrame, config: IMUConfig, sensor: str, watch_side: str) -> pd.DataFrame:
    """
    Preprocesses IMU data by resampling and applying filters.

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
    watch_side: str
        The side of the watch where the data was collected. Must be one of:
        - "left": Data was collected from the left wrist.
        - "right": Data was collected from the right wrist.

    Returns
    -------
    pd.DataFrame
        The preprocessed accelerometer and or gyroscope data with the following transformations:
        - Resampled data at the specified frequency.
        - Filtered accelerometer data with high-pass and low-pass filtering applied.
    
    Notes
    -----
    - The function applies Butterworth filters to accelerometer data, both high-pass and low-pass.
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
        
    # Resample the data to the specified frequency
    df = resample_data(
        df=df,
        time_column=DataColumns.TIME,
        values_column_names=values_colnames,
        sampling_frequency=config.sampling_frequency,
        resampling_frequency=config.sampling_frequency
    )

    # Invert the IMU data if the watch was worn on the right wrist
    df = invert_watch_side(df, watch_side, sensor)
    
    if sensor in ['accelerometer', 'both']:
      
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

    df = df[[DataColumns.TIME, *values_colnames]]

    return df


def preprocess_ppg_data(df_ppg: pd.DataFrame, df_acc: pd.DataFrame, ppg_config: PPGConfig, 
                        imu_config: IMUConfig, start_time_ppg: str, start_time_imu: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess PPG and IMU (accelerometer only) data by resampling, filtering, and aligning the data segments.

    Parameters
    ----------
    df_ppg : pd.DataFrame
        DataFrame containing PPG data.
    df_acc : pd.DataFrame
        DataFrame containing accelerometer from IMU data.
    ppg_config : PPGPreprocessingConfig
        Configuration object for PPG preprocessing.
    imu_config : IMUPreprocessingConfig
        Configuration object for IMU preprocessing.
    start_time_ppg : str
        iso8601 formatted start time of the PPG data.
    start_time_imu : str
        iso8601 formatted start time of the IMU data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Preprocessed PPG and IMU data as DataFrames.
    
    """

    # Extract overlapping segments
    df_ppg_overlapping, df_acc_overlapping = extract_overlapping_segments(df_ppg, df_acc, start_time_ppg, start_time_imu)
    
    # Resample accelerometer data
    df_acc_proc = resample_data(
        df=df_acc_overlapping,
        time_column=DataColumns.TIME,
        values_column_names = list(imu_config.d_channels_accelerometer.keys()),
        sampling_frequency=imu_config.sampling_frequency,
        resampling_frequency=imu_config.sampling_frequency
    )

    # Resample PPG data
    df_ppg_proc = resample_data(
        df=df_ppg_overlapping,
        time_column=DataColumns.TIME,
        values_column_names = list(ppg_config.d_channels_ppg.keys()),
        sampling_frequency=ppg_config.sampling_frequency,
        resampling_frequency=ppg_config.sampling_frequency
    )


    # Extract accelerometer data for filtering
    accel_data = df_acc_proc[imu_config.accelerometer_cols].values

    # Define filter configurations for high-pass and low-pass
    filter_renaming_configs = {
    "hp": {"result_columns": imu_config.accelerometer_cols, "replace_original": True}}

    # Apply filters in a loop
    for passband, filter_config in filter_renaming_configs.items():
        filtered_data = butterworth_filter(
        data=accel_data,
        order=imu_config.filter_order,
        cutoff_frequency=imu_config.lower_cutoff_frequency,
        passband=passband,
        sampling_frequency=imu_config.sampling_frequency,
        )

        # Replace or add new columns based on configuration
        df_acc_proc[filter_config["result_columns"]] = filtered_data
    
    # Extract accelerometer data for filtering
    ppg_data = df_ppg_proc[ppg_config.ppg_colname].values

    # Define filter configurations for high-pass and low-pass
    filter_renaming_configs = {
    "bandpass": {"result_columns": ppg_config.ppg_colname, "replace_original": True}}

    # Apply filters in a loop
    for passband, filter_config in filter_renaming_configs.items():
        filtered_data = butterworth_filter(
        data=ppg_data,
        order=ppg_config.filter_order,
        cutoff_frequency=[ppg_config.lower_cutoff_frequency, ppg_config.upper_cutoff_frequency],
        passband=passband,
        sampling_frequency=ppg_config.sampling_frequency,
        )

        # Replace or add new columns based on configuration
        df_ppg_proc[filter_config["result_columns"]] = filtered_data
    
    return df_ppg_proc, df_acc_proc




def extract_overlapping_segments(df_ppg: pd.DataFrame, df_acc: pd.DataFrame, start_time_ppg: str, start_time_acc: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract DataFrames with overlapping data segments between accelerometer (from the IMU) and PPG datasets based on their timestamps.

    Parameters
    ----------
    df_ppg : pd.DataFrame
        DataFrame containing PPG data.
    df_acc : pd.DataFrame
        DataFrame containing accelerometer data from the IMU.
    start_time_ppg : str
        iso8601 formatted start time of the PPG data.
    start_time_acc : str
        iso8601 formatted start time of the accelerometer data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames containing the overlapping segments (time and values) of PPG and accelerometer data.
    """
    # Convert start times to Unix timestamps
    datetime_ppg_start = datetime.fromisoformat(start_time_ppg.replace("Z", "+00:00"))
    start_unix_ppg = int(datetime_ppg_start.timestamp())
    datetime_acc_start = datetime.fromisoformat(start_time_acc.replace("Z", "+00:00"))
    start_acc_ppg = int(datetime_acc_start.timestamp())

    # Calculate the time in Unix timestamps for each dataset because the timestamps are relative to the start time
    ppg_time = df_ppg[DataColumns.TIME] + start_unix_ppg
    acc_time = df_acc[DataColumns.TIME] + start_acc_ppg

    # Determine the overlapping time interval
    start_time = max(ppg_time.iloc[0], acc_time.iloc[0])
    end_time = min(ppg_time.iloc[-1], acc_time.iloc[-1])

    # Extract indices for overlapping segments
    ppg_start_index = np.searchsorted(ppg_time, start_time, 'left')
    ppg_end_index = np.searchsorted(ppg_time, end_time, 'right') - 1
    acc_start_index = np.searchsorted(acc_time, start_time, 'left')
    acc_end_index = np.searchsorted(acc_time, end_time, 'right') - 1

    # Extract overlapping segments from DataFrames
    df_ppg_overlapping = df_ppg.iloc[ppg_start_index:ppg_end_index + 1]
    df_acc_overlapping = df_acc.iloc[acc_start_index:acc_end_index + 1]

    return df_ppg_overlapping, df_acc_overlapping