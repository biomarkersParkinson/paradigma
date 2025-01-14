import json
import numpy as np
import pandas as pd
import tsdf
from datetime import timedelta
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d
from typing import List, Tuple, Union

from paradigma.constants import TimeUnit, DataColumns
from paradigma.config import PPGConfig, IMUConfig
from paradigma.util import parse_iso8601_to_datetime, write_df_data, \
    read_metadata, extract_meta_from_tsdf_files


def resample_data(
    df: pd.DataFrame,
    time_column : str,
    values_column_names: List[str],
    resampling_frequency: int,
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
    resampling_frequency : int
        The frequency to which the data should be resampled (in Hz).

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
    The function uses cubic interpolation to resample the data to the specified frequency.
    It requires the input time array to be strictly increasing.
    """

    # Extract time and values from DataFrame
    time_abs_array = np.array(df[time_column])
    values_array = np.array(df[values_column_names])

    # Ensure the time array is strictly increasing
    if not np.all(np.diff(time_abs_array) > 0):
        raise ValueError("time_abs_array is not strictly increasing")

    # Resample the time data using the specified frequency
    t_resampled = np.arange(0, time_abs_array[-1], 1 / resampling_frequency)
    
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

def preprocess_imu_data(df: pd.DataFrame, config: IMUConfig, sensor: str) -> pd.DataFrame:
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
        values_column_names = values_colnames,
        resampling_frequency=config.sampling_frequency
    )
    
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


def preprocess_imu_data_io(path_to_input: str | Path, path_to_output: str | Path, 
                           config: IMUConfig, sensor: str) -> None:
    # Load data
    metadata_time, metadata_values = read_metadata(str(path_to_input), str(config.meta_filename),
                                                    str(config.time_filename), str(config.values_filename))
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

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
            metadata_time.units = [TimeUnit.RELATIVE_S]

            write_df_data(metadata_time, metadata_values, path_to_output, f'{sensor}_meta.json', df_sensor)


def scan_and_sync_segments(input_path_ppg: str | Path, input_path_imu: str | Path) -> Tuple[List[tsdf.TSDFMetadata], List[tsdf.TSDFMetadata]]:
    """
    Scan for available TSDF metadata files in the specified directories and synchronize the data segments based on the metadata start and end times.

    Parameters
    ----------
    input_path_ppg : str
        Path to the directory containing PPG data.
    input_path_imu : str
        Path to the directory containing IMU data.

    Returns
    -------
    Tuple[List[tsdf.TSDFMetadata], List[tsdf.TSDFMetadata]]
        A tuple containing lists of metadata objects for PPG and IMU data, respectively.
    """ 

    # Scan for available TSDF metadata files
    meta_ppg = extract_meta_from_tsdf_files(input_path_ppg)
    meta_imu = extract_meta_from_tsdf_files(input_path_imu)

    # Synchronize PPG and IMU data segments
    segments_ppg, segments_imu = synchronization(meta_ppg, meta_imu)  # Define `synchronization`
    
    assert len(segments_ppg) == len(segments_imu), 'Number of PPG and IMU segments do not match.'

    # Load metadata for every synced segment pair
    metadatas_ppg = [tsdf.load_metadata_from_path(meta_ppg[index]['tsdf_meta_fullpath']) for index in segments_ppg]
    metadatas_imu = [tsdf.load_metadata_from_path(meta_imu[index]['tsdf_meta_fullpath']) for index in segments_imu]

    return metadatas_ppg, metadatas_imu


def preprocess_ppg_data(df_ppg: pd.DataFrame, df_imu: pd.DataFrame, ppg_config: PPGConfig, 
                        imu_config: IMUConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess PPG and IMU (accelerometer only) data by resampling, filtering, and aligning the data segments.

    Parameters
    ----------
    tsdf_meta_ppg : tsdf.TSDFMetadata
        Metadata for the PPG data.
    tsdf_meta_imu : tsdf.TSDFMetadata
        Metadata for the IMU data.
    output_path : Union[str, Path]
        Path to store the preprocessed data.
    ppg_config : PPGPreprocessingConfig
        Configuration object for PPG preprocessing.
    imu_config : IMUPreprocessingConfig
        Configuration object for IMU preprocessing.
    store_locally : bool, optional
        Flag to store the preprocessed data locally, by default True.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Preprocessed PPG and IMU data as DataFrames.
    
    """

    # Drop the gyroscope columns from the IMU data
    cols_to_drop = df_imu.filter(regex='^gyroscope_').columns
    df_acc = df_imu.drop(cols_to_drop, axis=1)

    # Extract overlapping segments
    print(f"Original data shapes:\n- PPG data: {df_ppg.shape}\n- Accelerometer data: {df_acc.shape}")
    df_ppg_overlapping, df_acc_overlapping = extract_overlapping_segments(df_ppg, df_acc)
    print(f"Overlapping data shapes:\n- PPG data: {df_ppg_overlapping.shape}\n- Accelerometer data: {df_acc_overlapping.shape}")
    
    # Resample accelerometer data
    df_acc_proc = resample_data(
        df=df_acc_overlapping,
        time_column=DataColumns.TIME,
        values_column_names = list(imu_config.d_channels_accelerometer.keys()),
        resampling_frequency=imu_config.sampling_frequency
    )

    # Resample PPG data
    df_ppg_proc = resample_data(
        df=df_ppg_overlapping,
        time_column=DataColumns.TIME,
        values_column_names = list(ppg_config.d_channels_ppg.keys()),
        resampling_frequency=ppg_config.sampling_frequency
    )

    # apply Butterworth filter to accelerometer data
    for col in imu_config.d_channels_accelerometer.keys():

        for result, side_pass in zip(['filt', 'grav'], ['hp', 'lp']):
            df_acc_proc[f'{result}_{col}'] = butterworth_filter(
                data=np.array(df_acc_proc[col]),
                order=imu_config.filter_order,
                cutoff_frequency=imu_config.lower_cutoff_frequency,
                passband=side_pass,
                sampling_frequency=imu_config.sampling_frequency,
                )

        df_acc_proc = df_acc_proc.drop(columns=[col])
        df_acc_proc = df_acc_proc.rename(columns={f'filt_{col}': col})

    for col in ppg_config.d_channels_ppg.keys():
        df_ppg_proc[f'filt_{col}'] = butterworth_filter(
            data=np.array(df_ppg_proc[col]),
            order=ppg_config.filter_order,
            cutoff_frequency=[ppg_config.lower_cutoff_frequency, ppg_config.upper_cutoff_frequency],
            passband='band',
            sampling_frequency=ppg_config.sampling_frequency,
        )

        df_ppg_proc = df_ppg_proc.drop(columns=[col])
        df_ppg_proc = df_ppg_proc.rename(columns={f'filt_{col}': col})
    
    return df_ppg_proc, df_acc_proc

def preprocess_ppg_data_io(tsdf_meta_ppg: tsdf.TSDFMetadata, tsdf_meta_imu: tsdf.TSDFMetadata, 
                        output_path: Union[str, Path], ppg_config: PPGConfig, 
                           imu_config: IMUConfig) -> None:
    """	
    Preprocess PPG and IMU data by resampling, filtering, and aligning the data segments.

    Parameters
    ----------
    tsdf_meta_ppg : tsdf.TSDFMetadata
        Metadata for the PPG data.
    tsdf_meta_imu : tsdf.TSDFMetadata
        Metadata for the IMU data.
    output_path : Union[str, Path]
        Path to store the preprocessed data.
    ppg_config : PPGConfig
        Configuration object for PPG preprocessing.
    imu_config : IMUConfig
        Configuration object for IMU preprocessing.

    Returns
    -------
    None
    """ 

    # Load PPG data
    metadata_time_ppg = tsdf_meta_ppg[ppg_config.time_filename]
    metadata_values_ppg = tsdf_meta_ppg[ppg_config.values_filename]
    df_ppg = tsdf.load_dataframe_from_binaries([metadata_time_ppg, metadata_values_ppg], tsdf.constants.ConcatenationType.columns)

    # Load IMU data
    metadata_time_imu = tsdf_meta_imu[imu_config.time_filename]
    metadata_values_imu = tsdf_meta_imu[imu_config.values_filename]
    df_imu = tsdf.load_dataframe_from_binaries([metadata_time_imu, metadata_values_imu], tsdf.constants.ConcatenationType.columns)

    # Preprocess data
    df_ppg_proc, df_acc_proc = preprocess_ppg_data(df_ppg=df_ppg, df_imu=df_imu, ppg_config=ppg_config, imu_config=imu_config)

    # Store data
    metadata_values_imu.channels = list(imu_config.d_channels_accelerometer.keys())
    metadata_values_imu.units = list(imu_config.d_channels_accelerometer.values())
    metadata_values_imu.file_name = 'accelerometer_values.bin'
    metadata_time_imu.units = [TimeUnit.ABSOLUTE_MS]
    metadata_time_imu.file_name = 'accelerometer_time.bin'
    write_df_data(metadata_time_imu, metadata_values_imu, output_path, 'accelerometer_meta.json', df_acc_proc)

    metadata_values_ppg.channels = list(ppg_config.d_channels_ppg.keys())
    metadata_values_ppg.units = list(ppg_config.d_channels_ppg.values())
    metadata_values_ppg.file_name = 'PPG_values.bin'
    metadata_time_ppg.units = [TimeUnit.ABSOLUTE_MS]
    metadata_time_ppg.file_name = 'PPG_time.bin'
    write_df_data(metadata_time_ppg, metadata_values_ppg, output_path, 'PPG_meta.json', df_ppg_proc)

# TODO: ideally something like this should be possible directly in the tsdf library
def extract_meta_from_tsdf_files(tsdf_data_dir : str) -> List[dict]:
    """
    For each given TSDF directory, transcribe TSDF metadata contents to a list of dictionaries.
    
    Parameters
    ----------
    tsdf_data_dir : str
        Path to the directory containing TSDF metadata files.

    Returns
    -------
    List[Dict]
        List of dictionaries with metadata from each JSON file in the directory.

    Examples
    --------
    >>> extract_meta_from_tsdf_files('/path/to/tsdf_data')
    [{'start_iso8601': '2021-06-27T16:52:20Z', 'end_iso8601': '2021-06-27T17:52:20Z'}, ...]
    """
    metas = []
    
    # Collect all metadata JSON files in the specified directory
    meta_list = list(Path(tsdf_data_dir).rglob('*_meta.json'))
    for meta_file in meta_list:
        with open(meta_file, 'r') as file:
            json_obj = json.load(file)
            meta_data = {
                'tsdf_meta_fullpath': str(meta_file),
                'subject_id': json_obj['subject_id'],
                'start_iso8601': json_obj['start_iso8601'],
                'end_iso8601': json_obj['end_iso8601']
            }
            metas.append(meta_data)
    
    return metas


def synchronization(ppg_meta, imu_meta):
    """
    Synchronize PPG and IMU data segments based on their start and end times.

    Parameters
    ----------
    ppg_meta : list of dict
        List of dictionaries containing 'start_iso8601' and 'end_iso8601' keys for PPG data.
    imu_meta : list of dict
        List of dictionaries containing 'start_iso8601' and 'end_iso8601' keys for IMU data.

    Returns
    -------
    segment_ppg_total : list of int
        List of synchronized segment indices for PPG data.
    segment_imu_total : list of int
        List of synchronized segment indices for IMU data.
    """
    ppg_start_time = [parse_iso8601_to_datetime(t['start_iso8601']) for t in ppg_meta]
    imu_start_time = [parse_iso8601_to_datetime(t['start_iso8601']) for t in imu_meta]
    ppg_end_time = [parse_iso8601_to_datetime(t['end_iso8601']) for t in ppg_meta]
    imu_end_time = [parse_iso8601_to_datetime(t['end_iso8601']) for t in imu_meta]

    # Create a time vector covering the entire range
    time_vector_total = []
    current_time = min(min(ppg_start_time), min(imu_start_time))
    end_time = max(max(ppg_end_time), max(imu_end_time))
    while current_time <= end_time:
        time_vector_total.append(current_time)
        current_time += timedelta(seconds=1)
    
    time_vector_total = np.array(time_vector_total)

    # Initialize variables
    data_presence_ppg = np.zeros(len(time_vector_total), dtype=int)
    data_presence_ppg_idx = np.zeros(len(time_vector_total), dtype=int)
    data_presence_imu = np.zeros(len(time_vector_total), dtype=int)
    data_presence_imu_idx = np.zeros(len(time_vector_total), dtype=int)

    # Mark the segments of PPG data with 1
    for i, (start, end) in enumerate(zip(ppg_start_time, ppg_end_time)):
        indices = np.where((time_vector_total >= start) & (time_vector_total < end))[0]
        data_presence_ppg[indices] = 1
        data_presence_ppg_idx[indices] = i

    # Mark the segments of IMU data with 1
    for i, (start, end) in enumerate(zip(imu_start_time, imu_end_time)):
        indices = np.where((time_vector_total >= start) & (time_vector_total < end))[0]
        data_presence_imu[indices] = 1
        data_presence_imu_idx[indices] = i

    # Find the indices where both PPG and IMU data are present
    corr_indices = np.where((data_presence_ppg == 1) & (data_presence_imu == 1))[0]

    # Find the start and end indices of each segment
    corr_start_end = []
    if len(corr_indices) > 0:
        start_idx = corr_indices[0]
        for i in range(1, len(corr_indices)):
            if corr_indices[i] - corr_indices[i - 1] > 1:
                end_idx = corr_indices[i - 1]
                corr_start_end.append((start_idx, end_idx))
                start_idx = corr_indices[i]
        # Add the last segment
        corr_start_end.append((start_idx, corr_indices[-1]))

    # Extract the synchronized indices for each segment
    segment_ppg_total = []
    segment_imu_total = []
    for start_idx, end_idx in corr_start_end:
        segment_ppg = np.unique(data_presence_ppg_idx[start_idx:end_idx + 1])
        segment_imu = np.unique(data_presence_imu_idx[start_idx:end_idx + 1])
        if len(segment_ppg) > 1 and len(segment_imu) == 1:
            segment_ppg_total.extend(segment_ppg)
            segment_imu_total.extend([segment_imu[0]] * len(segment_ppg))
        elif len(segment_ppg) == 1 and len(segment_imu) > 1:
            segment_ppg_total.extend([segment_ppg[0]] * len(segment_imu))
            segment_imu_total.extend(segment_imu)
        elif len(segment_ppg) == len(segment_imu):
            segment_ppg_total.extend(segment_ppg)
            segment_imu_total.extend(segment_imu)
        else:
            continue

    return segment_ppg_total, segment_imu_total

def extract_overlapping_segments(df_ppg, df_acc, time_column_ppg='time', time_column_imu='time'):
    """
    Extract DataFrames with overlapping data segments between IMU and PPG datasets based on their timestamps.

    Parameters:
    df_ppg (pd.DataFrame): DataFrame containing PPG data with a time column in UNIX seconds.
    df_acc (pd.DataFrame): DataFrame containing IMU accelerometer data with a time column in UNIX seconds.
    time_column_ppg (str): Column name of the timestamp in the PPG DataFrame.
    time_column_imu (str): Column name of the timestamp in the IMU DataFrame.

    Returns:
    tuple: Tuple containing two DataFrames (df_ppg_overlapping, df_acc_overlapping) that contain only the data
    within the overlapping time segments.
    """
    ppg_time = df_ppg[time_column_ppg] 
    imu_time = df_acc[time_column_imu] 

    # Determine the overlapping time interval
    start_time = max(ppg_time.iloc[0], imu_time.iloc[0])
    end_time = min(ppg_time.iloc[-1], imu_time.iloc[-1])

    # Extract indices for overlapping segments
    ppg_start_index = np.searchsorted(ppg_time, start_time, 'left')
    ppg_end_index = np.searchsorted(ppg_time, end_time, 'right') - 1
    imu_start_index = np.searchsorted(imu_time, start_time, 'left')
    imu_end_index = np.searchsorted(imu_time, end_time, 'right') - 1

    # Extract overlapping segments from DataFrames
    df_ppg_overlapping = df_ppg.iloc[ppg_start_index:ppg_end_index + 1]
    df_acc_overlapping = df_acc.iloc[imu_start_index:imu_end_index + 1]

    return df_ppg_overlapping, df_acc_overlapping