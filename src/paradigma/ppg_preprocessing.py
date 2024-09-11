import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union
from datetime import datetime, timedelta

import tsdf
from paradigma.constants import DataUnits, TimeUnit, DataColumns
from paradigma.preprocessing_config import PPGPreprocessingConfig, IMUPreprocessingConfig
from paradigma.util import parse_iso8601_to_datetime, write_data
import paradigma.imu_preprocessing


def scan_and_sync_segments(input_path_ppg, input_path_imu):

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


def preprocess_ppg_data(tsdf_meta_ppg: tsdf.TSDFMetadata, tsdf_meta_imu: tsdf.TSDFMetadata, output_path: Union[str, Path], ppg_config: PPGPreprocessingConfig, imu_config: IMUPreprocessingConfig):

    # Load PPG data
    metadata_time_ppg = tsdf_meta_ppg[ppg_config.time_filename]
    metadata_samples_ppg = tsdf_meta_ppg[ppg_config.values_filename]
    df_ppg = tsdf.load_dataframe_from_binaries([metadata_time_ppg, metadata_samples_ppg], tsdf.constants.ConcatenationType.columns)

    # Load IMU data
    metadata_time_imu = tsdf_meta_imu[imu_config.time_filename]
    metadata_samples_imu = tsdf_meta_imu[imu_config.values_filename]
    df_imu = tsdf.load_dataframe_from_binaries([metadata_time_imu, metadata_samples_imu], tsdf.constants.ConcatenationType.columns)

    # Drop the gyroscope columns from the IMU data
    cols_to_drop = df_imu.filter(regex='^rotation_').columns
    df_imu.drop(cols_to_drop, axis=1, inplace=True)
    df_imu = df_imu.rename(columns={f'acceleration_{a}': f'accelerometer_{a}' for a in ['x', 'y', 'z']})

    # Transform the time arrays to absolute milliseconds
    start_time_ppg = parse_iso8601_to_datetime(metadata_time_ppg.start_iso8601).timestamp()
    df_imu[DataColumns.TIME] = paradigma.imu_preprocessing.transform_time_array(
        time_array=df_imu[DataColumns.TIME],
        scale_factor=1000, 
        input_unit_type = TimeUnit.DIFFERENCE_MS,
        output_unit_type = TimeUnit.ABSOLUTE_MS,
        start_time = start_time_ppg)

    start_time_imu = parse_iso8601_to_datetime(metadata_time_imu.start_iso8601).timestamp()
    df_ppg[DataColumns.TIME] = paradigma.imu_preprocessing.transform_time_array(
        time_array=df_ppg[DataColumns.TIME],
        scale_factor=1000, 
        input_unit_type = TimeUnit.DIFFERENCE_MS,
        output_unit_type = TimeUnit.ABSOLUTE_MS,
        start_time = start_time_imu)

    # Extract overlapping segments
    print("Shape of the original data:", df_ppg.shape, df_imu.shape)
    df_ppg_overlapping, df_imu_overlapping = extract_overlapping_segments(df_ppg, df_imu)
    print("Shape of the overlapping segments:", df_ppg_overlapping.shape, df_imu_overlapping.shape)

    # The following method is failing
    df_imu_proc = paradigma.imu_preprocessing.resample_data(
        df=df_imu_overlapping,
        time_column=DataColumns.TIME,
        time_unit_type=TimeUnit.ABSOLUTE_MS,
        unscaled_column_names = list(imu_config.d_channels_accelerometer.keys()),
        resampling_frequency=imu_config.sampling_frequency,
        scale_factors=metadata_samples_imu.scale_factors[0:3],
        start_time=start_time_imu)

    # metadata_samples_ppg.scale_factors - the data specifies 1, but it is not an obligatory tsdf field, maybe it should be optional parameter in `resample_data`
    df_ppg_proc = paradigma.imu_preprocessing.resample_data(
        df=df_ppg_overlapping,
        time_column=DataColumns.TIME,
        time_unit_type=TimeUnit.ABSOLUTE_MS,
        unscaled_column_names = list(ppg_config.d_channels_ppg.keys()),
        scale_factors=metadata_samples_imu.scale_factors,
        resampling_frequency=ppg_config.sampling_frequency,
        start_time = start_time_imu
        )

    # apply Butterworth filter to accelerometer data
    for col in imu_config.d_channels_accelerometer.keys():

        # change to correct units [g]
        if imu_config.acceleration_units == DataUnits.ACCELERATION:
            df_imu_proc[col] /= 9.81

        for result, side_pass in zip(['filt', 'grav'], ['hp', 'lp']):
            df_imu_proc[f'{result}_{col}'] = paradigma.imu_preprocessing.butterworth_filter(
                single_sensor_col=np.array(df_imu_proc[col]),
                order=imu_config.filter_order,
                cutoff_frequency=imu_config.lower_cutoff_frequency,
                passband=side_pass,
                sampling_frequency=imu_config.sampling_frequency,
                )

        df_imu_proc = df_imu_proc.drop(columns=[col])
        df_imu_proc = df_imu_proc.rename(columns={f'filt_{col}': col})

        for col in ppg_config.d_channels_ppg.keys():
            df_ppg_proc[f'filt_{col}'] = paradigma.imu_preprocessing.butterworth_filter(
                single_sensor_col=np.array(df_ppg_proc[col]),
                order=ppg_config.filter_order,
                cutoff_frequency=[ppg_config.lower_cutoff_frequency, ppg_config.upper_cutoff_frequency],
                passband='band',
                sampling_frequency=ppg_config.sampling_frequency,
            )

            df_ppg_proc = df_ppg_proc.drop(columns=[col])
            df_ppg_proc = df_ppg_proc.rename(columns={f'filt_{col}': col})

    df_imu_proc[DataColumns.TIME] = paradigma.imu_preprocessing.transform_time_array(
        time_array=df_imu_proc[DataColumns.TIME],
        scale_factor=1,
        input_unit_type=TimeUnit.ABSOLUTE_MS,
        output_unit_type=TimeUnit.RELATIVE_MS,
        start_time=start_time_ppg,
    )

    df_ppg_proc[DataColumns.TIME] = paradigma.imu_preprocessing.transform_time_array(
        time_array=df_ppg_proc[DataColumns.TIME],
        scale_factor=1,
        input_unit_type=TimeUnit.ABSOLUTE_MS,
        output_unit_type=TimeUnit.RELATIVE_MS,
        start_time=start_time_imu,
    )

    # Store data
    metadata_samples_imu.channels = list(imu_config.d_channels_accelerometer.keys())
    metadata_samples_imu.units = list(imu_config.d_channels_accelerometer.values())
    metadata_samples_imu.file_name = 'accelerometer_samples.bin'
    metadata_time_imu.units = [TimeUnit.ABSOLUTE_MS]
    metadata_time_imu.file_name = 'accelerometer_time.bin'
    write_data(metadata_time_imu, metadata_samples_imu, output_path, 'accelerometer_meta.json', df_imu_proc)

    metadata_samples_ppg.channels = list(ppg_config.d_channels_ppg.keys())
    metadata_samples_ppg.units = list(ppg_config.d_channels_ppg.values())
    metadata_samples_ppg.file_name = 'PPG_samples.bin'
    metadata_time_ppg.units = [TimeUnit.ABSOLUTE_MS]
    metadata_time_ppg.file_name = 'PPG_time.bin'
    write_data(metadata_time_ppg, metadata_samples_ppg, output_path, 'PPG_meta.json', df_ppg_proc)


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

def extract_overlapping_segments(df_ppg, df_imu, time_column_ppg='time', time_column_imu='time'):
    """
    Extract DataFrames with overlapping data segments between IMU and PPG datasets based on their timestamps.

    Parameters:
    df_ppg (pd.DataFrame): DataFrame containing PPG data with a time column in UNIX milliseconds.
    df_imu (pd.DataFrame): DataFrame containing IMU data with a time column in UNIX milliseconds.
    time_column_ppg (str): Column name of the timestamp in the PPG DataFrame.
    time_column_imu (str): Column name of the timestamp in the IMU DataFrame.

    Returns:
    tuple: Tuple containing two DataFrames (df_ppg_overlapping, df_imu_overlapping) that contain only the data
    within the overlapping time segments.
    """
    # Convert UNIX milliseconds to seconds
    ppg_time = df_ppg[time_column_ppg] / 1000  # Convert milliseconds to seconds
    imu_time = df_imu[time_column_imu] / 1000  # Convert milliseconds to seconds

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
    df_imu_overlapping = df_imu.iloc[imu_start_index:imu_end_index + 1]

    return df_ppg_overlapping, df_imu_overlapping
