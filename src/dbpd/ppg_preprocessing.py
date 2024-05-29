import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List

from datetime import datetime, timedelta

# Module methods

def tsdf_scan_meta(tsdf_data_full_path : str) -> List[dict]:
    """
    For each given TSDF directory, transcribe TSDF metadata contents to a list of dictionaries.
    
    Parameters
    ----------
    tsdf_data_full_path : str
        Full path to the directory containing TSDF metadata files.

    Returns
    -------
    List[Dict]
        List of dictionaries with metadata from each JSON file in the directory.

    Examples
    --------
    >>> tsdf_scan_meta('/path/to/tsdf_data')
    [{'start_iso8601': '2021-06-27T16:52:20Z', 'end_iso8601': '2021-06-27T17:52:20Z'}, ...]
    """
    tsdf = []
    
    # Collect all metadata JSON files in the specified directory
    meta_list = list(Path(tsdf_data_full_path).rglob('*_meta.json'))
    for meta_file in meta_list:
        with open(meta_file, 'r') as file:
            json_obj = json.load(file)
            meta_data = {
                'tsdf_meta_fullpath': str(meta_file),
                'subject_id': json_obj['subject_id'],
                'start_iso8601': json_obj['start_iso8601'],
                'end_iso8601': json_obj['end_iso8601']
            }
            tsdf.append(meta_data)
    
    return tsdf


def convert_iso8601_to_datetime(date_str):
        """
        Convert a date string to a datetime object.

        Parameters
        ----------
        date_str : str
            Date string in the format '%d-%b-%Y %H:%M:%S %Z'.

        Returns
        -------
        datetime
            A datetime object corresponding to the input date string.

        Examples
        --------
        >>> convert_to_datetime('27-Jun-2021 16:52:20 UTC')
        datetime.datetime(2021, 6, 27, 16, 52, 20, tzinfo=<UTC>)
        """
        return datetime.strptime(date_str, '%d-%b-%Y %H:%M:%S %Z')

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
    ppg_start_time = [convert_iso8601_to_datetime(t['start_iso8601']) for t in ppg_meta]
    imu_start_time = [convert_iso8601_to_datetime(t['start_iso8601']) for t in imu_meta]
    ppg_end_time = [convert_iso8601_to_datetime(t['end_iso8601']) for t in ppg_meta]
    imu_end_time = [convert_iso8601_to_datetime(t['end_iso8601']) for t in imu_meta]

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