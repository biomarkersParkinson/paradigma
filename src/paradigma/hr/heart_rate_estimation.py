import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dateutil import parser

from datetime import datetime
import os

import tsdf
import tsdf.constants
from paradigma.heart_rate_util import extract_ppg_features, calculate_power_ratio, read_PPG_quality_classifier
from paradigma.ppg_preprocessing import preprocess_ppg_data, scan_and_sync_segments
from paradigma.preprocessing_config import IMUPreprocessingConfig, PPGPreprocessingConfig
from paradigma.util import load_metadata_list, read_metadata, write_np_data
from paradigma.constants import DataColumns, UNIX_TICKS_MS, DataUnits, TimeUnit

def estimate_heart_rate_from_raw(input_path: str, raw_input_path: str, output_path: str, ppg_config:PPGPreprocessingConfig, imu_config: IMUPreprocessingConfig) -> None:
    # Load metadata and sync data (assuming similar steps to your other examples)
    metadatas_ppg, metadatas_imu = scan_and_sync_segments(os.path.join(raw_input_path, 'ppg'),
                                                       os.path.join(raw_input_path, 'imu'))
    df_ppg_preprocessed, _ = preprocess_ppg_data(metadatas_ppg[0], metadatas_imu[0],
                    "",
                    ppg_config, imu_config)
    estimate_heart_rate(input_path, df_ppg_preprocessed, output_path)


def estimate_heart_rate_from_raw_preprocessed(input_path: str, raw_input_path: str, output_path: str) -> None:
    # Load metadata and sync data (assuming similar steps to your other examples)
    metadata_ppg_list = load_metadata_list(raw_input_path, "PPG_meta.json", ["PPG_time.bin", "PPG_samples.bin"])
    df_ppg_preprocessed = tsdf.load_dataframe_from_binaries(metadata_ppg_list, tsdf.constants.ConcatenationType.columns)

    estimate_heart_rate(input_path, df_ppg_preprocessed, output_path)

def estimate_heart_rate(input_path: str, df_ppg_preprocessed: pd.DataFrame, output_path: str, hr_config:PPGPreprocessingConfig) -> None:
    # Load metadata and sync data (assuming similar steps to your other examples)
    metadata_classification_list = load_metadata_list(input_path, "classification_sqa_meta.json", ["classification_sqa_time.bin", "classification_sqa_ppg.bin", "classification_sqa_imu.bin"])
    df_classification = tsdf.load_dataframe_from_binaries(metadata_classification_list, tsdf.constants.ConcatenationType.columns)
    # arr_ppg = df_ppg[DataColumns.PPG].to_numpy()
    # relative_time_ppg = df_ppg[DataColumns.TIME].to_numpy()
    
    # Other metadata loading steps for sync data (assuming similar to PPG loading)
    # Note: you may need to adjust based on your file structure
    metadata_sync_list = load_metadata_list(input_path, "classification_sqa_meta.json", ["classification_sqa_sync.bin"])
    df_ppg_sync = tsdf.load_dataframe_from_binaries(metadata_sync_list, tsdf.constants.ConcatenationType.columns)


    # Parameters for HR analysis
    min_window_length = 10
    min_hr_samples = min_window_length * config.fs_ppg
    threshold_sqa = 0.5

    # Heart rate estimation parameters
    hr_est_length = 2
    hr_est_samples = hr_est_length * config.fs_ppg

    # Time-frequency distribution parameters
    tfd_length = 10
    kern_type = 'sep'
    win_type_doppler = 'hamm'
    win_type_lag = 'hamm'
    win_length_doppler = 1
    win_length_lag = 8
    doppler_samples = config.fs_ppg * win_length_doppler
    lag_samples = win_length_lag * config.fs_ppg
    kern_params = [{'doppler_samples': doppler_samples, 'win_type_doppler': win_type_doppler}, 
                   {'lag_samples': lag_samples, 'win_type_lag': win_type_lag}]

    # Moving average filter struct initialization
    MA = {
        'value': 0,  # Set to 1 if using moving average filter
        'window': 30,
        'FC': (1 / 30) * np.ones(30)  # Moving average filter coefficients
    }

    v_hr_ppg = []
    t_hr_unix = []

    # Loop over synchronized segments
    for n in range(len(data_sync)):
        ppg_indices = data_sync[n, 0:2]  # Get PPG segment indices
        ppg_segment = data_sync[n, 2]  # Segment number

        # Load TSDF metadata
        meta_path_ppg = f"{input_path}/{ppg_segment}/tsdf_meta.json"
        metadata_list_ppg, data_list_ppg = load_tsdf_metadata_from_path(meta_path_ppg)

        # Extract relevant data (e.g., time and sample indices)
        time_idx_ppg = tsdf_values_idx(metadata_list_ppg, 'time')
        values_idx_ppg = tsdf_values_idx(metadata_list_ppg, 'samples')

        t_iso_ppg = metadata_list_ppg[time_idx_ppg]["start_iso8601"]
        datetime_ppg = datetime.strptime(t_iso_ppg, "%Y-%m-%dT%H:%M:%S.%fZ")
        ts_ppg = datetime_ppg.timestamp() * config.unix_ticks_ms

        t_ppg = np.cumsum(np.array(data_list_ppg[time_idx_ppg])) + ts_ppg
        tr_ppg = (t_ppg - ts_ppg) / config.unix_ticks_ms

        v_ppg = np.array(data_list_ppg[values_idx_ppg])

        # Get the PPG segment
        v_ppg = v_ppg[ppg_indices[0]:ppg_indices[1]]
        tr_ppg = tr_ppg[ppg_indices[0]:ppg_indices[1]]

        # Synchronization timestamp
        ts_sync = ts_ppg + tr_ppg[0] * config.unix_ticks_ms
        tr_ppg = tr_ppg - tr_ppg[0]

        fs_ppg_est = 1 / np.median(np.diff(tr_ppg))  # Estimate sampling frequency

        # Check if the segment is too short
        if len(v_ppg) < config.fs_ppg * min_window_length:
            print('Warning: Sample is of insufficient length!')
            continue
        else:
            v_ppg_pre, tr_ppg_pre = preprocessing_ppg(tr_ppg, v_ppg, config.fs_ppg)

        # Select correct classification data
        class_ppg_segment = ppg_post_prob[class_start:class_end]
        class_acc_segment = imu_label[class_start:class_end]

        # Assign window-level probabilities to individual samples
        data_prob_sample = sample_prob_final(class_ppg_segment, class_acc_segment, config.fs_ppg)

        # SQA label calculation
        sqa_label = np.where(data_prob_sample > threshold_sqa, 1, 0)

        # Extract HR segments based on SQA label
        v_start_idx, v_end_idx = extract_hr_segments(sqa_label, min_hr_samples)

        for i in range(len(v_start_idx)):
            # Relevant PPG segment
            rel_ppg = v_ppg_pre[v_start_idx[i]:v_end_idx[i]]
            rel_time = tr_ppg_pre[v_start_idx[i]:v_end_idx[i]]

            # Check whether the epoch can be extended by 2s on both sides
            if v_start_idx[i] < 2 * config.fs_ppg or v_end_idx[i] > len(v_ppg_pre) - 2 * config.fs_ppg:
                continue

            rel_ppg_spwvd = v_ppg_pre[v_start_idx[i] - 2 * config.fs_ppg : v_end_idx[i] + 2 * config.fs_ppg]
            hr_est = PPG_TFD_HR(rel_ppg_spwvd, tfd_length, MA, config.fs_ppg, kern_type, kern_params)

            # Corresponding HR estimation time array
            hr_time = rel_time[::hr_est_samples]
            t_epoch_unix = np.array(hr_time) * config.unix_ticks_ms + ts_sync

            # Save output
            v_hr_ppg.append(hr_est)
            t_hr_unix.append(t_epoch_unix)

    # Convert lists to numpy arrays
    v_hr_ppg = np.concatenate(v_hr_ppg)
    t_hr_unix = np.concatenate(t_hr_unix)

    # Save the hr output in tsdf format
    data_hr_est = {
        1: (t_hr_unix / config.unix_ticks_ms).astype(np.int32),
        2: np.array(v_hr_ppg, dtype=np.float32)
    }

    # Save location and metadata
    location = os.path.join(output_path, "5.quantification", "ppg")
    os.makedirs(location, exist_ok=True)
    metafile_pre_template = metadata_list_ppg[time_idx_ppg]

    # If no HR data exists, use default empty time
    if t_hr_unix.size == 0:
        start_time_iso = datetime.utcfromtimestamp(0).isoformat() + 'Z'
        end_time_iso = datetime.utcfromtimestamp(0).isoformat() + 'Z'
    else:
        start_time_iso = datetime.utcfromtimestamp(t_hr_unix[0] / config.unix_ticks_ms).isoformat() + 'Z'
        end_time_iso = datetime.utcfromtimestamp(t_hr_unix[-1] / config.unix_ticks_ms).isoformat() + 'Z'

    metafile_pre_template["start_iso8601"] = start_time_iso
    metafile_pre_template["end_iso8601"] = end_time_iso

    # Time and HR metadata
    metafile_time = metafile_pre_template.copy()
    metafile_time["channels"] = ['time']
    metafile_time["units"] = ['time_absolute_unix_s']
    metafile_time["file_name"] = 'hr_est_time.bin'

    metafile_values_hr = metafile_pre_template.copy()
    metafile_values_hr["channels"] = ['HR estimates']
    metafile_values_hr["units"] = ['min^-1']
    metafile_values_hr["freq_sampling_original"] = round(fs_ppg_est, 2)
    metafile_values_hr["file_name"] = 'hr_est_values.bin'

    # Save metadata and data
    meta_class = [metafile_time, metafile_values_hr]
    mat_metadata_file_name = "hr_est_meta.json"
    save_tsdf_data(meta_class, data_hr_est, location, mat_metadata_file_name)
