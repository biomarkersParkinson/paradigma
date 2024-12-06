import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime
import os

import tsdf
import tsdf.constants
from paradigma.ppg_preprocessing import preprocess_ppg_data, scan_and_sync_segments
from paradigma.preprocessing_config import IMUPreprocessingConfig, PPGPreprocessingConfig
from paradigma.heart_rate.heart_rate_analysis_config import HeartRateExtractionConfig
from paradigma.util import load_metadata_list
from paradigma.heart_rate.tfd import nonsep_gdtfd

def estimate_heart_rate_from_raw(input_path: str, raw_input_path: str, output_path: str, ppg_config:PPGPreprocessingConfig, imu_config: IMUPreprocessingConfig) -> None:
    # Load metadata and sync data (assuming similar steps to your other examples)
    metadatas_ppg, metadatas_imu = scan_and_sync_segments(os.path.join(raw_input_path, 'ppg'),
                                                       os.path.join(raw_input_path, 'imu'))
    df_ppg_preprocessed, _ = preprocess_ppg_data(metadatas_ppg[0], metadatas_imu[0],
                    "",
                    ppg_config, imu_config)
    estimate_heart_rate_old(input_path, df_ppg_preprocessed, output_path)


def estimate_heart_rate_from_raw_preprocessed(input_path: str, raw_input_path: str, config:PPGPreprocessingConfig, output_path: str) -> None:
    # Load metadata and sync data (assuming similar steps to your other examples)
    metadata_ppg_list = load_metadata_list(raw_input_path, "PPG_meta.json", ["PPG_time.bin", "PPG_samples.bin"])
    df_ppg_preprocessed = tsdf.load_dataframe_from_binaries(metadata_ppg_list, tsdf.constants.ConcatenationType.columns)

    estimate_heart_rate_old(df, df_ppg_preprocessed, output_path)


def estimate_heart_rate_old(input_path: str, df_ppg_preprocessed: pd.DataFrame, output_path: str, config:PPGPreprocessingConfig) -> None:
    # Load metadata and sync data (assuming similar steps to your other examples)
    metadata_classification_list = load_metadata_list(input_path, "classification_sqa_meta.json", ["classification_sqa_time.bin", "classification_sqa_ppg.bin", "classification_sqa_imu.bin"])
    df_classification = tsdf.load_dataframe_from_binaries(metadata_classification_list, tsdf.constants.ConcatenationType.columns)
    # arr_ppg = df_ppg[DataColumns.PPG].to_numpy()
    # relative_time_ppg = df_ppg[DataColumns.TIME].to_numpy()
    
    df_hr_est = estimate_heart_rate(df, df_ppg_preprocessed, config, output_path)


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
        class_acc_segment = acc_label[class_start:class_end]

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

    # # If no HR data exists, use default empty time
    # if t_hr_unix.size == 0:
    #     start_time_iso = datetime.utcfromtimestamp(0).isoformat() + 'Z'
    #     end_time_iso = datetime.utcfromtimestamp(0).isoformat() + 'Z'
    # else:
    #     start_time_iso = datetime.utcfromtimestamp(t_hr_unix[0] / config.unix_ticks_ms).isoformat() + 'Z'
    #     end_time_iso = datetime.utcfromtimestamp(t_hr_unix[-1] / config.unix_ticks_ms).isoformat() + 'Z'

    # metafile_pre_template["start_iso8601"] = start_time_iso
    # metafile_pre_template["end_iso8601"] = end_time_iso

    # # Time and HR metadata
    # metafile_time = metafile_pre_template.copy()
    # metafile_time["channels"] = ['time']
    # metafile_time["units"] = ['time_absolute_unix_s']
    # metafile_time["file_name"] = 'hr_est_time.bin'

    # metafile_values_hr = metafile_pre_template.copy()
    # metafile_values_hr["channels"] = ['HR estimates']
    # metafile_values_hr["units"] = ['min^-1']
    # metafile_values_hr["freq_sampling_original"] = round(fs_ppg_est, 2)
    # metafile_values_hr["file_name"] = 'hr_est_values.bin'

    # # Save metadata and data
    # meta_class = [metafile_time, metafile_values_hr]
    # mat_metadata_file_name = "hr_est_meta.json"
    # save_tsdf_data(meta_class, data_hr_est, location, mat_metadata_file_name)


def assign_sqa_label(ppg_prob, config: HeartRateExtractionConfig, acc_label=None) -> np.ndarray:
    """
    Assigns a signal quality label to every individual data point.

    Parameters:
    - ppg_prob: numpy array of probabilities for PPG
    - acc_label: numpy array of labels for accelerometer (optional)
    - fs: Sampling frequency

    Returns:
    - sqa_label: numpy array of signal quality assessment labels
    """
    # Default _label to ones if not provided
    if acc_label is None:
        acc_label = np.ones(len(ppg_prob))


    # Number of samples in an epoch
    fs = config.sampling_frequency
    samples_per_epoch = config.sqa_window_length_s * fs

    # Calculate number of samples to shift for each epoch
    samples_shift = config.sqa_window_step_size_s * fs
    n_samples = int((len(ppg_prob) + config.sqa_window_overlap_s) * fs)
    data_prob = np.zeros(n_samples)
    data_label_imu = np.zeros(n_samples, dtype=np.int8)

    for i in range(n_samples):
        # Start and end indices for current epoch
        start_idx = int(np.floor((i - (samples_per_epoch - samples_shift)) / fs))
        end_idx = int(np.floor(i / fs))

        # Correct for first and last epochs
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(ppg_prob):
            end_idx = len(ppg_prob)

        # Extract probabilities and labels for the current epoch
        prob = ppg_prob[start_idx:end_idx+1]
        label_imu = acc_label[start_idx:end_idx+1]

        # Calculate mean probability and majority voting for labels
        data_prob[i] = np.mean(prob)
        data_label_imu[i] = int(np.mean(label_imu) >= 0.5)

    # Set probability to zero if majority IMU label is 0
    data_prob[data_label_imu == 0] = 0
    sqa_label = (data_prob > config.threshold_sqa).astype(np.int8)

    return sqa_label

def extract_hr_segments(sqa_label: np.ndarray, min_hr_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts heart rate segments based on the SQA label.

    Parameters
    ----------
    sqa_label : np.ndarray
        The signal quality assessment label.
    min_hr_samples : int
        The minimum number of samples required for a heart rate segment.

    Returns
    -------
    Tuple[v_start_idx_long, v_end_idx_long]
        The start and end indices of the heart rate segments.
    """
    # Find the start and end indices of the heart rate segments
    v_start_idx = np.where(np.diff(sqa_label) == 1)[0] + 1
    v_end_idx = np.where(np.diff(sqa_label) == -1)[0] + 1

    # Check if the first segment is a start or end
    if sqa_label[0] == 1:
        v_start_idx = np.insert(v_start_idx, 0, 0)
    if sqa_label[-1] == 1:
        v_end_idx = np.append(v_end_idx, len(sqa_label))

    # Check if the segments are long enough
    v_start_idx_long = v_start_idx[(v_end_idx - v_start_idx) >= min_hr_samples]
    v_end_idx_long = v_end_idx[(v_end_idx - v_start_idx) >= min_hr_samples]

    return v_start_idx_long, v_end_idx_long

def extract_hr_from_segment(ppg: np.ndarray, tfd_length: int, fs: int, kern_type: str, kern_params: dict) -> np.ndarray:
    """Extracts heart rate from the time-frequency distribution of the PPG signal.

    Parameters
    ----------
    ppg : np.ndarray
        The preprocessed PPG segment with 2 seconds of padding on both sides to reduce boundary effects.
    tfd_length : int
        Length of each segment (in seconds) to calculate the time-frequency distribution.
    fs : int
        The sampling frequency of the PPG signal.
    MA : int
        The moving average window length.
    kern_type : str
        Type of TFD kernel to use (e.g., 'wvd' for Wigner-Ville distribution).
    kern_params : dict
        Parameters for the specified kernel. Not required for 'wvd', but relevant for other 
        kernels like 'spwvd' or 'swvd'. Default is None.

    Returns
    -------
    np.ndarray
        The estimated heart rate.
    """

    # Constants to handle boundary effects
    edge_padding = 4  # Additional 4 seconds (2 seconds on both sides)
    extended_epoch_length = tfd_length + edge_padding

    # Calculate the actual segment length without padding
    original_segment_length = (len(ppg) - edge_padding * fs) / fs

    # Determine the number of tfd_length-second segments
    if original_segment_length > extended_epoch_length:
        num_segments = int(original_segment_length // tfd_length)
    else:
        num_segments = 1  # For shorter segments (< 30s)

    # Split the PPG signal into segments
    ppg_segments = []
    for i in range(num_segments):
        if i != num_segments - 1:  # For all segments except the last
            start_idx = int(i * tfd_length * fs)
            end_idx = int((i + 1) * tfd_length * fs + edge_padding * fs)
        else:  # For the last segment
            start_idx = int(i * tfd_length * fs)
            end_idx = len(ppg)
        ppg_segments.append(ppg[start_idx:end_idx])

    hr_est_from_ppg = []
    for segment in ppg_segments:
    # Calculate the time-frequency distribution
        hr_tfd = extract_hr_with_tfd(segment, fs, kern_type, kern_params)
        hr_est_from_ppg.extend(hr_tfd)

    return hr_est_from_ppg

def extract_hr_with_tfd(ppg: np.ndarray, fs: int, kern_type: str, kern_params: dict) -> np.ndarray:
    """
    Estimate heart rate (HR) from a PPG segment using a TFD method with optional 
    moving average filtering.

    Parameters
    ----------
    ppg_segment : array-like
        Segment of the PPG signal to process.
    fs : int
        Sampling frequency in Hz.
    kern_type : str
        Type of TFD kernel to use.
    kern_params : dict
        Parameters for the specified kernel.

    Returns
    -------
    hr_smooth_tfd : list
        Estimated HR values (in beats per minute) for each 2-second segment of the PPG signal.
    """
   # Generate the TFD matrix using the specified kernel
    tfd = nonsep_gdtfd(ppg, kern_type, kern_params)  # Returns an NxN matrix

    # Get time and frequency axes for the TFD
    num_time_samples, num_freq_bins = tfd.shape
    time_axis = np.arange(num_time_samples) / fs
    freq_axis = np.linspace(0, 0.5, num_freq_bins) * fs

    # Estimate HR by identifying the max frequency in the TFD
    max_freq_indices = np.argmax(tfd, axis=0)

    hr_smooth_tfd = []
    for i in range(2, int(len(ppg) / fs) - 4, 2):  # Skip the first and last 2 seconds
        relevant_indices = (time_axis >= i) & (time_axis < i + 2)
        avg_frequency = np.mean(freq_axis[max_freq_indices[relevant_indices]])
        hr_smooth_tfd.append(60 * avg_frequency)  # Convert frequency to BPM

    return hr_smooth_tfd
