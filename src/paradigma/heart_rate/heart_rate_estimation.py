import numpy as np
from typing import Tuple

from paradigma.config import HeartRateConfig
from paradigma.heart_rate.tfd import nonsep_gdtfd

def assign_sqa_label(ppg_prob: np.ndarray, config: HeartRateConfig, acc_label=None) -> np.ndarray:
    """
    Assigns a signal quality label to every individual data point.

    Parameters:
    - ppg_prob: numpy array of probabilities for PPG
    - config: HeartRateConfig object containing configuration parameters
    - acc_label: numpy array of labels for accelerometer (optional)

    Returns:
    - sqa_label: numpy array of signal quality assessment labels 
    """
    # Default _label to ones if not provided
    if acc_label is None:
        acc_label = np.ones(len(ppg_prob))

    # Number of samples in an epoch
    fs = config.sampling_frequency
    samples_per_epoch = config.window_length_s * fs

    # Calculate number of samples to shift for each epoch
    samples_shift = config.window_step_length_s * fs
    n_samples = int(np.round(len(ppg_prob) + config.window_overlap_s) * fs)
    data_prob = np.zeros(n_samples)
    data_label_imu = np.zeros(n_samples, dtype=np.int8)

    for i in range(n_samples):
        # Start and end indices for current epoch
        start_idx = max(0, int((i - (samples_per_epoch - samples_shift)) // fs)) # max to handle first epochs
        end_idx = min(int(i // fs), len(ppg_prob))  # min to handle last epochs  

        # Extract probabilities and labels for the current epoch
        prob = ppg_prob[start_idx:end_idx+1]
        label_imu = acc_label[start_idx:end_idx+1]

        # Calculate mean probability and majority voting for labels
        data_prob[i] = np.mean(prob)
        data_label_imu[i] = int(np.mean(label_imu) >= 0.5)

    # Set probability to zero if majority IMU label is 0
    data_prob[data_label_imu == 0] = 0
    sqa_label = data_prob >= config.threshold_sqa

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
    v_start_idx = np.where(np.diff(sqa_label.astype(int)) == 1)[0] + 1
    v_end_idx = np.where(np.diff(sqa_label.astype(int)) == -1)[0] + 1

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
    edge_padding = 4 * fs  # Additional 4 seconds (2 seconds on both sides)
    tfd_length = tfd_length * fs  # Convert tfd_length to samples

    # Calculate the actual segment length without padding
    original_segment_length = len(ppg) - edge_padding

    # Determine the number of tfd_length-second segments
    num_segments = max(1, int(original_segment_length // tfd_length))

    # Split the PPG signal into segments
    ppg_segments = []
    for i in range(num_segments):
        if i != num_segments - 1:  # For all segments except the last
            start_idx = int(i * tfd_length)
            end_idx = int((i + 1) * tfd_length + edge_padding)
        else:  # For the last segment
            start_idx = int(i * tfd_length)
            end_idx = len(ppg)
        ppg_segments.append(ppg[start_idx:end_idx])

    hr_est_from_ppg = np.array([])
    for segment in ppg_segments:
    # Calculate the time-frequency distribution
        hr_tfd = extract_hr_with_tfd(segment, fs, kern_type, kern_params)
        hr_est_from_ppg = np.concatenate((hr_est_from_ppg, hr_tfd))  # Append the HR estimates

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
    hr_smooth_tfd : np.ndarray
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

    hr_smooth_tfd = np.array([])
    for i in range(2, int(len(ppg) / fs) - 4 + 1, 2):  # Skip the first and last 2 seconds, add 1 to include the last segment (similar behavior as in matlab)
        relevant_indices = (time_axis >= i) & (time_axis < i + 2)
        avg_frequency = np.mean(freq_axis[max_freq_indices[relevant_indices]])
        hr_smooth_tfd = np.concatenate((hr_smooth_tfd, [60 * avg_frequency]))  # Convert frequency to BPM

    return hr_smooth_tfd