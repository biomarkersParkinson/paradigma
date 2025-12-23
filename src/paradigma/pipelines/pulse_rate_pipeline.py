from typing import List

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.signal.windows import hamming, hann

from paradigma.classification import ClassifierPackage
from paradigma.config import PulseRateConfig
from paradigma.constants import DataColumns
from paradigma.feature_extraction import (
    compute_auto_correlation,
    compute_dominant_frequency,
    compute_relative_power,
    compute_signal_to_noise_ratio,
    compute_spectral_entropy,
    compute_statistics,
)
from paradigma.pipelines.pulse_rate_utils import (
    assign_sqa_label,
    extract_pr_from_segment,
    extract_pr_segments,
)
from paradigma.segmenting import WindowedDataExtractor, tabulate_windows
from paradigma.util import aggregate_parameter


def extract_signal_quality_features(
    df_ppg: pd.DataFrame,
    ppg_config: PulseRateConfig,
    df_acc: pd.DataFrame | None = None,
    acc_config: PulseRateConfig | None = None,
) -> pd.DataFrame:
    """
    Extract signal quality features from the PPG signal.

    Features are computed from both temporal and spectral domains:
      - Temporal: variance, mean, median, kurtosis, skewness, signal-to-noise ratio, autocorrelation
      - Spectral: dominant frequency, relative power, spectral entropy

    Optionally, features from a simultaneously recorded accelerometer signal
    can be included to assess motion artifacts.

    Parameters
    ----------
    df_ppg : pd.DataFrame
        DataFrame containing the PPG signal.
    ppg_config: PulseRateConfig
        Configuration for the signal quality feature extraction of the PPG signal.
    df_acc : pd.DataFrame, optional, default=None
        DataFrame containing accelerometer signals.
    acc_config: PulseRateConfig, optional, default=None
        Configuration for the signal quality feature extraction of the accelerometer signals.

    Returns
    -------
    df_features : pd.DataFrame
        DataFrame containing extracted signal quality features for each window.

    """
    # Group sequences of timestamps into windows
    ppg_windowed_colnames = [ppg_config.time_colname, ppg_config.ppg_colname]
    ppg_windowed = tabulate_windows(
        df=df_ppg,
        columns=ppg_windowed_colnames,
        window_length_s=ppg_config.window_length_s,
        window_step_length_s=ppg_config.window_step_length_s,
        fs=ppg_config.sampling_frequency,
    )

    # Extract data from the windowed PPG signal
    extractor = WindowedDataExtractor(ppg_windowed_colnames)
    idx_time = extractor.get_index(ppg_config.time_colname)
    idx_ppg = extractor.get_index(ppg_config.ppg_colname)
    start_time_ppg = np.min(
        ppg_windowed[:, :, idx_time], axis=1
    )  # Start time of the window is relative to the first datapoint in the PPG data
    ppg_values_windowed = ppg_windowed[:, :, idx_ppg]

    df_features = pd.DataFrame(start_time_ppg, columns=[ppg_config.time_colname])

    if df_acc is not None and acc_config is not None:

        acc_windowed_colnames = [
            acc_config.time_colname
        ] + acc_config.accelerometer_colnames
        acc_windowed = tabulate_windows(
            df=df_acc,
            columns=acc_windowed_colnames,
            window_length_s=acc_config.window_length_s,
            window_step_length_s=acc_config.window_step_length_s,
            fs=acc_config.sampling_frequency,
        )

        # Extract data from the windowed accelerometer signal
        extractor = WindowedDataExtractor(acc_windowed_colnames)
        idx_acc = extractor.get_slice(acc_config.accelerometer_colnames)
        acc_values_windowed = acc_windowed[:, :, idx_acc]

        # Compute periodicity feature of the accelerometer signal
        df_accelerometer_feature = extract_accelerometer_feature(
            acc_values_windowed, ppg_values_windowed, acc_config
        )
        # Combine the accelerometer feature with the previously computed features
        df_features = pd.concat([df_features, df_accelerometer_feature], axis=1)

    # Compute features of the temporal domain of the PPG signal
    df_temporal_features = extract_temporal_domain_features(
        ppg_values_windowed,
        ppg_config,
        quality_stats=["var", "mean", "median", "kurtosis", "skewness"],
    )

    # Combine temporal features with the start time
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # Compute features of the spectral domain of the PPG signal
    df_spectral_features = extract_spectral_domain_features(
        ppg_values_windowed, ppg_config
    )

    # Combine the spectral features with the previously computed temporal features
    df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def signal_quality_classification(
    df: pd.DataFrame, config: PulseRateConfig, clf_package: ClassifierPackage
) -> pd.DataFrame:
    """
    Classify PPG signal quality using a pre-trained logistic regression classifier.

    - A probability close to 1 indicates a high-quality signal, while a probability close to 0
    indicates a low-quality signal.
    - The classifier is trained on features extracted from the PPG signal.
    - The features are extracted using the extract_signal_quality_features function.
    - The accelerometer signal is used to determine the signal quality based on the power
    ratio of the accelerometer signal and returns a binary label based on a threshold.
    - A value of 1 on the indicates no/minor periodic motion influence of the accelerometer
    on the PPG signal, 0 indicates major periodic motion influence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the PPG and accelerometer features.
    config : PulseRateConfig
        Configuration containing column names and thresholds for classification.
    clf_package : ClassifierPackage
        Pre-trained classifier and feature scaler.

    Returns
    -------
    pd.DataFrame
        DataFrame including predicted PPG signal quality probabilities and
        optional accelerometer labels.
    """
    # Set classifier
    clf = clf_package.classifier  # Load the logistic regression classifier

    # Apply scaling to relevant columns
    scaled_features = clf_package.transform_features(
        df.loc[:, clf.feature_names_in]
    )  # Apply scaling to the features

    # Make predictions for PPG signal quality assessment, and assign the probabilities to the DataFrame and drop the features
    df[DataColumns.PRED_SQA_PROBA] = clf.predict_proba(scaled_features)[:, 0]
    keep_cols = [config.time_colname, DataColumns.PRED_SQA_PROBA]

    if DataColumns.ACC_POWER_RATIO in df.columns:
        df[DataColumns.PRED_SQA_ACC_LABEL] = (
            df[DataColumns.ACC_POWER_RATIO] < config.threshold_sqa_accelerometer
        ).astype(
            int
        )  # Assign accelerometer label to the DataFrame based on the threshold
        keep_cols += [DataColumns.PRED_SQA_ACC_LABEL]

    return df[keep_cols]


def estimate_pulse_rate(
    df_sqa: pd.DataFrame, df_ppg_preprocessed: pd.DataFrame, config: PulseRateConfig
) -> pd.DataFrame:
    """
    Estimate pulse rate from the PPG signal using a time-frequency domain approach.

    Pulse rate estimates are computed only for segments marked as high-quality
    by the signal quality assessment. Segment boundaries are extended by 2 seconds
    on each side to improve estimation accuracy.

    Parameters
    ----------
    df_sqa : pd.DataFrame
        DataFrame containing window-level signal quality predictions.
    df_ppg_preprocessed : pd.DataFrame
        Preprocessed PPG signal data.
    config : PulseRateConfig
        Configuration including sampling frequency, time column, PPG column,
        TFD parameters, and kernel settings.

    Returns
    -------
    pd.DataFrame
        DataFrame containing pulse rate estimations.
    """

    # Extract NumPy arrays for faster operations
    ppg_post_prob = df_sqa[DataColumns.PRED_SQA_PROBA].to_numpy()

    if DataColumns.PRED_SQA_ACC_LABEL in df_sqa.columns:
        acc_label = df_sqa[DataColumns.PRED_SQA_ACC_LABEL].to_numpy()
    else:
        acc_label = None

    ppg_preprocessed = df_ppg_preprocessed.values
    time_idx = df_ppg_preprocessed.columns.get_loc(
        config.time_colname
    )  # Get the index of the time column
    ppg_idx = df_ppg_preprocessed.columns.get_loc(
        config.ppg_colname
    )  # Get the index of the PPG column

    # Assign window-level probabilities to individual samples
    sqa_label = assign_sqa_label(
        ppg_post_prob, config, acc_label
    )  # assigns a signal quality label to every individual data point
    v_start_idx, v_end_idx = extract_pr_segments(
        sqa_label, config.min_pr_samples
    )  # extracts pulse rate segments based on the SQA label

    v_pr_rel = np.array([])
    t_pr_rel = np.array([])

    edge_add = (
        2 * config.sampling_frequency
    )  # Add 2s on both sides of the segment for PR estimation
    step_size = config.pr_est_samples  # Step size for PR estimation

    # Estimate the maximum size for preallocation
    valid_segments = (v_start_idx >= edge_add) & (
        v_end_idx <= len(ppg_preprocessed) - edge_add
    )  # check if the segments are valid, e.g. not too close to the edges (2s)
    valid_start_idx = v_start_idx[valid_segments]  # get the valid start indices
    valid_end_idx = v_end_idx[valid_segments]  # get the valid end indices
    max_size = np.sum(
        (valid_end_idx - valid_start_idx) // step_size
    )  # maximum size for preallocation

    # Preallocate arrays
    v_pr_rel = np.empty(max_size, dtype=float)
    t_pr_rel = np.empty(max_size, dtype=float)

    # Track current position
    pr_pos = 0

    for start_idx, end_idx in zip(valid_start_idx, valid_end_idx):
        # Extract extended PPG segment
        extended_ppg_segment = ppg_preprocessed[
            start_idx - edge_add : end_idx + edge_add, ppg_idx
        ]

        # Estimate pulse rate
        pr_est = extract_pr_from_segment(
            extended_ppg_segment,
            config.tfd_length,
            config.sampling_frequency,
            config.kern_type,
            config.kern_params,
        )
        n_pr = len(pr_est)  # Number of pulse rate estimates
        end_idx_time = (
            n_pr * step_size + start_idx
        )  # Calculate end index for time, different from end_idx since it is always a multiple of step_size, while end_idx is not

        # Extract relative time for PR estimates
        pr_time = ppg_preprocessed[start_idx:end_idx_time:step_size, time_idx]

        # Insert into preallocated arrays
        v_pr_rel[pr_pos : pr_pos + n_pr] = pr_est
        t_pr_rel[pr_pos : pr_pos + n_pr] = pr_time
        pr_pos += n_pr

    df_pr = pd.DataFrame({"time": t_pr_rel, "pulse_rate": v_pr_rel})

    return df_pr


def aggregate_pulse_rate(
    pr_values: np.ndarray, aggregates: List[str] = ["mode", "99p"]
) -> dict:
    """
    Aggregate pulse rate estimates using specified aggregation methods.

    Parameters
    ----------
    pr_values : np.ndarray
        Array containing pulse rate estimates
    aggregates : List[str], default=['mode', '99p']
        List of aggregation methods to apply.

    Returns
    -------
    dict
        Dictionary with:
        - "metadata": number of pulse rate estimates
        - "pr_aggregates": aggregated results per method
    """
    # Initialize the dictionary for the aggregated results
    aggregated_results = {}

    # Initialize the dictionary for the aggregated results with the metadata
    aggregated_results = {
        "metadata": {"nr_pr_est": len(pr_values)},
        "pr_aggregates": {},
    }
    for aggregate in aggregates:
        aggregated_results["pr_aggregates"][f"{aggregate}_{DataColumns.PULSE_RATE}"] = (
            aggregate_parameter(pr_values, aggregate)
        )

    return aggregated_results


def extract_temporal_domain_features(
    ppg_windowed: np.ndarray,
    config: PulseRateConfig,
    quality_stats: List[str] = ["mean", "std"],
) -> pd.DataFrame:
    """
    Compute temporal domain features for windowed PPG data.

    Parameters
    ----------
    ppg_windowed: np.ndarray
        Windowed PPG signal.
    config: PulseRateConfig
        Configuration containing sampling frequency and other parameters.
    quality_stats: List[str], optional, default=['mean', 'std']
        Statistics to compute for the gravity component of the accelerometer signal.

    Returns
    -------
    pd.DataFrame
        DataFrame with temporal domain features for each window.

    Notes
    -----
    - The features are added to the dataframe. Therefore the original dataframe is modified,
    and the modified dataframe is returned.
    """

    feature_dict = {}
    for stat in quality_stats:
        feature_dict[stat] = compute_statistics(ppg_windowed, stat, abs_stats=True)

    feature_dict["signal_to_noise"] = compute_signal_to_noise_ratio(ppg_windowed)
    feature_dict["auto_corr"] = compute_auto_correlation(
        ppg_windowed, config.sampling_frequency
    )
    return pd.DataFrame(feature_dict)


def extract_spectral_domain_features(
    ppg_windowed: np.ndarray,
    config: PulseRateConfig,
) -> pd.DataFrame:
    """
    Compute spectral domain features for windowed PPG data.

    Features include dominant frequency, relative power, and spectral entropy
    computed using Welch's method.

    Parameters
    ----------
    ppg_windowed: np.ndarray
        Windowed PPG signal.
    config: PulseRateConfig
        Configuration including sampling frequency, window settings, and FFT parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with spectral domain features for each window.
    """
    d_features = {}

    window = hamming(config.window_length_welch, sym=True)

    n_samples_window = ppg_windowed.shape[1]

    freqs, psd = welch(
        ppg_windowed,
        fs=config.sampling_frequency,
        window=window,
        noverlap=config.overlap_welch_window,
        nfft=max(256, 2 ** int(np.log2(n_samples_window))),
        detrend=False,
        axis=1,
    )

    # Calculate each feature using the computed PSD and frequency array
    d_features["f_dom"] = compute_dominant_frequency(freqs, psd)
    d_features["rel_power"] = compute_relative_power(freqs, psd, config)
    d_features["spectral_entropy"] = compute_spectral_entropy(psd, n_samples_window)

    return pd.DataFrame(d_features)


def extract_acc_power_feature(
    f1: np.ndarray, PSD_acc: np.ndarray, f2: np.ndarray, PSD_ppg: np.ndarray
) -> np.ndarray:
    """
    Compute accelerometer power in the PPG frequency range.

    Parameters
    ----------
    f1: np.ndarray
        Frequency bins of the accelerometer signal.
    PSD_acc: np.ndarray
        Power spectral density of the accelerometer signal.
    f2: np.ndarray
        Frequency bins of the PPG signal.
    PSD_ppg: np.ndarray
        Power spectral density of the PPG signal.

    Returns
    -------
    np.ndarray
        Accelerometer power feature in the PPG frequency range.
    """

    # Find the index of the maximum PSD value in the PPG signal
    max_PPG_psd_idx = np.argmax(PSD_ppg, axis=1)
    max_PPG_freq_psd = f2[max_PPG_psd_idx]

    # Find the neighboring indices of the maximum PSD value in the PPG signal
    df_idx = np.column_stack(
        (max_PPG_psd_idx - 1, max_PPG_psd_idx, max_PPG_psd_idx + 1)
    )

    # Find the index of the closest frequency in the accelerometer signal to the first harmonic of the PPG frequency
    corr_acc_psd_fh_idx = np.argmin(np.abs(f1[:, None] - max_PPG_freq_psd * 2), axis=0)
    fh_idx = np.column_stack(
        (corr_acc_psd_fh_idx - 1, corr_acc_psd_fh_idx, corr_acc_psd_fh_idx + 1)
    )

    # Compute the power in the ranges corresponding to the PPG frequency
    acc_power_PPG_range = np.trapz(
        PSD_acc[np.arange(PSD_acc.shape[0])[:, None], df_idx], f1[df_idx], axis=1
    ) + np.trapz(
        PSD_acc[np.arange(PSD_acc.shape[0])[:, None], fh_idx], f1[fh_idx], axis=1
    )

    # Compute the total power across the entire frequency range
    acc_power_total = np.trapz(PSD_acc, f1)

    # Compute the power ratio of the accelerometer signal in the PPG frequency range
    acc_power_ratio = acc_power_PPG_range / acc_power_total

    return acc_power_ratio


def extract_accelerometer_feature(
    acc_windowed: np.ndarray, ppg_windowed: np.ndarray, config: PulseRateConfig
) -> pd.DataFrame:
    """
    Extract accelerometer features from the accelerometer signal in the PPG frequency range.

    Parameters
    ----------
    acc_windowed: np.ndarray
        Windowed accelerometer signal.
    ppg_windowed: np.ndarray
        Windowed PPG signal corresponding to the accelerometer windows.
    config: PulseRateConfig
        Configuration containing sampling frequency, sensor type, and Welch parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with the relative power accelerometer feature.
    """

    if config.sensor not in ["imu", "ppg"]:
        raise ValueError("Sensor not recognized.")

    d_freq = {}
    d_psd = {}
    for sensor in ["imu", "ppg"]:
        config.set_sensor(sensor)

        if sensor == "imu":
            windows = acc_windowed
        else:
            windows = ppg_windowed

        window_type = hann(config.window_length_welch, sym=True)
        d_freq[sensor], d_psd[sensor] = welch(
            windows,
            fs=config.sampling_frequency,
            window=window_type,
            noverlap=config.overlap_welch_window,
            nfft=config.nfft,
            detrend=False,
            axis=1,
        )

    d_psd["imu"] = np.sum(d_psd["imu"], axis=2)  # Sum the PSDs of the three axes

    acc_power_ratio = extract_acc_power_feature(
        d_freq["imu"], d_psd["imu"], d_freq["ppg"], d_psd["ppg"]
    )

    return pd.DataFrame(acc_power_ratio, columns=["acc_power_ratio"])
