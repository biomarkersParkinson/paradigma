import json
import logging
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.signal.windows import hamming, hann

from paradigma.classification import ClassifierPackage
from paradigma.config import PPGConfig, PulseRateConfig
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
from paradigma.preprocessing import preprocess_ppg_data
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
    The features are extracted from the temporal and spectral domain of the
    PPG signal. The temporal domain features include variance, mean, median,
    kurtosis, skewness, signal-to-noise ratio, and autocorrelation. The
    spectral domain features include the dominant frequency, relative power,
    spectral entropy.

    Parameters
    ----------
    df_ppg : pd.DataFrame
        The DataFrame containing the PPG signal.
    df_acc : pd.DataFrame
        The DataFrame containing the accelerometer signal.
    ppg_config: PulseRateConfig
        The configuration for the signal quality feature extraction of the PPG
        signal.
    acc_config: PulseRateConfig
        The configuration for the signal quality feature extraction of the
        accelerometer signal.

    Returns
    -------
    df_features : pd.DataFrame
        The DataFrame containing the extracted signal quality features.

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
    # Start time of the window is relative to the first datapoint in the PPG
    # data
    start_time_ppg = np.min(ppg_windowed[:, :, idx_time], axis=1)
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
    Classify the signal quality of the PPG signal using a logistic regression
    classifier. A probability close to 1 indicates a high-quality signal,
    while a probability close to 0 indicates a low-quality signal. The
    classifier is trained on features extracted from the PPG signal. The
    features are extracted using the extract_signal_quality_features
    function. The accelerometer signal is used to determine the signal
    quality based on the power ratio of the accelerometer signal and returns
    a binary label based on a threshold. A value of 1 on the indicates
    no/minor periodic motion influence of the accelerometer on the PPG
    signal, 0 indicates major periodic motion influence.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the PPG features and the accelerometer
        feature for signal quality classification.
    config : PulseRateConfig
        The configuration for the signal quality classification.
    clf_package : ClassifierPackage
        The classifier package containing the classifier and scaler.

    Returns
    -------
    df_sqa pd.DataFrame
        The DataFrame containing the PPG signal quality predictions (both
        probabilities of the PPG signal quality classification and the
        accelerometer label based on the threshold).
    """
    # Set classifier
    clf = clf_package.classifier  # Load the logistic regression classifier

    # Apply scaling to relevant columns
    scaled_features = clf_package.transform_features(
        df.loc[:, clf.feature_names_in]
    )  # Apply scaling to the features

    # Make predictions for PPG signal quality assessment, and assign the
    # probabilities to the DataFrame and drop the features
    df[DataColumns.PRED_SQA_PROBA] = clf.predict_proba(scaled_features)[:, 0]
    keep_cols = [config.time_colname, DataColumns.PRED_SQA_PROBA]

    if DataColumns.ACC_POWER_RATIO in df.columns:
        # Assign accelerometer label to the DataFrame based on the threshold
        df[DataColumns.PRED_SQA_ACC_LABEL] = (
            df[DataColumns.ACC_POWER_RATIO] < config.threshold_sqa_accelerometer
        ).astype(int)
        keep_cols += [DataColumns.PRED_SQA_ACC_LABEL]

    return df[keep_cols]


def estimate_pulse_rate(
    df_sqa: pd.DataFrame, df_ppg_preprocessed: pd.DataFrame, config: PulseRateConfig
) -> pd.DataFrame:
    """
    Estimate the pulse rate from the PPG signal using the time-frequency domain method.

    Parameters
    ----------
    df_sqa : pd.DataFrame
        The DataFrame containing the signal quality assessment predictions.
    df_ppg_preprocessed : pd.DataFrame
        The DataFrame containing the preprocessed PPG signal.
    config : PulseRateConfig
        The configuration for the pulse rate estimation.

    Returns
    -------
    df_pr : pd.DataFrame
        The DataFrame containing the pulse rate estimations.
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
        # Calculate end index for time, different from end_idx since it is
        # always a multiple of step_size, while end_idx is not
        end_idx_time = n_pr * step_size + start_idx

        # Extract relative time for PR estimates
        pr_time = ppg_preprocessed[start_idx:end_idx_time:step_size, time_idx]

        # Insert into preallocated arrays
        v_pr_rel[pr_pos : pr_pos + n_pr] = pr_est
        t_pr_rel[pr_pos : pr_pos + n_pr] = pr_time
        pr_pos += n_pr

    df_pr = pd.DataFrame({"time": t_pr_rel, "pulse_rate": v_pr_rel})

    return df_pr


def aggregate_pulse_rate(
    pr_values: np.ndarray, aggregates: list[str] = ["mode", "99p"]
) -> dict:
    """
    Aggregate the pulse rate estimates using the specified aggregation methods.

    Parameters
    ----------
    pr_values : np.ndarray
        The array containing the pulse rate estimates
    aggregates : List[str]
        The list of aggregation methods to be used for the pulse rate
        estimates. The default is ['mode', '99p'].

    Returns
    -------
    aggregated_results : dict
        The dictionary containing the aggregated results of the pulse rate estimates.
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
    quality_stats: list[str] = ["mean", "std"],
) -> pd.DataFrame:
    """
    Compute temporal domain features for the ppg signal. The features are
    added to the dataframe. Therefore the original dataframe is modified,
    and the modified dataframe is returned.

    Parameters
    ----------
    ppg_windowed: np.ndarray
        The dataframe containing the windowed accelerometer signal

    config: PulseRateConfig
        The configuration object containing the parameters for the feature extraction

    quality_stats: list, optional
        The statistics to be computed for the gravity component of the
        accelerometer signal (default: ['mean', 'std'])

    Returns
    -------
    pd.DataFrame
        The dataframe with the added temporal domain features.
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
    Calculate the spectral features (dominant frequency, relative power, and
    spectral entropy) for each segment of a PPG signal using a single
    Welch's method computation. The features are added to the dataframe.
    Therefore the original dataframe is modified, and the modified dataframe
    is returned.

    Parameters
    ----------
    ppg_windowed: np.ndarray
        The dataframe containing the windowed ppg signal

    config: PulseRateConfig
        The configuration object containing the parameters for the feature extraction

    Returns
    -------
    pd.DataFrame
        The dataframe with the added spectral domain features.
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
    f1: np.ndarray,
    psd_acc: np.ndarray,
    f2: np.ndarray,
    psd_ppg: np.ndarray,
) -> np.ndarray:
    """
    Extract the accelerometer power feature in the PPG frequency range.

    Parameters
    ----------
    f1: np.ndarray
        The frequency bins of the accelerometer signal.
    psd_acc: np.ndarray
        The power spectral density of the accelerometer signal.
    f2: np.ndarray
        The frequency bins of the PPG signal.
    psd_ppg: np.ndarray
        The power spectral density of the PPG signal.

    Returns
    -------
    np.ndarray
        The accelerometer power feature in the PPG frequency range
    """

    # Find the index of the maximum PSD value in the PPG signal
    max_ppg_psd_idx = np.argmax(psd_ppg, axis=1)
    max_ppg_freq_psd = f2[max_ppg_psd_idx]

    # Find the neighboring indices of the maximum PSD value in the PPG signal
    df_idx = np.column_stack(
        (max_ppg_psd_idx - 1, max_ppg_psd_idx, max_ppg_psd_idx + 1)
    )

    # Find the index of the closest frequency in the accelerometer signal
    # to the first harmonic of the PPG frequency
    corr_acc_psd_fh_idx = np.argmin(np.abs(f1[:, None] - max_ppg_freq_psd * 2), axis=0)
    fh_idx = np.column_stack(
        (corr_acc_psd_fh_idx - 1, corr_acc_psd_fh_idx, corr_acc_psd_fh_idx + 1)
    )

    # Compute the power in the ranges corresponding to the PPG frequency
    acc_power_ppg_range = np.trapezoid(
        psd_acc[np.arange(psd_acc.shape[0])[:, None], df_idx], f1[df_idx], axis=1
    ) + np.trapezoid(
        psd_acc[np.arange(psd_acc.shape[0])[:, None], fh_idx], f1[fh_idx], axis=1
    )

    # Compute the total power across the entire frequency range
    acc_power_total = np.trapezoid(psd_acc, f1)

    # Compute the power ratio of the accelerometer signal in the PPG frequency range
    acc_power_ratio = acc_power_ppg_range / acc_power_total

    return acc_power_ratio


def extract_accelerometer_feature(
    acc_windowed: np.ndarray, ppg_windowed: np.ndarray, config: PulseRateConfig
) -> pd.DataFrame:
    """
    Extract accelerometer features from the accelerometer signal in the PPG
    frequency range.

    Parameters
    ----------
    acc_windowed: np.ndarray
        The dataframe containing the windowed accelerometer signal

    ppg_windowed: np.ndarray
        The dataframe containing the corresponding windowed ppg signal

    config: PulseRateConfig
        The configuration object containing the parameters for the feature extraction

    Returns
    -------
    pd.DataFrame
        The dataframe with the relative power accelerometer feature.
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


def run_pulse_rate_pipeline(
    df_ppg_prepared: pd.DataFrame,
    output_dir: str | Path,
    store_intermediate: list[str] = [],
    pulse_rate_config: PulseRateConfig | None = None,
    ppg_config: PPGConfig | None = None,
    logging_level: int = logging.INFO,
    custom_logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    High-level pulse rate analysis pipeline for a single segment.

    This function implements the complete pulse rate analysis workflow from the
    pulse rate tutorial:
    1. Preprocess PPG and accelerometer data (accelerometer is optional)
    2. Extract signal quality features
    3. Signal quality classification
    4. Pulse rate estimation
    5. Quantify pulse rate (select relevant columns)

    Parameters
    ----------
    df_ppg_prepared : pd.DataFrame
        Prepared sensor data with time and PPG column.
    output_dir : str or Path
        Output directory for intermediate results (required)
    store_intermediate : list of str, default []
        Which intermediate results to store.
    pulse_rate_config : PulseRateConfig, optional
        Pulse rate analysis configuration
    ppg_config : PPGConfig, optional
        PPG preprocessing configuration
    logging_level : int, default logging.INFO
        Logging level using standard logging constants
    custom_logger : logging.Logger, optional
        Custom logger instance

    Returns
    -------
    pd.DataFrame
        Quantified pulse rate data with columns:
        - time: timestamp
        - pulse_rate: pulse rate estimate
        - signal_quality: quality assessment (if available)
    """
    # Setup logger
    active_logger = (
        custom_logger if custom_logger is not None else logging.getLogger(__name__)
    )
    if custom_logger is None:
        active_logger.setLevel(logging_level)

    if pulse_rate_config is None:
        pulse_rate_config = PulseRateConfig()
    if ppg_config is None:
        ppg_config = PPGConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input data columns (PPG is required, accelerometer is optional)
    required_columns = [DataColumns.TIME, DataColumns.PPG]
    missing_columns = [
        col for col in required_columns if col not in df_ppg_prepared.columns
    ]
    if missing_columns:
        active_logger.warning(
            f"Missing required columns for pulse rate pipeline: {missing_columns}"
        )
        return pd.DataFrame()

    # Step 1: Preprocess PPG and accelerometer data (following tutorial)
    active_logger.info("Step 1: Preprocessing PPG and accelerometer data")
    try:
        # Separate PPG data (always available)
        ppg_cols = [DataColumns.TIME, DataColumns.PPG]
        df_ppg = df_ppg_prepared[ppg_cols].copy()

        # Preprocess the data
        df_ppg_proc, _ = preprocess_ppg_data(
            df_ppg=df_ppg,
            ppg_config=ppg_config,
            verbose=1 if logging_level <= logging.INFO else 0,
        )

        if "preprocessing" in store_intermediate:
            preprocessing_dir = output_dir / "preprocessing"
            preprocessing_dir.mkdir(exist_ok=True)
            df_ppg_proc.to_parquet(preprocessing_dir / "ppg_preprocessed.parquet")
            active_logger.info(f"Saved preprocessed data to {preprocessing_dir}")

    except Exception as e:
        active_logger.error(f"Preprocessing failed: {e}")
        return pd.DataFrame()

    # Step 2: Extract signal quality features
    active_logger.info("Step 2: Extracting signal quality features")
    try:
        df_features = extract_signal_quality_features(df_ppg_proc, pulse_rate_config)

        if "pulse_rate" in store_intermediate:
            pulse_rate_dir = output_dir / "pulse_rate"
            pulse_rate_dir.mkdir(exist_ok=True)
            df_features.to_parquet(pulse_rate_dir / "signal_quality_features.parquet")
            active_logger.info(f"Saved signal quality features to {pulse_rate_dir}")

    except Exception as e:
        active_logger.error(f"Feature extraction failed: {e}")
        return pd.DataFrame()

    # Step 3: Signal quality classification
    active_logger.info("Step 3: Signal quality classification")
    try:
        classifier_path = files("paradigma.assets") / "ppg_quality_clf_package.pkl"
        classifier_package = ClassifierPackage.load(classifier_path)

        df_classified = signal_quality_classification(
            df_features, pulse_rate_config, classifier_package
        )

    except Exception as e:
        active_logger.error(f"Signal quality classification failed: {e}")
        return pd.DataFrame()

    # Step 4: Pulse rate estimation
    active_logger.info("Step 4: Pulse rate estimation")
    try:
        df_pulse_rates = estimate_pulse_rate(
            df_sqa=df_classified,
            df_ppg_preprocessed=df_ppg_proc,
            config=pulse_rate_config,
        )

    except Exception as e:
        active_logger.error(f"Pulse rate estimation failed: {e}")
        return pd.DataFrame()

    # Step 5: Quantify pulse rate (select relevant columns and apply quality filtering)
    active_logger.info("Step 5: Quantifying pulse rate")

    # Select quantification columns
    quantification_columns = []
    if DataColumns.TIME in df_pulse_rates.columns:
        quantification_columns.append(DataColumns.TIME)
    if DataColumns.PULSE_RATE in df_pulse_rates.columns:
        quantification_columns.append(DataColumns.PULSE_RATE)
    if "signal_quality" in df_pulse_rates.columns:
        quantification_columns.append("signal_quality")

    # Use available columns
    available_columns = [
        col for col in quantification_columns if col in df_pulse_rates.columns
    ]
    if not available_columns:
        active_logger.warning("No valid quantification columns found")
        return pd.DataFrame()

    df_quantification = df_pulse_rates[available_columns].copy()

    # Apply quality filtering if signal quality is available
    if (
        "signal_quality" in df_quantification.columns
        and DataColumns.PULSE_RATE in df_quantification.columns
    ):
        quality_threshold = getattr(pulse_rate_config, "threshold_sqa", 0.5)
        low_quality_mask = df_quantification["signal_quality"] < quality_threshold
        df_quantification.loc[low_quality_mask, DataColumns.PULSE_RATE] = np.nan

    if "quantification" in store_intermediate:
        quantification_dir = output_dir / "quantification"
        quantification_dir.mkdir(exist_ok=True)
        df_quantification.to_parquet(
            quantification_dir / "pulse_rate_quantification.parquet"
        )

        # Save quantification metadata
        valid_pulse_rates = (
            df_quantification[DataColumns.PULSE_RATE].dropna()
            if DataColumns.PULSE_RATE in df_quantification.columns
            else pd.Series(dtype=float)
        )
        quantification_meta = {
            "total_windows": len(df_quantification),
            "valid_pulse_rate_estimates": len(valid_pulse_rates),
            "columns": list(df_quantification.columns),
        }
        with open(quantification_dir / "pulse_rate_quantification_meta.json", "w") as f:
            json.dump(quantification_meta, f, indent=2)

        active_logger.info(f"Saved pulse rate quantification to {quantification_dir}")

    pulse_rate_estimates = (
        len(df_quantification[DataColumns.PULSE_RATE].dropna())
        if DataColumns.PULSE_RATE in df_quantification.columns
        else 0
    )
    active_logger.info(
        f"Pulse rate analysis completed: {pulse_rate_estimates} valid pulse "
        f"rate estimates from {len(df_quantification)} total windows"
    )

    return df_quantification
