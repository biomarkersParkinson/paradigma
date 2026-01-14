import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import signal

from paradigma.classification import ClassifierPackage
from paradigma.config import IMUConfig, TremorConfig
from paradigma.constants import DataColumns
from paradigma.feature_extraction import (
    compute_mfccs,
    compute_power_in_bandwidth,
    compute_total_power,
    extract_frequency_peak,
    extract_tremor_power,
)
from paradigma.preprocessing import preprocess_imu_data
from paradigma.segmenting import WindowedDataExtractor, tabulate_windows
from paradigma.util import aggregate_parameter


def extract_tremor_features(df: pd.DataFrame, config: TremorConfig) -> pd.DataFrame:
    """
    This function groups sequences of timestamps into windows and subsequently extracts
    tremor features from windowed gyroscope data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing sensor data, which includes time and gyroscope data. The data should be
        structured with the necessary columns as specified in the `config`.

    config : TremorConfig
        Configuration object containing parameters for feature extraction, including column names for time, gyroscope data,
        as well as settings for windowing, and feature computation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing extracted tremor features and a column corresponding to time.

    Notes
    -----
    - This function groups the data into windows based on timestamps.
    - The input DataFrame must include columns as specified in the `config` object for proper feature extraction.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the required columns as specified in the configuration or if any step in the feature extraction fails.
    """
    # group sequences of timestamps into windows
    windowed_colnames = [config.time_colname] + config.gyroscope_colnames
    windowed_data = tabulate_windows(
        df,
        windowed_colnames,
        config.window_length_s,
        config.window_step_length_s,
        config.sampling_frequency,
    )

    extractor = WindowedDataExtractor(windowed_colnames)

    # Extract the start time and gyroscope data from the windowed data
    idx_time = extractor.get_index(config.time_colname)
    idx_gyro = extractor.get_slice(config.gyroscope_colnames)

    # Extract data
    start_time = np.min(windowed_data[:, :, idx_time], axis=1)
    windowed_gyro = windowed_data[:, :, idx_gyro]

    df_features = pd.DataFrame(start_time, columns=[config.time_colname])

    # transform the signals from the temporal domain to the spectral domain and extract tremor features
    df_spectral_features = extract_spectral_domain_features(windowed_gyro, config)

    # Combine spectral features with the start time
    df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def detect_tremor(
    df: pd.DataFrame, config: TremorConfig, full_path_to_classifier_package: str | Path
) -> pd.DataFrame:
    """
    Detects tremor in the input DataFrame using a pre-trained classifier and applies a threshold to the predicted probabilities.

    This function performs the following steps:
    1. Loads the pre-trained classifier and scaling parameters from the provided directory.
    2. Scales the relevant features in the input DataFrame (`df`) using the loaded scaling parameters.
    3. Makes predictions using the classifier to estimate the probability of tremor.
    4. Applies a threshold to the predicted probabilities to classify whether tremor is detected or not.
    5. Checks for rest tremor by verifying the frequency of the peak and below tremor power.
    6. Adds the predicted probabilities and the classification result to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing extracted tremor features. The DataFrame must include
        the necessary columns as specified in the classifier's feature names.

    config : TremorConfig
        Configuration object containing settings for tremor detection, including the frequency range for rest tremor.

    full_path_to_classifier_package : str | Path
        The path to the directory containing the classifier file, threshold value, scaler parameters, and other necessary input
        files for tremor detection.

    Returns
    -------
    pd.DataFrame
        The input DataFrame (`df`) with two additional columns:
        - `PRED_TREMOR_PROBA`: Predicted probability of tremor based on the classifier.
        - `PRED_TREMOR_LOGREG`: Binary classification result (True for tremor, False for no tremor), based on the threshold applied to `PRED_TREMOR_PROBA`.
        - `PRED_TREMOR_CHECKED`: Binary classification result (True for tremor, False for no tremor), after performing extra checks for rest tremor on `PRED_TREMOR_LOGREG`.
        - `PRED_ARM_AT_REST`: Binary classification result (True for arm at rest or stable posture, False for significant arm movement), based on the power below tremor.

    Notes
    -----
    - The threshold used to classify tremor is loaded from a file and applied to the predicted probabilities.

    Raises
    ------
    FileNotFoundError
        If the classifier, scaler, or threshold files are not found at the specified paths.
    ValueError
        If the DataFrame does not contain the expected features for prediction or if the prediction fails.

    """

    # Load the classifier package
    clf_package = ClassifierPackage.load(full_path_to_classifier_package)

    # Set classifier
    clf = clf_package.classifier
    feature_names_scaling = clf_package.scaler.feature_names_in_
    feature_names_predictions = clf.feature_names_in_

    # Apply scaling to relevant columns
    scaled_features = clf_package.transform_features(df.loc[:, feature_names_scaling])

    # Replace scaled features in a copy of the relevant features for prediction
    X = df.loc[:, feature_names_predictions].copy()
    X.loc[:, feature_names_scaling] = scaled_features

    # Get the tremor probability
    df[DataColumns.PRED_TREMOR_PROBA] = clf_package.predict_proba(X)

    # Make prediction based on pre-defined threshold
    df[DataColumns.PRED_TREMOR_LOGREG] = (
        df[DataColumns.PRED_TREMOR_PROBA] >= clf_package.threshold
    ).astype(int)

    # Perform extra checks for rest tremor
    peak_check = (df[DataColumns.FREQ_PEAK] >= config.fmin_rest_tremor) & (
        df[DataColumns.FREQ_PEAK] <= config.fmax_rest_tremor
    )  # peak within 3-7 Hz
    df[DataColumns.PRED_ARM_AT_REST] = (
        df[DataColumns.BELOW_TREMOR_POWER] <= config.movement_threshold
    ).astype(
        int
    )  # arm at rest or in stable posture
    df[DataColumns.PRED_TREMOR_CHECKED] = (
        (df[DataColumns.PRED_TREMOR_LOGREG] == 1)
        & peak_check
        & df[DataColumns.PRED_ARM_AT_REST]
    ).astype(int)

    return df


def aggregate_tremor(df: pd.DataFrame, config: TremorConfig):
    """
    Quantifies the amount of tremor time and tremor power, aggregated over all windows in the input dataframe.
    Tremor time is calculated as the number of the detected tremor windows, as percentage of the number of windows
    without significant non-tremor movement (at rest). For tremor power the following aggregates are derived:
    the median, mode and percentile of tremor power specified in the configuration object.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the tremor predictions and computed tremor power.
        The DataFrame must also contain a datatime column ('time_dt').

    config : TremorConfig
        Configuration object containing the percentile for aggregating tremor power.

    Returns
    -------
    dict
        A dictionary with the aggregated tremor time and tremor power measures, as well as the number of valid days,
        the total number of windows, and the number of windows at rest available in the input dataframe.

    Notes
    -----
    - Tremor power is converted to log scale, after adding a constant of 1, so that zero tremor power
    corresponds to a value of 0 in log scale.
    - The modal tremor power is computed based on gaussian kernel density estimation.

    """
    nr_valid_days = (
        df["time_dt"].dt.date.unique().size
    )  # number of valid days in the input dataframe
    nr_windows_total = df.shape[0]  # number of windows in the input dataframe

    # remove windows with detected non-tremor arm movements to control for the amount of arm activities performed
    df_filtered = df.loc[df.pred_arm_at_rest == 1]
    nr_windows_rest = df_filtered.shape[
        0
    ]  # number of windows without non-tremor arm movement

    if (
        nr_windows_rest == 0
    ):  # if no windows without non-tremor arm movement are detected
        raise Warning("No windows without non-tremor arm movement are detected.")

    # calculate tremor time
    n_windows_tremor = np.sum(df_filtered[DataColumns.PRED_TREMOR_CHECKED])
    perc_windows_tremor = (
        n_windows_tremor / nr_windows_rest * 100
    )  # as percentage of total measured time without non-tremor arm movement

    aggregated_tremor_power = (
        {}
    )  # initialize dictionary to store aggregated tremor power measures

    if (
        n_windows_tremor == 0
    ):  # if no tremor is detected, the tremor power measures are set to NaN
        aggregated_tremor_power["median_tremor_power"] = np.nan
        aggregated_tremor_power["mode_binned_tremor_power"] = np.nan
        aggregated_tremor_power["90p_tremor_power"] = np.nan

    else:

        # calculate aggregated tremor power measures
        tremor_power = df_filtered.loc[
            df_filtered[DataColumns.PRED_TREMOR_CHECKED] == 1, DataColumns.TREMOR_POWER
        ]
        tremor_power = np.log10(tremor_power + 1)  # convert to log scale

        for aggregate in config.aggregates_tremor_power:
            aggregate_name = f"{aggregate}_tremor_power"
            aggregated_tremor_power[aggregate_name] = aggregate_parameter(
                tremor_power, aggregate, config.evaluation_points_tremor_power
            )

    # store aggregates in json format
    d_aggregates = {
        "metadata": {
            "nr_valid_days": nr_valid_days,
            "nr_windows_total": nr_windows_total,
            "nr_windows_rest": nr_windows_rest,
        },
        "aggregated_tremor_measures": {
            "perc_windows_tremor": perc_windows_tremor,
            "median_tremor_power": aggregated_tremor_power["median_tremor_power"],
            "modal_tremor_power": aggregated_tremor_power["mode_binned_tremor_power"],
            "90p_tremor_power": aggregated_tremor_power["90p_tremor_power"],
        },
    }

    return d_aggregates


def extract_spectral_domain_features(data: np.ndarray, config) -> pd.DataFrame:
    """
    Compute spectral domain features from the gyroscope data.

    This function computes Mel-frequency cepstral coefficients (MFCCs), the frequency of the peak,
    the tremor power, and the below tremor power based on the total power spectral density of the windowed gyroscope data.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array where each row corresponds to a window of gyroscope data.
    config : object
        Configuration object containing settings such as sampling frequency, window type,
        and MFCC parameters.

    Returns
    -------
    pd.DataFrame
        The feature dataframe containing the extracted spectral features, including
        MFCCs, the frequency of the peak, the tremor power and below tremor power for each window.

    """

    # Initialize a dictionary to hold the results
    feature_dict = {}

    # Initialize parameters
    sampling_frequency = config.sampling_frequency
    segment_length_psd_s = config.segment_length_psd_s
    segment_length_spectrogram_s = config.segment_length_spectrogram_s
    overlap_fraction = config.overlap_fraction
    spectral_resolution = config.spectral_resolution
    window_type = "hann"

    # Compute the power spectral density
    segment_length_n = sampling_frequency * segment_length_psd_s
    overlap_n = segment_length_n * overlap_fraction
    window = signal.get_window(window_type, segment_length_n, fftbins=False)
    nfft = sampling_frequency / spectral_resolution

    freqs, psd = signal.welch(
        x=data,
        fs=sampling_frequency,
        window=window,
        nperseg=segment_length_n,
        noverlap=overlap_n,
        nfft=nfft,
        detrend=False,
        scaling="density",
        axis=1,
    )

    # Compute the spectrogram
    segment_length_n = sampling_frequency * segment_length_spectrogram_s
    overlap_n = segment_length_n * overlap_fraction
    window = signal.get_window(window_type, segment_length_n)

    f, t, S1 = signal.stft(
        x=data,
        fs=sampling_frequency,
        window=window,
        nperseg=segment_length_n,
        noverlap=overlap_n,
        boundary=None,
        axis=1,
    )

    # Compute total power in the PSD and the total spectrogram (summed over the three axes)
    total_psd = compute_total_power(psd)
    total_spectrogram = np.sum(np.abs(S1) * sampling_frequency, axis=2)

    # Compute the MFCC's
    config.mfcc_low_frequency = config.fmin_mfcc
    config.mfcc_high_frequency = config.fmax_mfcc
    config.mfcc_n_dct_filters = config.n_dct_filters_mfcc
    config.mfcc_n_coefficients = config.n_coefficients_mfcc

    mfccs = compute_mfccs(
        total_power_array=total_spectrogram,
        config=config,
        total_power_type="spectrogram",
        rounding_method="round",
        multiplication_factor=1,
    )

    # Combine the MFCCs into the features DataFrame
    mfcc_colnames = [f"mfcc_{x}" for x in range(1, config.mfcc_n_coefficients + 1)]
    for i, colname in enumerate(mfcc_colnames):
        feature_dict[colname] = mfccs[:, i]

    # Compute the frequency of the peak, non-tremor power and tremor power
    feature_dict[DataColumns.FREQ_PEAK] = extract_frequency_peak(
        freqs, total_psd, config.fmin_peak_search, config.fmax_peak_search
    )
    feature_dict[DataColumns.BELOW_TREMOR_POWER] = compute_power_in_bandwidth(
        freqs,
        total_psd,
        config.fmin_below_rest_tremor,
        config.fmax_below_rest_tremor,
        include_max=False,
        spectral_resolution=config.spectral_resolution,
        cumulative_sum_method="sum",
    )
    feature_dict[DataColumns.TREMOR_POWER] = extract_tremor_power(
        freqs, total_psd, config.fmin_rest_tremor, config.fmax_rest_tremor
    )

    return pd.DataFrame(feature_dict)


def run_tremor_pipeline(
    df_prepared: pd.DataFrame,
    output_dir: str | Path,
    store_intermediate: List[str] = [],
    tremor_config: TremorConfig | None = None,
    imu_config: IMUConfig | None = None,
    verbosity: int = 1,
) -> pd.DataFrame:
    """
    High-level tremor analysis pipeline for a single segment.

    This function implements the complete tremor analysis workflow from the
    tremor tutorial:
    1. Preprocess gyroscope data
    2. Extract tremor features
    3. Detect tremor
    4. Quantify tremor (select relevant columns)

    Parameters
    ----------
    df_prepared : pd.DataFrame
        Prepared sensor data with time and gyroscope columns
    output_dir : str or Path
        Output directory for intermediate results (required)
    store_intermediate : list of str, default []
        Which intermediate results to store
    tremor_config : TremorConfig, optional
        Tremor analysis configuration
    imu_config : IMUConfig, optional
        IMU preprocessing configuration

    Returns
    -------
    pd.DataFrame
        Quantified tremor data with columns:
        - time: timestamp
        - pred_arm_at_rest: arm at rest prediction
        - pred_tremor_checked: tremor detection result
        - tremor_power: tremor power measure

    """
    logger = logging.getLogger(__name__)

    if tremor_config is None:
        tremor_config = TremorConfig()
    if imu_config is None:
        imu_config = IMUConfig()

    output_dir = Path(output_dir)

    # Validate input data columns
    required_columns = [
        DataColumns.TIME,
        DataColumns.GYROSCOPE_X,
        DataColumns.GYROSCOPE_Y,
        DataColumns.GYROSCOPE_Z,
    ]
    missing_columns = [
        col for col in required_columns if col not in df_prepared.columns
    ]
    if missing_columns:
        logger.warning(
            f"Missing required columns for tremor pipeline: {missing_columns}"
        )
        return pd.DataFrame()

    # Step 1: Preprocess gyroscope data (following tutorial)
    if verbosity >= 1:
        logger.info("Step 1: Preprocessing gyroscope data")
    df_preprocessed = preprocess_imu_data(
        df_prepared,
        imu_config,
        sensor="gyroscope",
        watch_side="left",  # Watch side is unimportant for tremor detection
    )

    if "preprocessing" in store_intermediate:
        preprocessing_dir = output_dir / "preprocessing"
        preprocessing_dir.mkdir(exist_ok=True)
        df_preprocessed.to_parquet(preprocessing_dir / "tremor_preprocessed.parquet")
        logger.info(f"Saved preprocessed data to {preprocessing_dir}")

    # Step 2: Extract tremor features
    if verbosity >= 1:
        logger.info("Step 2: Extracting tremor features")
    df_features = extract_tremor_features(df_preprocessed, tremor_config)

    if "tremor" in store_intermediate:
        tremor_dir = output_dir / "tremor"
        tremor_dir.mkdir(exist_ok=True)
        df_features.to_parquet(tremor_dir / "tremor_features.parquet")
        logger.info(f"Saved tremor features to {tremor_dir}")

    # Step 3: Detect tremor
    if verbosity >= 1:
        logger.info("Step 3: Detecting tremor")
    try:
        from importlib.resources import files

        classifier_path = files("paradigma.assets") / "tremor_detection_clf_package.pkl"
        df_predictions = detect_tremor(df_features, tremor_config, classifier_path)
    except Exception as e:
        logger.error(f"Tremor detection failed: {e}")
        return pd.DataFrame()

    # Step 4: Quantify tremor (following tutorial pattern)
    if verbosity >= 1:
        logger.info("Step 4: Quantifying tremor")

    # Select quantification columns as in the tutorial
    quantification_columns = [
        tremor_config.time_colname,
        DataColumns.PRED_ARM_AT_REST,
        DataColumns.PRED_TREMOR_CHECKED,
        DataColumns.TREMOR_POWER,
    ]

    # Check if all required columns exist
    available_columns = [
        col for col in quantification_columns if col in df_predictions.columns
    ]
    if len(available_columns) != len(quantification_columns):
        missing = set(quantification_columns) - set(available_columns)
        logger.warning(f"Missing quantification columns: {missing}")
        # Use available columns
        quantification_columns = available_columns

    df_quantification = df_predictions[quantification_columns].copy()

    # Set tremor power to None for non-tremor windows (following tutorial)
    if (
        DataColumns.TREMOR_POWER in df_quantification.columns
        and DataColumns.PRED_TREMOR_CHECKED in df_quantification.columns
    ):
        df_quantification.loc[
            df_quantification[DataColumns.PRED_TREMOR_CHECKED] == 0,
            DataColumns.TREMOR_POWER,
        ] = None

    if "quantification" in store_intermediate:
        quantification_dir = output_dir / "quantification"
        quantification_dir.mkdir(exist_ok=True)
        df_quantification.to_parquet(
            quantification_dir / "tremor_quantification.parquet"
        )

        # Save quantification metadata
        quantification_meta = {
            "total_windows": len(df_quantification),
            "tremor_windows": (
                int(df_quantification[DataColumns.PRED_TREMOR_CHECKED].sum())
                if DataColumns.PRED_TREMOR_CHECKED in df_quantification.columns
                else 0
            ),
            "columns": list(df_quantification.columns),
        }
        with open(quantification_dir / "tremor_quantification_meta.json", "w") as f:
            json.dump(quantification_meta, f, indent=2)

        if verbosity >= 2:
            logger.info(f"Saved tremor quantification to {quantification_dir}")

    tremor_windows = (
        int(df_quantification[DataColumns.PRED_TREMOR_CHECKED].sum())
        if DataColumns.PRED_TREMOR_CHECKED in df_quantification.columns
        else 0
    )
    if verbosity >= 1:
        logger.info(
            f"Tremor analysis completed: {tremor_windows} tremor windows detected from {len(df_quantification)} total windows"
        )

    return df_quantification
