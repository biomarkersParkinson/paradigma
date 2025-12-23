from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from paradigma.classification import ClassifierPackage
from paradigma.config import TremorConfig
from paradigma.constants import DataColumns
from paradigma.feature_extraction import (
    compute_mfccs,
    compute_power_in_bandwidth,
    compute_total_power,
    extract_frequency_peak,
    extract_tremor_power,
)
from paradigma.segmenting import WindowedDataExtractor, tabulate_windows
from paradigma.util import aggregate_parameter


def extract_tremor_features(df: pd.DataFrame, config: TremorConfig) -> pd.DataFrame:
    """
    Extract tremor features from windowed gyroscope data.

    Steps:
    1. Segment sequences of timestamps into overlapping windows.
    2. Extract spectral tremor features from each window.

    Parameters
    ----------
    df : pd.DataFrame
        Input sensor data containing time and gyroscope columns as defined in `config`.
    config : TremorConfig
        Configuration object specifying columns, windowing parameters, and
        feature extraction settings.

    Returns
    -------
    pd.DataFrame
        DataFrame containing extracted tremor features and start time of each window.
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
    Detect tremor using a pre-trained classifier and additional post-processing checks.

    Steps:
    1. Load classifier and scaling parameters from the specified package.
    2. Scale features in the input DataFrame.
    3. Predict tremor probability using the classifier.
    4. Apply threshold to classify tremor presence.
    5. Perform post-processing checks for rest tremor based on peak frequency and
    below-tremor power.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with tremor features.
    config : TremorConfig
        Configuration object containing tremor detection parameters
        (e.g., rest tremor frequency range).
    full_path_to_classifier_package : str | Path
        Directory containing the pre-trained classifier, scaler, and threshold.

    Returns
    -------
    pd.DataFrame
        Input DataFrame (`df`) with two additional columns:
        - `PRED_TREMOR_PROBA`: Predicted probability of tremor based on the classifier.
        - `PRED_TREMOR_LOGREG`: Binary classification result (True for tremor,
        False for no tremor), based on the threshold applied to `PRED_TREMOR_PROBA`.
        - `PRED_TREMOR_CHECKED`: Binary classification result (True for tremor,
        False for no tremor), after performing extra checks for rest tremor on
        `PRED_TREMOR_LOGREG`.
        - `PRED_ARM_AT_REST`: Binary classification result (True for arm at rest
        or stable posture, False for significant arm movement), based on the
        power below tremor.
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
    peak_check = (df["freq_peak"] >= config.fmin_rest_tremor) & (
        df["freq_peak"] <= config.fmax_rest_tremor
    )  # peak within 3-7 Hz
    df[DataColumns.PRED_ARM_AT_REST] = (
        df["below_tremor_power"] <= config.movement_threshold
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
    Quantifies the amount of tremor time and tremor power, aggregated over all windows
    in the input dataframe.

    Tremor time is calculated as the number of the detected tremor windows, as percentage
    of the number of windows without significant non-tremor movement (at rest). For tremor
    power the following aggregates are derived: the median, mode and percentile of tremor
    power specified in the configuration object.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with tremor predictions and computed tremor power.
        Must also contain a datetime column ('time_dt').
    config : TremorConfig
        Configuration object specifying aggregation settings and percentiles.

    Returns
    -------
    dict
        Dictionary containing:
        - `metadata`: number of valid days, total windows, windows at rest.
        - `aggregated_tremor_measures`: tremor percentage and power aggregates (median, modal, 90th percentile).

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
    n_windows_tremor = np.sum(df_filtered["pred_tremor_checked"])
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
            df_filtered["pred_tremor_checked"] == 1, "tremor_power"
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
    Compute spectral features from windowed gyroscope data.

    Features extracted include:
    - Mel-frequency cepstral coefficients (MFCCs)
    - Peak frequency
    - Tremor power
    - Below-tremor power

    Parameters
    ----------
    data : np.ndarray
        2D array where each row corresponds to a window of gyroscope data.
    config : object
        Configuration object containing settings such as sampling frequency, window type,
        and MFCC parameters.

    Returns
    -------
    pd.DataFrame
        Feature dataframe containing the extracted spectral features, including
        MFCCs, the frequency of the peak, the tremor power and below tremor power
        for each window.
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
    feature_dict["freq_peak"] = extract_frequency_peak(
        freqs, total_psd, config.fmin_peak_search, config.fmax_peak_search
    )
    feature_dict["below_tremor_power"] = compute_power_in_bandwidth(
        freqs,
        total_psd,
        config.fmin_below_rest_tremor,
        config.fmax_below_rest_tremor,
        include_max=False,
        spectral_resolution=config.spectral_resolution,
        cumulative_sum_method="sum",
    )
    feature_dict["tremor_power"] = extract_tremor_power(
        freqs, total_psd, config.fmin_rest_tremor, config.fmax_rest_tremor
    )

    return pd.DataFrame(feature_dict)
