import json
import logging
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import periodogram

from paradigma.classification import ClassifierPackage
from paradigma.config import GaitConfig, IMUConfig
from paradigma.constants import DataColumns
from paradigma.feature_extraction import (
    compute_angle,
    compute_dominant_frequency,
    compute_mfccs,
    compute_peak_angular_velocity,
    compute_power_in_bandwidth,
    compute_range_of_motion,
    compute_statistics,
    compute_std_euclidean_norm,
    compute_total_power,
    extract_angle_extremes,
    pca_transform_gyroscope,
    remove_moving_average_angle,
)
from paradigma.preprocessing import preprocess_imu_data
from paradigma.segmenting import (
    WindowedDataExtractor,
    create_segments,
    discard_segments,
    tabulate_windows,
)
from paradigma.util import aggregate_parameter, merge_predictions_with_timestamps

logger = logging.getLogger(__name__)

# Only configure basic logging if no handlers exist
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _empty_arm_swing_df(df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        DataColumns.GAIT_SEGMENT_NR,
        DataColumns.RANGE_OF_MOTION,
        DataColumns.PEAK_VELOCITY,
    ]
    if DataColumns.DATA_SEGMENT_NR in df.columns:
        expected_columns.append(DataColumns.DATA_SEGMENT_NR)
    return pd.DataFrame(columns=expected_columns)


def extract_gait_features(df: pd.DataFrame, config: GaitConfig) -> pd.DataFrame:
    """
    Extracts gait features from accelerometer and gravity sensor data in the
    input DataFrame by computing temporal and spectral features.

    This function performs the following steps:
    1. Groups sequences of timestamps into windows, using accelerometer and
       gravity data.
    2. Computes temporal domain features such as mean and standard deviation
       for accelerometer and gravity data.
    3. Transforms the signals from the temporal domain to the spectral
       domain using the Fast Fourier Transform (FFT).
    4. Computes spectral domain features for the accelerometer data.
    5. Combines both temporal and spectral features into a final DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing gait data, which includes time,
        accelerometer, and gravity sensor data. The data should be
        structured with the necessary columns as specified in the `config`.

    onfig : GaitConfig
        Configuration object containing parameters for feature extraction,
        including column names for time, accelerometer data, and gravity
        data, as well as settings for windowing, and feature computation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing extracted gait features, including temporal
        and spectral domain features. The DataFrame will have columns
        corresponding to time, statistical features of the accelerometer and
        gravity data, and spectral features of the accelerometer data.

    Notes
    -----
    - This function groups the data into windows based on timestamps and
      applies Fast Fourier Transform to compute spectral features.
    - The temporal features are extracted from the accelerometer and gravity
      data, and include statistics like mean and standard deviation.
    - The input DataFrame must include columns as specified in the `config`
      object for proper feature extraction.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the required columns as
        specified in the configuration or if any step in the feature
        extraction fails.
    """
    # Group sequences of timestamps into windows
    windowed_colnames = (
        [config.time_colname] + config.accelerometer_colnames + config.gravity_colnames
    )
    windowed_data = tabulate_windows(
        df=df,
        columns=windowed_colnames,
        window_length_s=config.window_length_s,
        window_step_length_s=config.window_step_length_s,
        fs=config.sampling_frequency,
    )

    extractor = WindowedDataExtractor(windowed_colnames)

    idx_time = extractor.get_index(config.time_colname)
    idx_acc = extractor.get_slice(config.accelerometer_colnames)
    idx_grav = extractor.get_slice(config.gravity_colnames)

    # Extract data
    start_time = np.min(windowed_data[:, :, idx_time], axis=1)
    windowed_acc = windowed_data[:, :, idx_acc]
    windowed_grav = windowed_data[:, :, idx_grav]

    df_features = pd.DataFrame(start_time, columns=[config.time_colname])

    # Compute statistics of the temporal domain signals (mean, std) for
    # accelerometer and gravity
    df_temporal_features = extract_temporal_domain_features(
        config=config,
        windowed_acc=windowed_acc,
        windowed_grav=windowed_grav,
        grav_stats=["mean", "std"],
    )

    # Combine temporal features with the start time
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # Transform the accelerometer data to the spectral domain using FFT and
    # extract spectral features
    df_spectral_features = extract_spectral_domain_features(
        config=config, sensor="accelerometer", windowed_data=windowed_acc
    )

    # Combine the spectral features with the previously computed temporal features
    df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def detect_gait(
    df: pd.DataFrame, clf_package: ClassifierPackage, parallel: bool = False
) -> pd.Series:
    """
    Detects gait activity in the input DataFrame using a pre-trained
    classifier and applies a threshold to classify results.

    This function performs the following steps:
    1. Loads the pre-trained classifier and scaling parameters from the
       specified directory.
    2. Scales the relevant features in the input DataFrame (`df`) using the
       loaded scaling parameters.
    3. Predicts the probability of gait activity for each sample in the
       DataFrame using the classifier.
    4. Applies a threshold to the predicted probabilities to determine
       whether gait activity is present.
    5. Returns predicted probabilities

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features extracted from gait data. It
        must include the necessary columns as specified in the classifier's
        feature names.

    clf_package : ClassifierPackage
        The pre-trained classifier package containing the classifier,
        threshold, and scaler.

    parallel : bool, optional, default=False
        If `True`, enables parallel processing during classification. If
        `False`, the classifier uses a single core.

    Returns
    -------
    pd.Series
        A Series containing the predicted probabilities of gait activity for
        each sample in the input DataFrame.
    """
    # Set classifier
    clf = clf_package.classifier
    if not parallel and hasattr(clf, "n_jobs"):
        clf.n_jobs = 1

    feature_names_scaling = clf_package.scaler.feature_names_in_
    feature_names_predictions = clf.feature_names_in_

    # Apply scaling to relevant columns
    scaled_features = clf_package.transform_features(df.loc[:, feature_names_scaling])

    # Replace scaled features in a copy of the relevant features for prediction
    x_features = df.loc[:, feature_names_predictions].copy()
    x_features.loc[:, feature_names_scaling] = scaled_features

    # Make prediction and add the probability of gait activity to the DataFrame
    pred_gait_proba_series = clf_package.predict_proba(x_features)

    return pred_gait_proba_series


def extract_arm_activity_features(
    df: pd.DataFrame,
    config: GaitConfig,
) -> pd.DataFrame:
    """
    Extract features related to arm activity from a time-series DataFrame.

    This function processes a DataFrame containing accelerometer, gravity,
    and gyroscope signals, and extracts features related to arm activity by
    performing the following steps:
    1. Computes the angle and velocity from gyroscope data.
    2. Filters the data to include only predicted gait segments.
    3. Groups the data into segments based on consecutive timestamps and
       pre-specified gaps.
    4. Removes segments that do not meet predefined criteria.
    5. Creates fixed-length windows from the time series data.
    6. Extracts angle-related features, temporal domain features, and
       spectral domain features.

    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame containing accelerometer, gravity, and
        gyroscope data of predicted gait.

    config : ArmActivityFeatureExtractionConfig
        Configuration object containing column names and parameters
        for feature extraction.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted arm activity features,
        including angle, velocity, temporal, and spectral features.
    """
    # Group consecutive timestamps into segments, with new segments
    # starting after a pre-specified gap. If data_segment_nr exists,
    # create gait segments per data segment to preserve both
    has_data_segments = DataColumns.DATA_SEGMENT_NR in df.columns

    if has_data_segments:
        df_list = []
        gait_segment_offset = 0

        for data_seg_nr in sorted(df[DataColumns.DATA_SEGMENT_NR].unique()):
            df_seg = df[df[DataColumns.DATA_SEGMENT_NR] == data_seg_nr].copy()

            # Create gait segments within this data segment
            df_seg[DataColumns.GAIT_SEGMENT_NR] = create_segments(
                time_array=df_seg[DataColumns.TIME].values,
                max_segment_gap_s=config.max_segment_gap_s,
            )

            # Offset gait segment numbers to be unique across data segments
            if gait_segment_offset > 0:
                df_seg[DataColumns.GAIT_SEGMENT_NR] += gait_segment_offset
            gait_segment_offset = df_seg[DataColumns.GAIT_SEGMENT_NR].max() + 1

            df_list.append(df_seg)

        df = pd.concat(df_list, ignore_index=True)
    else:
        df[DataColumns.GAIT_SEGMENT_NR] = create_segments(
            time_array=df[DataColumns.TIME], max_segment_gap_s=config.max_segment_gap_s
        )

    # Remove segments that do not meet predetermined criteria
    df = discard_segments(
        df=df,
        segment_nr_colname=DataColumns.GAIT_SEGMENT_NR,
        min_segment_length_s=config.min_segment_length_s,
        format="timestamps",
        fs=config.sampling_frequency,
    )

    # Create windows of fixed length and step size from the time series per segment
    windowed_data = []
    df_grouped = df.groupby(DataColumns.GAIT_SEGMENT_NR)
    windowed_colnames = (
        [config.time_colname]
        + config.accelerometer_colnames
        + config.gravity_colnames
        + config.gyroscope_colnames
    )

    # Collect windows from all segments in a list for faster concatenation
    for _, group in df_grouped:
        windows = tabulate_windows(
            df=group,
            columns=windowed_colnames,
            window_length_s=config.window_length_s,
            window_step_length_s=config.window_step_length_s,
            fs=config.sampling_frequency,
        )
        if len(windows) > 0:  # Skip if no windows are created
            windowed_data.append(windows)

    # If no windows were created, raise an error
    if not windowed_data:
        print("No windows were created from the given data.")
        return pd.DataFrame()

    # Concatenate the windows into one array at the end
    windowed_data = np.concatenate(windowed_data, axis=0)

    # Slice columns for accelerometer, gravity, gyroscope, angle, and velocity
    extractor = WindowedDataExtractor(windowed_colnames)

    idx_time = extractor.get_index(config.time_colname)
    idx_acc = extractor.get_slice(config.accelerometer_colnames)
    idx_grav = extractor.get_slice(config.gravity_colnames)
    idx_gyro = extractor.get_slice(config.gyroscope_colnames)

    # Extract data
    start_time = np.min(windowed_data[:, :, idx_time], axis=1)
    windowed_acc = windowed_data[:, :, idx_acc]
    windowed_grav = windowed_data[:, :, idx_grav]
    windowed_gyro = windowed_data[:, :, idx_gyro]

    # Initialize DataFrame for features
    df_features = pd.DataFrame(start_time, columns=[config.time_colname])

    # Extract temporal domain features (e.g., mean, std for accelerometer and gravity)
    df_temporal_features = extract_temporal_domain_features(
        config=config,
        windowed_acc=windowed_acc,
        windowed_grav=windowed_grav,
        grav_stats=["mean", "std"],
    )
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # Extract spectral domain features for accelerometer and gyroscope signals
    for sensor_name, windowed_sensor in zip(
        ["accelerometer", "gyroscope"], [windowed_acc, windowed_gyro]
    ):
        df_spectral_features = extract_spectral_domain_features(
            config=config, sensor=sensor_name, windowed_data=windowed_sensor
        )
        df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def filter_gait(
    df: pd.DataFrame, clf_package: ClassifierPackage, parallel: bool = False
) -> pd.Series:
    """
    Filters gait data to identify windows with no other arm activity using
    a pre-trained classifier.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features extracted from gait data.
    clf_package: ClassifierPackage
        The pre-trained classifier package containing the classifier,
        threshold, and scaler.
    parallel : bool, optional, default=False
        If `True`, enables parallel processing.

    Returns
    -------
    pd.Series
        A Series containing the predicted probabilities.
    """
    if df.shape[0] == 0:
        raise ValueError("No data found in the input DataFrame.")

    # Set classifier
    clf = clf_package.classifier
    if not parallel and hasattr(clf, "n_jobs"):
        clf.n_jobs = 1

    feature_names_scaling = clf_package.scaler.feature_names_in_
    feature_names_predictions = clf.feature_names_in_

    # Apply scaling to relevant columns
    scaled_features = clf_package.transform_features(df.loc[:, feature_names_scaling])

    # Replace scaled features in a copy of the relevant features for prediction
    x_features = df.loc[:, feature_names_predictions].copy()
    x_features.loc[:, feature_names_scaling] = scaled_features

    # Make predictions
    pred_no_other_arm_activity_proba_series = clf_package.predict_proba(x_features)

    return pred_no_other_arm_activity_proba_series


def quantify_arm_swing(
    df: pd.DataFrame,
    fs: int | None = None,
    filtered: bool = True,
    max_segment_gap_s: float = 1.5,
    min_segment_length_s: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """
    Quantify arm swing parameters for segments of motion based on gyroscope data.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the raw sensor data of predicted gait
        timestamps. Should include a column for predicted no other arm
        activity based on a fitted threshold if filtered is True.

    fs : int
        The sampling frequency of the sensor data.

    filtered : bool, optional, default=True
        If `True`, the gyroscope data is filtered to only include predicted
        no other arm activity.

    max_segment_gap_s : float, optional, default=1.5
        The maximum gap in seconds between consecutive timestamps to group
        them into segments.

    min_segment_length_s : float, optional, default=1.5
        The minimum length in seconds for a segment to be considered valid.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        A tuple containing a dataframe with quantified arm swing parameters
        and a dictionary containing metadata for each segment.
    """
    # Group consecutive timestamps into segments, with new segments starting
    # after a pre-specified gap. Segments are made based on predicted gait
    df["unfiltered_segment_nr"] = create_segments(
        time_array=df[DataColumns.TIME], max_segment_gap_s=max_segment_gap_s
    )

    # fs is deprecated
    fs = 1 / df[DataColumns.TIME].diff().median()

    # Remove segments that do not meet predetermined criteria
    df = discard_segments(
        df=df,
        segment_nr_colname="unfiltered_segment_nr",
        min_segment_length_s=min_segment_length_s,
        fs=fs,
        format="timestamps",
    )

    if df.empty:
        raise ValueError(
            "No segments found in the input data after discarding segments "
            "of invalid shape."
        )

    # Create dictionary of gait segment number and duration
    gait_segment_duration_dict = {
        segment_nr: len(group[DataColumns.TIME]) / fs
        for segment_nr, group in df.groupby("unfiltered_segment_nr", sort=False)
    }

    # If no arm swing data is remaining, return an empty dictionary
    if filtered and df.loc[df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY] == 1].empty:
        raise ValueError("No gait without other arm activities to quantify.")
    elif filtered:
        # Filter the DataFrame to only include predicted no other arm activity (1)
        df = df.loc[df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY] == 1].reset_index(
            drop=True
        )

        # Group consecutive timestamps into segments of filtered gait
        df["filtered_segment_nr"] = create_segments(
            time_array=df[DataColumns.TIME], max_segment_gap_s=max_segment_gap_s
        )

        # Remove segments that do not meet predetermined criteria
        df = discard_segments(
            df=df,
            segment_nr_colname="filtered_segment_nr",
            min_segment_length_s=min_segment_length_s,
            fs=fs,
        )

        if df.empty:
            raise ValueError(
                "No filtered gait segments found in the input data after "
                "discarding segments of invalid shape."
            )

    grouping_colname = "filtered_segment_nr" if filtered else "unfiltered_segment_nr"

    arm_swing_quantified = []
    segment_meta = {
        "all": {"duration_s": len(df[DataColumns.TIME]) / fs},
        "per_segment": {},
    }

    # PCA is fitted on only predicted gait without other arm activity if
    # filtered, otherwise it is fitted on the entire gyroscope data
    df[DataColumns.VELOCITY] = pca_transform_gyroscope(
        df=df,
        y_gyro_colname=DataColumns.GYROSCOPE_Y,
        z_gyro_colname=DataColumns.GYROSCOPE_Z,
    )

    # Group and process segments
    for segment_nr, group in df.groupby(grouping_colname, sort=False):
        if filtered:
            gait_segment_nr = group["unfiltered_segment_nr"].iloc[
                0
            ]  # Each filtered segment is contained within an unfiltered segment
        else:
            gait_segment_nr = segment_nr

        try:
            gait_segment_duration_s = gait_segment_duration_dict[gait_segment_nr]
        except KeyError:
            logger.warning(
                "Segment %s (filtered = %s) not found in gait segment "
                "duration dictionary. Skipping this segment.",
                gait_segment_nr,
                filtered,
            )
            logger.debug(
                "Available segments: %s", list(gait_segment_duration_dict.keys())
            )
            continue

        time_array = group[DataColumns.TIME].to_numpy()
        velocity_array = group[DataColumns.VELOCITY].to_numpy()

        # Integrate the angular velocity to obtain an estimation of the angle
        angle_array = compute_angle(
            time_array=time_array,
            velocity_array=velocity_array,
        )

        # Detrend angle using moving average
        angle_array = remove_moving_average_angle(
            angle_array=angle_array,
            fs=fs,
        )

        segment_meta["per_segment"][segment_nr] = {
            "start_time_s": float(time_array.min()),
            "end_time_s": float(time_array.max()),
            "duration_unfiltered_segment_s": gait_segment_duration_s,
        }

        if filtered:
            segment_meta["per_segment"][segment_nr]["duration_filtered_segment_s"] = (
                len(time_array) / fs
            )

        if angle_array.size > 0:
            angle_extrema_indices, _, _ = extract_angle_extremes(
                angle_array=angle_array,
                sampling_frequency=fs,
                max_frequency_activity=1.75,
            )

            if len(angle_extrema_indices) > 1:  # Requires at minimum 2 peaks
                try:
                    rom = compute_range_of_motion(
                        angle_array=angle_array,
                        extrema_indices=angle_extrema_indices,
                    )
                except Exception as e:
                    # Handle the error, set RoM to NaN, and log the error
                    print(
                        f"Error computing range of motion for segment "
                        f"{segment_nr}: {e}"
                    )
                    rom = np.array([np.nan])

                try:
                    pav = compute_peak_angular_velocity(
                        velocity_array=velocity_array,
                        angle_extrema_indices=angle_extrema_indices,
                    )
                except Exception as e:
                    # Handle the error, set pav to NaN, and log the error
                    print(
                        f"Error computing peak angular velocity for segment "
                        f"{segment_nr}: {e}"
                    )
                    pav = np.array([np.nan])

                params_dict = {
                    DataColumns.GAIT_SEGMENT_NR: segment_nr,
                    DataColumns.RANGE_OF_MOTION: rom,
                    DataColumns.PEAK_VELOCITY: pav,
                }

                # Add data_segment_nr if it exists in the input data
                if DataColumns.DATA_SEGMENT_NR in group.columns:
                    params_dict[DataColumns.DATA_SEGMENT_NR] = group[
                        DataColumns.DATA_SEGMENT_NR
                    ].iloc[0]

                df_params_segment = pd.DataFrame(params_dict)

                arm_swing_quantified.append(df_params_segment)

    if not arm_swing_quantified:
        # No valid arm swing segments found, return empty DataFrame
        arm_swing_quantified = _empty_arm_swing_df(df)
    else:
        arm_swing_quantified = pd.concat(arm_swing_quantified, ignore_index=True)

    return arm_swing_quantified, segment_meta


def aggregate_arm_swing_params(
    df_arm_swing_params: pd.DataFrame,
    segment_meta: dict,
    segment_cats: list[tuple],
    aggregates: list[str] = ["median"],
) -> dict:
    """
    Aggregate the quantification results for arm swing parameters.

    Parameters
    ----------
    df_arm_swing_params : pd.DataFrame
        A dataframe containing the arm swing parameters to be aggregated

    segment_meta : dict
        A dictionary containing metadata for each segment.

    segment_cats : List[tuple]
        A list of tuples defining the segment categories, where each tuple
        contains the lower and upper bounds for the segment duration.
    aggregates : List[str], optional
        A list of aggregation methods to apply to the quantification
        results.

    Returns
    -------
    dict
        A dictionary containing the aggregated quantification results for
        arm swing parameters.
    """
    arm_swing_parameters = [DataColumns.RANGE_OF_MOTION, DataColumns.PEAK_VELOCITY]

    aggregated_results = {}
    for segment_cat_range in segment_cats:
        segment_cat_str = f"{segment_cat_range[0]}_{segment_cat_range[1]}"
        cat_segments = [
            x
            for x in segment_meta.keys()
            if segment_meta[x]["duration_unfiltered_segment_s"] >= segment_cat_range[0]
            and segment_meta[x]["duration_unfiltered_segment_s"] < segment_cat_range[1]
        ]

        if len(cat_segments) > 0:
            # For each segment, use 'duration_filtered_segment_s' if present,
            # else 'duration_unfiltered_segment_s'
            aggregated_results[segment_cat_str] = {
                "duration_s": sum(
                    [
                        (
                            segment_meta[x]["duration_filtered_segment_s"]
                            if "duration_filtered_segment_s" in segment_meta[x]
                            else segment_meta[x]["duration_unfiltered_segment_s"]
                        )
                        for x in cat_segments
                    ]
                )
            }

            df_arm_swing_params_cat = df_arm_swing_params.loc[
                df_arm_swing_params[DataColumns.GAIT_SEGMENT_NR].isin(cat_segments)
            ]

            # Aggregate across all segments
            aggregates_per_segment = ["median", "mean"]

            for arm_swing_parameter in arm_swing_parameters:
                for aggregate in aggregates:
                    if aggregate in ["std", "cov"]:
                        per_segment_agg = []
                        # If the aggregate is 'cov' (coefficient of variation),
                        # we also compute the mean and standard deviation per
                        # segment
                        segment_groups = dict(
                            tuple(
                                df_arm_swing_params_cat.groupby(
                                    DataColumns.GAIT_SEGMENT_NR
                                )
                            )
                        )
                        for segment_nr in cat_segments:
                            segment_df = segment_groups.get(segment_nr)
                            if segment_df is not None:
                                per_segment_agg.append(
                                    aggregate_parameter(
                                        segment_df[arm_swing_parameter], aggregate
                                    )
                                )

                        # Drop nans
                        per_segment_agg = np.array(per_segment_agg)
                        per_segment_agg = per_segment_agg[~np.isnan(per_segment_agg)]

                        for segment_level_aggregate in aggregates_per_segment:
                            key = (
                                f"{segment_level_aggregate}_{aggregate}_"
                                f"{arm_swing_parameter}"
                            )
                            aggregated_results[segment_cat_str][key] = (
                                aggregate_parameter(
                                    per_segment_agg, segment_level_aggregate
                                )
                            )
                    else:
                        aggregated_results[segment_cat_str][
                            f"{aggregate}_{arm_swing_parameter}"
                        ] = aggregate_parameter(
                            df_arm_swing_params_cat[arm_swing_parameter], aggregate
                        )

        else:
            # If no segments are found for this category, initialize with NaN
            aggregated_results[segment_cat_str] = {
                "duration_s": 0,
            }

    return aggregated_results


def extract_temporal_domain_features(
    config,
    windowed_acc: np.ndarray,
    windowed_grav: np.ndarray,
    grav_stats: list[str] = ["mean"],
) -> pd.DataFrame:
    """
    Compute temporal domain features for the accelerometer signal.

    This function calculates various statistical features for the gravity signal
    and computes the standard deviation of the accelerometer's Euclidean norm.

    Parameters
    ----------
    config : object
        Configuration object containing the accelerometer and gravity column names.
    windowed_acc : numpy.ndarray
        A 2D numpy array of shape (N, M) where N is the number of windows and M is
        the number of accelerometer values per window.
    windowed_grav : numpy.ndarray
        A 2D numpy array of shape (N, M) where N is the number of windows and M is
        the number of gravity signal values per window.
    grav_stats : list of str, optional
        A list of statistics to compute for the gravity signal (default is ['mean']).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the computed features, with each row corresponding
        to a window and each column representing a specific feature.
    """
    # Compute gravity statistics (e.g., mean, std, etc.)
    feature_dict = {}
    for stat in grav_stats:
        stats_result = compute_statistics(data=windowed_grav, statistic=stat)
        for i, col in enumerate(config.gravity_colnames):
            feature_dict[f"{col}_{stat}"] = stats_result[:, i]

    # Compute standard deviation of the Euclidean norm of the accelerometer signal
    feature_dict["accelerometer_std_norm"] = compute_std_euclidean_norm(
        data=windowed_acc
    )

    return pd.DataFrame(feature_dict)


def extract_spectral_domain_features(
    windowed_data: np.ndarray,
    config,
    fs,
    sensor: str,
) -> pd.DataFrame:
    """
    Compute spectral domain features for a sensor's data.

    This function computes the periodogram, extracts power in specific
    frequency bands, calculates the dominant frequency, and computes
    Mel-frequency cepstral coefficients (MFCCs) for a given sensor's
    windowed data.

    Parameters
    ----------
    windowed_data : numpy.ndarray
        A 2D numpy array where each row corresponds to a window of sensor data.

    config : object
        Configuration object containing settings such as window type,
        frequency bands, and MFCC parameters.

    sensor : str
        The name of the sensor (e.g., 'accelerometer', 'gyroscope').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the computed spectral features, with each row
        corresponding to a window and each column representing a specific
        feature.
    """
    # Initialize a dictionary to hold the results
    feature_dict = {}

    # Compute periodogram (power spectral density)
    freqs, psd = periodogram(x=windowed_data, fs=fs, window=config.window_type, axis=1)

    # Compute power in specified frequency bands
    for band_name, band_freqs in config.d_frequency_bandwidths.items():
        band_powers = compute_power_in_bandwidth(
            freqs=freqs,
            psd=psd,
            fmin=band_freqs[0],
            fmax=band_freqs[1],
            include_max=False,
        )
        for i, col in enumerate(config.axes):
            feature_dict[f"{sensor}_{col}_{band_name}"] = band_powers[:, i]

    # Compute dominant frequency for each axis
    dominant_frequencies = compute_dominant_frequency(
        freqs=freqs,
        psd=psd,
        fmin=config.spectrum_low_frequency,
        fmax=config.spectrum_high_frequency,
    )

    # Add dominant frequency features to the feature_dict
    for axis, freq in zip(config.axes, dominant_frequencies.T):
        feature_dict[f"{sensor}_{axis}_dominant_frequency"] = freq

    # Compute total power in the PSD
    total_power_psd = compute_total_power(psd)

    # Compute MFCCs
    mfccs = compute_mfccs(
        total_power_array=total_power_psd, config=config, multiplication_factor=4
    )

    # Combine the MFCCs into the features DataFrame
    mfcc_colnames = [
        f"{sensor}_mfcc_{x}" for x in range(1, config.mfcc_n_coefficients + 1)
    ]
    for i, colname in enumerate(mfcc_colnames):
        feature_dict[colname] = mfccs[:, i]

    return pd.DataFrame(feature_dict)


def run_gait_pipeline(
    df_prepared: pd.DataFrame,
    watch_side: str,
    output_dir: str | Path,
    imu_config: IMUConfig | None = None,
    gait_config: GaitConfig | None = None,
    arm_activity_config: GaitConfig | None = None,
    store_intermediate: list[str] | None = None,
    segment_number_offset_filtered: int = 0,
    segment_number_offset_unfiltered: int = 0,
    logging_level: int = logging.INFO,
    custom_logger: logging.Logger | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """
    Run the complete gait analysis pipeline on prepared data (steps 1-6).

    This function implements the gait analysis workflow as described in the tutorials:
    1. Preprocessing
    2. Gait feature extraction
    3. Gait detection
    4. Arm activity feature extraction
    5. Filtering gait
    6. Arm swing quantification

    Step 7 (aggregation) should be done after processing all segments.

    Parameters
    ----------
    df_prepared : pd.DataFrame
        Prepared IMU data with time, accelerometer, and gyroscope columns.
        Should contain columns: time, accelerometer_x/y/z, gyroscope_x/y/z.
        Will be preprocessed as step 1 of the pipeline.
    watch_side : str
        Side of the watch ('left' or 'right') to configure preprocessing accordingly.
    output_dir : str or Path
        Directory to save intermediate results (required)
    imu_config : IMUConfig, optional
        Configuration for IMU data preprocessing.
        If None, uses default IMUConfig.
    gait_config : GaitConfig, optional
        Configuration for gait feature extraction and detection.
        If None, uses default GaitConfig(step="gait").
    arm_activity_config : GaitConfig, optional
        Configuration for arm activity feature extraction and filtering.
        If None, uses default GaitConfig(step="arm_activity").
    store_intermediate : list[str], optional
        Steps of which intermediate results should be stored:
        - 'preprocessing': Store preprocessed data after step 1
        - 'gait': Store gait features and predictions after step 3
        - 'arm_activity': Store arm activity features and predictions after step 5
        - 'quantification': Store arm swing quantification results after step 6
        If empty, only returns the final quantified results.
    segment_number_offset_filtered : int, optional, default=0
        Offset to add to filtered segment numbers to avoid conflicts when
        concatenating multiple data segments. Used for proper segment numbering
        across multiple files.
    segment_number_offset_unfiltered : int, optional, default=0
        Offset to add to unfiltered segment numbers to avoid conflicts when
        concatenating multiple data segments. Used for proper segment numbering
        across multiple files.
    logging_level : int, default logging.INFO
        Logging level using standard logging constants (logging.DEBUG, logging.INFO,
        etc.)
    custom_logger : logging.Logger, optional
        Custom logger instance. If provided, logging_level is ignored.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, dict]]
        A tuple containing two dictionaries:
        - First dict contains quantified arm swing parameters with keys:
            - 'filtered': DataFrame with arm swings from clean gait only
            - 'unfiltered': DataFrame with arm swings from all gait
          Each DataFrame has columns:
            - gait_segment_nr: Gait segment number within this data segment
            - Various arm swing metrics (range of motion, peak angular velocity, etc.)
            - Additional metadata columns
        - Second dict contains gait segment metadata with keys:
            - 'filtered': Metadata for filtered quantification
            - 'unfiltered': Metadata for unfiltered quantification
          Each metadata dict contains information about each detected gait segment

    Notes
    -----
    This function processes a single contiguous data segment. For multiple segments,
    call this function for each segment, then use aggregate_arm_swing_params()
    on the concatenated results.

    The function follows the exact workflow from the gait analysis tutorial:
    https://github.com/biomarkersParkinson/paradigma/blob/main/docs/
    tutorials/gait_analysis.ipynb
    """
    # Setup logger
    active_logger = custom_logger if custom_logger is not None else logger
    if custom_logger is None:
        active_logger.setLevel(logging_level)

    if store_intermediate is None:
        store_intermediate = []

    # Set default configurations
    if imu_config is None:
        imu_config = IMUConfig()
    if gait_config is None:
        gait_config = GaitConfig(step="gait")
    if arm_activity_config is None:
        arm_activity_config = GaitConfig(step="arm_activity")

    output_dir = Path(output_dir)

    # Validate input data has required columns
    required_columns = [
        DataColumns.TIME,
        DataColumns.ACCELEROMETER_X,
        DataColumns.ACCELEROMETER_Y,
        DataColumns.ACCELEROMETER_Z,
        DataColumns.GYROSCOPE_X,
        DataColumns.GYROSCOPE_Y,
        DataColumns.GYROSCOPE_Z,
    ]
    missing_columns = [
        col for col in required_columns if col not in df_prepared.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Step 1: Preprocess data
    active_logger.info("Step 1: Preprocessing IMU data")

    df_preprocessed = preprocess_imu_data(
        df=df_prepared,
        config=imu_config,
        sensor="both",
        watch_side=watch_side,
    )

    if "preprocessing" in store_intermediate:
        preprocessing_dir = output_dir / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        df_preprocessed.to_parquet(
            preprocessing_dir / "preprocessed_data.parquet", index=False
        )
        active_logger.debug(
            f"Saved preprocessed data to "
            f"{preprocessing_dir / 'preprocessed_data.parquet'}"
        )

    # Step 2: Extract gait features
    active_logger.info("Step 2: Extracting gait features")
    df_gait = extract_gait_features(df_preprocessed, gait_config)

    if "gait" in store_intermediate:
        gait_dir = output_dir / "gait"
        gait_dir.mkdir(parents=True, exist_ok=True)
        df_gait.to_parquet(gait_dir / "gait_features.parquet", index=False)
        active_logger.debug(
            f"Saved gait features to {gait_dir / 'gait_features.parquet'}"
        )

    # Step 3: Detect gait
    active_logger.info("Step 3: Detecting gait")
    try:
        classifier_path = files("paradigma.assets") / "gait_detection_clf_package.pkl"
        classifier_package_gait = ClassifierPackage.load(classifier_path)
    except Exception as e:
        active_logger.error(f"Could not load gait detection classifier: {e}")
        raise RuntimeError("Gait detection classifier not available")

    gait_proba = detect_gait(df_gait, classifier_package_gait, parallel=False)
    df_gait[DataColumns.PRED_GAIT_PROBA] = gait_proba

    # Merge predictions back with timestamps
    df_gait_with_time = merge_predictions_with_timestamps(
        df_ts=df_preprocessed,
        df_predictions=df_gait,
        pred_proba_colname=DataColumns.PRED_GAIT_PROBA,
        window_length_s=gait_config.window_length_s,
    )

    # Add binary prediction column
    df_gait_with_time[DataColumns.PRED_GAIT] = (
        df_gait_with_time[DataColumns.PRED_GAIT_PROBA]
        >= classifier_package_gait.threshold
    ).astype(int)

    if "gait" in store_intermediate:
        gait_dir = output_dir / "gait"
        gait_dir.mkdir(parents=True, exist_ok=True)
        df_gait_with_time.to_parquet(gait_dir / "gait_predictions.parquet", index=False)
        active_logger.info(
            f"Saved gait predictions to {gait_dir / 'gait_predictions.parquet'}"
        )

    # Filter to only gait periods
    df_gait_only = df_gait_with_time.loc[
        df_gait_with_time[DataColumns.PRED_GAIT] == 1
    ].reset_index(drop=True)

    if len(df_gait_only) == 0:
        active_logger.warning("No gait detected in this segment")
        empty_df_filtered = _empty_arm_swing_df(df_prepared)
        empty_df_unfiltered = _empty_arm_swing_df(df_prepared)
        empty_meta_filtered = {"all": {"duration_s": 0}, "per_segment": {}}
        empty_meta_unfiltered = {"all": {"duration_s": 0}, "per_segment": {}}
        return {"filtered": empty_df_filtered, "unfiltered": empty_df_unfiltered}, {
            "filtered": empty_meta_filtered,
            "unfiltered": empty_meta_unfiltered,
        }

    # Step 4: Extract arm activity features
    active_logger.info("Step 4: Extracting arm activity features")
    df_arm_activity = extract_arm_activity_features(df_gait_only, arm_activity_config)

    if "arm_activity" in store_intermediate:
        arm_activity_dir = output_dir / "arm_activity"
        arm_activity_dir.mkdir(parents=True, exist_ok=True)
        df_arm_activity.to_parquet(
            arm_activity_dir / "arm_activity_features.parquet", index=False
        )
        active_logger.debug(
            f"Saved arm activity features to "
            f"{arm_activity_dir / 'arm_activity_features.parquet'}"
        )

    # Step 5: Filter gait (remove other arm activities)
    active_logger.info("Step 5: Filtering gait")
    try:
        classifier_path = files("paradigma.assets") / "gait_filtering_clf_package.pkl"
        classifier_package_arm_activity = ClassifierPackage.load(classifier_path)
    except Exception as e:
        active_logger.error(f"Could not load arm activity classifier: {e}")
        raise RuntimeError("Arm activity classifier not available")

    # Filter gait returns probabilities which we add to the arm activity features
    arm_activity_probabilities = filter_gait(
        df_arm_activity, classifier_package_arm_activity, parallel=False
    )

    df_arm_activity[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = (
        arm_activity_probabilities
    )

    # Merge predictions back with timestamps
    df_arm_activity = merge_predictions_with_timestamps(
        df_ts=df_gait_only,
        df_predictions=df_arm_activity,
        pred_proba_colname=DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA,
        window_length_s=arm_activity_config.window_length_s,
    )

    # Add binary prediction column
    filt_threshold = classifier_package_arm_activity.threshold
    df_arm_activity[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY] = (
        df_arm_activity[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= filt_threshold
    ).astype(int)

    if "arm_activity" in store_intermediate:
        arm_activity_dir = output_dir / "arm_activity"
        arm_activity_dir.mkdir(parents=True, exist_ok=True)
        df_arm_activity.to_parquet(
            arm_activity_dir / "filtered_gait.parquet", index=False
        )
        active_logger.debug(
            f"Saved filtered gait to {arm_activity_dir / 'filtered_gait.parquet'}"
        )

    # Step 6a: Quantify arm swing (unfiltered - all gait)
    # Always compute unfiltered quantification, even if there's no clean gait
    active_logger.info("Step 6a: Quantifying arm swing (unfiltered)")
    try:
        quantified_arm_swing_unfiltered, gait_segment_meta_unfiltered = (
            quantify_arm_swing(
                df=df_arm_activity,
                filtered=False,  # Quantify all gait
                max_segment_gap_s=arm_activity_config.max_segment_gap_s,
                min_segment_length_s=arm_activity_config.min_segment_length_s,
            )
        )
    except ValueError as exc:
        active_logger.warning(
            "Arm swing quantification (unfiltered) failed (%s). "
            "Returning empty unfiltered arm swing results.",
            exc,
        )
        quantified_arm_swing_unfiltered = _empty_arm_swing_df(df_arm_activity)
        gait_segment_meta_unfiltered = {
            "all": {"duration_s": 0},
            "per_segment": {},
        }

    # Check if there's clean gait for filtered quantification
    if (
        len(
            df_arm_activity.loc[
                df_arm_activity[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY] == 1
            ]
        )
        == 0
    ):
        active_logger.warning("No clean gait data remaining after filtering")
        # Set empty filtered results but continue to save/offset logic
        quantified_arm_swing_filtered = _empty_arm_swing_df(df_arm_activity)
        gait_segment_meta_filtered = {
            "all": {"duration_s": 0},
            "per_segment": {},
        }
    else:
        # Step 6b: Quantify arm swing (filtered - clean gait only)
        active_logger.info("Step 6b: Quantifying arm swing (filtered)")
        quantified_arm_swing_filtered, gait_segment_meta_filtered = quantify_arm_swing(
            df=df_arm_activity,
            filtered=True,  # Quantify clean gait only
            max_segment_gap_s=arm_activity_config.max_segment_gap_s,
            min_segment_length_s=arm_activity_config.min_segment_length_s,
        )

    # Apply segment number offsets for multi-file processing
    if segment_number_offset_unfiltered > 0:
        if DataColumns.GAIT_SEGMENT_NR in quantified_arm_swing_unfiltered.columns:
            quantified_arm_swing_unfiltered = quantified_arm_swing_unfiltered.copy()
            quantified_arm_swing_unfiltered[
                DataColumns.GAIT_SEGMENT_NR
            ] += segment_number_offset_unfiltered

        if (
            gait_segment_meta_unfiltered
            and "per_segment" in gait_segment_meta_unfiltered
            and gait_segment_meta_unfiltered["per_segment"]
        ):
            updated_per_segment_meta = {}
            for seg_id, meta in gait_segment_meta_unfiltered["per_segment"].items():
                new_seg_id = seg_id + segment_number_offset_unfiltered
                updated_per_segment_meta[new_seg_id] = meta
            gait_segment_meta_unfiltered["per_segment"] = updated_per_segment_meta

    if segment_number_offset_filtered > 0:
        if DataColumns.GAIT_SEGMENT_NR in quantified_arm_swing_filtered.columns:
            quantified_arm_swing_filtered = quantified_arm_swing_filtered.copy()
            quantified_arm_swing_filtered[
                DataColumns.GAIT_SEGMENT_NR
            ] += segment_number_offset_filtered

        if (
            gait_segment_meta_filtered
            and "per_segment" in gait_segment_meta_filtered
            and gait_segment_meta_filtered["per_segment"]
        ):
            updated_per_segment_meta = {}
            for seg_id, meta in gait_segment_meta_filtered["per_segment"].items():
                updated_per_segment_meta[seg_id + segment_number_offset_filtered] = meta
            gait_segment_meta_filtered["per_segment"] = updated_per_segment_meta

    if "quantification" in store_intermediate:
        quantification_dir = output_dir / "quantification"
        quantification_dir.mkdir(parents=True, exist_ok=True)

        # Save unfiltered quantification
        quantified_arm_swing_unfiltered.to_parquet(
            quantification_dir / "arm_swing_quantified_unfiltered.parquet", index=False
        )
        with open(quantification_dir / "gait_segment_meta_unfiltered.json", "w") as f:
            json.dump(gait_segment_meta_unfiltered, f, indent=2)

        # Save filtered quantification
        quantified_arm_swing_filtered.to_parquet(
            quantification_dir / "arm_swing_quantified_filtered.parquet", index=False
        )
        with open(quantification_dir / "gait_segment_meta_filtered.json", "w") as f:
            json.dump(gait_segment_meta_filtered, f, indent=2)

        active_logger.debug(
            f"Saved unfiltered quantification to "
            f"{quantification_dir / 'arm_swing_quantified_unfiltered.parquet'}"
        )
        active_logger.debug(
            f"Saved filtered quantification to "
            f"{quantification_dir / 'arm_swing_quantified_filtered.parquet'}"
        )

    active_logger.info(
        f"Gait analysis pipeline completed. Found "
        f"{len(quantified_arm_swing_unfiltered)} unfiltered arm swings and "
        f"{len(quantified_arm_swing_filtered)} filtered arm swings."
    )

    return {
        "filtered": quantified_arm_swing_filtered,
        "unfiltered": quantified_arm_swing_unfiltered,
    }, {
        "filtered": gait_segment_meta_filtered,
        "unfiltered": gait_segment_meta_unfiltered,
    }
