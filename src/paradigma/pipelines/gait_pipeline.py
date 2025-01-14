import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.signal import periodogram
from typing import List, Tuple
import tsdf

from paradigma.classification import ClassifierPackage
from paradigma.constants import DataColumns, TimeUnit
from paradigma.config import GaitFeatureExtractionConfig, GaitDetectionConfig, ArmActivityFeatureExtractionConfig, \
    FilteringGaitConfig
from paradigma.feature_extraction import compute_statistics, compute_std_euclidean_norm, \
    compute_power_in_bandwidth, compute_dominant_frequency, compute_total_power, compute_mfccs, \
    pca_transform_gyroscope, compute_angle, remove_moving_average_angle, \
    extract_angle_extremes, compute_range_of_motion, compute_peak_angular_velocity
from paradigma.segmenting import tabulate_windows, create_segments, discard_segments, categorize_segments, WindowedDataExtractor
from paradigma.util import aggregate_parameter, merge_predictions_with_timestamps, read_metadata, write_df_data, get_end_iso8601


def extract_gait_features(
        df: pd.DataFrame,
        config: GaitFeatureExtractionConfig
    ) -> pd.DataFrame:
    """
    Extracts gait features from accelerometer and gravity sensor data in the input DataFrame by computing temporal and spectral features.

    This function performs the following steps:
    1. Groups sequences of timestamps into windows, using accelerometer and gravity data.
    2. Computes temporal domain features such as mean and standard deviation for accelerometer and gravity data.
    3. Transforms the signals from the temporal domain to the spectral domain using the Fast Fourier Transform (FFT).
    4. Computes spectral domain features for the accelerometer data.
    5. Combines both temporal and spectral features into a final DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing gait data, which includes time, accelerometer, and gravity sensor data. The data should be
        structured with the necessary columns as specified in the `config`.

    config : GaitFeatureExtractionConfig
        Configuration object containing parameters for feature extraction, including column names for time, accelerometer data, and
        gravity data, as well as settings for windowing, and feature computation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing extracted gait features, including temporal and spectral domain features. The DataFrame will have
        columns corresponding to time, statistical features of the accelerometer and gravity data, and spectral features of the
        accelerometer data.
    
    Notes
    -----
    - This function groups the data into windows based on timestamps and applies Fast Fourier Transform to compute spectral features.
    - The temporal features are extracted from the accelerometer and gravity data, and include statistics like mean and standard deviation.
    - The input DataFrame must include columns as specified in the `config` object for proper feature extraction.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the required columns as specified in the configuration or if any step in the feature extraction fails.
    """
    # Group sequences of timestamps into windows
    windowed_cols = [DataColumns.TIME] + config.accelerometer_cols + config.gravity_cols
    windowed_data = tabulate_windows(
        df=df, 
        columns=windowed_cols,
        window_length_s=config.window_length_s,
        window_step_length_s=config.window_step_length_s,
        fs=config.sampling_frequency
    )

    extractor = WindowedDataExtractor(windowed_cols)

    idx_time = extractor.get_index(DataColumns.TIME)
    idx_acc = extractor.get_slice(config.accelerometer_cols)
    idx_grav = extractor.get_slice(config.gravity_cols)

    # Extract data
    start_time = np.min(windowed_data[:, :, idx_time], axis=1)
    windowed_acc = windowed_data[:, :, idx_acc]
    windowed_grav = windowed_data[:, :, idx_grav]

    df_features = pd.DataFrame(start_time, columns=[DataColumns.TIME])
    
    # Compute statistics of the temporal domain signals (mean, std) for accelerometer and gravity
    df_temporal_features = extract_temporal_domain_features(
        config=config, 
        windowed_acc=windowed_acc,
        windowed_grav=windowed_grav,
        grav_stats=['mean', 'std']
    )

    # Combine temporal features with the start time
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # Transform the accelerometer data to the spectral domain using FFT and extract spectral features
    df_spectral_features = extract_spectral_domain_features(
        config=config, 
        sensor='accelerometer', 
        windowed_data=windowed_acc
    )

    # Combine the spectral features with the previously computed temporal features
    df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def extract_gait_features_io(
        config: GaitFeatureExtractionConfig,
        path_to_input: str | Path, 
        path_to_output: str | Path
    ) -> None:
    # Load data
    metadata_time, metadata_values = read_metadata(path_to_input, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Extract gait features
    df_features = extract_gait_features(df=df, config=config)

    # Store data
    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_features[DataColumns.TIME][-1:].values[0] + config.window_length_s))

    metadata_values.file_name = 'gait_values.bin'
    metadata_time.file_name = 'gait_time.bin'
    metadata_values.end_iso8601 = end_iso8601
    metadata_time.end_iso8601 = end_iso8601
    
    metadata_values.channels = list(config.d_channels_values.keys())
    metadata_values.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = [TimeUnit.RELATIVE_S]

    write_df_data(metadata_time, metadata_values, path_to_output, 'gait_meta.json', df_features)


def detect_gait(
        df: pd.DataFrame, 
        clf_package: ClassifierPackage, 
        parallel: bool=False
    ) -> pd.Series:
    """
    Detects gait activity in the input DataFrame using a pre-trained classifier and applies a threshold to classify results.

    This function performs the following steps:
    1. Loads the pre-trained classifier and scaling parameters from the specified directory.
    2. Scales the relevant features in the input DataFrame (`df`) using the loaded scaling parameters.
    3. Predicts the probability of gait activity for each sample in the DataFrame using the classifier.
    4. Applies a threshold to the predicted probabilities to determine whether gait activity is present.
    5. Returns predicted probabilities

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features extracted from gait data. It must include the necessary columns 
        as specified in the classifier's feature names.

    clf_package : ClassifierPackage
        The pre-trained classifier package containing the classifier, threshold, and scaler.

    parallel : bool, optional, default=False
        If `True`, enables parallel processing during classification. If `False`, the classifier uses a single core.

    Returns
    -------
    pd.Series
        A Series containing the predicted probabilities of gait activity for each sample in the input DataFrame.
    """
    # Set classifier
    clf = clf_package.classifier
    if not parallel and hasattr(clf, 'n_jobs'):
        clf.n_jobs = 1

    feature_names_scaling = clf_package.scaler.feature_names_in_
    feature_names_predictions = clf.feature_names_in_

    # Apply scaling to relevant columns
    scaled_features = clf_package.transform_features(df.loc[:, feature_names_scaling])

    # Replace scaled features in a copy of the relevant features for prediction
    X = df.loc[:, feature_names_predictions].copy()
    X.loc[:, feature_names_scaling] = scaled_features

    # Make prediction and add the probability of gait activity to the DataFrame
    pred_gait_proba_series = clf_package.predict_proba(X)

    return pred_gait_proba_series


def detect_gait_io(
        config: GaitDetectionConfig, 
        path_to_input: str | Path, 
        path_to_output: str | Path, 
        full_path_to_classifier_package: str | Path, 
    ) -> None:
    
    # Load the data
    metadata_time, metadata_values = read_metadata(path_to_input, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    clf_package = ClassifierPackage.load(full_path_to_classifier_package)

    df[DataColumns.PRED_GAIT_PROBA] = detect_gait(
        df=df, 
        clf_package=clf_package
    )

    # Prepare the metadata
    metadata_values.file_name = 'gait_values.bin'
    metadata_time.file_name = 'gait_time.bin'

    metadata_values.channels = [DataColumns.PRED_GAIT_PROBA]
    metadata_values.units = ['probability']

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = [TimeUnit.RELATIVE_S]

    write_df_data(metadata_time, metadata_values, path_to_output, 'gait_meta.json', df)


def extract_arm_activity_features(
        config: ArmActivityFeatureExtractionConfig,
        df_timestamps: pd.DataFrame, 
        df_predictions: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
    """
    Extract features related to arm activity from a time-series DataFrame.

    This function processes a DataFrame containing accelerometer, gravity, and gyroscope signals, 
    and extracts features related to arm activity by performing the following steps:
    1. Merges the gait predictions with timestamps by expanding overlapping windows into individual timestamps.
    2. Computes the angle and velocity from gyroscope data.
    3. Filters the data to include only predicted gait segments.
    4. Groups the data into segments based on consecutive timestamps and pre-specified gaps.
    5. Removes segments that do not meet predefined criteria.
    6. Creates fixed-length windows from the time series data.
    7. Extracts angle-related features, temporal domain features, and spectral domain features.

    Parameters
    ----------
    config : ArmActivityFeatureExtractionConfig
        Configuration object containing column names and parameters for feature extraction.

    df_timestamps : pd.DataFrame
        A DataFrame containing the raw sensor data, including accelerometer, gravity, and gyroscope columns.

    df_predictions : pd.DataFrame
        A DataFrame containing the predicted probabilities for gait activity per window.

    path_to_classifier_input : str | Path
        The path to the directory containing the classifier files and other necessary input files for feature extraction.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted arm activity features, including angle, velocity, 
        temporal, and spectral features.
    """
    if not any(df_predictions[DataColumns.PRED_GAIT_PROBA] >= threshold):
        raise ValueError("No gait detected in the input data.")
    
    # Merge gait predictions with timestamps
    gait_preprocessing_config = GaitFeatureExtractionConfig()
    df = merge_predictions_with_timestamps(
        df_ts=df_timestamps, 
        df_predictions=df_predictions, 
        pred_proba_colname=DataColumns.PRED_GAIT_PROBA,
        window_length_s=gait_preprocessing_config.window_length_s,
        fs=gait_preprocessing_config.sampling_frequency
    )
    
    # Add a column for predicted gait based on a fitted threshold
    df[DataColumns.PRED_GAIT] = (df[DataColumns.PRED_GAIT_PROBA] >= threshold).astype(int)

    # Filter the DataFrame to only include predicted gait (1)
    df = df.loc[df[DataColumns.PRED_GAIT]==1].reset_index(drop=True)

    # Group consecutive timestamps into segments, with new segments starting after a pre-specified gap
    df[DataColumns.SEGMENT_NR] = create_segments(
        time_array=df[DataColumns.TIME], 
        max_segment_gap_s=config.max_segment_gap_s
    )

    # Remove segments that do not meet predetermined criteria
    df = discard_segments(
        df=df,
        segment_nr_colname=DataColumns.SEGMENT_NR,
        min_segment_length_s=config.min_segment_length_s,
        fs=config.sampling_frequency,
        format='timestamps'
    )

    # Create windows of fixed length and step size from the time series per segment
    windowed_data = []
    df_grouped = df.groupby(DataColumns.SEGMENT_NR)
    windowed_cols = (
        [DataColumns.TIME] + 
        config.accelerometer_cols + 
        config.gravity_cols + 
        config.gyroscope_cols
    )

    # Collect windows from all segments in a list for faster concatenation
    for _, group in df_grouped:
        windows = tabulate_windows(
            df=group, 
            columns=windowed_cols,
            window_length_s=config.window_length_s,
            window_step_length_s=config.window_step_length_s,
            fs=config.sampling_frequency
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
    extractor = WindowedDataExtractor(windowed_cols)

    idx_time = extractor.get_index(DataColumns.TIME)
    idx_acc = extractor.get_slice(config.accelerometer_cols)
    idx_grav = extractor.get_slice(config.gravity_cols)
    idx_gyro = extractor.get_slice(config.gyroscope_cols)

    # Extract data
    start_time = np.min(windowed_data[:, :, idx_time], axis=1)
    windowed_acc = windowed_data[:, :, idx_acc]
    windowed_grav = windowed_data[:, :, idx_grav]
    windowed_gyro = windowed_data[:, :, idx_gyro]

    # Initialize DataFrame for features
    df_features = pd.DataFrame(start_time, columns=[DataColumns.TIME])

    # Extract temporal domain features (e.g., mean, std for accelerometer and gravity)
    df_temporal_features = extract_temporal_domain_features(
        config=config, 
        windowed_acc=windowed_acc, 
        windowed_grav=windowed_grav, 
        grav_stats=['mean', 'std']
    )
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # Extract spectral domain features for accelerometer and gyroscope signals
    for sensor_name, windowed_sensor in zip(['accelerometer', 'gyroscope'], [windowed_acc, windowed_gyro]):
        df_spectral_features = extract_spectral_domain_features(
            config=config, 
            sensor=sensor_name, 
            windowed_data=windowed_sensor
        )
        df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def extract_arm_activity_features_io(
        config: ArmActivityFeatureExtractionConfig, 
        path_to_timestamp_input: str | Path, 
        path_to_prediction_input: str | Path, 
        full_path_to_classifier_package: str | Path, 
        path_to_output: str | Path
    ) -> None:
    # Load accelerometer and gyroscope data
    dfs = []
    for sensor in ['accelerometer', 'gyroscope']:
        config.set_sensor(sensor)
        meta_ts_filename = f'{sensor}_meta.json'
        values_ts_filename = f'{sensor}_values.bin'
        time_ts_filename = f'{sensor}_time.bin'

        metadata_ts_dict = tsdf.load_metadata_from_path(os.path.join(path_to_timestamp_input, meta_ts_filename))
        metadata_ts_time = metadata_ts_dict[time_ts_filename]
        metadata_ts_values = metadata_ts_dict[values_ts_filename]
        dfs.append(tsdf.load_dataframe_from_binaries([metadata_ts_time, metadata_ts_values], tsdf.constants.ConcatenationType.columns))

    df_ts = pd.merge(dfs[0], dfs[1], on=DataColumns.TIME)

    # Load gait predictions
    meta_pred_filename = 'gait_meta.json'
    values_pred_filename = 'gait_values.bin'
    time_pred_filename = 'gait_time.bin'

    metadata_pred_dict = tsdf.load_metadata_from_path(os.path.join(path_to_prediction_input, meta_pred_filename))
    metadata_pred_time = metadata_pred_dict[time_pred_filename]
    metadata_pred_values = metadata_pred_dict[values_pred_filename]

    df_pred_gait = tsdf.load_dataframe_from_binaries([metadata_pred_time, metadata_pred_values], tsdf.constants.ConcatenationType.columns)

    clf_package = ClassifierPackage.load(full_path_to_classifier_package)

    # Extract arm activity features
    df_features = extract_arm_activity_features(
        config=config,
        df_timestamps=df_ts, 
        df_predictions=df_pred_gait, 
        threshold=clf_package.threshold
    )

    end_iso8601 = get_end_iso8601(metadata_ts_values.start_iso8601, df_features[DataColumns.TIME][-1:].values[0] + config.window_length_s)

    metadata_ts_values.end_iso8601 = end_iso8601
    metadata_ts_values.file_name = 'arm_activity_values.bin'
    metadata_ts_time.end_iso8601 = end_iso8601
    metadata_ts_time.file_name = 'arm_activity_time.bin'

    metadata_ts_values.channels = list(config.d_channels_values.keys())
    metadata_ts_values.units = list(config.d_channels_values.values())

    metadata_ts_time.channels = [DataColumns.TIME]
    metadata_ts_time.units = [TimeUnit.RELATIVE_S]

    write_df_data(metadata_ts_time, metadata_ts_values, path_to_output, 'arm_activity_meta.json', df_features)


def filter_gait(
        df: pd.DataFrame, 
        clf_package: ClassifierPackage, 
        parallel: bool=False
    ) -> pd.Series:
    """
    Filters gait data to identify windows with no other arm activity using a pre-trained classifier.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features extracted from gait data.
    full_path_to_classifier_package : str | Path
        The path to the pre-trained classifier file.
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
    if not parallel and hasattr(clf, 'n_jobs'):
        clf.n_jobs = 1

    feature_names_scaling = clf_package.scaler.feature_names_in_
    feature_names_predictions = clf.feature_names_in_

    # Apply scaling to relevant columns
    scaled_features = clf_package.transform_features(df.loc[:, feature_names_scaling])

    # Replace scaled features in a copy of the relevant features for prediction
    X = df.loc[:, feature_names_predictions].copy()
    X.loc[:, feature_names_scaling] = scaled_features

    # Make predictions
    pred_no_other_arm_activity_proba_series = clf_package.predict_proba(X)

    return pred_no_other_arm_activity_proba_series


def filter_gait_io(
        config: FilteringGaitConfig, 
        path_to_input: str | Path, 
        path_to_output: str | Path, 
        full_path_to_classifier_package: str | Path, 
    ) -> None:
    # Load the data
    metadata_time, metadata_values = read_metadata(path_to_input, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    clf_package = ClassifierPackage.load(filepath=full_path_to_classifier_package)

    df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = filter_gait(
        df=df, 
        clf_package=clf_package
    )

    # Prepare the metadata
    metadata_values.file_name = 'arm_activity_values.bin'
    metadata_time.file_name = 'arm_activity_time.bin'

    metadata_values.channels = [DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]
    metadata_values.units = ['probability']

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = [TimeUnit.RELATIVE_S]

    write_df_data(metadata_time, metadata_values, path_to_output, 'arm_activity_meta.json', df)


def quantify_arm_swing(
        df_timestamps: pd.DataFrame, 
        df_predictions: pd.DataFrame, 
        classification_threshold: float, 
        window_length_s: float,
        max_segment_gap_s: float, 
        min_segment_length_s: float,
        fs: int,
        dfs_to_quantify: List[str] | str = ['unfiltered', 'filtered'],
    ) -> Tuple[dict[str, pd.DataFrame], dict]:
    """
    Quantify arm swing parameters for segments of motion based on gyroscope data.

    Parameters
    ----------
    df_timestamps : pd.DataFrame
        A DataFrame containing the raw sensor data, including gyroscope columns.

    df_predictions : pd.DataFrame
        A DataFrame containing the predicted probabilities for no other arm activity per window.

    classification_threshold : float
        The threshold used to classify no other arm activity based on the predicted probabilities.

    window_length_s : float
        The length of the window used for feature extraction.

    max_segment_gap_s : float
        The maximum gap allowed between segments.

    min_segment_length_s : float
        The minimum length required for a segment to be considered valid.

    fs : int
        The sampling frequency of the sensor data.

    dfs_to_quantify : List[str] | str, optional
        The DataFrames to quantify arm swing parameters for. Options are 'unfiltered' and 'filtered', with 'unfiltered' being predicted gait, and 
        'filtered' being predicted gait without other arm activities.

    Returns
    -------
    Tuple[dict, dict]
        A tuple containing a dictionary with quantified arm swing parameters for dfs_to_quantify, 
        and a dictionary containing metadata for each segment.
    """
    if not any(df_predictions[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= classification_threshold):
        raise ValueError("No gait without other arm activity detected in the input data.")
    
    if isinstance(dfs_to_quantify, str):
        dfs_to_quantify = [dfs_to_quantify]
    elif not isinstance(dfs_to_quantify, list):
        raise ValueError("dfs_to_quantify must be either 'unfiltered', 'filtered', or a list containing both.")

    valid_values = {'unfiltered', 'filtered'}
    if set(dfs_to_quantify) - valid_values:
        raise ValueError(
            f"Invalid value in dfs_to_quantify: {dfs_to_quantify}. "
            f"Valid options are 'unfiltered', 'filtered', or both in a list."
        ) 
    
    # Merge arm activity predictions with timestamps
    df = merge_predictions_with_timestamps(
        df_ts=df_timestamps, 
        df_predictions=df_predictions, 
        pred_proba_colname=DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA,
        window_length_s=window_length_s, 
        fs=fs
    )

    # Add a column for predicted no other arm activity based on a fitted threshold
    df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY] = (
        df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= classification_threshold
    ).astype(int)

    # Group consecutive timestamps into segments, with new segments starting after a pre-specified gap
    # Segments are made based on predicted gait
    df[DataColumns.SEGMENT_NR] = create_segments(
        time_array=df[DataColumns.TIME], 
        max_segment_gap_s=max_segment_gap_s
    )

    # Remove segments that do not meet predetermined criteria
    df = discard_segments(
        df=df,
        segment_nr_colname=DataColumns.SEGMENT_NR,
        min_segment_length_s=min_segment_length_s,
        fs=fs,
        format='timestamps'
    )

    if df.empty:
        raise ValueError("No segments found in the input data.")

    # If no arm swing data is remaining, return an empty dictionary
    if df.loc[df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY]==1].empty:
        
        if 'filtered' in dfs_to_quantify and len(dfs_to_quantify) == 1:
            raise ValueError("No gait without other arm activities to quantify.")
        
        dfs_to_quantify = [x for x in dfs_to_quantify if x != 'filtered']

    df[DataColumns.SEGMENT_CAT] = categorize_segments(
        df=df,
        fs=fs
    )

    df[DataColumns.VELOCITY] = pca_transform_gyroscope(
        df=df,
        y_gyro_colname=DataColumns.GYROSCOPE_Y,
        z_gyro_colname=DataColumns.GYROSCOPE_Z,
        pred_colname=DataColumns.PRED_NO_OTHER_ARM_ACTIVITY
    )

    # Group and process segments
    arm_swing_quantified = {}
    segment_meta = {}

    # If both unfiltered and filtered gait are to be quantified, start with the unfiltered data
    # and subset to get filtered data afterwards.
    dfs_to_quantify = sorted(dfs_to_quantify)

    for df_name in dfs_to_quantify:    
        if df_name == 'filtered':
            # Filter the DataFrame to only include predicted no other arm activity (1)
            df = df.loc[df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY]==1].reset_index(drop=True)

            # Group consecutive timestamps into segments, with new segments starting after a pre-specified gap
            # Now segments are based on predicted gait without other arm activity for subsequent processes
            df[DataColumns.SEGMENT_NR] = create_segments(
                time_array=df[DataColumns.TIME], 
                max_segment_gap_s=max_segment_gap_s
            )

        arm_swing_quantified[df_name] = []
        segment_meta[df_name] = {}

        for segment_nr, group in df.groupby(DataColumns.SEGMENT_NR, sort=False):
            segment_cat = group[DataColumns.SEGMENT_CAT].iloc[0]
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

            segment_meta[df_name][segment_nr] = {
                'time_s': len(angle_array) / fs,
                DataColumns.SEGMENT_CAT: segment_cat
            }

            if angle_array.size > 0:  
                angle_extrema_indices, _, _ = extract_angle_extremes(
                    angle_array=angle_array,
                    sampling_frequency=fs,
                    max_frequency_activity=1.75
                )

                if len(angle_extrema_indices) > 1:  # Requires at minimum 2 peaks
                    try:
                        rom = compute_range_of_motion(
                            angle_array=angle_array,
                            extrema_indices=angle_extrema_indices,
                        )
                    except Exception as e:
                        # Handle the error, set RoM to NaN, and log the error
                        print(f"Error computing range of motion for segment {segment_nr}: {e}")
                        rom = np.array([np.nan])

                    try:
                        pav = compute_peak_angular_velocity(
                            velocity_array=velocity_array,
                            angle_extrema_indices=angle_extrema_indices
                        )
                    except Exception as e:
                        # Handle the error, set pav to NaN, and log the error
                        print(f"Error computing peak angular velocity for segment {segment_nr}: {e}")
                        pav = np.array([np.nan])

                    df_params_segment = pd.DataFrame({
                        DataColumns.SEGMENT_NR: segment_nr,
                        DataColumns.RANGE_OF_MOTION: rom,
                        DataColumns.PEAK_VELOCITY: pav
                    })

                    arm_swing_quantified[df_name].append(df_params_segment)

        arm_swing_quantified[df_name] = pd.concat(arm_swing_quantified[df_name], ignore_index=True)
            
    return {df_name: arm_swing_quantified[df_name] for df_name in dfs_to_quantify}, segment_meta


def aggregate_arm_swing_params(df_arm_swing_params: pd.DataFrame, segment_meta: dict, aggregates: List[str] = ['median']) -> dict:
    """
    Aggregate the quantification results for arm swing parameters.
    
    Parameters
    ----------
    df_arm_swing_params : pd.DataFrame
        A dataframe containing the arm swing parameters to be aggregated

    segment_meta : dict
        A dictionary containing metadata for each segment.
        
    aggregates : List[str], optional
        A list of aggregation methods to apply to the quantification results.
        
    Returns
    -------
    dict
        A dictionary containing the aggregated quantification results for arm swing parameters.
    """
    uq_segment_cats = set([segment_meta[x][DataColumns.SEGMENT_CAT] for x in df_arm_swing_params[DataColumns.SEGMENT_NR].unique()])

    aggregated_results = {}
    for segment_cat in uq_segment_cats:
        cat_segments = [x for x in segment_meta.keys() if segment_meta[x][DataColumns.SEGMENT_CAT] == segment_cat]

        aggregated_results[segment_cat] = {
            'time_s': sum([segment_meta[x]['time_s'] for x in cat_segments])
        }

        df_arm_swing_params_cat = df_arm_swing_params[df_arm_swing_params[DataColumns.SEGMENT_NR].isin(cat_segments)]
        
        for aggregate in aggregates:
            aggregated_results[segment_cat][f'{aggregate}_{DataColumns.RANGE_OF_MOTION}'] = aggregate_parameter(df_arm_swing_params_cat[DataColumns.RANGE_OF_MOTION], aggregate)
            aggregated_results[segment_cat][f'{aggregate}_forward_{DataColumns.PEAK_VELOCITY}'] = aggregate_parameter(df_arm_swing_params_cat[DataColumns.PEAK_VELOCITY], aggregate)

    return aggregated_results


def extract_temporal_domain_features(
        config, 
        windowed_acc: np.ndarray, 
        windowed_grav: np.ndarray, 
        grav_stats: List[str] = ['mean']
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
        for i, col in enumerate(config.gravity_cols):
            feature_dict[f'{col}_{stat}'] = stats_result[:, i]

    # Compute standard deviation of the Euclidean norm of the accelerometer signal
    feature_dict['accelerometer_std_norm'] = compute_std_euclidean_norm(data=windowed_acc)

    return pd.DataFrame(feature_dict)


def extract_spectral_domain_features(
        config, 
        sensor: str, 
        windowed_data: np.ndarray
    ) -> pd.DataFrame:
    """
    Compute spectral domain features for a sensor's data.

    This function computes the periodogram, extracts power in specific frequency bands, 
    calculates the dominant frequency, and computes Mel-frequency cepstral coefficients (MFCCs) 
    for a given sensor's windowed data.

    Parameters
    ----------
    config : object
        Configuration object containing settings such as sampling frequency, window type, 
        frequency bands, and MFCC parameters.
    sensor : str
        The name of the sensor (e.g., 'accelerometer', 'gyroscope').
    windowed_data : numpy.ndarray
        A 2D numpy array where each row corresponds to a window of sensor data.

    Returns
    -------
    dict
        The updated feature dictionary containing the extracted spectral features, including 
        power in frequency bands, dominant frequencies, and MFCCs for each window.
    """
    # Initialize a dictionary to hold the results
    feature_dict = {}

    # Compute periodogram (power spectral density)
    freqs, psd = periodogram(
        x=windowed_data, 
        fs=config.sampling_frequency, 
        window=config.window_type, 
        axis=1
    )

    # Compute power in specified frequency bands
    for band_name, band_freqs in config.d_frequency_bandwidths.items():
        band_powers = compute_power_in_bandwidth(
            freqs=freqs,
            psd=psd, 
            fmin=band_freqs[0],
            fmax=band_freqs[1],
        )
        for i, col in enumerate(config.axes):
            feature_dict[f'{sensor}_{col}_{band_name}'] = band_powers[:, i]

    # Compute dominant frequency for each axis
    dominant_frequencies = compute_dominant_frequency(
        freqs=freqs, 
        psd=psd, 
        fmin=config.spectrum_low_frequency, 
        fmax=config.spectrum_high_frequency
    )

    # Add dominant frequency features to the feature_dict
    for axis, freq in zip(config.axes, dominant_frequencies.T):
        feature_dict[f'{sensor}_{axis}_dominant_frequency'] = freq

    # Compute total power in the PSD
    total_power_psd = compute_total_power(psd)

    # Compute MFCCs
    mfccs = compute_mfccs(
        total_power_array=total_power_psd,
        config=config,
        multiplication_factor=4
    )

    # Combine the MFCCs into the features DataFrame
    mfcc_colnames = [f'{sensor}_mfcc_{x}' for x in range(1, config.mfcc_n_coefficients + 1)]
    for i, colname in enumerate(mfcc_colnames):
        feature_dict[colname] = mfccs[:, i]

    return pd.DataFrame(feature_dict)