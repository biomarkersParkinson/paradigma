import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.preprocessing import StandardScaler

import tsdf

from paradigma.constants import DataColumns
from paradigma.gait.gait_analysis_config import GaitFeatureExtractionConfig, GaitDetectionConfig, \
    ArmActivityFeatureExtractionConfig, FilteringGaitConfig
from paradigma.gait.feature_extraction import extract_temporal_domain_features, \
    extract_spectral_domain_features, compute_angle_and_velocity_from_gyro, extract_angle_features
from paradigma.segmenting import tabulate_windows, create_segments, discard_segments
from paradigma.util import get_end_iso8601, write_df_data, read_metadata


def extract_gait_features(df: pd.DataFrame, config: GaitFeatureExtractionConfig) -> pd.DataFrame:
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
    window_cols = [config.time_colname] + config.accelerometer_cols + config.gravity_cols
    data_windowed = tabulate_windows(config, df, window_cols)

    idx_time = window_cols.index(config.time_colname)
    idx_acc = slice(1, 4)
    idx_grav = slice(4, 7)

    # Extract the start time, accelerometer, and gravity data from the windowed data
    start_time = np.min(data_windowed[:, :, idx_time], axis=1)
    accel_windowed = data_windowed[:, :, idx_acc]
    grav_windowed = data_windowed[:, :, idx_grav]

    df_features = pd.DataFrame(start_time, columns=[config.time_colname])
    
    # Compute statistics of the temporal domain signals (mean, std) for accelerometer and gravity
    df_temporal_features = extract_temporal_domain_features(
        config=config, 
        windowed_acc=accel_windowed,
        windowed_grav=grav_windowed,
        grav_stats=['mean', 'std']
    )

    # Combine temporal features with the start time
    df_features= pd.concat([df_features, df_temporal_features], axis=1)

    # Transform the accelerometer data to the spectral domain using FFT and extract spectral features
    df_spectral_features = extract_spectral_domain_features(
        config=config, 
        sensor='accelerometer', 
        windowed_data=accel_windowed
    )

    # Combine the spectral features with the previously computed temporal features
    df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def extract_gait_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: GaitFeatureExtractionConfig) -> None:
    # Load data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Extract gait features
    df_features = extract_gait_features(df, config)

    # Store data
    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_features[config.time_colname][-1:].values[0] + config.window_length_s))

    metadata_samples.file_name = 'gait_values.bin'
    metadata_time.file_name = 'gait_time.bin'
    metadata_samples.end_iso8601 = end_iso8601
    metadata_time.end_iso8601 = end_iso8601
    
    metadata_samples.channels = list(config.d_channels_values.keys())
    metadata_samples.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'gait_meta.json', df_features)


def detect_gait(df: pd.DataFrame, config: GaitDetectionConfig, path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:
    """
    Detects gait activity in the input DataFrame using a pre-trained classifier and applies a threshold to the predicted probabilities.

    This function performs the following steps:
    1. Loads the pre-trained classifier and scaling parameters from the provided directory.
    2. Scales the relevant features in the input DataFrame (`df`) using the loaded scaling parameters.
    3. Makes predictions using the classifier to estimate the probability of gait activity.
    4. Applies a threshold to the predicted probabilities to classify whether gait activity is detected or not.
    5. Adds the predicted probabilities and the classification result to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features extracted from gait data. The DataFrame must include
        the necessary columns as specified in the classifier's feature names.

    config : GaitDetectionConfig
        Configuration object containing the classifier file name, threshold file name, and other settings for gait detection.

    path_to_classifier_input : Union[str, Path]
        The path to the directory containing the classifier file, threshold value, scaler parameters, and other necessary input
        files for gait detection.

    Returns
    -------
    pd.DataFrame
        The input DataFrame (`df`) with two additional columns:
        - `PRED_GAIT_PROBA`: Predicted probability of gait activity based on the classifier.
        - `PRED_GAIT`: Binary classification result (True for gait, False for no gait), based on the threshold applied to `PRED_GAIT_PROBA`.
    
    Notes
    -----
    - The classifier should output probabilities for gait activity, which are used to filter the gait data.
    - The threshold used to classify gait activity is loaded from a file and applied to the predicted probabilities.

    Raises
    ------
    FileNotFoundError
        If the classifier, scaler, or threshold files are not found at the specified paths.
    ValueError
        If the DataFrame does not contain the expected features for prediction or if the prediction fails.
    """
    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', config.classifier_file_name))

    # Load and apply scaling parameters
    with open(os.path.join(path_to_classifier_input, 'scalers', 'gait_detection_scaler_params.json'), 'r') as f:
        scaler_params = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.var_ = scaler_params['var']
    scaler.scale_ = scaler_params['scale']
    scaler.feature_names_in_ = scaler_params['features']

    # Scale the features in the DataFrame
    df[scaler_params['features']] = scaler.transform(df[scaler_params['features']])

    # Extract the relevant features for prediction
    X = df.loc[:, clf.feature_names_in_]

    # Make prediction and add the probability of gait activity to the DataFrame
    df[DataColumns.PRED_GAIT_PROBA] = clf.predict_proba(X)[:, 1]

    return df


def detect_gait_io(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: GaitDetectionConfig) -> None:
    
    # Load the data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    df = detect_gait(df, config, path_to_classifier_input)

    # Prepare the metadata
    metadata_samples.file_name = 'gait_values.bin'
    metadata_time.file_name = 'gait_time.bin'

    metadata_samples.channels = ['pred_gait_proba']
    metadata_samples.units = ['probability']

    metadata_time.channels = [config.time_colname]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'gait_meta.json', df)


def extract_arm_activity_features(
        df_timestamps: pd.DataFrame, 
        df_predictions: pd.DataFrame,
        config: ArmActivityFeatureExtractionConfig,
        input_path_classifiers: Union[str, Path],
    ) -> pd.DataFrame:
    """
    Extract features related to arm activity from a time-series DataFrame.

    This function processes a DataFrame containing accelerometer, gravity, and gyroscope signals, 
    and extracts features related to arm activity by performing the following steps:
    1. Adds a "random" prediction of gait.
    2. Computes the angle and velocity from gyroscope data.
    3. Filters the data to include only predicted gait segments.
    4. Groups the data into segments based on consecutive timestamps and pre-specified gaps.
    5. Removes segments that do not meet predefined criteria.
    6. Creates fixed-length windows from the time series data.
    7. Extracts angle-related features, temporal domain features, and spectral domain features.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the raw sensor data, including accelerometer, gravity, and gyroscope columns.
    
    config : ArmActivityFeatureExtractionConfig
        Configuration object containing column names and parameters for feature extraction.

    classification_threshold : float
        The threshold value used to classify a window as gait based on the predicted probabilities.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted arm activity features, including angle, velocity, 
        temporal, and spectral features.
    """

    # Load classification threshold
    gait_detection_config = GaitDetectionConfig()
    with open(os.path.join(input_path_classifiers, 'thresholds', gait_detection_config.thresholds_file_name), 'r') as f:
        classification_threshold = float(f.read())

    # Merge gait predictions with timestamps
    gait_preprocessing_config = GaitFeatureExtractionConfig()
    df = merge_predictions_with_timestamps(df_timestamps, df_predictions, gait_preprocessing_config)

    # Add a column for predicted gait based on a fitted threshold
    df[config.pred_gait_colname] = (df[config.pred_gait_proba_colname] >= classification_threshold).astype(int)
    
    # Compute angle and velocity from gyroscope data
    df[config.angle_colname], df[config.velocity_colname] = compute_angle_and_velocity_from_gyro(config, df)

    # Filter the DataFrame to only include predicted gait (1)
    df = df.loc[df[config.pred_gait_colname]==1].reset_index(drop=True)

    # Group consecutive timestamps into segments, with new segments starting after a pre-specified gap
    df[config.segment_nr_colname] = create_segments(config=config, df=df)

    # Remove segments that do not meet predetermined criteria
    df = discard_segments(config=config, df=df)

    # Create windows of fixed length and step size from the time series per segment
    windowed_data = []
    df_grouped = df.groupby(config.segment_nr_colname)
    windowed_cols = (
        [config.time_colname] + 
        config.accelerometer_cols + 
        config.gravity_cols + 
        config.gyroscope_cols + 
        [config.angle_colname, config.velocity_colname]
    )

    # Collect windows from all segments in a list for faster concatenation
    for _, group in df_grouped:
        windows = tabulate_windows(config=config, df=group, columns=windowed_cols)
        if len(windows) > 0:  # Skip if no windows are created
            windowed_data.append(windows)

    # If no windows were created, raise an error
    if not windowed_data:
        raise ValueError("No windows were created from the given data.")

    # Concatenate the windows into one array at the end
    windowed_data = np.concatenate(windowed_data, axis=0)

    # Slice columns for accelerometer, gravity, gyroscope, angle, and velocity
    n_acc_cols = len(config.accelerometer_cols)
    n_grav_cols = len(config.gravity_cols)
    n_gyro_cols = len(config.gyroscope_cols)

    idx_time = windowed_cols.index(config.time_colname)
    idx_acc = slice(0, n_acc_cols)
    idx_grav = slice(n_acc_cols, n_acc_cols + n_grav_cols)
    idx_gyro = slice(n_acc_cols + n_grav_cols, n_acc_cols + n_grav_cols + n_gyro_cols)
    idx_angle = windowed_cols.index(config.angle_colname)
    idx_velocity = windowed_cols.index(config.velocity_colname)

    # Extract windows for accelerometer, gravity, gyroscope, angle, and velocity
    start_time = np.min(windowed_data[:, :, idx_time], axis=1)
    windowed_acc = windowed_data[:, :, idx_acc]
    windowed_grav = windowed_data[:, :, idx_grav]
    windowed_gyro = windowed_data[:, :, idx_gyro]
    windowed_angle = windowed_data[:, :, idx_angle]
    windowed_velocity = windowed_data[:, :, idx_velocity]

    # Initialize DataFrame for features
    df_features = pd.DataFrame(start_time, columns=[config.time_colname])

    # Extract angle features
    df_angle_features = extract_angle_features(config, windowed_angle, windowed_velocity)    
    df_features = pd.concat([df_features, df_angle_features], axis=1)

    # Extract temporal domain features (e.g., mean, std for accelerometer and gravity)
    df_temporal_features = extract_temporal_domain_features(config, windowed_acc, windowed_grav, grav_stats=['mean', 'std'])
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # Extract spectral domain features for accelerometer and gyroscope signals
    for sensor_name, windowed_sensor in zip(['accelerometer', 'gyroscope'], [windowed_acc, windowed_gyro]):
        df_spectral_features = extract_spectral_domain_features(config, sensor_name, windowed_sensor)
        df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def extract_arm_activity_features_io(input_path_timestamps: Union[str, Path], input_path_predictions: Union[str, Path], input_path_classifiers: Union[str, Path], output_path: Union[str, Path], config: ArmActivityFeatureExtractionConfig) -> None:
    # Load accelerometer and gyroscope data
    dfs = []
    for sensor in ['accelerometer', 'gyroscope']:
        config.set_sensor(sensor)
        meta_ts_filename = f'{sensor}_meta.json'
        values_ts_filename = f'{sensor}_samples.bin'
        time_ts_filename = f'{sensor}_time.bin'

        metadata_ts_dict = tsdf.load_metadata_from_path(os.path.join(input_path_timestamps, meta_ts_filename))
        metadata_ts_time = metadata_ts_dict[time_ts_filename]
        metadata_ts_samples = metadata_ts_dict[values_ts_filename]
        dfs.append(tsdf.load_dataframe_from_binaries([metadata_ts_time, metadata_ts_samples], tsdf.constants.ConcatenationType.columns))

    df_ts = pd.merge(dfs[0], dfs[1], on=config.time_colname)

    # Load gait predictions
    meta_pred_filename = 'gait_meta.json'
    values_pred_filename = 'gait_values.bin'
    time_pred_filename = 'gait_time.bin'

    metadata_pred_dict = tsdf.load_metadata_from_path(os.path.join(input_path_predictions, meta_pred_filename))
    metadata_pred_time = metadata_pred_dict[time_pred_filename]
    metadata_pred_samples = metadata_pred_dict[values_pred_filename]

    df_pred_gait = tsdf.load_dataframe_from_binaries([metadata_pred_time, metadata_pred_samples], tsdf.constants.ConcatenationType.columns)

    # Extract arm activity features
    df_features = extract_arm_activity_features(df_ts, df_pred_gait, config, input_path_classifiers)

    end_iso8601 = get_end_iso8601(metadata_ts_samples.start_iso8601, 
                                df_features[config.time_colname][-1:].values[0] + config.window_length_s)

    metadata_ts_samples.end_iso8601 = end_iso8601
    metadata_ts_samples.file_name = 'arm_activity_values.bin'
    metadata_ts_time.end_iso8601 = end_iso8601
    metadata_ts_time.file_name = 'arm_activity_time.bin'

    metadata_ts_samples.channels = list(config.d_channels_values.keys())
    metadata_ts_samples.units = list(config.d_channels_values.values())

    metadata_ts_time.channels = [config.time_colname]
    metadata_ts_time.units = ['relative_time_ms']

    write_df_data(metadata_ts_time, metadata_ts_samples, output_path, 'arm_activity_meta.json', df_features)


def filter_gait(df: pd.DataFrame, config: FilteringGaitConfig, path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:
    """
    Filters gait data using a pre-trained classifier and scales the features before making predictions.

    This function performs the following steps:
    1. Loads the pre-trained classifier and feature scaling parameters from the provided directory.
    2. Scales the relevant features in the input DataFrame (`df`) using the loaded scaling parameters.
    3. Makes predictions using the classifier to filter gait data based on the model's classification.
    4. Adds the predicted probabilities (for gait activity) to the DataFrame as a new column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features extracted from gait data. The DataFrame must include
        the necessary columns as specified in the classifier's feature names.

    config : FilteringGaitConfig
        Configuration object containing the classifier file name and other settings for gait filtering.

    path_to_classifier_input : Union[str, Path]
        The path to the directory containing the classifier file, scaler parameters, and other necessary input
        files for filtering gait activity.

    Returns
    -------
    pd.DataFrame
        The input DataFrame (`df`) with an additional column containing the predicted probabilities of gait activity
        based on the classifier's output. The new column is labeled according to `DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA`.
    
    Notes
    -----
    - The function expects the pre-trained classifier and scaling parameters to be located at the specified paths.
    - The classifier should output probabilities for gait activity, which are used to filter the gait data.

    Raises
    ------
    FileNotFoundError
        If the classifier or scaler parameter files are not found at the specified paths.
    ValueError
        If the DataFrame does not contain the expected features for prediction.
    """
    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', config.classifier_file_name))

    # Load and apply scaling parameters
    with open(os.path.join(path_to_classifier_input, 'scalers', 'gait_filtering_scaler_params.json'), 'r') as f:
        scaler_params = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.var_ = scaler_params['var']
    scaler.scale_ = scaler_params['scale']
    scaler.feature_names_in_ = scaler_params['features']

    # Scale the features in the DataFrame
    df[scaler_params['features']] = scaler.transform(df[scaler_params['features']])

    # Extract the relevant features for prediction
    X = df.loc[:, clf.feature_names_in_]

    # Make prediction and add the probability of gait activity to the DataFrame
    df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = clf.predict_proba(X)[:, 1]

    return df

def filter_gait_io(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: FilteringGaitConfig) -> None:
    # Load the data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    df = filter_gait(df, config, path_to_classifier_input)

    # Prepare the metadata
    metadata_samples.file_name = 'arm_activity_values.bin'
    metadata_time.file_name = 'arm_activity_time.bin'

    metadata_samples.channels = [DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]
    metadata_samples.units = ['probability']

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'arm_activity_meta.json', df)


# def quantify_arm_swing(df: pd.DataFrame, config: ArmSwingQuantificationConfig) -> pd.DataFrame:

#     # temporarily for testing: manually determine predictions
#     df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = np.concatenate([np.repeat([1], df.shape[0]//3), np.repeat([0], df.shape[0]//3), np.repeat([1], df.shape[0] - 2*df.shape[0]//3)], axis=0)

#     # keep only predicted arm swing
#     # TODO: Aggregate overlapping windows for probabilities
#     df_filtered = df.loc[df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]>=0.5].copy().reset_index(drop=True)

#     del df

#     # create peak angular velocity
#     df_filtered.loc[:, f'peak_{config.velocity_colname}'] = df_filtered.loc[:, [f'forward_peak_{config.velocity_colname}_mean', f'backward_peak_{config.velocity_colname}_mean']].mean(axis=1)
#     df_filtered = df_filtered.drop(columns=[f'forward_peak_{config.velocity_colname}_mean', f'backward_peak_{config.velocity_colname}_mean'])

#     # Aggregate arm swing parameters per segment to obtain estimates for varying segment durations
#     df_segments = df_filtered.copy()
#     df_segments[DataColumns.SEGMENT_NR] = create_segments(
#         config=config,
#         df=df_segments
#     )

#     df_segments = discard_segments(
#         config=config,
#         df=df_segments
#     )

#     # Quantify arm swing
#     df_segment_aggregates = aggregate_segments(
#         df=df_segments,
#         time_colname=DataColumns.TIME,
#         segment_nr_colname=DataColumns.SEGMENT_NR,
#         window_step_size_s=config.window_step_length_s,
#         metrics=['range_of_motion', f'peak_{config.velocity_colname}'],
#         aggregates=['median'],
#         quantiles=[0.95]
#     )

#     df_segment_aggregates['segment_duration_ms'] = (df_segment_aggregates['segment_duration_s'] * 1000).round().astype(int)
#     df_segment_aggregates = df_segment_aggregates.drop(columns=[DataColumns.SEGMENT_NR])

#     df_weekly_median = df_filtered[['range_of_motion', f'peak_{config.velocity_colname}']].agg('median')
#     df_weekly_quantile = df_filtered[['range_of_motion', f'peak_{config.velocity_colname}']].quantile(0.95)
#     df_weekly_aggregates = pd.concat([df_weekly_median, df_weekly_quantile], axis=1)

#     return df_weekly_aggregates, df_segment_aggregates


# def quantify_arm_swing_io(path_to_feature_input: Union[str, Path], path_to_prediction_input: Union[str, Path], output_path: Union[str, Path], config: ArmSwingQuantificationConfig) -> None:
#     # Load the features & predictions
#     metadata_time, metadata_samples = read_metadata(path_to_feature_input, config.meta_filename, config.time_filename, config.values_filename)
#     df_features = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

#     metadata_dict = tsdf.load_metadata_from_path(os.path.join(path_to_prediction_input, config.meta_filename))
#     metadata_time = metadata_dict[config.time_filename]
#     metadata_samples = metadata_dict[config.values_filename]
#     df_predictions = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

#     # Validate
#     # Dataframes have same length
#     assert df_features.shape[0] == df_predictions.shape[0]

#     # Dataframes have same time column
#     assert df_features[DataColumns.TIME].equals(df_predictions[DataColumns.TIME])

#     # Subset features
#     feature_cols = [DataColumns.TIME, 'range_of_motion', f'forward_peak_{config.velocity_colname}_mean', f'backward_peak_{config.velocity_colname}_mean']
#     df_features = df_features[feature_cols]

#     # Concatenate features and predictions
#     df = pd.concat([df_features, df_predictions[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]], axis=1)

#     df_weekly_aggregates, df_segment_aggregates = quantify_arm_swing(df, config)

#     # Store data
#     df_segment_aggregates.to_pickle(os.path.join(output_path, 'segment_aggregates.pkl'))
#     df_weekly_aggregates.to_pickle(os.path.join(output_path, 'weekly_aggregates.pkl'))


# def aggregate_weekly_arm_swing():
#     pass

def merge_predictions_with_timestamps(df_ts, df_predictions, config):
    """
    Merges prediction probabilities with timestamps by expanding overlapping windows 
    into individual timestamps and averaging probabilities per unique timestamp.

    Parameters:
    ----------
    df_ts : pd.DataFrame
        DataFrame containing timestamps to be merged with predictions.
        Must include the timestamp column specified in `config.time_colname`.
    df_predictions : pd.DataFrame
        DataFrame containing prediction windows with start times and probabilities.
        Must include:
        - A column for window start times (defined by `config.time_colname`).
        - A column for prediction probabilities (defined by `config.pred_gait_proba_colname`).
    config : object
        Configuration object containing the following attributes:
        - time_colname (str): Column name for timestamps.
        - pred_gait_proba_colname (str): Column name for prediction probabilities.
        - window_length_s (float): Length of each prediction window in seconds.
        - sampling_frequency (float): Frequency of data sampling (Hz).

    Returns:
    -------
    pd.DataFrame
        Updated `df_ts` with an additional column for averaged prediction probabilities.

    Steps:
    ------
    1. Expand prediction windows into individual timestamps.
    2. Round timestamps to mitigate floating-point inaccuracies.
    3. Aggregate probabilities by unique timestamps.
    4. Merge the aggregated probabilities with the input `df_ts`.

    Notes:
    ------
    - Ensure that timestamps in `df_ts` and `df_predictions` align appropriately before merging.
    """
    # Step 1: Expand each window into individual timestamps
    expanded_data = []
    for _, row in df_predictions.iterrows():
        start_time = row[config.time_colname]
        proba = row[config.pred_gait_proba_colname]
        timestamps = np.arange(start_time, start_time + config.window_length_s, 1/config.sampling_frequency)
        expanded_data.extend(zip(timestamps, [proba] * len(timestamps)))

    # Create a new DataFrame with expanded timestamps
    expanded_df = pd.DataFrame(expanded_data, columns=[config.time_colname, config.pred_gait_proba_colname])

    # Step 2: Round timestamps to avoid floating-point inaccuracies
    expanded_df[config.time_colname] = expanded_df[config.time_colname].round(2)
    df_ts[config.time_colname] = df_ts[config.time_colname].round(2)

    # Step 3: Aggregate by unique timestamps and calculate the mean probability
    expanded_df = expanded_df.groupby(config.time_colname, as_index=False)[config.pred_gait_proba_colname].mean()

    df_ts = pd.merge(left=df_ts, right=expanded_df, how='left', on=config.time_colname)
    df_ts = df_ts.dropna(subset=[config.pred_gait_proba_colname])

    return df_ts