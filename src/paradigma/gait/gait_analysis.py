import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.preprocessing import StandardScaler

import tsdf

from paradigma.constants import DataColumns
from paradigma.config import GaitFeatureExtractionConfig, GaitDetectionConfig, \
    ArmActivityFeatureExtractionConfig, FilteringGaitConfig, ArmSwingQuantificationConfig
from paradigma.gait.feature_extraction import extract_temporal_domain_features, \
    extract_spectral_domain_features
from paradigma.segmenting import tabulate_windows, create_segments, discard_segments, categorize_segments
from paradigma.util import get_end_iso8601, write_df_data, read_metadata, WindowedDataExtractor


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
    windowed_cols = [DataColumns.TIME] + config.accelerometer_cols + config.gravity_cols
    windowed_data = tabulate_windows(
        df=df, 
        columns=windowed_cols,
        window_length_s=config.window_length_s,
        window_step_length_s=config.window_step_length_s,
        sampling_frequency=config.sampling_frequency
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


def extract_gait_features_io(path_to_preprocessed_input: Union[str, Path], path_to_output: Union[str, Path], 
                             config: GaitFeatureExtractionConfig) -> None:
    # Load data
    metadata_time, metadata_values = read_metadata(path_to_preprocessed_input, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Extract gait features
    df_features = extract_gait_features(df, config)

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
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_values, path_to_output, 'gait_meta.json', df_features)


def detect_gait(df: pd.DataFrame, path_to_classifier_input: Union[str, Path], 
                classifier_filename: str, scaler_filename: str, parallel: bool=False) -> pd.DataFrame:
    """
    Detects gait activity in the input DataFrame using a pre-trained classifier and applies a threshold to classify results.

    This function performs the following steps:
    1. Loads the pre-trained classifier and scaling parameters from the specified directory.
    2. Scales the relevant features in the input DataFrame (`df`) using the loaded scaling parameters.
    3. Predicts the probability of gait activity for each sample in the DataFrame using the classifier.
    4. Applies a threshold to the predicted probabilities to determine whether gait activity is present.
    5. Adds the predicted probabilities and classification results as new columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features extracted from gait data. It must include the necessary columns 
        as specified in the classifier's feature names.

    path_to_classifier_input : Union[str, Path]
        The path to the directory containing the classifier, scaler parameters, and other required files.

    classifier_filename : str
        The name of the file containing the pre-trained classifier, located in the `classifiers` subdirectory.

    scaler_filename : str
        The name of the file containing the scaling parameters, located in the `scalers` subdirectory.

    parallel : bool, optional, default=False
        If `True`, enables parallel processing during classification. If `False`, the classifier uses a single core.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with two additional columns:
        - `gait_probability`: The predicted probabilities of gait activity.
        - `gait_detected`: A binary classification result (1 for gait detected, 0 otherwise) based on the threshold.

    Notes
    -----
    - The function expects the pre-trained classifier and scaling parameters to be located in specific 
      subdirectories (`classifiers` and `scalers`) under `path_to_classifier_input`.
    - The threshold used for classification is hardcoded or can be loaded from a configuration file, 
      depending on implementation.

    Raises
    ------
    FileNotFoundError
        If the classifier or scaler parameter files are not found at the specified paths.
    ValueError
        If the DataFrame does not contain the required features for prediction.
    """
    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', classifier_filename))

    if not parallel:
        clf.n_jobs = 1

    # Load and apply scaling parameters
    with open(os.path.join(path_to_classifier_input, 'scalers', scaler_filename), 'r') as f:
        scaler_params = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.var_ = scaler_params['var']
    scaler.scale_ = scaler_params['scale']
    scaler.feature_names_in_ = scaler_params['features']

    df_scaled = df.copy()

    # Scale the features in the DataFrame
    df_scaled[scaler_params['features']] = scaler.transform(df_scaled[scaler_params['features']])

    # Extract the relevant features for prediction
    X = df_scaled.loc[:, clf.feature_names_in_]

    # Make prediction and add the probability of gait activity to the DataFrame
    pred_gait_proba_series = clf.predict_proba(X)[:, 1]

    return pred_gait_proba_series


def detect_gait_io(path_to_input_features: Union[str, Path], path_to_output: Union[str, Path], 
                   path_to_classifier_input: Union[str, Path], config: GaitDetectionConfig) -> None:
    
    # Load the data
    metadata_time, metadata_values = read_metadata(path_to_input_features, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    df[DataColumns.PRED_GAIT_PROBA] = detect_gait(df, config, path_to_classifier_input)

    # Prepare the metadata
    metadata_values.file_name = 'gait_values.bin'
    metadata_time.file_name = 'gait_time.bin'

    metadata_values.channels = ['pred_gait_proba']
    metadata_values.units = ['probability']

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_values, path_to_output, 'gait_meta.json', df)


def extract_arm_activity_features(
        df_timestamps: pd.DataFrame, 
        df_predictions: pd.DataFrame,
        config: ArmActivityFeatureExtractionConfig,
        path_to_classifier_input: Union[str, Path],
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
    df_timestamps : pd.DataFrame
        A DataFrame containing the raw sensor data, including accelerometer, gravity, and gyroscope columns.

    df_predictions : pd.DataFrame
        A DataFrame containing the predicted probabilities for gait activity per window.
    
    config : ArmActivityFeatureExtractionConfig
        Configuration object containing column names and parameters for feature extraction.

    path_to_classifier_input : Union[str, Path]
        The path to the directory containing the classifier files and other necessary input files for feature extraction.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted arm activity features, including angle, velocity, 
        temporal, and spectral features.
    """

    # Load classification threshold
    gait_detection_config = GaitDetectionConfig()
    with open(os.path.join(path_to_classifier_input, 'thresholds', gait_detection_config.thresholds_file_name), 'r') as f:
        classification_threshold = float(f.read())

    # Merge gait predictions with timestamps
    gait_preprocessing_config = GaitFeatureExtractionConfig()
    df = merge_predictions_with_timestamps(df_timestamps, df_predictions, gait_preprocessing_config)

    # Add a column for predicted gait based on a fitted threshold
    df[DataColumns.PRED_GAIT] = (df[DataColumns.PRED_GAIT_PROBA] >= classification_threshold).astype(int)

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
        sampling_frequency=config.sampling_frequency,
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
            sampling_frequency=config.sampling_frequency
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
    df_temporal_features = extract_temporal_domain_features(config, windowed_acc, windowed_grav, grav_stats=['mean', 'std'])
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # Extract spectral domain features for accelerometer and gyroscope signals
    for sensor_name, windowed_sensor in zip(['accelerometer', 'gyroscope'], [windowed_acc, windowed_gyro]):
        df_spectral_features = extract_spectral_domain_features(config, sensor_name, windowed_sensor)
        df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def extract_arm_activity_features_io(path_to_timestamp_input: Union[str, Path], path_to_prediction_input: Union[str, Path], 
                                     path_to_classifier_input: Union[str, Path], path_to_output: Union[str, Path], 
                                     config: ArmActivityFeatureExtractionConfig) -> None:
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

    # Extract arm activity features
    df_features = extract_arm_activity_features(df_ts, df_pred_gait, config, path_to_classifier_input)

    end_iso8601 = get_end_iso8601(metadata_ts_values.start_iso8601, 
                                df_features[DataColumns.TIME][-1:].values[0] + config.window_length_s)

    metadata_ts_values.end_iso8601 = end_iso8601
    metadata_ts_values.file_name = 'arm_activity_values.bin'
    metadata_ts_time.end_iso8601 = end_iso8601
    metadata_ts_time.file_name = 'arm_activity_time.bin'

    metadata_ts_values.channels = list(config.d_channels_values.keys())
    metadata_ts_values.units = list(config.d_channels_values.values())

    metadata_ts_time.channels = [DataColumns.TIME]
    metadata_ts_time.units = ['relative_time_ms']

    write_df_data(metadata_ts_time, metadata_ts_values, path_to_output, 'arm_activity_meta.json', df_features)


def filter_gait(df: pd.DataFrame, path_to_classifier_input: Union[str, Path], 
                classifier_filename: str, scaler_filename: str, parallel: bool=False) -> pd.DataFrame:
    """
    Filters gait data to identify periods with no other arm activity using a pre-trained classifier.

    This function performs the following steps:
    1. Loads a pre-trained classifier and feature scaling parameters from the specified directory.
    2. Scales the relevant features in the input DataFrame (`df`) using the loaded scaling parameters.
    3. Makes predictions with the classifier to estimate the probability of no other arm activity during gait.
    4. Returns a Series containing the predicted probabilities.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features extracted from gait data. It must include the necessary 
        columns as specified in the classifier's feature names.

    path_to_classifier_input : Union[str, Path]
        The path to the directory containing the classifier, scaler parameters, and other necessary input files.

    classifier_filename : str
        The name of the file containing the pre-trained classifier, located in the `classifiers` subdirectory.

    scaler_filename : str
        The name of the file containing the scaling parameters, located in the `scalers` subdirectory.

    parallel : bool, optional, default=False
        If `True`, enables parallel processing during classification. If `False`, the classifier uses a 
        single core for predictions.

    Returns
    -------
    pd.Series
        A Series containing the predicted probabilities of no other arm activity during gait for each sample 
        in the input DataFrame.

    Notes
    -----
    - The function expects the pre-trained classifier and scaling parameters to be located in specific 
      subdirectories (`classifiers` and `scalers`) under `path_to_classifier_input`.
    - The classifier should output probabilities indicating the likelihood of no other arm activity during gait.

    Raises
    ------
    FileNotFoundError
        If the classifier or scaler parameter files are not found at the specified paths.
    ValueError
        If the DataFrame does not contain the required features for prediction.
    """
    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', classifier_filename))

    if not parallel:
        clf.n_jobs = 1

    # Load and apply scaling parameters
    with open(os.path.join(path_to_classifier_input, 'scalers', scaler_filename), 'r') as f:
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
    pred_no_other_arm_activity_proba_series = clf.predict_proba(X)[:, 1]

    return pred_no_other_arm_activity_proba_series


def filter_gait_io(path_to_feature_input: Union[str, Path], path_to_classifier_input: Union[str, Path], 
                   path_to_output: Union[str, Path], config: FilteringGaitConfig) -> None:
    # Load the data
    metadata_time, metadata_values = read_metadata(path_to_feature_input, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = filter_gait(df, config, path_to_classifier_input)

    # Prepare the metadata
    metadata_values.file_name = 'arm_activity_values.bin'
    metadata_time.file_name = 'arm_activity_time.bin'

    metadata_values.channels = [DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]
    metadata_values.units = ['probability']

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_values, path_to_output, 'arm_activity_meta.json', df)


def quantify_arm_swing(df_features: pd.DataFrame, df_predictions: pd.DataFrame, config: ArmSwingQuantificationConfig,
                       path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:

    # Expand prediction windows from start time to start time + window length
    # This is done to ensure that each timestamp has a corresponding probability
    expanded_data = []
    for _, row in df_predictions.iterrows():
        start_time = row[DataColumns.TIME]
        proba = row[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]
        timestamps = np.arange(start_time, start_time + config.window_length_s, config.window_step_length_s)
        expanded_data.extend(zip(timestamps, [proba] * len(timestamps)))

    # Create a new DataFrame with expanded timestamps
    expanded_df = pd.DataFrame(expanded_data, columns=[DataColumns.TIME, DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA])

    # Aggregate overlapping windows
    df_predictions = expanded_df.groupby(DataColumns.TIME, as_index=False)[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA].mean()

    # Keep only predicted arm swing based on a classification threshold
    filtering_gait_config = FilteringGaitConfig()
    with open(os.path.join(path_to_classifier_input, 'thresholds', filtering_gait_config.thresholds_file_name), 'r') as f:
        classification_threshold = float(f.read())

    df_filtered = df_predictions.loc[df_predictions[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]>=classification_threshold].copy().reset_index(drop=True)

    if df_filtered.shape[0] == 0:
        raise ValueError("No arm swing detected in the input data.")
    
    del df_predictions

    # Merge features into predictions
    df_filtered = pd.merge(df_filtered, df_features, on=DataColumns.TIME)

    # Construct peak angular velocity from the forward and backward velocities
    df_filtered.loc[:, DataColumns.PEAK_VELOCITY] = df_filtered.loc[:, [f'forward_peak_{DataColumns.VELOCITY}_mean', f'backward_peak_{DataColumns.VELOCITY}_mean']].mean(axis=1)
    df_filtered = df_filtered.drop(
        columns=[f'forward_peak_{DataColumns.VELOCITY}_mean', f'backward_peak_{DataColumns.VELOCITY}_mean']
    )

    # Create segments of predicted gait without other arm activities
    df_segments = df_filtered.copy()
    df_segments[DataColumns.SEGMENT_NR] = create_segments(
        time_array=df_segments[DataColumns.TIME],
        max_segment_gap_s=config.max_segment_gap_s
    )

    # Discard segments that are too short 
    df_segments = discard_segments(
        config=config,
        df=df_segments,
        format='windows'
    )

    # Categorize segments based on segment duration
    df_segments[DataColumns.SEGMENT_CAT] = categorize_segments(df_segments, config, 'windows')

    # Quantify the arm swing using the median and 95th percentile for the range of motion and the peak velocity
    # of each segment category
    df_segment_cat_aggregates = df_segments.groupby(DataColumns.SEGMENT_CAT).agg({
        DataColumns.RANGE_OF_MOTION: ['median', lambda x: np.percentile(x, 95)],
        DataColumns.PEAK_VELOCITY: ['median', lambda x: np.percentile(x, 95)],
    })

    # Rename the columns for clarity
    df_segment_cat_aggregates.columns = [
        f'{DataColumns.RANGE_OF_MOTION}_median', f'{DataColumns.RANGE_OF_MOTION}_95p',
        f'{DataColumns.PEAK_VELOCITY}_median', f'{DataColumns.PEAK_VELOCITY}_95p'
    ]

    # Compute total aggregates of all segments combined
    total_aggregates = pd.Series({
        f'{DataColumns.RANGE_OF_MOTION}_median': df_segments[DataColumns.RANGE_OF_MOTION].median(),
        f'{DataColumns.RANGE_OF_MOTION}_95p': np.percentile(df_segments[DataColumns.RANGE_OF_MOTION], 95),
        f'{DataColumns.PEAK_VELOCITY}_median': df_segments[DataColumns.PEAK_VELOCITY].median(),
        f'{DataColumns.PEAK_VELOCITY}_95p': np.percentile(df_segments[DataColumns.PEAK_VELOCITY], 95),
    }, name='total')

    # Append total aggregates to segment-level aggregates
    df_segment_cat_aggregates = pd.concat([df_segment_cat_aggregates, total_aggregates.to_frame().T])

    return df_segment_cat_aggregates


def quantify_arm_swing_io(path_to_feature_input: Union[str, Path], path_to_prediction_input: Union[str, Path], path_to_classifier_input: Union[str, Path], path_to_output: Union[str, Path], config: ArmSwingQuantificationConfig) -> None:
    # Load the features & predictions
    metadata_time, metadata_values = read_metadata(path_to_feature_input, config.meta_filename, config.time_filename, config.values_filename)
    df_features = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    metadata_dict = tsdf.load_metadata_from_path(os.path.join(path_to_prediction_input, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_values = metadata_dict[config.values_filename]
    df_predictions = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Subset features
    feature_cols = [DataColumns.TIME, DataColumns.RANGE_OF_MOTION, f'forward_peak_{DataColumns.VELOCITY}_mean', f'backward_peak_{DataColumns.VELOCITY}_mean']
    df_features = df_features[feature_cols]

    df_aggregates = quantify_arm_swing(df_features, df_predictions, config, path_to_classifier_input)

    # Store data as json
    with open(os.path.join(path_to_output, 'aggregates.json'), 'w') as f:
        json.dump(df_aggregates.to_dict(), f)
    


def merge_predictions_with_timestamps(df_ts, df_predictions, config):
    """
    Merges prediction probabilities with timestamps by expanding overlapping windows
    into individual timestamps and averaging probabilities per unique timestamp.

    Parameters:
    ----------
    df_ts : pd.DataFrame
        DataFrame containing timestamps to be merged with predictions.
        Must include the timestamp column specified in `DataColumns.TIME`.
    df_predictions : pd.DataFrame
        DataFrame containing prediction windows with start times and probabilities.
        Must include:
        - A column for window start times (defined by `DataColumns.TIME`).
        - A column for prediction probabilities (defined by `DataColumns.PRED_GAIT_PROBA`).
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
    1. Expand prediction windows into individual timestamps using NumPy broadcasting.
    2. Flatten the timestamps and prediction probabilities into single arrays.
    3. Aggregate probabilities by unique timestamps using pandas `groupby`.
    4. Merge the aggregated probabilities with the input `df_ts`.

    Notes:
    ------
    - Rounding is applied to timestamps to mitigate floating-point inaccuracies.
    - Fully vectorized for speed and scalability, avoiding any row-wise operations.
    """
    # Step 1: Generate all timestamps for prediction windows using NumPy broadcasting
    window_length = int(config.window_length_s * config.sampling_frequency)
    timestamps = (
        df_predictions[DataColumns.TIME].values[:, None] +
        np.arange(0, window_length) / config.sampling_frequency
    )
    
    # Flatten timestamps and probabilities into a single array for efficient processing
    flat_timestamps = timestamps.ravel()
    flat_proba = np.repeat(
        df_predictions[DataColumns.PRED_GAIT_PROBA].values,
        window_length
    )

    # Step 2: Create a DataFrame for expanded data
    expanded_df = pd.DataFrame({
        DataColumns.TIME: flat_timestamps,
        DataColumns.PRED_GAIT_PROBA: flat_proba
    })

    # Step 3: Round timestamps and aggregate probabilities
    expanded_df[DataColumns.TIME] = expanded_df[DataColumns.TIME].round(2)
    mean_proba = expanded_df.groupby(DataColumns.TIME, as_index=False).mean()

    # Step 4: Round timestamps in `df_ts` and merge
    df_ts[DataColumns.TIME] = df_ts[DataColumns.TIME].round(2)
    df_ts = pd.merge(df_ts, mean_proba, how='left', on=DataColumns.TIME)
    df_ts = df_ts.dropna(subset=[DataColumns.PRED_GAIT_PROBA])

    return df_ts

