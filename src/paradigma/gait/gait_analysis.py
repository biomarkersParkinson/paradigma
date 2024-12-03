import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import tsdf

from paradigma.constants import DataColumns
from paradigma.gait.gait_analysis_config import GaitFeatureExtractionConfig, GaitDetectionConfig, \
    ArmActivityFeatureExtractionConfig, FilteringGaitConfig, ArmSwingQuantificationConfig
from paradigma.gait.feature_extraction import extract_temporal_domain_features, \
    extract_spectral_domain_features, compute_angle_and_velocity_from_gyro, extract_angle_features
from paradigma.gait.quantification import aggregate_segments
from paradigma.segmenting import tabulate_windows, create_segments, discard_segments
from paradigma.util import get_end_iso8601, write_df_data, read_metadata


def extract_gait_features(df: pd.DataFrame, config: GaitFeatureExtractionConfig) -> pd.DataFrame:
    # group sequences of timestamps into windows
    l_window_cols = [config.time_colname] + config.l_accelerometer_cols + config.l_gravity_cols
    data_windowed = tabulate_windows(config, df, l_window_cols)

    idx_time = l_window_cols.index(config.time_colname)
    idx_acc = slice(1, 4)
    idx_grav = slice(4, 7)

    start_time = np.min(data_windowed[:, :, idx_time], axis=1)
    accel_windowed = data_windowed[:, :, idx_acc]
    grav_windowed = data_windowed[:, :, idx_grav]

    df_features = pd.DataFrame(start_time, columns=[config.time_colname])
    
    # compute statistics of the temporal domain signals
    df_temporal_features = extract_temporal_domain_features(
        config=config, 
        windowed_acc=accel_windowed,
        windowed_grav=grav_windowed,
        l_grav_stats=['mean', 'std']
    )

    df_features= pd.concat([df_features, df_temporal_features], axis=1)

    # transform the signals from the temporal domain to the spectral domain using the fast fourier transform
    # and extract spectral features
    df_spectral_features = extract_spectral_domain_features(
        config=config, 
        sensor='accelerometer', 
        windowed_data=accel_windowed
    )

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
    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', config.classifier_file_name))
    with open(os.path.join(path_to_classifier_input, 'thresholds', config.thresholds_file_name), 'r') as f:
        threshold = float(f.read())

    # Scale features
    with open(os.path.join(path_to_classifier_input, 'scalers', 'gait_detection_scaler_params.json'), 'r') as f:
        scaler_params = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.var_ = scaler_params['var']
    scaler.scale_ = scaler_params['scale']
    scaler.feature_names_in_ = scaler_params['features']

    df[scaler_params['features']] = scaler.transform(df[scaler_params['features']])

    # Prepare the data
    X = df.loc[:, clf.feature_names_in_]

    # Make prediction
    df[DataColumns.PRED_GAIT_PROBA] = clf.predict_proba(X)[:, 1]
    df[DataColumns.PRED_GAIT] = df[DataColumns.PRED_GAIT_PROBA] >= threshold

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


def extract_arm_activity_features(df: pd.DataFrame, config: ArmActivityFeatureExtractionConfig) -> pd.DataFrame:

    # temporary add "random" predictions
    df[config.pred_gait_colname] = np.concatenate([np.repeat([1], df.shape[0]//3), np.repeat([0], df.shape[0]//3), np.repeat([1], df.shape[0] + 1 - 2*df.shape[0]//3)], axis=0)
    
    df[config.angle_colname], df[config.velocity_colname] = compute_angle_and_velocity_from_gyro(config, df)

    # use only predicted gait for the subsequent steps
    df = df.loc[df[config.pred_gait_colname]==1].reset_index(drop=True)

    # group consecutive timestamps into segments with new segments starting after a pre-specified gap
    df[config.segment_nr_colname] = create_segments(
        config=config,
        df=df
    )

    # remove any segments that do not adhere to predetermined criteria
    df = discard_segments(
        config=config,
        df=df
    )

    # create windows of a fixed length and step size from the time series per segment
    windowed_data = []
    df_grouped = df.groupby(config.segment_nr_colname)
    l_windowed_cols = (
        [config.time_colname] + 
        config.l_accelerometer_cols + 
        config.l_gravity_cols + 
        config.l_gyroscope_cols + 
        [config.angle_colname, config.velocity_colname]
    )

    for _, group in df_grouped:
        windows = tabulate_windows(
            config=config,
            df=group,
            columns=l_windowed_cols
        )
        if len(windows) > 0:  # Skip if no windows are created
            windowed_data.append(windows)

    if len(windowed_data) > 0:
        windowed_data = np.concatenate(windowed_data, axis=0)
    else:
        raise ValueError("No windows were created from the given data.")

    n_acc_cols = len(config.l_accelerometer_cols)
    n_grav_cols = len(config.l_gravity_cols)
    n_gyro_cols = len(config.l_gyroscope_cols)

    idx_time = l_windowed_cols.index(config.time_colname)
    idx_acc = slice(0, n_acc_cols)
    idx_grav = slice(n_acc_cols, n_acc_cols + n_grav_cols)
    idx_gyro = slice(n_acc_cols + n_grav_cols, n_acc_cols + n_grav_cols + n_gyro_cols)
    idx_angle = l_windowed_cols.index(config.angle_colname)
    idx_velocity = l_windowed_cols.index(config.velocity_colname)

    start_time = np.min(windowed_data[:, :, idx_time], axis=1)
    windowed_acc = windowed_data[:, :, idx_acc]
    windowed_grav = windowed_data[:, :, idx_grav]
    windowed_gyro = windowed_data[:, :, idx_gyro]
    windowed_angle = windowed_data[:, :, idx_angle]
    windowed_velocity = windowed_data[:, :, idx_velocity]

    df_features = pd.DataFrame(start_time, columns=[config.time_colname])

    df_angle_features = extract_angle_features(config, windowed_angle, windowed_velocity)    
    df_features = pd.concat([df_features, df_angle_features], axis=1)

    # compute statistics of the temporal domain accelerometer signals
    df_temporal_features = extract_temporal_domain_features(config, windowed_acc, windowed_grav, l_grav_stats=['mean', 'std'])
    df_features = pd.concat([df_features, df_temporal_features], axis=1)

    # transform the accelerometer and gyroscope signals from the temporal domain to the spectral domain
    # using the fast fourier transform and extract spectral features
    for sensor_name, windowed_sensor in zip(['accelerometer', 'gyroscope'], [windowed_acc, windowed_gyro]):
        df_spectral_features = extract_spectral_domain_features(config, sensor_name, windowed_sensor)
        df_features = pd.concat([df_features, df_spectral_features], axis=1)

    return df_features


def extract_arm_activity_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: ArmActivityFeatureExtractionConfig) -> None:
    # load accelerometer and gyroscope data
    l_dfs = []
    for sensor in ['accelerometer', 'gyroscope']:
        config.set_sensor(sensor)
        meta_filename = f'{sensor}_meta.json'
        values_filename = f'{sensor}_samples.bin'
        time_filename = f'{sensor}_time.bin'

        metadata_dict = tsdf.load_metadata_from_path(os.path.join(input_path, meta_filename))
        metadata_time = metadata_dict[time_filename]
        metadata_samples = metadata_dict[values_filename]
        l_dfs.append(tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns))

    df = pd.merge(l_dfs[0], l_dfs[1], on=config.time_colname)

    # TODO: Load gait prediction and merge

    df_features = extract_arm_activity_features(df, config)

    end_iso8601 = get_end_iso8601(metadata_samples.start_iso8601, 
                                df_features[config.time_colname][-1:].values[0] + config.window_length_s)

    metadata_samples.end_iso8601 = end_iso8601
    metadata_samples.file_name = 'arm_activity_values.bin'
    metadata_time.end_iso8601 = end_iso8601
    metadata_time.file_name = 'arm_activity_time.bin'

    metadata_samples.channels = list(config.d_channels_values.keys())
    metadata_samples.units = list(config.d_channels_values.values())

    metadata_time.channels = [config.time_colname]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'arm_activity_meta.json', df_features)


def filter_gait(df: pd.DataFrame, config: FilteringGaitConfig, path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:
    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', config.classifier_file_name))

    # Scale features
    with open(os.path.join(path_to_classifier_input, 'scalers', 'gait_filtering_scaler_params.json'), 'r') as f:
        scaler_params = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.var_ = scaler_params['var']
    scaler.scale_ = scaler_params['scale']
    scaler.feature_names_in_ = scaler_params['features']

    df[scaler_params['features']] = scaler.transform(df[scaler_params['features']])

    # Scale features
    with open(os.path.join(path_to_classifier_input, 'scalers', 'gait_filtering_scaler_params.json'), 'r') as f:
        scaler_params = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.var_ = scaler_params['var']
    scaler.scale_ = scaler_params['scale']
    scaler.feature_names_in_ = scaler_params['features']

    df[scaler_params['features']] = scaler.transform(df[scaler_params['features']])

    X = df.loc[:, clf.feature_names_in_]

    # Make prediction
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
#         l_metrics=['range_of_motion', f'peak_{config.velocity_colname}'],
#         l_aggregates=['median'],
#         l_quantiles=[0.95]
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
#     l_feature_cols = [DataColumns.TIME, 'range_of_motion', f'forward_peak_{config.velocity_colname}_mean', f'backward_peak_{config.velocity_colname}_mean']
#     df_features = df_features[l_feature_cols]

#     # Concatenate features and predictions
#     df = pd.concat([df_features, df_predictions[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]], axis=1)

#     df_weekly_aggregates, df_segment_aggregates = quantify_arm_swing(df, config)

#     # Store data
#     df_segment_aggregates.to_pickle(os.path.join(output_path, 'segment_aggregates.pkl'))
#     df_weekly_aggregates.to_pickle(os.path.join(output_path, 'weekly_aggregates.pkl'))


# def aggregate_weekly_arm_swing():
#     pass
