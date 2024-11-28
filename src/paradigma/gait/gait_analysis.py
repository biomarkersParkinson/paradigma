import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tsdf

from paradigma.constants import DataColumns
from paradigma.gait.gait_analysis_config import GaitFeatureExtractionConfig, GaitDetectionConfig, \
    ArmActivityFeatureExtractionConfig, FilteringGaitConfig, ArmSwingQuantificationConfig
from paradigma.gait.feature_extraction import extract_temporal_domain_features, \
    extract_spectral_domain_features, compute_angle_and_velocity_from_gyro, extract_angle_features
from paradigma.gait.quantification import aggregate_segments
from paradigma.src.paradigma.segmenting import tabulate_windows, create_segments, discard_segments
from paradigma.util import get_end_iso8601, write_df_data, read_metadata


def extract_gait_features(df: pd.DataFrame, config: GaitFeatureExtractionConfig) -> pd.DataFrame:
    # group sequences of timestamps into windows
    l_window_cols = [config.time_colname] + config.l_accelerometer_cols + config.l_gravity_cols
    data_windowed = tabulate_windows(config, df, l_window_cols)

    start_time = np.min(data_windowed[:, :, 0], axis=1)
    accel_windowed = data_windowed[:, :, 1:4]
    grav_windowed = data_windowed[:, :, 4:7]

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

    # Prepare the data
    clf.feature_names_in_ = [f'{x}_power_below_gait' for x in config.l_accelerometer_cols] + \
                            [f'{x}_power_gait' for x in config.l_accelerometer_cols] + \
                            [f'{x}_power_tremor' for x in config.l_accelerometer_cols] + \
                            [f'{x}_power_above_tremor' for x in config.l_accelerometer_cols] + \
                            ['std_norm_acc'] + [f'cc_{i}_accelerometer' for i in range(1, 13)] + [f'grav_{x}_{y}' for x in config.l_accelerometer_cols for y in ['mean', 'std']] + \
                            [f'{x}_dominant_frequency' for x in config.l_accelerometer_cols]
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

    idx_acc = slice(0, n_acc_cols)
    idx_grav = slice(n_acc_cols, n_acc_cols + n_grav_cols)
    idx_gyro = slice(n_acc_cols + n_grav_cols, n_acc_cols + n_grav_cols + n_gyro_cols)
    idx_angle = -2
    idx_velocity = -1

    windowed_acc = windowed_data[:, :, idx_acc]
    windowed_grav = windowed_data[:, :, idx_grav]
    windowed_gyro = windowed_data[:, :, idx_gyro]
    windowed_angle = windowed_data[:, :, idx_angle]
    windowed_velocity = windowed_data[:, :, idx_velocity]

    df_features = extract_angle_features(config, windowed_angle, windowed_velocity)    

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


def filter_gait(df: pd.DataFrame, config: FilteringGaitConfig, clf: Union[LogisticRegression, RandomForestClassifier]) -> pd.DataFrame:

    # Prepare the data
    clf.feature_names_in_ = ['std_norm_acc'] + [f'{x}_power_below_gait' for x in config.l_accelerometer_cols] + \
                            [f'{x}_power_gait' for x in config.l_accelerometer_cols] + \
                            [f'{x}_power_tremor' for x in config.l_accelerometer_cols] + \
                            [f'{x}_power_above_tremor' for x in config.l_accelerometer_cols] + \
                            [f'cc_{i}_accelerometer' for i in range(1, 13)] + [f'cc_{i}_gyroscope' for i in range(1, 13)] + \
                            [f'grav_{x}_mean' for x in config.l_accelerometer_cols] +  [f'grav_{x}_std' for x in config.l_accelerometer_cols] + \
                            [f'{config.angle_colname}_dominant_frequency'] + [f'{x}_dominant_frequency' for x in config.l_accelerometer_cols]
    X = df.loc[:, clf.feature_names_in_]

    # Make prediction
    df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = clf.predict_proba(X)[:, 1]

    return df

def filter_gait_io(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: FilteringGaitConfig) -> None:
    # Load the data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Load the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', config.classifier_file_name))

    df = filter_gait(df, config, clf)

    # Prepare the metadata
    metadata_samples.file_name = 'arm_activity_values.bin'
    metadata_time.file_name = 'arm_activity_time.bin'

    metadata_samples.channels = [DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]
    metadata_samples.units = ['probability']

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'arm_activity_meta.json', df)


def quantify_arm_swing(df: pd.DataFrame, config: ArmSwingQuantificationConfig) -> pd.DataFrame:

    # temporarily for testing: manually determine predictions
    df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = np.concatenate([np.repeat([1], df.shape[0]//3), np.repeat([0], df.shape[0]//3), np.repeat([1], df.shape[0] - 2*df.shape[0]//3)], axis=0)

    # keep only predicted arm swing
    # TODO: Aggregate overlapping windows for probabilities
    df_filtered = df.loc[df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]>=0.5].copy().reset_index(drop=True)

    del df

    # create peak angular velocity
    df_filtered.loc[:, f'peak_{config.velocity_colname}'] = df_filtered.loc[:, [f'forward_peak_{config.velocity_colname}_mean', f'backward_peak_{config.velocity_colname}_mean']].mean(axis=1)
    df_filtered = df_filtered.drop(columns=[f'forward_peak_{config.velocity_colname}_mean', f'backward_peak_{config.velocity_colname}_mean'])

    # Segmenting


    df_filtered[DataColumns.SEGMENT_NR] = create_segments(
        df=df_filtered,
        time_column_name=DataColumns.TIME,
        gap_threshold_s=config.segment_gap_s
    )
    df_filtered = discard_segments(
        df=df_filtered,
        segment_nr_colname=DataColumns.SEGMENT_NR,
        min_length_segment_s=config.min_segment_length_s,
        sampling_frequency=config.sampling_frequency
    )

    # Quantify arm swing
    df_aggregates = aggregate_segments(
        df=df_filtered,
        time_colname=DataColumns.TIME,
        segment_nr_colname=DataColumns.SEGMENT_NR,
        window_step_size_s=config.window_step_size,
        l_metrics=['range_of_motion', f'peak_{config.velocity_colname}'],
        l_aggregates=['median'],
        l_quantiles=[0.95]
    )

    df_aggregates['segment_duration_ms'] = (df_aggregates['segment_duration_s'] * 1000).round().astype(int)
    df_aggregates = df_aggregates.drop(columns=[DataColumns.SEGMENT_NR])

    return df_aggregates


def quantify_arm_swing_io(path_to_feature_input: Union[str, Path], path_to_prediction_input: Union[str, Path], output_path: Union[str, Path], config: ArmSwingQuantificationConfig) -> None:
    # Load the features & predictions
    metadata_time, metadata_samples = read_metadata(path_to_feature_input, config.meta_filename, config.time_filename, config.values_filename)
    df_features = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    metadata_dict = tsdf.load_metadata_from_path(os.path.join(path_to_prediction_input, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_samples = metadata_dict[config.values_filename]
    df_predictions = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Validate
    # Dataframes have same length
    assert df_features.shape[0] == df_predictions.shape[0]

    # Dataframes have same time column
    assert df_features[DataColumns.TIME].equals(df_predictions[DataColumns.TIME])

    # Subset features
    l_feature_cols = [DataColumns.TIME, 'range_of_motion', f'forward_peak_{config.velocity_colname}_mean', f'backward_peak_{config.velocity_colname}_mean']
    df_features = df_features[l_feature_cols]

    # Concatenate features and predictions
    df = pd.concat([df_features, df_predictions[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]], axis=1)

    df_aggregates = quantify_arm_swing(df, config)

    # Store data
    metadata_samples.file_name = 'arm_swing_values.bin'
    metadata_time.file_name = 'arm_swing_time.bin'

    metadata_samples.channels = ['range_of_motion_median', 'range_of_motion_quantile_95',
                                    f'peak_{config.velocity_colname}_median', f'peak_{config.velocity_colname}_quantile_95']
    metadata_samples.units = ['deg', 'deg', 'deg/s', 'deg/s']

    metadata_time.channels = [DataColumns.TIME, 'segment_duration_ms']
    metadata_time.units = ['relative_time_ms', 'ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'arm_swing_meta.json', df_aggregates)


def aggregate_weekly_arm_swing():
    pass
