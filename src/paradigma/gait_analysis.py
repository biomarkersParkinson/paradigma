import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

import tsdf

from paradigma.constants import DataColumns
from paradigma.gait_analysis_config import GaitFeatureExtractionConfig, GaitDetectionConfig, \
    ArmSwingFeatureExtractionConfig, ArmSwingDetectionConfig, ArmSwingQuantificationConfig
from paradigma.feature_extraction import extract_temporal_domain_features, \
    extract_spectral_domain_features, pca_transform_gyroscope, compute_angle, \
    remove_moving_average_angle, extract_range_of_motion, extract_angle_extremes, \
    extract_peak_angular_velocity, signal_to_ffts, get_dominant_frequency, compute_perc_power
from paradigma.quantification import aggregate_segments
from paradigma.windowing import tabulate_windows, create_segments, discard_segments
from paradigma.util import get_end_iso8601, write_data, read_metadata


def extract_gait_features(input_path: Union[str, Path], output_path: Union[str, Path], config: GaitFeatureExtractionConfig) -> None:
    # load data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # group sequences of timestamps into windows
    df_windowed = tabulate_windows(
        df=df,
        window_size=config.window_length_s * config.sampling_frequency,
        step_size=config.window_step_size_s * config.sampling_frequency,
        time_column_name=config.time_colname,
        list_value_cols=config.l_data_point_level_cols,
        )

    # compute statistics of the temporal domain signals
    df_windowed = extract_temporal_domain_features(config, df_windowed, l_gravity_stats=['mean', 'std'])

    # transform the signals from the temporal domain to the spectral domain using the fast fourier transform
    # and extract spectral features
    df_windowed = extract_spectral_domain_features(config, df_windowed, config.sensor, config.l_accelerometer_cols)

    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_windowed[config.time_colname][-1:].values[0] + config.window_length_s))

    metadata_samples.end_iso8601 = end_iso8601
    metadata_samples.file_name = 'gait_values.bin'
    metadata_time.end_iso8601 = end_iso8601
    metadata_time.file_name = 'gait_time.bin'

    metadata_samples.channels = list(config.d_channels_values.keys())
    metadata_samples.units = list(config.d_channels_values.values())

    metadata_time.channels = ['time']
    metadata_time.units = ['relative_time_ms']

    write_data(metadata_time, metadata_samples, output_path, 'gait_meta.json', df_windowed)


def detect_gait(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: GaitDetectionConfig) -> None:
    
    # Load the data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, config.classifier_file_name))
    with open(os.path.join(path_to_classifier_input, config.thresholds_file_name), 'r') as f:
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
    df['pred_gait_proba'] = clf.predict_proba(X)[:, 1]
    df['pred_gait'] = df['pred_gait_proba'] > threshold

    # Prepare the metadata
    metadata_samples.file_name = 'gait_values.bin'
    metadata_time.file_name = 'gait_time.bin'

    metadata_samples.channels = ['pred_gait_proba']
    metadata_samples.units = ['probability']

    metadata_time.channels = [config.time_colname]
    metadata_time.units = ['relative_time_ms']

    write_data(metadata_time, metadata_samples, output_path, 'gait_meta.json', df)


def extract_arm_swing_features(input_path: Union[str, Path], output_path: Union[str, Path], config: ArmSwingFeatureExtractionConfig) -> None:
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

    # temporary add "random" predictions
    df[config.pred_gait_colname] = np.concatenate([np.repeat([1], df.shape[0]//3), np.repeat([0], df.shape[0]//3), np.repeat([1], df.shape[0] + 1 - 2*df.shape[0]//3)], axis=0)

    # perform principal component analysis on the gyroscope signals to obtain the angular velocity in the
    # direction of the swing of the arm 
    df[config.velocity_colname] = pca_transform_gyroscope(
        df=df,
        y_gyro_colname=DataColumns.GYROSCOPE_Y,
        z_gyro_colname=DataColumns.GYROSCOPE_Z,
        pred_gait_colname=config.pred_gait_colname
    )

    # integrate the angular velocity to obtain an estimation of the angle
    df[config.angle_colname] = compute_angle(
        velocity_col=df[config.velocity_colname],
        time_col=df[config.time_colname]
    )

    # remove the moving average from the angle to account for possible drift caused by the integration
    # of noise in the angular velocity
    df[config.angle_smooth_colname] = remove_moving_average_angle(
        angle_col=df[config.angle_colname],
        sampling_frequency=config.sampling_frequency
    )

    # use only predicted gait for the subsequent steps
    df = df.loc[df[config.pred_gait_colname]==1].reset_index(drop=True)

    # group consecutive timestamps into segments with new segments starting after a pre-specified gap
    df[config.segment_nr_colname] = create_segments(
        df=df,
        time_column_name=config.time_colname,
        gap_threshold_s=config.segment_gap_s
    )

    # remove any segments that do not adhere to predetermined criteria
    df_segments = discard_segments(
        df=df,
        segment_nr_colname=config.segment_nr_colname,
        min_length_segment_s=config.window_length_s,
        sampling_frequency=config.sampling_frequency
    )

    # create windows of a fixed length and step size from the time series per segment
    # df, window_size, step_size, time_column_name, single_value_cols=None, list_value_cols=None, agg_func='first'
    l_dfs = [
        tabulate_windows(
            df=df_segments[df_segments[config.segment_nr_colname] == segment_nr].reset_index(drop=True),
            window_size=int(config.window_length_s * config.sampling_frequency),
            step_size=int(config.window_step_size_s * config.sampling_frequency),
            time_column_name=config.time_colname,
            single_value_cols=[config.segment_nr_colname],
            list_value_cols=config.l_data_point_level_cols,
        )
        for segment_nr in df_segments[config.segment_nr_colname].unique()
    ]
    l_dfs = [df for df in l_dfs if not df.empty]

    df_windowed = pd.concat(l_dfs).reset_index(drop=True)

    # transform the angle from the temporal domain to the spectral domain using the fast fourier transform
    df_windowed['angle_freqs'], df_windowed['angle_fft'] = signal_to_ffts(
        sensor_col=df_windowed[config.angle_colname],
        window_type=config.window_type,
        sampling_frequency=config.sampling_frequency)

    # obtain the dominant frequency of the angle signal in the frequency band of interest
    # defined by the highest peak in the power spectrum
    df_windowed['angle_dominant_frequency'] = df_windowed.apply(
        lambda x: get_dominant_frequency(signal_ffts=x['angle_fft'],
                                        signal_freqs=x['angle_freqs'],
                                        fmin=config.power_band_low_frequency,
                                        fmax=config.power_band_high_frequency
                                        ), axis=1
    )

    df_windowed = df_windowed.drop(columns=['angle_fft', 'angle_freqs'])

    # compute the percentage of power in the frequency band of interest (i.e., the frequency band of the arm swing)
    df_windowed['angle_perc_power'] = df_windowed[config.angle_colname].apply(
        lambda x: compute_perc_power(
            sensor_col=x,
            fmin_band=config.power_band_low_frequency,
            fmax_band=config.power_band_high_frequency,
            fmin_total=config.spectrum_low_frequency,
            fmax_total=config.spectrum_high_frequency,
            sampling_frequency=config.sampling_frequency,
            window_type=config.window_type
            )
    )

    df_windowed = extract_angle_extremes(
        df=df_windowed,
        angle_colname=config.angle_colname,
        dominant_frequency_colname=f'{config.angle_colname}_dominant_frequency',
        sampling_frequency=config.sampling_frequency
    )

    # calculate the change in angle between consecutive extrema (minima and maxima) of the angle signal inside the window
    df_windowed['angle_amplitudes'] = extract_range_of_motion(
        angle_extrema_values_col=df_windowed['angle_extrema_values']
    )

    df_windowed = df_windowed.drop(columns=['angle_extrema_values'])

    # aggregate the changes in angle between consecutive extrema to obtain the range of motion
    df_windowed['range_of_motion'] = df_windowed['angle_amplitudes'].apply(lambda x: np.mean(x) if len(x) > 0 else 0).replace(np.nan, 0)

    df_windowed = df_windowed.drop(columns=['angle_amplitudes'])

    # compute the forward and backward peak angular velocity using the extrema of the angular velocity
    extract_peak_angular_velocity(
        df=df_windowed,
        velocity_colname=config.velocity_colname,
        angle_minima_colname='angle_minima',
        angle_maxima_colname='angle_maxima'
    )

    df_windowed = df_windowed.drop(columns=['angle_minima','angle_maxima', 'angle_new_minima',
                                            'angle_new_maxima', config.velocity_colname])

    # compute aggregated measures of the peak angular velocity
    for dir in ['forward', 'backward']:
        df_windowed[f'{dir}_peak_ang_vel_mean'] = df_windowed[f'{dir}_peak_ang_vel'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
        df_windowed[f'{dir}_peak_ang_vel_std'] = df_windowed[f'{dir}_peak_ang_vel'].apply(lambda x: np.std(x) if len(x) > 0 else 0)

        df_windowed = df_windowed.drop(columns=[f'{dir}_peak_ang_vel'])

    # compute statistics of the temporal domain accelerometer signals
    df_windowed = extract_temporal_domain_features(config, df_windowed, l_gravity_stats=['mean', 'std'])

    # transform the accelerometer and gyroscope signals from the temporal domain to the spectral domain
    #  using the fast fourier transform and extract spectral features
    for sensor, l_sensor_colnames in zip(['accelerometer', 'gyroscope'], [config.l_accelerometer_cols, config.l_gyroscope_cols]):
        df_windowed = extract_spectral_domain_features(config, df_windowed, sensor, l_sensor_colnames)

    end_iso8601 = get_end_iso8601(metadata_samples.start_iso8601, 
                                df_windowed[config.time_colname][-1:].values[0] + config.window_length_s)

    metadata_samples.end_iso8601 = end_iso8601
    metadata_samples.file_name = 'arm_swing_values.bin'
    metadata_time.end_iso8601 = end_iso8601
    metadata_time.file_name = 'arm_swing_time.bin'

    metadata_samples.channels = list(config.d_channels_values.keys())
    metadata_samples.units = list(config.d_channels_values.values())

    metadata_time.channels = [config.time_colname]
    metadata_time.units = ['relative_time_ms']

    write_data(metadata_time, metadata_samples, output_path, 'arm_swing_meta.json', df_windowed)


def detect_arm_swing(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: ArmSwingDetectionConfig) -> None:

    config.pred_arm_swing = DataColumns.PRED_ARM_SWING

    # Load the data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, config.classifier_file_name))

    # Prepare the data
    clf.feature_names_in_ = ['std_norm_acc'] + [f'{x}_power_below_gait' for x in config.l_accelerometer_cols] + \
                            [f'{x}_power_gait' for x in config.l_accelerometer_cols] + \
                            [f'{x}_power_tremor' for x in config.l_accelerometer_cols] + \
                            [f'{x}_power_above_tremor' for x in config.l_accelerometer_cols] + \
                            [f'cc_{i}_accelerometer' for i in range(1, 13)] + [f'cc_{i}_gyroscope' for i in range(1, 13)] + \
                            [f'grav_{x}_mean' for x in config.l_accelerometer_cols] +  [f'grav_{x}_std' for x in config.l_accelerometer_cols] + \
                            ['range_of_motion', 'forward_peak_ang_vel_mean', 'backward_peak_ang_vel_mean', 'forward_peak_ang_vel_std', 
                            'backward_peak_ang_vel_std', 'angle_perc_power', 'angle_dominant_frequency'] + \
                            [f'{x}_dominant_frequency' for x in config.l_accelerometer_cols]
                            
    X = df.loc[:, clf.feature_names_in_]

    # Make prediction
    # df['pred_arm_swing_proba'] = clf.predict_proba(X)[:, 1]
    df[config.pred_arm_swing] = clf.predict(X)

    # Prepare the metadata
    metadata_samples.file_name = 'arm_swing_values.bin'
    metadata_time.file_name = 'arm_swing_time.bin'

    metadata_samples.channels = [config.pred_arm_swing]
    metadata_samples.units = ['boolean']

    metadata_time.channels = [config.time_colname]
    metadata_time.units = ['relative_time_ms']

    write_data(metadata_time, metadata_samples, output_path, 'arm_swing_meta.json', df)


def quantify_arm_swing(path_to_feature_input: Union[str, Path], path_to_prediction_input: Union[str, Path], output_path: Union[str, Path], config: ArmSwingQuantificationConfig) -> None:

    config.sampling_frequency = 100

    # Load the features & predictions
    metadata_time, metadata_samples = read_metadata(path_to_feature_input, config.meta_filename, config.time_filename, config.values_filename)
    df_features = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    metadata_dict = tsdf.load_metadata_from_path(os.path.join(path_to_prediction_input, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_samples = metadata_dict[config.values_filename]
    df_predictions = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Validate
    # dataframes have same length
    assert df_features.shape[0] == df_predictions.shape[0]

    # dataframes have same time column
    assert df_features[config.time_colname].equals(df_predictions[config.time_colname])

    # Prepare the data

    # subset features
    l_feature_cols = [config.time_colname, 'range_of_motion', 'forward_peak_ang_vel_mean', 'backward_peak_ang_vel_mean']
    df_features = df_features[l_feature_cols]

    # concatenate features and predictions
    df = pd.concat([df_features, df_predictions[config.pred_arm_swing_colname]], axis=1)

    # temporarily for testing: manually determine predictions
    df[config.pred_arm_swing_colname] = np.concatenate([np.repeat([1], df.shape[0]//3), np.repeat([0], df.shape[0]//3), np.repeat([1], df.shape[0] - 2*df.shape[0]//3)], axis=0)

    # keep only predicted arm swing
    df_arm_swing = df.loc[df[config.pred_arm_swing_colname]==1].copy().reset_index(drop=True)

    del df

    # create peak angular velocity
    df_arm_swing.loc[:, 'peak_ang_vel'] = df_arm_swing.loc[:, ['forward_peak_ang_vel_mean', 'backward_peak_ang_vel_mean']].mean(axis=1)
    df_arm_swing = df_arm_swing.drop(columns=['forward_peak_ang_vel_mean', 'backward_peak_ang_vel_mean'])

    # Segmenting

    df_arm_swing[config.segment_nr_colname] = create_segments(
        df=df_arm_swing,
        time_column_name=config.time_colname,
        gap_threshold_s=config.segment_gap_s 
    )
    df_arm_swing = discard_segments(
        df=df_arm_swing,
        segment_nr_colname=config.segment_nr_colname,
        min_length_segment_s=config.min_segment_length_s,
        sampling_frequency=config.sampling_frequency
    )

    # Quantify arm swing
    df_aggregates = aggregate_segments(
        df=df_arm_swing,
        time_colname=config.time_colname,
        segment_nr_colname=config.segment_nr_colname,
        window_step_size_s=config.window_step_size,
        l_metrics=['range_of_motion', 'peak_ang_vel'],
        l_aggregates=['median'],
        l_quantiles=[0.95]
    )

    df_aggregates['segment_duration_ms'] = df_aggregates['segment_duration_s'] * 1000
    df_aggregates = df_aggregates.drop(columns=[config.segment_nr_colname])

    # Store data
    metadata_samples.file_name = 'arm_swing_values.bin'
    metadata_time.file_name = 'arm_swing_time.bin'

    metadata_samples.channels = ['range_of_motion_median', 'range_of_motion_quantile_95',
                                    'peak_ang_vel_median', 'peak_ang_vel_quantile_95']
    metadata_samples.units = ['deg', 'deg', 'deg/s', 'deg/s']

    metadata_time.channels = [config.time_colname, 'segment_duration_ms']
    metadata_time.units = ['relative_time_ms', 'ms']

    write_data(metadata_time, metadata_samples, output_path, 'arm_swing_meta.json', df_aggregates)


def aggregate_weekly_arm_swing():
    pass
