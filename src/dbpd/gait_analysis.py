import os

import tsdf

from dbpd.gait_analysis_config import *
from dbpd.feature_extraction import *
from dbpd.quantification import *
from dbpd.windowing import *
from dbpd.util import get_end_iso8601, write_data


def extract_gait_features(input_path: str, output_path: str, config: GaitFeatureExtractionConfig) -> None:
    metadata_dict = tsdf.load_metadata_from_path(os.path.join(input_path, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_samples = metadata_dict[config.values_filename]

    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)
    df_windowed = tabulate_windows(
        df=df,
        time_column_name='time',
        data_point_level_cols=config.l_data_point_level_cols,
        window_length_s=config.window_length_s,
        window_step_size_s=config.window_step_size_s,
        sampling_frequency=config.sampling_frequency
        )

    print(df_windowed.shape)

    # extract mean and std of gravity acceleration
    for col in config.l_gravity_cols:
        for stat in ['mean', 'std']:
            df_windowed[f'{col}_{stat}'] = generate_statistics(
                sensor_col=df_windowed[col],
                statistic=stat
                )

    # extract standard deviation of the Euclidean norm of the three axes
    df_windowed['std_norm_acc'] = generate_std_norm(
        df=df_windowed,
        cols=config.l_accelerometer_cols
        )
    
    for col in config.l_accelerometer_cols:

        # fast fourier transforms
        df_windowed[f'{col}_freqs'], df_windowed[f'{col}_fft'] = signal_to_ffts(
            sensor_col=df_windowed[col],
            window_type=config.window_type,
            sampling_frequency=config.sampling_frequency
            )

        # compute power in distinct frequency bandwidths
        for bandwidth in config.d_frequency_bandwidths.keys():
            df_windowed[col+'_'+bandwidth] = df_windowed.apply(lambda x: compute_power_in_bandwidth(
                sensor_col=x[col],
                fmin=config.d_frequency_bandwidths[bandwidth][0],
                fmax=config.d_frequency_bandwidths[bandwidth][1],
                sampling_frequency=config.sampling_frequency,
                window_type=config.window_type,
                ), axis=1
            )

        # extract dominant frequency
        df_windowed[col+'_dominant_frequency'] = df_windowed.apply(lambda x: get_dominant_frequency(
            signal_ffts=x[col+'_fft'], 
            signal_freqs=x[col+'_freqs'],
            fmin=config.d_frequency_bandwidths[bandwidth][0],
            fmax=config.d_frequency_bandwidths[bandwidth][1]
            ), axis=1
        )

    for bandwidth in config.d_frequency_bandwidths.keys():
        df_windowed['total_acc_'+bandwidth] = df_windowed.apply(lambda x: sum(x[y+'_'+bandwidth] for y in config.l_accelerometer_cols), axis=1)

    df_windowed['total_accel_power'] = compute_power(
        df=df_windowed,
        fft_cols=[f'{col}_fft' for col in config.l_accelerometer_cols])

    cc_cols = generate_cepstral_coefficients(
        total_power_col=df_windowed['total_accel_power'],
        window_length_s=config.window_length_s,
        sampling_frequency=config.sampling_frequency,
        low_frequency=config.low_frequency,
        high_frequency=config.high_frequency,
        filter_length=config.filter_length,
        n_dct_filters=config.n_dct_filters
        )

    df_windowed = pd.concat([df_windowed, cc_cols], axis=1)

    df_windowed = df_windowed.rename(columns={f'cc_{cc_nr}': f'cc_{cc_nr}_acc' for cc_nr in range(1,17)}).rename(columns={'window_start': 'time'})

    df_windowed = df_windowed.drop(columns=[f'{col}{x}' for x in ['', '_freqs', '_fft', '_fft_power'] for col in config.l_accelerometer_cols] + ['total_accel_power', 'window_nr', 'window_end'] + config.l_gravity_cols + config.l_accelerometer_cols)


    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_windowed['time'][-1:].values[0] + config.window_length_s))

    metadata_samples.__setattr__('end_iso8601', end_iso8601)
    metadata_samples.__setattr__('file_name', 'gait_values.bin')
    metadata_samples.__setattr__('file_dir_path', output_path)
    metadata_time.__setattr__('end_iso8601', end_iso8601)
    metadata_time.__setattr__('file_name', 'gait_time.bin')
    metadata_time.__setattr__('file_dir_path', output_path)

    metadata_samples.__setattr__('channels', list(config.d_channels_values.keys()))
    metadata_samples.__setattr__('units', list(config.d_channels_values.values()))

    metadata_time.__setattr__('channels', ['time'])
    metadata_time.__setattr__('units', ['relative_time_ms'])
    metadata_time.__setattr__('data_type', np.int64)

    write_data(metadata_time, metadata_samples, output_path, 'gait_meta.json', df_windowed)


def detect_gait(input_path: str, output_path: str, path_to_classifier_input: str, config: GaitDetectionConfig) -> None:
    
    # Load the data
    metadata_dict = tsdf.load_metadata_from_path(os.path.join(input_path, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_samples = metadata_dict[config.values_filename]
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, config.classifier_file_name))
    with open(os.path.join(path_to_classifier_input, config.thresholds_file_name), 'r') as f:
        thresholds_str = f.read()
    threshold = np.mean([float(x) for x in thresholds_str.split(' ')])

    # Prepare the data
    clf.feature_names_in_ = [f'{x}_power_below_gait' for x in config.l_accel_cols] + \
                            [f'{x}_power_gait' for x in config.l_accel_cols] + \
                            [f'{x}_power_tremor' for x in config.l_accel_cols] + \
                            [f'{x}_power_above_tremor' for x in config.l_accel_cols] + \
                            ['std_norm_acc'] + [f'cc_{i}_acc' for i in range(1, 17)] + [f'grav_{x}_{y}' for x in config.l_accel_cols for y in ['mean', 'std']] + \
                            [f'{x}_dominant_frequency' for x in config.l_accel_cols]
    X = df.loc[:, clf.feature_names_in_]

    # Make prediction
    df['pred_gait_proba'] = clf.predict_proba(X)[:, 1]
    df['pred_gait'] = df['pred_gait_proba'] > threshold

    # Prepare the metadata
    metadata_samples.__setattr__('file_name', 'gait_values.bin')
    metadata_samples.__setattr__('file_dir_path', output_path)
    metadata_time.__setattr__('file_name', 'gait_time.bin')
    metadata_time.__setattr__('file_dir_path', output_path)

    metadata_samples.__setattr__('channels', ['pred_gait_proba'])
    metadata_samples.__setattr__('units', ['probability'])
    metadata_samples.__setattr__('data_type', np.float32)
    metadata_samples.__setattr__('bits', 32)

    metadata_time.__setattr__('channels', ['time'])
    metadata_time.__setattr__('units', ['relative_time_ms'])
    metadata_time.__setattr__('data_type', np.int32)
    metadata_time.__setattr__('bits', 32)

    write_data(metadata_time, metadata_samples, output_path, 'gait_meta.json', df)


def extract_arm_swing_features(input_path: str, output_path: str, config: ArmSwingFeatureExtractionConfig) -> None:
    # load accelerometer and gyroscope data
    l_dfs = []
    for sensor in ['accelerometer', 'gyroscope']:
        meta_filename = f'{sensor}_meta.json'
        values_filename = f'{sensor}_samples.bin'
        time_filename = f'{sensor}_time.bin'

        metadata_dict = tsdf.load_metadata_from_path(os.path.join(input_path, meta_filename))
        metadata_time = metadata_dict[time_filename]
        metadata_samples = metadata_dict[values_filename]
        l_dfs.append(tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns))

    df = pd.merge(l_dfs[0], l_dfs[1], on='time')

    # Prepare data

    # temporary add "random" predictions
    df[config.pred_gait_colname] = np.concatenate([np.repeat([1], df.shape[0]//3), np.repeat([0], df.shape[0]//3), np.repeat([1], df.shape[0] + 1 - 2*df.shape[0]//3)], axis=0)

    # Process data

    df[config.velocity_colname] = pca_transform_gyroscope(
        df=df, 
        y_gyro_colname=DataColumns.GYROSCOPE_Y,
        z_gyro_colname=DataColumns.GYROSCOPE_Z,
        pred_gait_colname=config.pred_gait_colname
    )

    df[config.angle_colname] = compute_angle(
        velocity_col=df[config.velocity_colname],
        time_col=df[config.time_colname]
    )

    df[config.angle_smooth_colname] = remove_moving_average_angle(
        angle_col=df[config.angle_colname],
        sampling_frequency=config.sampling_frequency
    )

    df = df.loc[df[config.pred_gait_colname]==1].reset_index(drop=True)

    df_segments = create_segments(
        df=df,
        time_colname=config.time_colname,
        segment_nr_colname='segment_nr',
        minimum_gap_s=3
    )

    df_segments = discard_segments(
        df=df_segments,
        time_colname=config.time_colname,
        segment_nr_colname='segment_nr',
        minimum_segment_length_s=3
    )

    l_dfs = []
    for segment_nr in df_segments[config.segment_nr_colname].unique():
        df_single_segment = df_segments.loc[df_segments[config.segment_nr_colname]==segment_nr].copy().reset_index(drop=True)
        l_dfs.append(tabulate_windows(
            df=df_single_segment,
            time_column_name=config.time_colname,
            segment_nr_colname=config.segment_nr_colname,
            data_point_level_cols=config.l_data_point_level_cols,
            window_length_s=config.window_length_s,
            window_step_size_s=config.window_step_size_s,
            segment_nr=segment_nr,
            sampling_frequency=config.sampling_frequency,
            )
        )
    df_windowed = pd.concat(l_dfs).reset_index(drop=True)

    del df, df_segments


    df_windowed['angle_freqs'], df_windowed['angle_fft'] = signal_to_ffts(
        sensor_col=df_windowed[config.angle_smooth_colname],
        window_type=config.window_type,
        sampling_frequency=config.sampling_frequency)

    df_windowed['gyroscope_dominant_frequency'] = df_windowed.apply(
        lambda x: get_dominant_frequency(signal_ffts=x['angle_fft'],
                                        signal_freqs=x['angle_freqs'],
                                        fmin=config.power_band_low_frequency,
                                        fmax=config.power_band_high_frequency
                                        ), axis=1
    )

    df_windowed = df_windowed.drop(columns=['angle_fft', 'angle_freqs'])

    df_windowed['angle_perc_power'] = df_windowed[config.angle_smooth_colname].apply(
        lambda x: compute_perc_power(
            sensor_col=x,
            fmin_band=config.power_band_low_frequency,
            fmax_band=config.power_band_high_frequency,
            fmin_total=config.power_total_low_frequency,
            fmax_total=config.power_total_high_frequency,
            sampling_frequency=config.sampling_frequency,
            window_type=config.window_type
            )
    )

    # note to eScience: why are the columns 'angle_new_minima', 'angle_new_maxima', 
    # 'angle_minima_deleted' and 'angle_maxima deleted' created here? Should a copy
    # of 'df_windowed' be created inside 'extract_angle_extremes' to prevent this from
    # happening?
    extract_angle_extremes(
        df=df_windowed,
        angle_colname=config.angle_smooth_colname,
        dominant_frequency_colname='gyroscope_dominant_frequency',
        sampling_frequency=config.sampling_frequency
    )

    df_windowed = df_windowed.drop(columns=[config.angle_smooth_colname])

    df_windowed['angle_amplitudes'] = extract_range_of_motion(
        angle_extrema_values_col=df_windowed['angle_extrema_values']
    )

    df_windowed = df_windowed.drop(columns=['angle_extrema_values'])

    df_windowed['range_of_motion'] = df_windowed['angle_amplitudes'].apply(lambda x: np.mean(x) if len(x) > 0 else 0).replace(np.nan, 0)

    df_windowed = df_windowed.drop(columns=['angle_amplitudes'])

    extract_peak_angular_velocity(
        df=df_windowed,
        velocity_colname=config.velocity_colname,
        angle_minima_colname='angle_minima',
        angle_maxima_colname='angle_maxima'
    )

    df_windowed = df_windowed.drop(columns=['angle_minima','angle_maxima', 'angle_new_minima',
                                            'angle_new_maxima', config.velocity_colname])

    for dir in ['forward', 'backward']:
        df_windowed[f'{dir}_peak_ang_vel_mean'] = df_windowed[f'{dir}_peak_ang_vel'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
        df_windowed[f'{dir}_peak_ang_vel_std'] = df_windowed[f'{dir}_peak_ang_vel'].apply(lambda x: np.std(x) if len(x) > 0 else 0)

        df_windowed = df_windowed.drop(columns=[f'{dir}_peak_ang_vel'])

    df_windowed['std_norm_acc'] = generate_std_norm(
        df=df_windowed,
        cols=[DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]
    )

    for col in [x for x in df_windowed.columns if 'grav' in x]:
        for stat in ['mean', 'std']:
            df_windowed[f'{col}_{stat}'] = generate_statistics(
                sensor_col=df_windowed[col],
                statistic=stat
            )

    for col in [DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z,
                DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z]:
        df_windowed[f'{col}_freqs'], df_windowed[f'{col}_fft'] = signal_to_ffts(
            sensor_col=df_windowed[col],
            window_type=config.window_type,
            sampling_frequency=config.sampling_frequency
        )

        for bandwidth, frequencies in config.d_frequency_bandwidths.items():
            df_windowed[col+'_'+bandwidth] = df_windowed[col].apply(
                lambda x: compute_power_in_bandwidth(
                    sensor_col=x,
                    fmin=frequencies[0],
                    fmax=frequencies[1],
                    sampling_frequency=config.sampling_frequency,
                    window_type=config.window_type,
                    )
                )

        # dominant frequency
        df_windowed[col+'_dominant_frequency'] = df_windowed.apply(
            lambda x: get_dominant_frequency(
                signal_ffts=x[col+'_fft'], 
                signal_freqs=x[col+'_freqs'],
                fmin=config.power_total_low_frequency,
                fmax=config.power_total_high_frequency
            ), axis=1
        )

    # cepstral coefficients
    for sensor in ['accelerometer', 'gyroscope']:
        if sensor == 'accelerometer':
            fft_cols = [f'{col}_fft' for col in [DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]]
        else:
            fft_cols = [f'{col}_fft' for col in [DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z]]

        df_windowed['total_power'] = compute_power(
            df=df_windowed,
            fft_cols=fft_cols
        )

        cc_cols = generate_cepstral_coefficients(
            total_power_col=df_windowed['total_power'],
            window_length_s=config.window_length_s,
            sampling_frequency=config.sampling_frequency,
            low_frequency=config.power_total_low_frequency,
            high_frequency=config.power_total_high_frequency,
            filter_length=config.filter_length,
            n_dct_filters=config.n_dct_filters
        )

        df_windowed = pd.concat([df_windowed, cc_cols], axis=1)

        for i in range(config.n_dct_filters):
            df_windowed = df_windowed.rename(columns={f'cc_{i+1}': f'cc_{i+1}_{sensor}'})

    # TODO: instead, just extract the columns that are needed?
    l_drop_cols = [DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z,
                DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z,
                f'grav_{DataColumns.ACCELEROMETER_X}', f'grav_{DataColumns.ACCELEROMETER_Y}', f'grav_{DataColumns.ACCELEROMETER_Z}',
                f'{DataColumns.ACCELEROMETER_X}_fft', f'{DataColumns.ACCELEROMETER_Y}_fft', f'{DataColumns.ACCELEROMETER_Z}_fft',
                f'{DataColumns.GYROSCOPE_X}_fft', f'{DataColumns.GYROSCOPE_Y}_fft', f'{DataColumns.GYROSCOPE_Z}_fft',
                f'{DataColumns.ACCELEROMETER_X}_freqs', f'{DataColumns.ACCELEROMETER_Y}_freqs', f'{DataColumns.ACCELEROMETER_Z}_freqs',
                f'{DataColumns.GYROSCOPE_X}_freqs', f'{DataColumns.GYROSCOPE_Y}_freqs', f'{DataColumns.GYROSCOPE_Z}_freqs',
                f'{DataColumns.ACCELEROMETER_X}_fft_power', f'{DataColumns.ACCELEROMETER_Y}_fft_power', f'{DataColumns.ACCELEROMETER_Z}_fft_power',
                f'{DataColumns.GYROSCOPE_X}_fft_power', f'{DataColumns.GYROSCOPE_Y}_fft_power', f'{DataColumns.GYROSCOPE_Z}_fft_power',
                'total_power', 'gyroscope_dominant_frequency', 'window_nr', 'window_end']

    df_windowed = df_windowed.drop(columns=l_drop_cols).rename(columns={'window_start': 'time'})

    # Store data

    end_iso8601 = get_end_iso8601(metadata_samples.start_iso8601, 
                                df_windowed['time'][-1:].values[0] + config.window_length_s)

    metadata_samples.__setattr__('end_iso8601', end_iso8601)
    metadata_samples.__setattr__('file_name', 'arm_swing_values.bin')
    metadata_samples.__setattr__('file_dir_path', output_path)
    metadata_time.__setattr__('end_iso8601', end_iso8601)
    metadata_time.__setattr__('file_name', 'arm_swing_time.bin')
    metadata_time.__setattr__('file_dir_path', output_path)

    metadata_samples.__setattr__('channels', list(config.d_channels_values.keys()))
    metadata_samples.__setattr__('units', list(config.d_channels_values.values()))
    metadata_samples.__setattr__('data_type', np.float32)
    metadata_samples.__setattr__('bits', 32)

    metadata_time.__setattr__('channels', ['time'])
    metadata_time.__setattr__('units', ['relative_time_ms'])
    metadata_time.__setattr__('data_type', np.int32)
    metadata_time.__setattr__('bits', 32)

    write_data(metadata_time, metadata_samples, output_path, 'arm_swing_meta.json', df_windowed)


def detect_arm_swing(input_path: str, output_path: str, path_to_classifier_input: str, config: ArmSwingDetectionConfig) -> None:
    # Load the data
    metadata_dict = tsdf.load_metadata_from_path(os.path.join(input_path, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_samples = metadata_dict[config.values_filename]
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Initialize the classifier
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, config.classifier_file_name))

    # Prepare the data
    clf.feature_names_in_ = [f'{x}_power_below_gait' for x in config.l_accel_cols] + \
                            [f'{x}_power_gait' for x in config.l_accel_cols] + \
                            [f'{x}_power_tremor' for x in config.l_accel_cols] + \
                            [f'{x}_power_above_tremor' for x in config.l_accel_cols] + \
                            [f'grav_{x}_{y}' for x in config.l_accel_cols for y in ['mean', 'std']] + \
                            ['std_norm_acc'] + [f'cc_{i}_accelerometer' for i in range(1, 17)] + \
                            [f'{x}_dominant_frequency' for x in config.l_accel_cols] + \
                            [f'cc_{i}_gyroscope' for i in range(1, 17)] + \
                            [f'{x}_dominant_frequency' for x in config.l_gyro_cols] + \
                            ['range_of_motion', 'forward_peak_ang_vel_mean',
                            'forward_peak_ang_vel_std', 'backward_peak_ang_vel_mean',
                            'backward_peak_ang_vel_std', 'angle_perc_power']
    X = df.loc[:, clf.feature_names_in_]

    # Make prediction
    # df['pred_arm_swing_proba'] = clf.predict_proba(X)[:, 1]
    df['pred_arm_swing'] = clf.predict(X)

    # Prepare the metadata
    metadata_samples.__setattr__('file_name', 'arm_swing_values.bin')
    metadata_samples.__setattr__('file_dir_path', output_path)
    metadata_time.__setattr__('file_name', 'arm_swing_time.bin')
    metadata_time.__setattr__('file_dir_path', output_path)

    metadata_samples.__setattr__('channels', ['pred_arm_swing'])
    metadata_samples.__setattr__('units', ['boolean'])
    metadata_samples.__setattr__('data_type', np.int8)
    metadata_samples.__setattr__('bits', 8)

    metadata_time.__setattr__('channels', ['time'])
    metadata_time.__setattr__('units', ['relative_time_ms'])
    metadata_time.__setattr__('data_type', np.int32)
    metadata_time.__setattr__('bits', 32)

    write_data(metadata_time, metadata_samples, output_path, 'arm_swing_meta.json', df)


def quantify_arm_swing(path_to_feature_input: str, path_to_prediction_input: str, output_path: str, config: ArmSwingQuantificationConfig) -> None:
    # Load the features & predictions
    metadata_dict = tsdf.load_metadata_from_path(os.path.join(path_to_feature_input, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_samples = metadata_dict[config.values_filename]
    df_features = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    metadata_dict = tsdf.load_metadata_from_path(os.path.join(path_to_prediction_input, config.meta_filename))
    metadata_time = metadata_dict[config.time_filename]
    metadata_samples = metadata_dict[config.values_filename]
    df_predictions = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Validate
    # dataframes have same length
    assert df_features.shape[0] == df_predictions.shape[0]

    # dataframes have same time column
    assert df_features['time'].equals(df_predictions['time'])

    # Prepare the data

    # subset features
    l_feature_cols = ['time', 'range_of_motion', 'forward_peak_ang_vel_mean', 'backward_peak_ang_vel_mean']
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

    df_arm_swing = create_segments(
        df=df_arm_swing,
        time_colname='time',
        segment_nr_colname='segment_nr',
        minimum_gap_s=config.segment_gap_s
    )
    df_arm_swing = discard_segments(
        df=df_arm_swing,
        time_colname='time',
        segment_nr_colname='segment_nr',
        minimum_segment_length_s=config.min_segment_length_s
    )

    # Quantify arm swing
    df_aggregates = aggregate_segments(
        df=df_arm_swing,
        time_colname='time',
        segment_nr_colname='segment_nr',
        window_step_size_s=config.window_step_size,
        l_metrics=['range_of_motion', 'peak_ang_vel'],
        l_aggregates=['median'],
        l_quantiles=[0.95]
    )

    df_aggregates['segment_duration_ms'] = df_aggregates['segment_duration_s'] * 1000
    df_aggregates = df_aggregates.drop(columns=['segment_nr'])

    # Store data
    metadata_samples.__setattr__('file_name', 'arm_swing_values.bin')
    metadata_samples.__setattr__('file_dir_path', output_path)
    metadata_time.__setattr__('file_name', 'arm_swing_time.bin')
    metadata_time.__setattr__('file_dir_path', output_path)

    metadata_samples.__setattr__('channels', ['range_of_motion_median', 'range_of_motion_quantile_95',
                                            'peak_ang_vel_median', 'peak_ang_vel_quantile_95'])
    metadata_samples.__setattr__('units', ['deg', 'deg', 'deg/s', 'deg/s'])
    metadata_samples.__setattr__('data_type', np.float32)
    metadata_samples.__setattr__('bits', 32)

    metadata_time.__setattr__('channels', ['time', 'segment_duration_ms'])
    metadata_time.__setattr__('units', ['relative_time_ms', 'ms'])
    metadata_time.__setattr__('data_type', np.int32)
    metadata_time.__setattr__('bits', 32)

    write_data(metadata_time, metadata_samples, output_path, 'arm_swing_meta.json', df_aggregates)


def aggregate_weekly_arm_swing():
    pass

