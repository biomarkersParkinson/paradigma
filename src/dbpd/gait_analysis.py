import os

import tsdf

from dbpd.gait_analysis_config import *
from dbpd.feature_extraction import *
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

def extract_arm_swing_features():
    pass

def detect_arm_swing():
    pass

def quantify_arm_swing():
    pass

def aggregate_weekly_arm_swing():
    pass

