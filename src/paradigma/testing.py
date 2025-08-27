import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tsdf
from typing import List

from paradigma.classification import ClassifierPackage
from paradigma.config import IMUConfig, PPGConfig, GaitConfig, TremorConfig, PulseRateConfig
from paradigma.constants import DataColumns, TimeUnit
from paradigma.pipelines.gait_pipeline import extract_gait_features, detect_gait, \
    extract_arm_activity_features, filter_gait
from paradigma.pipelines.tremor_pipeline import extract_tremor_features, detect_tremor, \
    aggregate_tremor
from paradigma.pipelines.pulse_rate_pipeline import extract_signal_quality_features, signal_quality_classification, \
    aggregate_pulse_rate
from paradigma.preprocessing import preprocess_imu_data, preprocess_ppg_data
from paradigma.util import read_metadata, write_df_data, get_end_iso8601, merge_predictions_with_timestamps


def preprocess_imu_data_io(path_to_input: str | Path, path_to_output: str | Path, 
                           config: IMUConfig, sensor: str, watch_side: str) -> None:
    # Load data
    metadata_time, metadata_values = read_metadata(str(path_to_input), str(config.meta_filename),
                                                    str(config.time_filename), str(config.values_filename))
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Preprocess data
    df = preprocess_imu_data(df=df, config=config, sensor=sensor, watch_side=watch_side)

    # Store data
    for sensor, units in zip(['accelerometer', 'gyroscope'], ['g', config.rotation_units]):
        if any(sensor in col for col in df.columns):
            df_sensor = df[[DataColumns.TIME] + [x for x in df.columns if sensor in x]]

            metadata_values.channels = [x for x in df.columns if sensor in x]
            metadata_values.units = list(np.repeat(units, len(metadata_values.channels)))
            metadata_values.scale_factors = []
            metadata_values.file_name = f'{sensor}_values.bin'

            metadata_time.file_name = f'{sensor}_time.bin'
            metadata_time.units = [TimeUnit.RELATIVE_S]

            write_df_data(metadata_time, metadata_values, path_to_output, f'{sensor}_meta.json', df_sensor)


def preprocess_ppg_data_io(path_to_input_ppg: str | Path, path_to_input_imu: str | Path,
                           output_path: str | Path, ppg_config: PPGConfig, 
                           imu_config: IMUConfig) -> None:
    """	
    Preprocess PPG and IMU data by resampling, filtering, and aligning the data segments.

    Parameters
    ----------
    path_to_input_ppg : str | Path
        Path to the PPG data.
    path_to_input_imu : str | Path
        Path to the IMU data.
    output_path : str | Path
        Path to store the preprocessed data.
    ppg_config : PPGConfig
        Configuration object for PPG preprocessing.
    imu_config : IMUConfig
        Configuration object for IMU preprocessing.

    Returns
    -------
    None
    """ 

    # Load PPG data
        # Load data
    metadata_time_ppg, metadata_values_ppg = read_metadata(path_to_input_ppg, ppg_config.meta_filename,
                                                    ppg_config.time_filename, ppg_config.values_filename)
    df_ppg = tsdf.load_dataframe_from_binaries([metadata_time_ppg, metadata_values_ppg], tsdf.constants.ConcatenationType.columns)

    # Load IMU data
    metadata_time_imu, metadata_values_imu = read_metadata(path_to_input_imu, imu_config.meta_filename,
                                                    imu_config.time_filename, imu_config.values_filename)
    df_imu = tsdf.load_dataframe_from_binaries([metadata_time_imu, metadata_values_imu], tsdf.constants.ConcatenationType.columns)

    # Drop the gyroscope columns from the IMU data
    cols_to_drop = df_imu.filter(regex='^gyroscope_').columns
    df_acc = df_imu.drop(cols_to_drop, axis=1)

    # Preprocess data
    df_ppg_proc, df_acc_proc = preprocess_ppg_data(
        df_ppg=df_ppg, 
        df_acc=df_acc, 
        ppg_config=ppg_config, 
        imu_config=imu_config,
        start_time_ppg=metadata_time_ppg.start_iso8601,
        start_time_imu=metadata_time_imu.start_iso8601
    )

    # Store data
    metadata_values_imu.channels = list(imu_config.d_channels_accelerometer.keys())
    metadata_values_imu.units = list(imu_config.d_channels_accelerometer.values())
    metadata_values_imu.file_name = 'accelerometer_values.bin'
    metadata_time_imu.units = [TimeUnit.ABSOLUTE_MS]
    metadata_time_imu.file_name = 'accelerometer_time.bin'
    write_df_data(metadata_time_imu, metadata_values_imu, output_path, 'accelerometer_meta.json', df_acc_proc)

    metadata_values_ppg.channels = list(ppg_config.d_channels_ppg.keys())
    metadata_values_ppg.units = list(ppg_config.d_channels_ppg.values())
    metadata_values_ppg.file_name = 'PPG_values.bin'
    metadata_time_ppg.units = [TimeUnit.ABSOLUTE_MS]
    metadata_time_ppg.file_name = 'PPG_time.bin'
    write_df_data(metadata_time_ppg, metadata_values_ppg, output_path, 'PPG_meta.json', df_ppg_proc)


def extract_gait_features_io(
        config: GaitConfig,
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


def detect_gait_io(
        config: GaitConfig, 
        path_to_input: str | Path, 
        path_to_output: str | Path, 
        full_path_to_classifier_package: str | Path, 
    ) -> None:
    
    # Load the data
    config.set_filenames('gait')

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


def extract_arm_activity_features_io(
        config: GaitConfig, 
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

    gait_preprocessing_config = GaitConfig(step='gait')

    df = merge_predictions_with_timestamps(
        df_ts=df_ts, 
        df_predictions=df_pred_gait, 
        pred_proba_colname=DataColumns.PRED_GAIT_PROBA,
        window_length_s=gait_preprocessing_config.window_length_s,
        fs=gait_preprocessing_config.sampling_frequency
    )

    # Add a column for predicted gait based on a fitted threshold
    df[DataColumns.PRED_GAIT] = (df[DataColumns.PRED_GAIT_PROBA] >= clf_package.threshold).astype(int)

    # Filter the DataFrame to only include predicted gait (1)
    df = df.loc[df[DataColumns.PRED_GAIT]==1].reset_index(drop=True)

    # Extract arm activity features
    config = GaitConfig(step='arm_activity')
    df_features = extract_arm_activity_features(
        df=df, 
        config=config,
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


def filter_gait_io(
        config: GaitConfig, 
        path_to_input: str | Path, 
        path_to_output: str | Path, 
        full_path_to_classifier_package: str | Path, 
    ) -> None:
    # Load the data
    config.set_filenames('arm_activity')

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


def extract_tremor_features_io(input_path: str | Path, output_path: str | Path, config: TremorConfig) -> None:
    # Load data
    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Extract tremor features
    df_windowed = extract_tremor_features(df, config)

    # Store data
    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_windowed[DataColumns.TIME][-1:].values[0] + config.window_length_s))

    metadata_values.end_iso8601 = end_iso8601
    metadata_values.file_name = 'tremor_values.bin'
    metadata_time.end_iso8601 = end_iso8601
    metadata_time.file_name = 'tremor_time.bin'

    metadata_values.channels = list(config.d_channels_values.keys())
    metadata_values.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_values, output_path, 'tremor_meta.json', df_windowed)


def detect_tremor_io(input_path: str | Path, output_path: str | Path, path_to_classifier_input: str | Path, config: TremorConfig) -> None:
    
    # Load the data
    config.set_filenames('tremor')

    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    df = detect_tremor(df, config, path_to_classifier_input)

    # Prepare the metadata
    metadata_values.file_name = 'tremor_values.bin'
    metadata_time.file_name = 'tremor_time.bin'

    metadata_values.channels = list(config.d_channels_values.keys())
    metadata_values.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_values, output_path, 'tremor_meta.json', df)


def aggregate_tremor_io(path_to_feature_input: str | Path, path_to_prediction_input: str | Path, output_path: str | Path, config: TremorConfig) -> None:
    
    # Load the features & predictions
    metadata_time, metadata_values = read_metadata(path_to_feature_input, config.meta_filename, config.time_filename, config.values_filename)
    df_features = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    metadata_dict = tsdf.load_metadata_from_path(path_to_prediction_input / config.meta_filename)
    metadata_time = metadata_dict[config.time_filename]
    metadata_values = metadata_dict[config.values_filename]
    df_predictions = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Subset features
    df_features = df_features[['tremor_power', 'below_tremor_power']]

    # Concatenate predictions and tremor power
    df = pd.concat([df_predictions, df_features], axis=1)

    # Compute aggregated tremor measures
    d_aggregates = aggregate_tremor(df, config)

    # Save output
    with open(output_path / "tremor_aggregates.json", 'w') as json_file:
        json.dump(d_aggregates, json_file, indent=4)


def extract_signal_quality_features_io(input_path: str | Path, output_path: str | Path, ppg_config: PulseRateConfig, acc_config: PulseRateConfig) -> pd.DataFrame:
    """
    Extract signal quality features from the PPG signal and save them to a file.

    Parameters
    ----------
    input_path : str | Path
        The path to the directory containing the preprocessed PPG and accelerometer data.
    output_path : str | Path
        The path to the directory where the extracted features will be saved.
    ppg_config: PulseRateConfig
        The configuration for the signal quality feature extraction of the ppg signal.
    acc_config: PulseRateConfig
        The configuration for the signal quality feature extraction of the accelerometer signal.

    Returns
    -------
    df_windowed : pd.DataFrame
        The DataFrame containing the extracted signal quality features.

    """	
    # Load PPG data
    metadata_time, metadata_values = read_metadata(input_path, ppg_config.meta_filename, ppg_config.time_filename, ppg_config.values_filename)
    df_ppg = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)
    
    # Load IMU data
    metadata_time, metadata_values = read_metadata(input_path, acc_config.meta_filename, acc_config.time_filename, acc_config.values_filename)
    df_acc = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Extract signal quality features
    df_windowed = extract_signal_quality_features(df_ppg, df_acc, ppg_config, acc_config)
    
    # Save the extracted features
    #TO BE ADDED
    return df_windowed


def signal_quality_classification_io(input_path: str | Path, output_path: str | Path, path_to_classifier_input: str | Path, config: PulseRateConfig) -> None:
    
    # Load the data
    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df_windowed = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    df_sqa = signal_quality_classification(df_windowed, config, path_to_classifier_input)


def aggregate_pulse_rate_io(
        full_path_to_input: str | Path, 
        full_path_to_output: str | Path, 
        aggregates: List[str] = ['mode', '99p']
    ) -> None:
    """
    Extract pulse rate from the PPG signal and save the aggregated pulse rate estimates to a file.

    Parameters
    ----------
    input_path : str | Path
        The path to the directory containing the pulse rate estimates.
    output_path : str | Path
        The path to the directory where the aggregated pulse rate estimates will be saved.
    aggregates : List[str]
        The list of aggregation methods to be used for the pulse rate estimates. The default is ['mode', '99p'].
    """

    # Load the pulse rate estimates
    with open(full_path_to_input, 'r') as f:
        df_pr = json.load(f)
    
    # Aggregate the pulse rate estimates
    pr_values = df_pr['pulse_rate'].values
    df_pr_aggregates = aggregate_pulse_rate(pr_values, aggregates)

    # Save the aggregated pulse rate estimates
    with open(full_path_to_output, 'w') as json_file:
        json.dump(df_pr_aggregates, json_file, indent=4)