import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dateutil import parser
from typing import Union
from pathlib import Path

from datetime import datetime
import os

import tsdf
import tsdf.constants
from paradigma.hr.heart_rate_analysis_config import SignalQualityFeatureExtractionConfig, SignalQualityClassificationConfig
from paradigma.util import read_metadata, write_np_data, write_df_data
from paradigma.constants import DataColumns, UNIX_TICKS_MS, DataUnits, TimeUnit


def extract_signal_quality_features(df: pd.DataFrame, classifier_path: str, output_path: str, config: SignalQualityFeatureExtractionConfig) -> None:
    # load data
    metadata_time_ppg, metadata_samples_ppg = read_metadata(input_path, "PPG_meta.json", "PPG_time.bin", "PPG_samples.bin")
    df_ppg = tsdf.load_dataframe_from_binaries([metadata_time_ppg, metadata_samples_ppg], tsdf.constants.ConcatenationType.columns)
    arr_ppg = df_ppg[DataColumns.PPG].to_numpy()
    relative_time_ppg = df_ppg[DataColumns.TIME].to_numpy()
    
    metadata_time_acc, metadata_samples_acc = read_metadata(input_path, "accelerometer_meta.json", "accelerometer_time.bin", "accelerometer_samples.bin")
    df_acc = tsdf.load_dataframe_from_binaries([metadata_time_acc, metadata_samples_acc], tsdf.constants.ConcatenationType.columns)
    arr_acc = df_acc[[DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]].to_numpy()

    sampling_frequency_ppg = config.sampling_frequency_ppg
    sampling_frequency_imu = config.sampling_frequency_imu

    # Parameters
    epoch_length = 6  # in seconds
    overlap = 5  # in seconds

    # Number of samples in epoch
    samples_per_epoch_ppg = int(epoch_length * sampling_frequency_ppg)
    samples_per_epoch_acc = int(epoch_length * sampling_frequency_imu)

    # Calculate number of samples to shift for each epoch
    samples_shift_ppg = int((epoch_length - overlap) * sampling_frequency_ppg)
    samples_shift_acc = int((epoch_length - overlap) * sampling_frequency_imu)

    pwelchwin_acc = int(3 * sampling_frequency_imu)  # window length for pwelch
    pwelchwin_ppg = int(3 * sampling_frequency_ppg)  # window length for pwelch
    noverlap_acc = int(0.5 * pwelchwin_acc)  # overlap for pwelch
    noverlap_ppg = int(0.5 * pwelchwin_ppg)  # overlap for pwelch

    f_bin_res = 0.05  # the threshold is set based on this binning
    nfft_ppg = np.arange(0, sampling_frequency_ppg / 2, f_bin_res)  # frequency bins for pwelch ppg
    nfft_acc = np.arange(0, sampling_frequency_imu / 2, f_bin_res)  # frequency bins for pwelch imu

    features_ppg_scaled = []
    feature_acc = []
    t_unix_feat_total = []
    count = 0
    acc_idx = 0

    # Read the classifier (it contains mu and sigma)
    clf = read_PPG_quality_classifier(classifier_path)
    # not used here: lr_model = clf['model']
    arr_mu = clf['mu'][:, 0]
    arr_sigma = clf['sigma'][:, 0]

    scaler = StandardScaler()
    scaler.mean_ = arr_mu
    scaler.scale_ = arr_sigma

    ppg_start_time = parser.parse(metadata_time_ppg.start_iso8601)

    # Loop over 6s segments for both PPG and IMU and calculate features
    for i in range(0, len(arr_ppg) - samples_per_epoch_ppg + 1, samples_shift_ppg):
        if acc_idx + samples_per_epoch_acc > len(arr_acc):  # For the last epoch, check if the segment for IMU is too short (not 6 seconds)
            break
        else:
            acc_segment = arr_acc[acc_idx:acc_idx + samples_per_epoch_acc, :]  # Extract the IMU window (6 seconds)

        ppg_segment = arr_ppg[i:i + samples_per_epoch_ppg]  # Extract the PPG window (6 seconds)

        count += 1

        # Feature extraction + scaling
        features = extract_ppg_features(ppg_segment, sampling_frequency_ppg)
        features = features.reshape(1, -1)
        features_ppg_scaled.append(scaler.transform(features)[0])

        # Calculating PSD (power spectral density) of IMU and PPG
        f1, pxx1 = welch(acc_segment, sampling_frequency_imu, window='hann', nperseg=pwelchwin_acc, noverlap=None, nfft=len(nfft_acc))
        PSD_imu = np.sum(pxx1, axis=0)  # sum over the three axes
        f2, pxx2 = welch(ppg_segment, sampling_frequency_ppg, window='hann', nperseg=pwelchwin_ppg, noverlap=None, nfft=len(nfft_ppg))
        PSD_ppg = np.sum(pxx2)  # this does nothing, equal to PSD_ppg = pxx2

        # IMU feature extraction
        feature_acc.append(calculate_power_ratio(f1, PSD_imu, f2, PSD_ppg))  # Calculate the power ratio of the accelerometer signal in the PPG frequency range

        # time channel
        t_unix_feat_total.append((relative_time_ppg[i] + ppg_start_time.timestamp()) * UNIX_TICKS_MS)  # Save in absolute unix time ms
        acc_idx += samples_shift_acc  # update IMU_idx

    # Convert lists to numpy arrays
    features_ppg_scaled = np.array(features_ppg_scaled)
    feature_acc = np.array(feature_acc)
    t_unix_feat_total = np.array(t_unix_feat_total)

    # Synchronization information
    # TODO: store this, as this is needed for the HR pipeline
    # v_sync_ppg_total = np.array([
    #     ppg_indices[0],  # start index
    #     ppg_indices[1],  # end index
    #     segment_ppg[0],  # Segment index
    #     count  # Number of epochs in the segment
    # ])

    metadata_features_ppg = metadata_samples_ppg
    metadata_features_ppg.file_name = "features_ppg_samples.bin"
    metadata_features_ppg.channels = [
    DataColumns.VARIANCE,
    DataColumns.MEAN,
    DataColumns.MEDIAN,
    DataColumns.KURTOSIS,
    DataColumns.SKEWNESS,
    DataColumns.DOMINANT_FREQUENCY,
    DataColumns.RELATIVE_POWER,
    DataColumns.SPECTRAL_ENTROPY,
    DataColumns.SIGNAL_NOISE_RATIO,
    DataColumns.SECOND_HIGHEST_PEAK
]
    metadata_features_ppg.units = [
    DataUnits.NONE,  # variance
    DataUnits.NONE,  # mean
    DataUnits.NONE,  # median
    DataUnits.NONE,  # kurtosis
    DataUnits.NONE,  # skewness
    DataUnits.FREQUENCY,  # dominant_frequency (measured in Hz)
    DataUnits.NONE,  # relative_power (unitless measure)
    DataUnits.NONE,  # spectral_entropy (unitless measure)
    DataUnits.NONE,  # signal_noise_ratio (unitless measure)
    DataUnits.NONE   # second_highest_peak (depends on the feature, assume no unit)
]
    metadata_time_ppg.file_name = "features_ppg_time.bin"
    metadata_time_ppg.channels = [DataColumns.TIME]
    metadata_time_ppg.units = [TimeUnit.RELATIVE_MS]
    
    
    metadata_features_acc = metadata_samples_acc
    metadata_features_acc.file_name = "features_acc_samples.bin"
    metadata_features_acc.channels = [DataColumns.POWER_RATIO]
    metadata_features_acc.units = [DataUnits.NONE]

    metadata_time_acc.file_name = "features_acc_time.bin"
    metadata_time_acc.channels = [DataColumns.TIME]
    metadata_time_acc.units = [TimeUnit.RELATIVE_MS]

    write_np_data(metadata_time_ppg, t_unix_feat_total, metadata_features_ppg, 
                  features_ppg_scaled, output_path, 'features_ppg_meta.json')
    write_np_data(metadata_time_acc, t_unix_feat_total, metadata_features_acc,
                  feature_acc, output_path, 'feature_acc_meta.json')


def extract_signal_quality_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: SignalQualityFeatureExtractionConfig) -> None:
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Extract gait features
    df_windowed = extract_signal_quality_features(df, config)

    # Store data
    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_windowed[config.time_colname][-1:].values[0] + config.window_length_s))

    metadata_samples.end_iso8601 = end_iso8601
    metadata_samples.file_name = 'gait_values.bin'
    metadata_time.end_iso8601 = end_iso8601
    metadata_time.file_name = 'gait_time.bin'

    metadata_samples.channels = list(config.d_channels_values.keys())
    metadata_samples.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'gait_meta.json', df_windowed)

def signal_quality_classification(input_path: str, classifier_path: str, output_path: str, config: SignalQualityClassificationConfig) -> None:
    # load data
    metadata_time_ppg, metadata_samples_ppg = read_metadata(input_path, "features_ppg_meta.json", "features_ppg_time.bin", "features_ppg_samples.bin")
    df_ppg = tsdf.load_dataframe_from_binaries([metadata_time_ppg, metadata_samples_ppg], tsdf.constants.ConcatenationType.columns)
    arr_ppg = df_ppg[DataColumns.PPG].to_numpy()
    relative_time_ppg = df_ppg[DataColumns.TIME].to_numpy()
    
    metadata_time_acc, metadata_samples_acc = read_metadata(input_path, "feature_acc_meta.json", "feature_acc_time.bin", "feature_acc_samples.bin")
    df_acc = tsdf.load_dataframe_from_binaries([metadata_time_acc, metadata_samples_acc], tsdf.constants.ConcatenationType.columns)
    arr_acc = df_acc[[DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]].to_numpy()



    # Read the classifier (it contains mu and sigma)
    clf = read_PPG_quality_classifier(classifier_path)
    lr_model = clf['model']
    arr_mu = clf['mu'][:, 0]
    arr_sigma = clf['sigma'][:, 0]

    # Load preprocessed features and scaling model (assuming these are passed in or processed earlier)
    # This is auto generated code to import the inputs (we already imported the inputs), this is here just to show the names used
    features_ppg_scaled = np.load(f"path_to_quality_features/features_ppg_scaled.npy")
    feature_acc = np.load(f"path_to_quality_features/feature_acc.npy")
    t_unix_feat_total = np.load(f"path_to_quality_features/t_unix_feat_total.npy")
    v_sync_ppg_total = np.load(f"path_to_quality_features/v_sync_ppg_total.npy")

    threshold_acc = 0.13  # Final threshold

    # Calculate posterior probability using Logistic Regression (scikit-learn)
    ppg_post_prob = lr_model.predict_proba(features_ppg_scaled)
    ppg_post_prob_HQ = ppg_post_prob[:, 0]

    # IMU classification based on threshold
    acc_label = feature_acc < threshold_acc  # boolean array

    # Storage of classification in tsdf
    data_class = {}
    unix_ticks_ms = 1000  # Assuming 1 ms per tick
    data_class[1] = (t_unix_feat_total / unix_ticks_ms).astype(np.int32)  # 32-bit integer
    data_class[2] = ppg_post_prob_HQ.astype(np.float32)  # 32-bit float
    data_class[3] = acc_label.astype(np.int8)  # 8-bit integer
    data_class[4] = v_sync_ppg_total.astype(np.int32)  # 64-bit integer for synchronization
    data_class[5] = feature_acc.astype(np.float32)  # 32-bit float

    # Time and metadata handling
    start_time_iso = datetime.utcfromtimestamp(t_unix_feat_total[0] / unix_ticks_ms).isoformat() + 'Z'
    end_time_iso = datetime.utcfromtimestamp(t_unix_feat_total[-1] / unix_ticks_ms).isoformat() + 'Z'

    # Create metadata templates (adjust this as needed for your system)
    metafile_pre_template = config.metadata_list_ppg[config.values_idx_ppg]  # Load template

    metafile_pre_template["start_iso8601"] = start_time_iso
    metafile_pre_template["end_iso8601"] = end_time_iso

    # Define metadata files
    meta_class = []

    # 1. Time metadata
    metafile_time = metafile_pre_template.copy()
    metafile_time["channels"] = ['time']
    metafile_time["units"] = ['time_absolute_unix_ms']
    metafile_time["file_name"] = 'classification_sqa_time.bin'
    meta_class.append(metafile_time)

    # 2. PPG post-probability metadata
    metafile_values_ppg = metafile_pre_template.copy()
    metafile_values_ppg["channels"] = ['post probability']
    metafile_values_ppg["units"] = ['probability']
    metafile_values_ppg["freq_sampling_original"] = config.fs_ppg_est  # Sampling rate in Hz
    metafile_values_ppg["file_name"] = 'classification_sqa_ppg.bin'
    meta_class.append(metafile_values_ppg)

    # 3. IMU classification metadata
    metafile_values_imu = metafile_pre_template.copy()
    metafile_values_imu["channels"] = ['accelerometer classification']
    metafile_values_imu["units"] = ['boolean_num']
    metafile_values_imu["freq_sampling_original"] = config.fs_imu_est  # Sampling rate in Hz
    metafile_values_imu["file_name"] = 'classification_sqa_imu.bin'
    meta_class.append(metafile_values_imu)

    # 4. Synchronization metadata
    metafile_sync = metafile_pre_template.copy()
    metafile_sync["channels"] = ['ppg start index', 'ppg end index', 'ppg segment index', 'number of windows']
    metafile_sync["units"] = ['index', 'index', 'index', 'none']
    metafile_sync["file_name"] = 'classification_sqa_sync.bin'
    meta_class.append(metafile_sync)

    # 5. IMU feature metadata
    metafile_values_imu_feat = metafile_pre_template.copy()
    metafile_values_imu_feat["channels"] = ['Relative power']
    metafile_values_imu_feat["units"] = ['none']
    metafile_values_imu_feat["file_name"] = 'classification_sqa_feat_imu.bin'
    metafile_values_imu_feat["bin_width"] = 2 * config.f_bin_res  # Bin width of the relative power ratio feature
    meta_class.append(metafile_values_imu_feat)

    # Define metadata file name
    mat_metadata_file_name = "classification_sqa_meta.json"

    # Save the data and metadata
    save_tsdf_data(meta_class, data_class, path_to_signal_quality, mat_metadata_file_name)
