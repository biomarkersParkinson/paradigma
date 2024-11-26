import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dateutil import parser
from typing import Union
from pathlib import Path

from datetime import datetime
import os
import pandas as pd

import tsdf
import tsdf.constants 
from paradigma.heart_rate.heart_rate_analysis_config import SignalQualityFeatureExtractionConfig, SignalQualityClassificationConfig, HeartRateExtractionConfig
from paradigma.util import read_metadata, load_metadata_list, write_np_data, write_df_data, get_end_iso8601
from paradigma.constants import DataColumns, UNIX_TICKS_MS, DataUnits, TimeUnit
from paradigma.windowing import tabulate_windows, create_segments, discard_segments
from paradigma.heart_rate.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features
from paradigma.heart_rate.signal_quality_predictions import classify_data_quality
from paradigma.heart_rate.heart_rate_estimation import assign_sqa_label, extract_hr_segments, extract_hr_from_segment

def extract_signal_quality_features(df: pd.DataFrame, config: SignalQualityFeatureExtractionConfig) -> pd.DataFrame:
    # Group sequences of timestamps into windows
    df_windowed = tabulate_windows(config, df)

    # Compute statistics of the temporal domain signals
    df_windowed = extract_temporal_domain_features(config, df_windowed, l_quality_stats=['var','mean', 'median', 'kurtosis', 'skewness'])
    
    # Compute statistics of the spectral domain signals
    df_windowed = extract_spectral_domain_features(config, df_windowed)
    return df_windowed


def extract_signal_quality_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: SignalQualityFeatureExtractionConfig) -> None:
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)
    
    # Extract gait features
    df_windowed = extract_signal_quality_features(df, config)
    return df_windowed


def signal_quality_classification(df: pd.DataFrame, config: SignalQualityClassificationConfig, path_to_classifier_input: Union[str, Path]) -> pd.DataFrame:
    """
    Classify the signal quality of the PPG signal using a logistic regression classifier.
    The classifier is trained on features extracted from the PPG signal.
    The features are extracted using the extract_signal_quality_features function.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the PPG signal.
    config : SignalQualityClassificationConfig
        The configuration for the signal quality classification.
    path_to_classifier_input : Union[str, Path]
        The path to the directory containing the classifier.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the PPG signal quality predictions.
    """
    
    clf = pd.read_pickle(os.path.join(path_to_classifier_input, 'classifiers', config.classifier_file_name))
    lr_clf = clf['model']
    mu = clf['mu']
    sigma = clf['sigma']

    # Prepare the data
    lr_clf.feature_names_in_ = ['var', 'mean', 'median', 'kurtosis', 'skewness', 'f_dom', 'rel_power', 'spectral_entropy', 'signal_to_noise', 'auto_corr']
    X = df.loc[:, lr_clf.feature_names_in_]

    # Normalize features using mu and sigma
    X_normalized = X.copy()
    for idx, feature in enumerate(lr_clf.feature_names_in_):
        X_normalized[feature] = (X[feature] - mu[idx]) / sigma[idx]

    # Make predictions for PPG signal quality assessment
    df[DataColumns.PRED_SQA_PROBA] = lr_clf.predict_proba(X_normalized)[:, 0]                   

    return df    



def signal_quality_classification_io(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: SignalQualityClassificationConfig) -> None:
    
    # Load the data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    df = signal_quality_classification(df, config, path_to_classifier_input)
    
    
    # load data
    metadata_time_ppg, metadata_samples_ppg = read_metadata(input_path, "features_ppg_meta.json", "features_ppg_time.bin", "features_ppg_samples.bin")
    # df_ppg = tsdf.load_dataframe_from_binaries([metadata_time_ppg, metadata_samples_ppg], tsdf.constants.ConcatenationType.columns)
    # arr_ppg = df_ppg[DataColumns.PPG].to_numpy()
    # relative_time_ppg = df_ppg[DataColumns.TIME].to_numpy()
    
    # metadata_time_acc, metadata_samples_acc = read_metadata(input_path, "feature_acc_meta.json", "feature_acc_time.bin", "feature_acc_samples.bin")
    # df_acc = tsdf.load_dataframe_from_binaries([metadata_time_acc, metadata_samples_acc], tsdf.constants.ConcatenationType.columns)
    # arr_acc = df_acc[[DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]].to_numpy()

    # # Load preprocessed features and scaling model (assuming these are passed in or processed earlier)
    # # This is auto generated code to import the inputs (we already imported the inputs), this is here just to show the names used
    # features_ppg_scaled = np.load(f"path_to_quality_features/features_ppg_scaled.npy")
    # feature_acc = np.load(f"path_to_quality_features/feature_acc.npy")
    # t_unix_feat_total = np.load(f"path_to_quality_features/t_unix_feat_total.npy")
    # v_sync_ppg_total = np.load(f"path_to_quality_features/v_sync_ppg_total.npy")

    # threshold_acc = 0.13  # Final threshold

    # # Calculate posterior probability using Logistic Regression (scikit-learn)
    # ppg_post_prob = lr_model.predict_proba(features_ppg_scaled)
    # ppg_post_prob_HQ = ppg_post_prob[:, 0]

    # # IMU classification based on threshold
    # acc_label = feature_acc < threshold_acc  # boolean array

    # # Storage of classification in tsdf
    # data_class = {}
    # unix_ticks_ms = 1000  # Assuming 1 ms per tick
    # data_class[1] = (t_unix_feat_total / unix_ticks_ms).astype(np.int32)  # 32-bit integer
    # data_class[2] = ppg_post_prob_HQ.astype(np.float32)  # 32-bit float
    # data_class[3] = acc_label.astype(np.int8)  # 8-bit integer
    # data_class[4] = v_sync_ppg_total.astype(np.int32)  # 64-bit integer for synchronization
    # data_class[5] = feature_acc.astype(np.float32)  # 32-bit float

    # # Time and metadata handling
    # start_time_iso = datetime.utcfromtimestamp(t_unix_feat_total[0] / unix_ticks_ms).isoformat() + 'Z'
    # end_time_iso = datetime.utcfromtimestamp(t_unix_feat_total[-1] / unix_ticks_ms).isoformat() + 'Z'

    # # Create metadata templates (adjust this as needed for your system)
    # metafile_pre_template = config.metadata_list_ppg[config.values_idx_ppg]  # Load template

    # metafile_pre_template["start_iso8601"] = start_time_iso
    # metafile_pre_template["end_iso8601"] = end_time_iso

    # # Define metadata files
    # meta_class = []

    # # 1. Time metadata
    # metafile_time = metafile_pre_template.copy()
    # metafile_time["channels"] = ['time']
    # metafile_time["units"] = ['time_absolute_unix_ms']
    # metafile_time["file_name"] = 'classification_sqa_time.bin'
    # meta_class.append(metafile_time)

    # # 2. PPG post-probability metadata
    # metafile_values_ppg = metafile_pre_template.copy()
    # metafile_values_ppg["channels"] = ['post probability']
    # metafile_values_ppg["units"] = ['probability']
    # metafile_values_ppg["freq_sampling_original"] = config.fs_ppg_est  # Sampling rate in Hz
    # metafile_values_ppg["file_name"] = 'classification_sqa_ppg.bin'
    # meta_class.append(metafile_values_ppg)

    # # 3. IMU classification metadata
    # metafile_values_imu = metafile_pre_template.copy()
    # metafile_values_imu["channels"] = ['accelerometer classification']
    # metafile_values_imu["units"] = ['boolean_num']
    # metafile_values_imu["freq_sampling_original"] = config.fs_imu_est  # Sampling rate in Hz
    # metafile_values_imu["file_name"] = 'classification_sqa_imu.bin'
    # meta_class.append(metafile_values_imu)

    # # 4. Synchronization metadata
    # metafile_sync = metafile_pre_template.copy()
    # metafile_sync["channels"] = ['ppg start index', 'ppg end index', 'ppg segment index', 'number of windows']
    # metafile_sync["units"] = ['index', 'index', 'index', 'none']
    # metafile_sync["file_name"] = 'classification_sqa_sync.bin'
    # meta_class.append(metafile_sync)

    # # 5. IMU feature metadata
    # metafile_values_imu_feat = metafile_pre_template.copy()
    # metafile_values_imu_feat["channels"] = ['Relative power']
    # metafile_values_imu_feat["units"] = ['none']
    # metafile_values_imu_feat["file_name"] = 'classification_sqa_feat_imu.bin'
    # metafile_values_imu_feat["bin_width"] = 2 * config.f_bin_res  # Bin width of the relative power ratio feature
    # meta_class.append(metafile_values_imu_feat)

    # # Define metadata file name
    # mat_metadata_file_name = "classification_sqa_meta.json"

    # Save the data and metadata
    #save_tsdf_data(meta_class, data_class, path_to_signal_quality, mat_metadata_file_name)

def estimate_heart_rate(df: pd.DataFrame, df_ppg_preprocessed: pd.DataFrame, config:HeartRateExtractionConfig, output_path:str) -> None:  
    # Assign window-level probabilities to individual samples
    ppg_post_prob = df.loc[:, DataColumns.PRED_SQA_PROBA].to_numpy()
    #acc_label = df.loc[:, DataColumns.ACCELEROMETER_LABEL].to_numpy() # Adjust later in data columns to get the correct label, should be first intergrated in feature extraction and classification

    sqa_label = assign_sqa_label(ppg_post_prob, config)
    v_start_idx, v_end_idx = extract_hr_segments(sqa_label, config.min_hr_samples)
    
    fs = config.fs_ppg

    for i in range(len(v_start_idx)):
            # Relevant PPG segment
            rel_segment = df_ppg_preprocessed[v_start_idx[i]:v_end_idx[i]]


            # Check whether the epoch can be extended by 2s on both sides 
            if v_start_idx[i] < 2 * fs or v_end_idx[i] > len(df_ppg_preprocessed) - 2 * fs:
                continue

            # Extract relevant PPG segment for HR estimation by adding 2s on both sides to overcome edge effects of the time-frequency method
            v_ppg_spwvd = df_ppg_preprocessed[DataColumns.PPG][v_start_idx[i] - 2 * fs : v_end_idx[i] + 2 * fs]

            hr_est = extract_hr_from_segment(v_ppg_spwvd, config.tfd_length, fs, config.kern_type, config.kern_params)

            # Corresponding HR estimation time array
            hr_time = rel_time[::hr_est_samples]
            t_epoch_unix = np.array(hr_time) * config.unix_ticks_ms + ts_sync

            # Save output
            v_hr_ppg.append(hr_est)
            t_hr_unix.append(t_epoch_unix)
    

    return df_hr_est

def estimate_heart_rate_io(input_path: str, raw_input_path: str, output_path: str) -> None:
    # Load metadata and sync data (assuming similar steps to your other examples)
    metadata_ppg_list = load_metadata_list(raw_input_path, "PPG_meta.json", ["PPG_time.bin", "PPG_samples.bin"])
    df_ppg_preprocessed = tsdf.load_dataframe_from_binaries(metadata_ppg_list, tsdf.constants.ConcatenationType.columns)

    estimate_heart_rate(input_path, df_ppg_preprocessed, output_path)