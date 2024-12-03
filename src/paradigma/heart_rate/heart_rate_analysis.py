from typing import Union
from pathlib import Path

import pandas as pd
import os

import tsdf
import tsdf.constants 
from paradigma.util import read_metadata
from paradigma.windowing import tabulate_windows
from paradigma.constants import DataColumns
from paradigma.heart_rate.heart_rate_analysis_config import SignalQualityFeatureExtractionConfig, SignalQualityClassificationConfig, HeartRateExtractionConfig
from paradigma.heart_rate.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features
from paradigma.heart_rate.heart_rate_estimation import assign_sqa_label, extract_hr_segments, extract_hr_from_segment

def extract_signal_quality_features(df: pd.DataFrame, config: SignalQualityFeatureExtractionConfig) -> pd.DataFrame:
    # Group sequences of timestamps into windows
    df_windowed = tabulate_windows(config, df)

    # Compute statistics of the temporal domain signals
    df_windowed = extract_temporal_domain_features(config, df_windowed, l_quality_stats=['var','mean', 'median', 'kurtosis', 'skewness'])
    
    # Compute statistics of the spectral domain signals
    df_windowed = extract_spectral_domain_features(config, df_windowed)

    df_windowed.drop(columns = ['green'], inplace=True)  # Drop the values channel since it is no longer needed
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
    df.drop(columns = lr_clf.feature_names_in_, inplace=True)  # Drop the features used for classification since they are no longer needed

    return df    



def signal_quality_classification_io(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: SignalQualityClassificationConfig) -> None:
    
    # Load the data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    df = signal_quality_classification(df, config, path_to_classifier_input)


def estimate_heart_rate(df: pd.DataFrame, df_ppg_preprocessed: pd.DataFrame, config:HeartRateExtractionConfig) -> pd.DataFrame:  
    """
    Estimate the heart rate from the PPG signal using the time-frequency domain method.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the PPG signal.
    df_ppg_preprocessed : pd.DataFrame
        The DataFrame containing the preprocessed PPG signal.
    config : HeartRateExtractionConfig
        The configuration for the heart rate estimation.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the heart rate estimations.
    """
       
    # Assign window-level probabilities to individual samples
    ppg_post_prob = df.loc[:, DataColumns.PRED_SQA_PROBA].to_numpy()
    #acc_label = df.loc[:, DataColumns.ACCELEROMETER_LABEL].to_numpy() # Adjust later in data columns to get the correct label, should be first intergrated in feature extraction and classification

    sqa_label = assign_sqa_label(ppg_post_prob, config)
    v_start_idx, v_end_idx = extract_hr_segments(sqa_label, config.min_hr_samples)
    
    fs = config.sampling_frequency

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
