from typing import Union
from pathlib import Path

import pandas as pd
import os

import tsdf
import tsdf.constants 
from paradigma.heart_rate.heart_rate_analysis_config import SignalQualityFeatureExtractionConfig
from paradigma.util import read_metadata
from paradigma.segmenting import tabulate_windows_legacy
from paradigma.heart_rate.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features
from paradigma.heart_rate.heart_rate_analysis_config import SignalQualityClassificationConfig
from paradigma.constants import DataColumns


def extract_signal_quality_features(df: pd.DataFrame, config: SignalQualityFeatureExtractionConfig) -> pd.DataFrame:
    # Group sequences of timestamps into windows
    df_windowed = tabulate_windows_legacy(config, df)

    # Compute statistics of the temporal domain signals
    df_windowed = extract_temporal_domain_features(config, df_windowed, quality_stats=['var','mean', 'median', 'kurtosis', 'skewness'])
    
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