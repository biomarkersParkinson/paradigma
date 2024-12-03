from typing import Union
from pathlib import Path

import pandas as pd

import tsdf
import tsdf.constants 
from paradigma.heart_rate.heart_rate_analysis_config import SignalQualityFeatureExtractionConfig
from paradigma.util import read_metadata
from paradigma.windowing import tabulate_windows
from paradigma.heart_rate.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features

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
