import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.linear_model import LogisticRegression

import tsdf

from paradigma.constants import DataColumns
from paradigma.tremor.tremor_analysis_config import TremorFeatureExtractionConfig
from paradigma.tremor.feature_extraction import extract_spectral_domain_features
from paradigma.windowing import tabulate_windows
from paradigma.util import get_end_iso8601, write_df_data, read_metadata


def extract_tremor_features(df: pd.DataFrame, config: TremorFeatureExtractionConfig) -> pd.DataFrame:
    # group sequences of timestamps into windows
    df_windowed = tabulate_windows(config,df)
    
    # transform the signals from the temporal domain to the spectral domain using the fast fourier transform
    # and extract spectral features
    df_windowed = extract_spectral_domain_features(config, df_windowed,config.l_gyroscope_cols)

    return df_windowed

def extract_tremor_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: TremorFeatureExtractionConfig) -> None:
    # Load data
    metadata_time, metadata_samples = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_samples], tsdf.constants.ConcatenationType.columns)

    # Extract tremor features
    df_windowed = extract_tremor_features(df, config)

    # Store data
    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_windowed[config.time_colname][-1:].values[0] + config.window_length_s))

    metadata_samples.end_iso8601 = end_iso8601
    metadata_samples.file_name = 'tremor_values.bin'
    metadata_time.end_iso8601 = end_iso8601
    metadata_time.file_name = 'tremor_time.bin'

    metadata_samples.channels = list(config.d_channels_values.keys())
    metadata_samples.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_samples, output_path, 'tremor_meta.json', df_windowed)


