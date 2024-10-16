import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.linear_model import LogisticRegression

import tsdf

from paradigma.constants import DataColumns
from paradigma.tremor_analysis_config import TremorFeatureExtractionConfig
from paradigma.feature_extraction import signal_to_ffts, compute_power, \
    generate_cepstral_coefficients,compute_power_in_bandwidth
from paradigma.windowing import tabulate_windows
from paradigma.util import get_end_iso8601, write_data, read_metadata


def extract_tremor_features(df: pd.DataFrame, config: TremorFeatureExtractionConfig) -> pd.DataFrame:
    # group sequences of timestamps into windows
    df_windowed = tabulate_windows(
        df=df,
        time_column_name=config.time_colname,
        data_point_level_cols=config.l_data_point_level_cols,
        window_length_s=config.window_length_s,
        window_step_size_s=config.window_step_size_s,
        sampling_frequency=config.sampling_frequency
        )
    
    for col in config.l_gyroscope_cols:

        # transform the temporal signal to the spectral domain using the fast fourier transform
        df_windowed[f'{col}_freqs'], df_windowed[f'{col}_fft'] = signal_to_ffts(
            sensor_col=df_windowed[col],
            window_type=config.window_type,
            sampling_frequency=config.sampling_frequency
            )

    # compute the power summed over the individual frequency bandwidths to obtain the total power
    df_windowed['total_power'] = compute_power(
        df=df_windowed,
        fft_cols=[f'{col}_fft' for col in config.l_gyroscope_cols])

    
    cc_cols = generate_cepstral_coefficients(
        total_power_col=df_windowed['total_power'],
        window_length_s=config.window_length_s,
        sampling_frequency=config.sampling_frequency,
        low_frequency=config.spectrum_low_frequency,
        high_frequency=config.spectrum_high_frequency,
        n_filters=config.n_dct_filters_cc,
        n_coefficients=config.n_coefficients_cc
        )

    df_windowed = pd.concat([df_windowed, cc_cols], axis=1)

    # compute the power in distinct frequency bandwidths
    df_windowed['arm_mov_power'] = df_windowed.apply(lambda x: compute_power_in_bandwidth(
                sensor_col=x['total_power'],
                fmin=config.d_frequency_bandwidths[0],
                fmax=config.d_frequency_bandwidths[1],
                sampling_frequency=config.sampling_frequency,
                window_type=config.window_type,
                ), axis=1
            )
 

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

    write_data(metadata_time, metadata_samples, output_path, 'tremor_meta.json', df_windowed)

