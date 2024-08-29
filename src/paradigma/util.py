import os
import numpy as np
import pandas as pd
from datetime import timedelta
from dateutil import parser
from typing import Tuple

import tsdf
from tsdf import TSDFMetadata

def parse_iso8601_to_datetime(date_str):
    return parser.parse(date_str)

def format_datetime_to_iso8601(datetime):
    return datetime.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'

def get_end_iso8601(start_iso8601, window_length_seconds):
    start_date = parser.parse(start_iso8601)
    end_date = start_date + timedelta(seconds=window_length_seconds)
    return format_datetime_to_iso8601(end_date)

def write_data(metadata_time: TSDFMetadata, metadata_samples: TSDFMetadata,
               output_path: str, output_filename: str, df: pd.DataFrame):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Make sure the iso8601 format is correctly set
    #TODO: this should be properly validated in the tsdf library instead
    start_date = parser.parse(metadata_time.start_iso8601)
    metadata_time.start_iso8601 = format_datetime_to_iso8601(start_date)
    end_date = parser.parse(metadata_time.end_iso8601)
    metadata_time.end_iso8601 = format_datetime_to_iso8601(end_date)
    start_date = parser.parse(metadata_samples.start_iso8601)
    metadata_samples.start_iso8601 = format_datetime_to_iso8601(start_date)
    end_date = parser.parse(metadata_samples.end_iso8601)
    metadata_samples.end_iso8601 = format_datetime_to_iso8601(end_date)

    # TODO: improve the way the metadata is stored at a different location
    metadata_time.file_dir_path = output_path
    metadata_samples.file_dir_path = output_path

    # store binaries and metadata
    tsdf.write_dataframe_to_binaries(output_path, df, [metadata_time, metadata_samples])
    tsdf.write_metadata([metadata_time, metadata_samples], output_filename)

def read_metadata(input_path: str, meta_filename: str, time_filename: str, values_filename: str) -> Tuple[TSDFMetadata, TSDFMetadata]:
    metadata_dict = tsdf.load_metadata_from_path(os.path.join(input_path, meta_filename))
    metadata_time = metadata_dict[time_filename]
    metadata_samples = metadata_dict[values_filename]
    return metadata_time, metadata_samples
