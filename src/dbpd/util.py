import os
import numpy as np
import pandas as pd
from datetime import timedelta
from dateutil import parser

import tsdf
from tsdf import TSDFMetadata

def get_end_iso8601(start_iso8601, window_length_seconds):
    start_date = parser.parse(start_iso8601)
    end_date = start_date + timedelta(seconds=window_length_seconds)
    # TODO: this is not valid iso8601:
    return end_date.strftime('%d-%b-%Y %H:%M:%S') + ' UTC'

def write_data(metadata_time: TSDFMetadata, metadata_samples: TSDFMetadata,
               output_path: str, output_filename: str, df: pd.DataFrame):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # store binaries and metadata
    tsdf.write_dataframe_to_binaries(output_path, df, [metadata_time, metadata_samples])
    tsdf.write_metadata([metadata_time, metadata_samples], output_filename)
