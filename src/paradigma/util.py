import os
import numpy as np
import pandas as pd
from datetime import timedelta
from dateutil import parser
from typing import List, Tuple

import tsdf
from tsdf import TSDFMetadata


def parse_iso8601_to_datetime(date_str):
    return parser.parse(date_str)


def format_datetime_to_iso8601(datetime):
    return datetime.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def get_end_iso8601(start_iso8601, window_length_seconds):
    start_date = parser.parse(start_iso8601)
    end_date = start_date + timedelta(seconds=window_length_seconds)
    return format_datetime_to_iso8601(end_date)


def write_np_data(
    metadata_time: TSDFMetadata,
    np_array_time: np.ndarray, 
    metadata_values: TSDFMetadata,
    np_array_values: np.ndarray,
    output_path: str,
    output_filename: str,
):
    """
    Write the numpy arrays to binary files and store the metadata.

    Parameters
    ----------
    metadata_time : TSDFMetadata
        Metadata for the time column.
    np_array_time : np.ndarray
        The numpy array for the time column.
    metadata_values : TSDFMetadata
        Metadata for the samples columns.
    np_array_values : np.ndarray
        The numpy array for the samples columns.
    output_path : str
        The path where the files will be stored.
    output_filename : str
        The filename for the metadata.

    """
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # TODO: improve the way the metadata is stored at a different location
    metadata_time.file_dir_path = output_path
    metadata_values.file_dir_path = output_path

    # store binaries and metadata
    time_tsdf = tsdf.write_binary_file(file_dir=output_path, file_name=metadata_time.file_name, data=np_array_time, metadata=metadata_time.get_plain_tsdf_dict_copy())

    samples_tsdf = tsdf.write_binary_file(file_dir=output_path, file_name=metadata_values.file_name, data=np_array_values, metadata=metadata_values.get_plain_tsdf_dict_copy())

    tsdf.write_metadata([time_tsdf, samples_tsdf], output_filename)


def write_df_data(
    metadata_time: TSDFMetadata,
    metadata_values: TSDFMetadata,
    output_path: str,
    output_filename: str,
    df: pd.DataFrame,
):
    """
    Write the Pandas DataFrame to binary files and store the metadata.

    Parameters
    ----------
    metadata_time : TSDFMetadata
        Metadata for the time column.
    metadata_values : TSDFMetadata
        Metadata for the samples columns.
    output_path : str
        The path where the files will be stored.
    output_filename : str
        The filename for the metadata.
    df : pd.DataFrame
        The DataFrame to be stored.

    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Make sure the iso8601 format is correctly set
    # TODO: this should be properly validated in the tsdf library instead
    start_date = parser.parse(metadata_time.start_iso8601)
    metadata_time.start_iso8601 = format_datetime_to_iso8601(start_date)
    end_date = parser.parse(metadata_time.end_iso8601)
    metadata_time.end_iso8601 = format_datetime_to_iso8601(end_date)
    start_date = parser.parse(metadata_values.start_iso8601)
    metadata_values.start_iso8601 = format_datetime_to_iso8601(start_date)
    end_date = parser.parse(metadata_values.end_iso8601)
    metadata_values.end_iso8601 = format_datetime_to_iso8601(end_date)

    # TODO: improve the way the metadata is stored at a different location
    metadata_time.file_dir_path = output_path
    metadata_values.file_dir_path = output_path

    # store binaries and metadata
    tsdf.write_dataframe_to_binaries(output_path, df, [metadata_time, metadata_values])
    tsdf.write_metadata([metadata_time, metadata_values], output_filename)

def read_metadata(
    input_path: str, meta_filename: str, time_filename: str, values_filename: str
) -> Tuple[TSDFMetadata, TSDFMetadata]:
    metadata_dict = tsdf.load_metadata_from_path(
        os.path.join(input_path, meta_filename)
    )
    metadata_time = metadata_dict[time_filename]
    metadata_values = metadata_dict[values_filename]
    return metadata_time, metadata_values

def load_metadata_list(
    dir_path: str, meta_filename: str, filenames: List[str]
) -> List[TSDFMetadata]:
    """
    Load the metadata objects from a metadata file according to the specified binaries.

    Parameters
    ----------
    dir_path : str
        The dir path where the metadata file is stored.
    meta_filename : str
        The filename of the metadata file.
    filenames : List[str]
        The list of binary files of which the metadata files need to be loaded
    
    """	
    metadata_dict = tsdf.load_metadata_from_path(
        os.path.join(dir_path, meta_filename)
    )
    metadata_list = []
    for filename in filenames:
        metadata_list.append(metadata_dict[filename])

    return metadata_list

class WindowedDataExtractor:
    """
    A utility class for extracting specific column indices and slices 
    from a list of windowed column names.
    Attributes
    ----------
    column_indices : dict
        A dictionary mapping column names to their indices.
    Methods
    -------
    get_index(col)
        Returns the index of a specific column.
    get_slice(cols)
        Returns a slice object for a range of consecutive columns.
    """

    def __init__(self, windowed_cols):
        """
        Initialize the WindowedDataExtractor.
        Parameters
        ----------
        windowed_cols : list of str
            A list of column names in the windowed data.
        Raises
         ------
        ValueError
            If the list of `windowed_cols` is empty.
        """
        if not windowed_cols:
            raise ValueError("The list of windowed columns cannot be empty.")
        self.column_indices = {col: idx for idx, col in enumerate(windowed_cols)}

    def get_index(self, col):
        """
        Get the index of a specific column.
        Parameters
        ----------
        col : str
            The name of the column to retrieve the index for.
        Returns
        -------
        int
            The index of the specified column.
        Raises
        ------
        ValueError
            If the column is not found in the `windowed_cols` list.
        """
        if col not in self.column_indices:
            raise ValueError(f"Column '{col}' not found in windowed_cols.")
        return self.column_indices[col]

    def get_slice(self, cols):
        """
        Get a slice object for a range of consecutive columns.
        Parameters
        ----------
        cols : list of str
            A list of consecutive column names to define the slice.
        Returns
        -------
        slice
            A slice object spanning the indices of the given columns.
        Raises
        ------
        ValueError
            If one or more columns in `cols` are not found in the `windowed_cols` list.
        """
        if not all(col in self.column_indices for col in cols):
            missing = [col for col in cols if col not in self.column_indices]
            raise ValueError(f"The following columns are missing from windowed_cols: {missing}")
        start_idx = self.column_indices[cols[0]]
        end_idx = self.column_indices[cols[-1]] + 1
        return slice(start_idx, end_idx)