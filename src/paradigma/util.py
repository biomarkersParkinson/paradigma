import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from dateutil import parser
from typing import List, Tuple

import tsdf
from tsdf import TSDFMetadata

from paradigma.constants import DataColumns, TimeUnit


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

def load_tsdf_dataframe(path_to_data, prefix, meta_suffix='meta.json', time_suffix='time.bin', values_suffix='values.bin'):
    meta_filename = f"{prefix}_{meta_suffix}"
    time_filename = f"{prefix}_{time_suffix}"
    values_filename = f"{prefix}_{values_suffix}"

    metadata_time, metadata_values = read_metadata(path_to_data, meta_filename, time_filename, values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    return df, metadata_time, metadata_values

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
# TODO: ideally something like this should be possible directly in the tsdf library
def extract_meta_from_tsdf_files(tsdf_data_dir : str) -> List[dict]:
    """
    For each given TSDF directory, transcribe TSDF metadata contents to a list of dictionaries.
    
    Parameters
    ----------
    tsdf_data_dir : str
        Path to the directory containing TSDF metadata files.

    Returns
    -------
    List[Dict]
        List of dictionaries with metadata from each JSON file in the directory.

    Examples
    --------
    >>> extract_meta_from_tsdf_files('/path/to/tsdf_data')
    [{'start_iso8601': '2021-06-27T16:52:20Z', 'end_iso8601': '2021-06-27T17:52:20Z'}, ...]
    """
    metas = []
    
    # Collect all metadata JSON files in the specified directory
    meta_list = list(Path(tsdf_data_dir).rglob('*_meta.json'))
    for meta_file in meta_list:
        with open(meta_file, 'r') as file:
            json_obj = json.load(file)
            meta_data = {
                'tsdf_meta_fullpath': str(meta_file),
                'subject_id': json_obj['subject_id'],
                'start_iso8601': json_obj['start_iso8601'],
                'end_iso8601': json_obj['end_iso8601']
            }
            metas.append(meta_data)
    
    return metas


def transform_time_array(
    time_array: pd.Series,
    input_unit_type: str,
    output_unit_type: str,
    start_time: float = 0.0,
) -> np.ndarray:
    """
    Transforms the time array to relative time (when defined in delta time) and scales the values.

    Parameters
    ----------
    time_array : pd.Series
        The time array to transform.
    input_unit_type : str
        The time unit type of the input time array.
    output_unit_type : str
        The time unit type of the output time array. ParaDigMa expects `TimeUnit.RELATIVE_S`.
    start_time : float, optional
        The start time of the time array in UNIX seconds (default is 0.0)

    Returns
    -------
    np.ndarray
        The transformed time array in seconds, with the specified time unit type.

    Notes
    -----
    - The function handles different time units (`TimeUnit.RELATIVE_MS`, `TimeUnit.RELATIVE_S`, `TimeUnit.ABSOLUTE_MS`, `TimeUnit.ABSOLUTE_S`, `TimeUnit.DIFFERENCE_MS`, `TimeUnit.DIFFERENCE_S`).
    - The transformation allows for scaling of the time array, converting between time unit types (e.g., relative, absolute, or difference).
    - When converting to `TimeUnit.RELATIVE_MS`, the function calculates the relative time starting from the provided or default start time.
    """
    input_units = input_unit_type.split('_')[-1].lower()
    output_units = output_unit_type.split('_')[-1].lower()

    if input_units == output_units:
        scale_factor = 1
    elif input_units == 's' and output_units == 'ms':
        scale_factor = 1e3
    elif input_units == 'ms' and output_units == 's':
        scale_factor = 1 / 1e3
    else:
        raise ValueError(f"Unsupported time units conversion: {input_units} to {output_units}")
    
    # Transform to relative time (`TimeUnit.RELATIVE_MS`) 
    if input_unit_type == TimeUnit.DIFFERENCE_MS or input_unit_type == TimeUnit.DIFFERENCE_S:
    # Convert a series of differences into cumulative sum to reconstruct original time series.
        time_array = np.cumsum(np.double(time_array))
    elif input_unit_type == TimeUnit.ABSOLUTE_MS or input_unit_type == TimeUnit.ABSOLUTE_S:
        # Set the start time if not provided.
        if np.isclose(start_time, 0.0, rtol=1e-09, atol=1e-09):
            start_time = time_array[0]
        # Convert absolute time stamps into a time series relative to start_time.
        time_array = (time_array - start_time) 

    # Transform the time array from `TimeUnit.RELATIVE_MS` to the specified time unit type
    if output_unit_type == TimeUnit.ABSOLUTE_MS or output_unit_type == TimeUnit.ABSOLUTE_S:
        # Converts time array to absolute time by adding the start time to each element.
        time_array = time_array + start_time
    elif output_unit_type == TimeUnit.DIFFERENCE_MS or output_unit_type == TimeUnit.DIFFERENCE_S:
        # Creates a new array starting with 0, followed by the differences between consecutive elements.
        time_array = np.diff(np.insert(time_array, 0, start_time))
    elif output_unit_type == TimeUnit.RELATIVE_MS or output_unit_type == TimeUnit.RELATIVE_S:
        # The array is already in relative format, do nothing.
        pass

    return time_array * scale_factor


def convert_units_accelerometer(data: np.ndarray, units: str) -> np.ndarray:
    """
    Convert acceleration data to g.

    Parameters
    ----------
    data : np.ndarray
        The acceleration data.

    units : str
        The unit of the data (currently supports g and m/s^2).

    Returns
    -------
    np.ndarray
        The acceleration data in g.

    """
    if units == "m/s^2":
        return data / 9.81
    elif units == "g":
        return data
    else:
        raise ValueError(f"Unsupported unit: {units}")
    

def convert_units_gyroscope(data: np.ndarray, units: str) -> np.ndarray:
    """
    Convert gyroscope data to deg/s.
    
    Parameters
    ----------
    data : np.ndarray
        The gyroscope data.
        
    units : str
        The unit of the data (currently supports deg/s and rad/s).
        
    Returns
    -------
    np.ndarray
        The gyroscope data in deg/s.
        
    """
    if units == "deg/s":
        return data
    elif units == "rad/s":
        return np.degrees(data)
    else:
        raise ValueError(f"Unsupported unit: {units}")
    

def invert_watch_side(df: pd.DataFrame, side: str) -> np.ndarray:
    """
    Invert the data based on the watch side.

    Parameters
    ----------
    df : pd.DataFrame
        The data.
    side : str
        The watch side (left or right).

    Returns
    -------
    pd.DataFrame
        The inverted data.

    """
    if side not in ["left", "right"]:
        raise ValueError(f"Unsupported side: {side}")
    elif side == "right":
        df[DataColumns.GYROSCOPE_Y] *= -1
        df[DataColumns.GYROSCOPE_Z] *= -1
        df[DataColumns.ACCELEROMETER_X] *= -1

    return df


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