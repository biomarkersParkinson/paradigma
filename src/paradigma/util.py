import os
import numpy as np
import pandas as pd
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

def load_tsdf_dataframe(path_to_data, sensor_name):
    path_to_sensor_data = os.path.join(path_to_data, sensor_name)
    meta_filename = f"{sensor_name.upper()}_meta.json"
    time_filename = f"{sensor_name.upper()}_time.bin"
    values_filename = f"{sensor_name.upper()}_values.bin"

    metadata_time, metadata_values = read_metadata(path_to_sensor_data, meta_filename, time_filename, values_filename)
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


def transform_time_array(
    time_array: pd.Series,
    input_units: str,
    input_unit_type: str,
    output_units: str,
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
        The time unit type of the output time array. ParaDigMa expects `TimeUnit.RELATIVE_MS`.
    start_time : float, optional
        The start time of the time array in UNIX seconds (default is 0.0)

    Returns
    -------
    np.ndarray
        The transformed time array in seconds, with the specified time unit type.

    Notes
    -----
    - The function handles different time units (`TimeUnit.DIFFERENCE_MS`, `TimeUnit.ABSOLUTE_MS`, `TimeUnit.RELATIVE_MS`).
    - The transformation allows for scaling of the time array, converting between time unit types (e.g., relative, absolute, or difference).
    - When converting to `TimeUnit.RELATIVE_MS`, the function calculates the relative time starting from the provided or default start time.
    """
    # Transform to relative time (`TimeUnit.RELATIVE_MS`) 
    if input_unit_type == TimeUnit.DIFFERENCE:
    # Convert a series of differences into cumulative sum to reconstruct original time series.
        time_array = np.cumsum(np.double(time_array))
    elif input_unit_type == TimeUnit.ABSOLUTE:
        # Set the start time if not provided.
        if np.isclose(start_time, 0.0, rtol=1e-09, atol=1e-09):
            start_time = time_array[0]
        # Convert absolute time stamps into a time series relative to start_time.
        time_array = (time_array - start_time) 

    # Transform the time array from `TimeUnit.RELATIVE_MS` to the specified time unit type
    if output_unit_type == TimeUnit.ABSOLUTE:
        # Converts time array to absolute time by adding the start time to each element.
        time_array = time_array + start_time
    elif output_unit_type == TimeUnit.DIFFERENCE:
        # Creates a new array starting with 0, followed by the differences between consecutive elements.
        time_array = np.diff(np.insert(time_array, 0, start_time))
    elif output_unit_type == TimeUnit.RELATIVE:
        # The array is already in relative format, do nothing.
        pass
    return time_array


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