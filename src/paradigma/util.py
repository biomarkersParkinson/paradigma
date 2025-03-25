import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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
    

def invert_watch_side(df: pd.DataFrame, side: str, sensor='both') -> np.ndarray:
    """
    Invert the data based on the watch side.

    Parameters
    ----------
    df : pd.DataFrame
        The data.
    side : str
        The watch side (left or right).
    sensor: str
        The sensor(s) to invert: 'accelerometer', 'gyroscope', or 'both'

    Returns
    -------
    pd.DataFrame
        The inverted data.

    """
    if side not in ["left", "right"]:
        raise ValueError(f"Unsupported side: {side}")
    if sensor not in ['accelerometer', 'gyroscope', 'both']:
        raise ValueError(f"Unsupported sensor: {sensor}")

    elif side == "right":
        if sensor in ['gyroscope', 'both']:
            df[DataColumns.GYROSCOPE_Y] *= -1
            df[DataColumns.GYROSCOPE_Z] *= -1
        if sensor in ['accelerometer', 'both']:
            df[DataColumns.ACCELEROMETER_X] *= -1

    return df

def aggregate_parameter(parameter: np.ndarray, aggregate: str) -> np.ndarray:
    """
    Aggregate a parameter based on the specified method.
    
    Parameters
    ----------
    parameter : np.ndarray
        The parameter to aggregate.
        
    aggregate : str
        The aggregation method to apply.
        
    Returns
    -------
    np.ndarray
        The aggregated parameter.
    """
    if aggregate == 'mean':
        return np.mean(parameter)
    elif aggregate == 'median':
        return np.median(parameter)
    elif aggregate == 'mode':
        unique_values, counts = np.unique(parameter, return_counts=True)
        return unique_values[np.argmax(counts)]
    elif aggregate == '90p':
        return np.percentile(parameter, 90)
    elif aggregate == '95p':
        return np.percentile(parameter, 95)
    elif aggregate == '99p':
        return np.percentile(parameter, 99)
    elif aggregate == 'std':
        return np.std(parameter)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregate}")

def merge_predictions_with_timestamps(
        df_ts: pd.DataFrame, 
        df_predictions: pd.DataFrame, 
        pred_proba_colname: str,
        window_length_s: float, 
        fs: int
    ) -> pd.DataFrame:
    """
    Merges prediction probabilities with timestamps by expanding overlapping windows
    into individual timestamps and averaging probabilities per unique timestamp.

    Parameters:
    ----------
    df_ts : pd.DataFrame
        DataFrame containing timestamps to be merged with predictions.
        Must include the timestamp column specified in `DataColumns.TIME`.

    df_predictions : pd.DataFrame
        DataFrame containing prediction windows with start times and probabilities.
        Must include:
        - A column for window start times (defined by `DataColumns.TIME`).
        - A column for prediction probabilities (defined by `DataColumns.PRED_GAIT_PROBA`).

    pred_proba_colname : str
        The column name for the prediction probabilities in `df_predictions`.

    window_length_s : float
        The length of the prediction window in seconds.

    fs : int
        The sampling frequency of the data.
        
    Returns:
    -------
    pd.DataFrame
        Updated `df_ts` with an additional column for averaged prediction probabilities.

    Steps:
    ------
    1. Expand prediction windows into individual timestamps using NumPy broadcasting.
    2. Flatten the timestamps and prediction probabilities into single arrays.
    3. Aggregate probabilities by unique timestamps using pandas `groupby`.
    4. Merge the aggregated probabilities with the input `df_ts`.

    Notes:
    ------
    - Rounding is applied to timestamps to mitigate floating-point inaccuracies.
    - Fully vectorized for speed and scalability, avoiding any row-wise operations.
    """
    # Step 1: Generate all timestamps for prediction windows using NumPy broadcasting
    window_length = int(window_length_s * fs)
    timestamps = (
        df_predictions[DataColumns.TIME].values[:, None] +
        np.arange(0, window_length) / fs
    )
    
    # Flatten timestamps and probabilities into a single array for efficient processing
    flat_timestamps = timestamps.ravel()
    flat_proba = np.repeat(
        df_predictions[pred_proba_colname].values,
        window_length
    )

    # Step 2: Create a DataFrame for expanded data
    expanded_df = pd.DataFrame({
        DataColumns.TIME: flat_timestamps,
        pred_proba_colname: flat_proba
    })

    # Step 3: Round timestamps and aggregate probabilities
    expanded_df[DataColumns.TIME] = expanded_df[DataColumns.TIME].round(2)
    mean_proba = expanded_df.groupby(DataColumns.TIME, as_index=False).mean()

    # Step 4: Round timestamps in `df_ts` and merge
    df_ts[DataColumns.TIME] = df_ts[DataColumns.TIME].round(2)
    df_ts = pd.merge(df_ts, mean_proba, how='left', on=DataColumns.TIME)
    df_ts = df_ts.dropna(subset=[pred_proba_colname])

    return df_ts


def select_hours(df: pd.DataFrame, select_hours_start: str, select_hours_end: str) -> pd.DataFrame:
    
    """
    Select hours of interest from the data to include in the aggregation step.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    select_hours_start: str
        The start time of the selected hours in "HH:MM" format.
    
    select_hours_end: str
            The end time of the selected hours in "HH:MM" format.

    Returns
    -------
    pd.DataFrame
        The selected data.

    """

    select_hours_start = datetime.strptime(select_hours_start, '%H:%M').time() # convert to time object
    select_hours_end = datetime.strptime(select_hours_end, '%H:%M').time() 
    df_subset = df[df['time_dt'].dt.time.between(select_hours_start, select_hours_end)] # select the hours of interest

    return df_subset

def select_days(df: pd.DataFrame, min_hours_per_day: int) -> pd.DataFrame:

    """
    Select days of interest from the data to include in the aggregation step.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with column 'time_dt' in which the date is stored.

    min_hours_per_day: int
        The minimum number of hours per day required for including the day in the aggregation step.


    Returns
    -------
    pd.DataFrame
        The selected data.

    """

    min_s_per_day = min_hours_per_day * 3600
    window_length_s = df['time_dt'].diff().dt.total_seconds()[1] # determine the length of the first window in seconds
    min_windows_per_day = min_s_per_day / window_length_s
    df_subset = df.groupby(df['time_dt'].dt.date).filter(lambda x: len(x) >= min_windows_per_day)

    return df_subset