from typing import List

import numpy as np
import pandas as pd

from paradigma.constants import DataColumns
from paradigma.util import deprecated


def tabulate_windows(
    df: pd.DataFrame,
    columns: List[str],
    window_length_s: float,
    window_step_length_s: float,
    fs: int,
) -> np.ndarray:
    """
    Split the given DataFrame into overlapping windows of specified length and step size.

    This function extracts windows of data from the specified columns of the DataFrame, based on
    the window length and step size provided in the configuration. The windows are returned in
    a 3D NumPy array, where the first dimension represents the window index, the second dimension
    represents the time steps within the window, and the third dimension represents the columns
    of the data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be windowed.
    columns : list of str
        A list of column names from the DataFrame that will be used for windowing.
    window_length_s : float
        The length of each window in seconds.
    window_step_length_s : float
        The step size between consecutive windows in seconds.
    fs : int
        The sampling frequency of the data in Hz.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (n_windows, window_size, n_columns), where:
        - `n_windows` is the number of windows that can be formed from the data.
        - `window_size` is the length of each window in terms of the number of time steps.
        - `n_columns` is the number of columns in the input DataFrame specified by `columns`.

        If the length of the data is shorter than the specified window size, an empty array is returned.

    Notes
    -----
    This function uses `np.lib.stride_tricks.sliding_window_view` to generate sliding windows of data.
    The step size is applied to extract windows at intervals.
    If the data is insufficient for at least one window, an empty array will be returned.

    Example
    -------
    config = Config(window_length_s=5, window_step_length_s=1, sampling_frequency=100)
    df = pd.DataFrame({'col1': np.random.randn(100), 'col2': np.random.randn(100)})
    columns = ['col1', 'col2']
    windows = tabulate_windows(config, df, columns)
    """
    window_size = int(window_length_s * fs)
    window_step_size = int(window_step_length_s * fs)
    n_columns = len(columns)

    data = df[columns].values

    # Check if data length is sufficient
    if len(data) < window_size:
        return np.empty(
            (0, window_size, n_columns)
        )  # Return an empty array if insufficient data

    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=(window_size, n_columns)
    )[::window_step_size].squeeze()

    # Ensure 3D shape (n_windows, window_size, n_columns)
    if windows.ndim == 2:  # Single window case
        windows = windows[np.newaxis, :, :]  # Add a new axis at the start

    return windows


def tabulate_windows_legacy(config, df, agg_func="first"):
    """
    Efficiently creates a windowed dataframe from the input dataframe using vectorized operations.

    Parameters
    ----------
    config : object
        A configuration object containing:
        - `window_length_s`: The number of seconds per window.
        - `window_step_length_s`: The number of seconds to shift between windows.
        - `sampling_frequency`: The sampling frequency in Hz.
        - `single_value_colnames`: List of column names where a single value (e.g., mean) is needed.
        - `list_value_colnames`: List of column names where all 600 values should be stored in a list.
    agg_func : str or callable, optional
        Aggregation function for single-value columns. Can be 'mean', 'first', or a custom callable.
        Default is 'first'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where each row corresponds to a window, containing:
        - `window_nr`: The window number (starting from 1).
        - `window_start`: The start time of the window.
        - `window_end`: The end time of the window.
        - Aggregated values for `single_value_colnames`.
        - Lists of values for `list_value_colnames`.

    """
    # If single_value_colnames or list_value_colnames is None, default to an empty list
    if config.single_value_colnames is None:
        config.single_value_colnames = []
    if config.list_value_colnames is None:
        config.list_value_colnames = []

    window_length = int(config.window_length_s * config.sampling_frequency)
    window_step_size = int(config.window_step_length_s * config.sampling_frequency)

    n_rows = len(df)
    if window_length > n_rows:
        raise ValueError(
            f"Window size ({window_length}) cannot be greater than the number of rows ({n_rows}) in the dataframe."
        )

    # Create indices for window start positions
    window_starts = np.arange(0, n_rows - window_length + 1, window_step_size)

    # Prepare the result for the final DataFrame
    result = []

    # Handle single value columns with vectorized operations
    agg_func_map = {
        "mean": np.mean,
        "first": lambda x: x[0],
    }

    # Check if agg_func is a callable (custom function) or get the function from the map
    if callable(agg_func):
        agg_func_np = agg_func
    else:
        agg_func_np = agg_func_map.get(
            agg_func, agg_func_map["mean"]
        )  # Default to 'mean' if agg_func is not recognized

    for window_nr, start in enumerate(window_starts, 1):
        end = start + window_length
        window = df.iloc[start:end]

        agg_data = {
            "window_nr": window_nr,
            "window_start": window[DataColumns.TIME].iloc[0],
            "window_end": window[DataColumns.TIME].iloc[-1],
        }

        # Aggregate single-value columns
        for col in config.single_value_colnames:
            if col in window.columns:  # Only process columns that exist in the window
                agg_data[col] = agg_func_np(window[col].values)

        # Collect list-value columns efficiently using numpy slicing
        for col in config.list_value_colnames:
            if col in window.columns:  # Only process columns that exist in the window
                agg_data[col] = window[col].values.tolist()

        result.append(agg_data)

    # Convert result list into a DataFrame
    windowed_df = pd.DataFrame(result)

    # Ensure the column order is as desired: window_nr, window_start, window_end, pre_or_post, and then the rest
    desired_order = (
        ["window_nr", "window_start", "window_end"]
        + config.single_value_colnames
        + config.list_value_colnames
    )

    return windowed_df[desired_order]


def create_segments(
    time_array: np.ndarray,
    max_segment_gap_s: float,
):
    # Calculate the difference between consecutive time values
    time_diff = np.diff(time_array, prepend=0.0)

    # Create a boolean mask for where the gap exceeds the threshold
    gap_exceeds = time_diff > max_segment_gap_s

    # Create the segment number based on the cumulative sum of the gap_exceeds mask
    segments = gap_exceeds.cumsum() + 1

    return segments


def discard_segments(
    df: pd.DataFrame,
    segment_nr_colname: str,
    min_segment_length_s: float,
    fs: int,
    format: str = "timestamps",
) -> pd.DataFrame:
    """
    Remove segments smaller than a specified size and reset segment enumeration.

    This function filters out segments from the DataFrame that are smaller than a
    given minimum size, based on the configuration. After removing small segments,
    the segment numbers are reset to start from 1.

    Parameters
    ----------
    config : object
        A configuration object containing:
        - `min_segment_length_s`: The minimum segment length in seconds.
        - `sampling_frequency`: The sampling frequency in Hz.
    df : pd.DataFrame
        The input DataFrame containing a segment column and time series data.
    format : str, optional
        The format of the input data, either 'timestamps' or 'windows'.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame where small segments have been removed and segment
        numbers have been reset to start from 1.

    Example
    -------
    config = Config(min_segment_length_s=2, sampling_frequency=100, segment_nr_colname='segment')
    df = pd.DataFrame({
        'segment': [1, 1, 2, 2, 2],
        'time': [0, 1, 2, 3, 4]
    })
    df_filtered = discard_segments(config, df)
    # Result:
    #   segment  time
    # 0       1     0
    # 1       1     1
    # 2       2     2
    # 3       2     3
    # 4       2     4
    """
    # Minimum segment size in number of samples
    if format == "timestamps":
        min_samples = int(min_segment_length_s * fs)
    elif format == "windows":
        min_samples = int(min_segment_length_s)
    else:
        raise ValueError("Invalid format. Must be 'timestamps' or 'windows'.")

    # Count samples per segment
    segment_counts = df.groupby(segment_nr_colname).size()

    # Filter rows for valid segments (>= min samples)
    counts_map = segment_counts.to_dict()
    df = df[df[segment_nr_colname].map(counts_map) >= min_samples].copy()

    if df.empty:
        raise ValueError(
            f"All segments were removed: no segment ≥ {min_samples} samples."
        )

    # Reset segment numbers
    df[segment_nr_colname] = pd.factorize(df[segment_nr_colname])[0] + 1

    return df


@deprecated("This will be removed in v1.1.")
def categorize_segments(df, fs, format="timestamps", window_step_length_s=None):
    """
    Categorize segments based on their duration.

    This function categorizes segments into four categories based on their duration
    in seconds. The categories are defined as:
    - Category 1: Segments shorter than 5 seconds
    - Category 2: Segments between 5 and 10 seconds
    - Category 3: Segments between 10 and 20 seconds
    - Category 4: Segments longer than 20 seconds

    The duration of each segment is calculated based on the sampling frequency and
    the number of rows (data points) in the segment.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the segment column with segment numbers.
    config : object
        A configuration object containing `sampling_frequency`.
    format : str, optional
        The format of the input data, either 'timestamps' or 'windows'.

    Returns
    -------
    pd.Series
        A Series containing the category for each segment:
        - 'short' for segments < 5 seconds
        - 'moderately_long' for segments between 5 and 10 seconds
        - 'long' for segments between 10 and 20 seconds
        - 'very_long' for segments > 20 seconds
    """
    if format == "windows" and window_step_length_s is None:
        raise ValueError("Window step length must be provided for 'windows' format.")

    # Define duration thresholds in seconds
    d_max_duration = {"short": 5, "moderately_long": 10, "long": 20}

    # Convert thresholds to rows if format is 'timestamps'
    if format == "timestamps":
        d_max_duration = {k: v * fs for k, v in d_max_duration.items()}

    # Count rows per segment
    segment_sizes = df[DataColumns.SEGMENT_NR].value_counts()

    # Convert segment sizes to duration in seconds
    if format == "windows":
        segment_sizes *= window_step_length_s

    # Group by the segment column and apply the categorization
    def categorize(segment_size):
        if segment_size < d_max_duration["short"]:
            return "short"
        elif segment_size < d_max_duration["moderately_long"]:
            return "moderately_long"
        elif segment_size < d_max_duration["long"]:
            return "long"
        else:
            return "very_long"

    # Apply categorization to the DataFrame
    return (
        df[DataColumns.SEGMENT_NR].map(segment_sizes).map(categorize).astype("category")
    )


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
    get_index(colname)
        Returns the index of a specific column name.
    get_slice(colnames)
        Returns a slice object for a range of consecutive column names.
    """

    def __init__(self, windowed_colnames: List[str]):
        """
        Initialize the WindowedDataExtractor.

        Parameters
        ----------
        windowed_colnames : list of str
            A list of column names in the windowed data.

        Raises
         ------
        ValueError
            If the list of `windowed_colnames` is empty.
        """
        if not windowed_colnames:
            raise ValueError("The list of windowed columns cannot be empty.")
        self.column_indices = {col: idx for idx, col in enumerate(windowed_colnames)}

    def get_index(self, colname: str) -> int:
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
            If the column is not found in the `windowed_colnames` list.
        """
        if colname not in self.column_indices:
            raise ValueError(f"Column name '{colname}' not found in windowed_colnames.")
        return self.column_indices[colname]

    def get_slice(self, colnames: List[str]) -> slice:
        """
        Get a slice object for a range of consecutive columns.

        Parameters
        ----------
        colnames : list of str
            A list of consecutive column names to define the slice.

        Returns
        -------
        slice
            A slice object spanning the indices of the given columns.

        Raises
        ------
        ValueError
            If one or more columns in `colnames` are not found in the `windowed_colnames` list.
        """
        if not all(col in self.column_indices for col in colnames):
            missing = [col for col in colnames if col not in self.column_indices]
            raise ValueError(
                f"The following columns are missing from windowed_colnames: {missing}"
            )
        start_idx = self.column_indices[colnames[0]]
        end_idx = self.column_indices[colnames[-1]] + 1
        return slice(start_idx, end_idx)
