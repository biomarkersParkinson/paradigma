import pandas as pd
import numpy as np

from typing import List
from paradigma.constants import DataColumns


import numpy as np

def tabulate_windows(
        config, 
        df: pd.DataFrame, 
        columns: List[str]
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
    config : object
        A configuration object containing `window_length_s` (window length in seconds), 
        `window_step_length_s` (step size in seconds), and `sampling_frequency` (sampling frequency in Hz).
    df : pd.DataFrame
        The input DataFrame containing the data to be windowed.
    columns : list of str
        A list of column names from the DataFrame that will be used for windowing.

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
    window_size = int(config.window_length_s * config.sampling_frequency)
    window_step_size = int(config.window_step_length_s * config.sampling_frequency)
    n_columns = len(columns)

    data = df[columns].values

    # Check if data length is sufficient
    if len(data) < window_size:
        return np.empty((0, window_size, n_columns))  # Return an empty array if insufficient data
    
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=(window_size, n_columns)
        )[::window_step_size].squeeze()
    
    # Ensure 3D shape (n_windows, window_size, n_columns)
    if windows.ndim == 2:  # Single window case
        windows = windows[np.newaxis, :, :]  # Add a new axis at the start

    return windows

def tabulate_windows_legacy(config, df, agg_func='first'):
    """
    Efficiently creates a windowed dataframe from the input dataframe using vectorized operations.
    
    Args:
        df: The input dataframe, where each row represents a timestamp (0.01 sec).
        window_length_s: The number of seconds per window.
        window_step_length_s: The number of seconds to shift between windows.
        single_value_cols: List of columns where a single value (e.g., mean) is needed.
        list_value_cols: List of columns where all 600 values should be stored in a list.
        agg_func: Aggregation function for single-value columns (e.g., 'mean', 'first').
        
    Returns:
        The windowed dataframe.
    """
    # If single_value_cols or list_value_cols is None, default to an empty list
    if config.single_value_cols is None:
        config.single_value_cols = []
    if config.list_value_cols is None:
        config.list_value_cols = []

    window_length = int(config.window_length_s * config.sampling_frequency)
    window_step_size = int(config.window_step_length_s * config.sampling_frequency)

    n_rows = len(df)
    if window_length > n_rows:
        raise ValueError(f"Window size ({window_length}) cannot be greater than the number of rows ({n_rows}) in the dataframe.")
    
    # Create indices for window start positions 
    window_starts = np.arange(0, n_rows - window_length + 1, window_step_size)
    
    # Prepare the result for the final DataFrame
    result = []
    
    # Handle single value columns with vectorized operations
    agg_func_map = {
        'mean': np.mean,
        'first': lambda x: x[0],
    }

    # Check if agg_func is a callable (custom function) or get the function from the map
    if callable(agg_func):
        agg_func_np = agg_func
    else:
        agg_func_np = agg_func_map.get(agg_func, agg_func_map['mean'])  # Default to 'mean' if agg_func is not recognized

        
    for window_nr, start in enumerate(window_starts, 1):
        end = start + window_length
        window = df.iloc[start:end]

        agg_data = {
            'window_nr': window_nr,
            'window_start': window[DataColumns.TIME].iloc[0],
            'window_end': window[DataColumns.TIME].iloc[-1],
        }
        
        # Aggregate single-value columns
        for col in config.single_value_cols:
            if col in window.columns:  # Only process columns that exist in the window
                agg_data[col] = agg_func_np(window[col].values)
        
        # Collect list-value columns efficiently using numpy slicing
        for col in config.list_value_cols:
            if col in window.columns:  # Only process columns that exist in the window
                agg_data[col] = window[col].values.tolist()

        result.append(agg_data)
    
    # Convert result list into a DataFrame
    windowed_df = pd.DataFrame(result)
    
    # Ensure the column order is as desired: window_nr, window_start, window_end, pre_or_post, and then the rest
    desired_order = ['window_nr', 'window_start', 'window_end'] + config.single_value_cols + config.list_value_cols
    
    return windowed_df[desired_order]


def create_segments(config, df: pd.DataFrame):
    """
    Create segments by detecting time gaps using Pandas operations.

    This function divides the input DataFrame into segments by identifying time gaps that 
    exceed a specified threshold. If the gap between consecutive time points exceeds 
    `max_segment_gap_s`, a new segment is started. The function returns a series of segment 
    numbers that correspond to each row in the DataFrame.

    Parameters
    ----------
    config : object
        A configuration object containing `time_colname` (the name of the time column in `df`) 
        and `max_segment_gap_s` (the maximum gap in seconds that defines a new segment).
    df : pd.DataFrame
        The input DataFrame containing the time column specified in `config`.

    Returns
    -------
    pd.Series
        A Pandas Series containing the segment number for each row in the input DataFrame.

    Notes
    -----
    - The function assumes that the time column is in ascending order.
    - If the time difference between consecutive rows exceeds `max_segment_gap_s`, a new segment 
      is started at that point.
    - The segment numbering starts at 1, and the same segment number is assigned to rows that are 
      within the same segment.

    Example
    -------
    config = Config(time_colname='time', max_segment_gap_s=2)
    df = pd.DataFrame({'time': [0, 1, 3, 4, 7, 8, 9]})
    segments = create_segments(config, df)
    # Result: Series([1, 1, 2, 2, 3, 3, 3])
    """
    # Calculate the difference between consecutive time values
    time_diff = df[DataColumns.TIME].diff().fillna(0.0)

    # Create a boolean mask for where the gap exceeds the threshold
    gap_exceeds = time_diff > config.max_segment_gap_s

    # Create the segment number based on the cumulative sum of the gap_exceeds mask
    segments = gap_exceeds.cumsum() + 1  # +1 to start enumeration from 1

    return segments


def create_segment_df(config, df: pd.DataFrame):
    """
    Create a DataFrame summarizing the start and end times for each segment.

    This function groups the input DataFrame by segment number and calculates the 
    start and end times for each segment. The start time is the minimum time 
    in each segment, and the end time is the maximum time in each segment. 
    It returns a DataFrame with these segment start and end times.

    Parameters
    ----------
    config : object
        A configuration object containing `segment_nr_colname` (the name of the segment 
        number column in `df`) and `time_colname` (the name of the time column in `df`).
    df : pd.DataFrame
        The input DataFrame containing the segment and time information.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - `segment_nr`: The segment number.
        - `time_start`: The minimum time value in each segment.
        - `time_end`: The maximum time value in each segment.

    Example
    -------
    config = Config(segment_nr_colname='segment', time_colname='time')
    df = pd.DataFrame({
        'segment': [1, 1, 2, 2],
        'time': [0, 5, 10, 15]
    })
    df_segment_times = create_segment_df(config, df)
    # Result:
    #   segment_nr  time_start  time_end
    # 0           1           0         5
    # 1           2          10        15
    """
    df_segment_times = df.groupby(DataColumns.SEGMENT_NR)[DataColumns.TIME].agg(
        time_start='min',  # Start time (min time in each segment)
        time_end='max'     # End time (max time in each segment)
    ).reset_index()

    return df_segment_times


def discard_segments(config, df, format='timestamps'):
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
    if format == 'timestamps':
        min_samples = config.min_segment_length_s * config.sampling_frequency
    elif format == 'windows':
        min_samples = config.min_segment_length_s
    else:
        raise ValueError("Invalid format. Must be 'timestamps' or 'windows'.")

    # Group by segment and filter out small segments in one step
    valid_segment_mask = (
        df.groupby(DataColumns.SEGMENT_NR)[DataColumns.SEGMENT_NR]
        .transform('size') >= min_samples
    )

    df = df[valid_segment_mask].copy()

    # Reset segment numbers in a single step
    unique_segments = pd.factorize(df[DataColumns.SEGMENT_NR])[0] + 1
    df[DataColumns.SEGMENT_NR] = unique_segments

    return df


def categorize_segments(df, config, format='timestamps'):
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
    # Define duration thresholds in seconds
    d_max_duration = {
        'short': 5,
        'moderately_long': 10,
        'long': 20
    }
    
    # Convert thresholds to rows if format is 'timestamps'
    if format == 'timestamps':
        d_max_duration = {k: v * config.sampling_frequency for k, v in d_max_duration.items()}

    # Count rows per segment
    segment_sizes = df[DataColumns.SEGMENT_NR].value_counts()

    # Convert segment sizes to duration in seconds
    if format == 'windows':
        segment_sizes *= config.window_step_length_s

    # Group by the segment column and apply the categorization
    def categorize(segment_size):
        if segment_size < d_max_duration['short']:
            return 'short'
        elif segment_size < d_max_duration['moderately_long']:
            return 'moderately_long'
        elif segment_size < d_max_duration['long']:
            return 'long'
        else:
            return 'very_long'

    # Apply categorization to the DataFrame
    return df[DataColumns.SEGMENT_NR].map(segment_sizes).map(categorize)
