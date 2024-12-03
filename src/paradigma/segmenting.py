import pandas as pd
import numpy as np

from typing import List


import numpy as np

def tabulate_windows(
        config, 
        df: pd.DataFrame, 
        columns: List[str]
    ) -> np.ndarray:
    """
    Splits the given DataFrame into overlapping windows of specified length and step size.

    This function extracts windows of data from the specified columns of the DataFrame, based on
    the window length and step size provided in the configuration. The windows are returned in
    a 3D numpy array, where the first dimension represents the window index, the second dimension
    represents the time steps within the window, and the third dimension represents the columns 
    of the data.

    Args:
        config: A configuration object containing `window_length_s` (window length in seconds), 
            `window_step_length_s` (step size in seconds), and `sampling_frequency` (sampling frequency in Hz).
        df: The input DataFrame containing the data to be windowed.
        columns: A list of column names from the DataFrame that will be used for windowing.

    Returns:
        A 3D numpy array of shape (n_windows, window_size, n_columns), where:
            - `n_windows` is the number of windows that can be formed from the data.
            - `window_size` is the length of each window in terms of the number of time steps.
            - `n_columns` is the number of columns in the input DataFrame specified by `columns`.
        
        If the length of the data is shorter than the specified window size, an empty array is returned.

    Notes
    -----
    The function uses `np.lib.stride_tricks.sliding_window_view` to generate sliding windows of data.
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
        window_size_s: The number of seconds per window.
        step_size_s: The number of seconds to shift between windows.
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
    window_step_size = int(config.window_step_size_s * config.sampling_frequency)

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
            'window_start': window[config.time_colname].iloc[0],
            'window_end': window[config.time_colname].iloc[-1],
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


def create_segments(config, df):
    """
    Creates segments by detecting time gaps using Pandas operations.

    Args:
        config: A configuration object containing `time_colname` and `max_segment_gap_s`.
        df: Input DataFrame with time column.

    Returns:
        A series of segment numbers.
    """
    # Calculate the difference between consecutive time values
    time_diff = df[config.time_colname].diff().fillna(0.0)

    # Create a boolean mask for where the gap exceeds the threshold
    gap_exceeds = time_diff > config.max_segment_gap_s

    # Create the segment number based on the cumulative sum of the gap_exceeds mask
    segments = gap_exceeds.cumsum() + 1  # +1 to start enumeration from 1

    return segments


def create_segment_df(config, df):
    df_segment_times = df.groupby(config.segment_nr_colname)[config.time_colname].agg(
        time_start='min',  # Start time (min time in each segment)
        time_end='max'     # End time (max time in each segment)
    ).reset_index()

    return df_segment_times


def discard_segments(config, df):
    """
    Removes segments from the dataframe that are smaller than a specified size,
    and resets the segment enumeration to start from 1.

    Args:
        df: The input dataframe with a segment column.
        segment_col: The name of the column that contains segment numbers.
        min_length_segment_s: The minimum length a segment must have (in seconds) to be retained.

    Returns:
        The filtered dataframe with small segments removed and segment numbers reset.
    """
    # Minimum segment size in number of samples
    min_samples = config.min_segment_length_s * config.sampling_frequency

    # Group by segment and filter out small segments in one step
    valid_segment_mask = (
        df.groupby(config.segment_nr_colname)[config.segment_nr_colname]
        .transform('size') >= min_samples
    )

    df = df[valid_segment_mask].copy()

    # Reset segment numbers in a single step
    unique_segments = pd.factorize(df[config.segment_nr_colname])[0] + 1
    df[config.segment_nr_colname] = unique_segments

    return df


def categorize_segments(df, segment_nr_colname, sampling_frequency):
    # Calculate the number of rows corresponding to 5, 10, and 20 seconds
    short_segments_max_duration = 5 * sampling_frequency  # 5 seconds 
    moderately_long_segments_max_duration = 10 * sampling_frequency  # 10 seconds
    long_segments_max_duration = 20 * sampling_frequency  # 20 seconds 

    # Group by the segment column and apply the categorization
    def categorize(segment_size):
        if segment_size < short_segments_max_duration:
            return 1
        elif segment_size < moderately_long_segments_max_duration:
            return 2
        elif segment_size < long_segments_max_duration:
            return 3
        else:
            return 4

    # Create the new category column
    segment_sizes = df[segment_nr_colname].value_counts().sort_index()

    return df[segment_nr_colname].map(segment_sizes).apply(categorize)
