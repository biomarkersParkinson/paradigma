import pandas as pd
import numpy as np
import math

from typing import Union, List


import numpy as np

def tabulate_windows(df, window_size, step_size, time_column_name, single_value_cols=None, list_value_cols=None, agg_func='first'):
    """
    Efficiently creates a windowed dataframe from the input dataframe using vectorized operations.
    
    Args:
        df (pd.DataFrame): The input dataframe, where each row represents a timestamp (0.01 sec).
        window_size (int): The number of rows per window (600 for 6 seconds).
        step_size (int): The number of rows to shift between windows (100 for 1 second shift).
        single_value_cols (list): List of columns where a single value (e.g., mean) is needed.
        list_value_cols (list): List of columns where all 600 values should be stored in a list.
        agg_func (str or function): Aggregation function for single-value columns (e.g., 'mean', 'first').
        
    Returns:
        pd.DataFrame: The windowed dataframe.
    """
    # If single_value_cols or list_value_cols is None, default to an empty list
    if single_value_cols is None:
        single_value_cols = []
    if list_value_cols is None:
        list_value_cols = []

    n_rows = len(df)
    if window_size > n_rows:
        raise ValueError(f"Window size ({window_size}) cannot be greater than the number of rows ({n_rows}) in the dataframe.")
    
    # Create indices for window start positions 
    window_starts = np.arange(0, n_rows - window_size + 1, step_size)
    
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
        end = start + window_size
        window = df.iloc[start:end]

        agg_data = {
            'window_nr': window_nr,
            'window_start': window[time_column_name].iloc[0],
            'window_end': window[time_column_name].iloc[-1],
        }
        
        # Aggregate single-value columns
        for col in single_value_cols:
            if col in window.columns:  # Only process columns that exist in the window
                agg_data[col] = agg_func_np(window[col].values)
        
        # Collect list-value columns efficiently using numpy slicing
        for col in list_value_cols:
            if col in window.columns:  # Only process columns that exist in the window
                agg_data[col] = window[col].values.tolist()

        result.append(agg_data)
    
    # Convert result list into a DataFrame
    windowed_df = pd.DataFrame(result)
    
    # Ensure the column order is as desired: window_nr, window_start, window_end, pre_or_post, and then the rest
    desired_order = ['window_nr', 'window_start', 'window_end'] + single_value_cols + list_value_cols
    
    return windowed_df[desired_order]


def create_segments(df, time_column_name, gap_threshold_s):
    """
    Adds a 'segment_nr' column to the dataframe, enumerating segments based on gaps 
    in the specified time column exceeding a given threshold.

    Args:
        df (pd.DataFrame): The input dataframe with a time column.
        time_column_name (str): The name of the time column to check for gaps.
        gap_threshold (float): The threshold for gaps in seconds.

    Returns:
        pd.Series: A series of length equal to the input dataframe, containing segment numbers.
    """
    # Calculate the difference between consecutive time values
    time_diff = df[time_column_name].diff().fillna(0.0)

    # Create a boolean mask for where the gap exceeds the threshold
    gap_exceeds = time_diff > gap_threshold_s

    # Create the segment number based on the cumulative sum of the gap_exceeds mask
    segments_series = gap_exceeds.cumsum() + 1  # +1 to start enumeration from 1

    return segments_series


def create_segment_df(df, time_column_name, segment_nr_colname):
    df_segment_times = df.groupby(segment_nr_colname)[time_column_name].agg(
        time_start='min',  # Start time (min time in each segment)
        time_end='max'     # End time (max time in each segment)
    ).reset_index()

    return df_segment_times


def discard_segments(df, segment_nr_colname, min_length_segment_s, sampling_frequency):
    """
    Removes segments from the dataframe that are smaller than a specified size,
    and resets the segment enumeration to start from 1.

    Args:
        df (pd.DataFrame): The input dataframe with a segment column.
        segment_col (str): The name of the column that contains segment numbers.
        min_length_segment_s (int): The minimum length a segment must have (in seconds) to be retained.

    Returns:
        pd.DataFrame: The filtered dataframe with small segments removed and segment numbers reset.
    """
    # Count the size of each segment
    segment_sizes = df[segment_nr_colname].value_counts()

    # Identify segments that are larger than or equal to the minimum size
    valid_segments = segment_sizes[segment_sizes >= min_length_segment_s * sampling_frequency].index

    # Filter the DataFrame to retain only valid segments
    filtered_df = df[df[segment_nr_colname].isin(valid_segments)].copy()

    # Reset the segment enumeration starting from 1
    filtered_df[segment_nr_colname] = pd.factorize(filtered_df[segment_nr_colname])[0] + 1

    return filtered_df


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
