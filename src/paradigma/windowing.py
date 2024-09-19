import pandas as pd
import numpy as np
import math

from typing import Union, List


def create_window(
        df: pd.DataFrame,
        time_column_name: str,
        window_nr: int,
        lower_index: int,
        upper_index: int,
        data_point_level_cols: list,
        segment_nr: int,
        sampling_frequency: int = 100
    ) -> list:
    """Transforms (a subset of) a dataframe into a single row

    Parameters
    ----------
    df: pd.DataFrame
        The original dataframe to be windowed
    time_column_name: str
        The name of the time column
    window_nr: int
        The identification of the window
    lower_index: int
        The dataframe index of the first sample to be windowed
    upper_index: int
        The dataframe index of the final sample to be windowed
    data_point_level_cols: list
        The columns in sensor_df that are to be kept as individual datapoints in a list instead of aggregates
    segment_nr: int
        The identification of the segment
    sampling_frequency: int, optional
        The sampling frequency (Hz) of the data (default: 100)

    Returns
    -------
    list
        Rows corresponding to single windows
    """
    t_start_window = df.iloc[lower_index][time_column_name]

    df_subset = df.iloc[lower_index:upper_index][data_point_level_cols].values
    t_end = (upper_index/sampling_frequency) + t_start_window

    if segment_nr is None:
        l_subset_squeezed = [window_nr+1, t_start_window, t_end] + df_subset.T.tolist()
    else:
        l_subset_squeezed = [segment_nr, window_nr+1, t_start_window, t_end] + df_subset.T.tolist()

    return l_subset_squeezed


def tabulate_windows(
        df: pd.DataFrame,
        time_column_name: str,
        data_point_level_cols: list,
        window_length_s: int = 6,
        window_step_size_s: int = 1,
        sampling_frequency: int = 100,
        segment_nr_colname: str = None,
        segment_nr: int = None,
    ) -> pd.DataFrame:
    """Compiles multiple windows into a single dataframe

    Parameters
    ----------
    df: pd.DataFrame
        The original dataframe to be windowed
    time_column_name: str
        The name of the time column
    data_point_level_cols: list
        The names of the columns that are to be kept as individual datapoints in a list instead of aggregates
    window_length_s: int, optional
        The number of seconds a window constitutes (default: 6)
    window_step_size_s: int, optional
        The number of seconds between the end of the previous and the start of the next window (default: 1)
    sampling_frequency: int, optional
        The sampling frequency of the data (default: 100)
    segment_nr_colname: str, optional
        The name of the column that identifies the segment; set to None if not applicable (default: None)
    segment_nr: int, optional
        The identification of the segment; set to None if not applicable (default: None)
    

    Returns
    -------
    pd.DataFrame
        Dataframe with each row corresponding to an individual window
    """
    window_length = int(sampling_frequency * window_length_s)
    window_step_size = int(sampling_frequency * window_step_size_s)

    df = df.reset_index(drop=True)

    n_windows = int(max(0, (df.shape[0] - window_length) // window_step_size + 1))
    l_windows = [None] * n_windows

    for window_nr in range(n_windows):
        lower = window_nr * window_step_size
        upper = lower + window_length
        l_windows[window_nr] = create_window(
            df=df,
            time_column_name=time_column_name,
            window_nr=window_nr,
            lower_index=lower,
            upper_index=upper,
            data_point_level_cols=data_point_level_cols,
            segment_nr=segment_nr,
            sampling_frequency=sampling_frequency
        )

    columns = ['window_nr', 'window_start', 'window_end'] + data_point_level_cols
    if segment_nr is not None:
        columns = [segment_nr_colname] + columns
            
    return pd.DataFrame(l_windows, columns=columns)


def create_segments(
        df: pd.DataFrame,
        time_colname: str,
        segment_nr_colname: str,
        minimum_gap_s: int,
    ) -> pd.DataFrame:
    """Create segments based on the time column of the dataframe. Segments are defined as continuous time periods.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to be segmented
    time_colname: str
        The name of the time column
    minimum_gap_s: int
        The minimum gap in seconds to split up the time periods into segments

    Returns
    -------
    pd.DataFrame
        The dataframe with additional columns related to segments
    """

    # Calculate the difference between consecutive time values
    time_diff = df[time_colname].diff()
    
    # Identify where the time gap exceeds the minimum to start a new segment
    df[segment_nr_colname] = (time_diff > minimum_gap_s).cumsum() + 1

    return df


def discard_segments(
        df: pd.DataFrame,
        time_colname: str,
        segment_nr_colname: str,
        minimum_segment_length_s: int,
    ) -> pd.DataFrame:
    """Discard segments that are shorter than a specified length.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing information about the segments
    time_colname: str
        The column name of the time column
    segment_nr_colname: str
        The column name of the column containing the segment numbers
    minimum_segment_length_s: int
        The minimum required length of a segment in seconds
    
    Returns
    -------
    pd.DataFrame
        The dataframe with segments that are longer than the specified length
    """
    # Compute segment lengths
    segment_lengths  = df.groupby(segment_nr_colname)[time_colname].apply(lambda x: x.max() - x.min())

    # Filter out short segments
    valid_segments = segment_lengths[segment_lengths > minimum_segment_length_s].index
    df_filtered = df[df[segment_nr_colname].isin(valid_segments)].copy()

    # Reorder segments starting at 1
    df_filtered[segment_nr_colname] = df_filtered[segment_nr_colname].astype('category').cat.codes + 1

    return df_filtered


def categorize_segments(
        df: pd.DataFrame,
        segment_nr_colname: str,
        sampling_frequency: int,
) -> pd.DataFrame:
    """Categorize segments based on their segment duration.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing information about the segments
    segment_nr: str
        The column name of the column containing the segment numbers
    sampling_frequency: int
        The sampling frequency of the data
    
    Returns
    -------
    pd.DataFrame
        The dataframe with segments categorized
    """
    
    # Calculate segment durations
    df['segment_duration_s'] = df.groupby(segment_nr_colname)[segment_nr_colname].transform('size')
    df['segment_duration_s'] /= sampling_frequency
    
    # Categorize segment durations using pd.cut
    bins = [0, 5, 10, 20, float('inf')]
    labels = [1, 2, 3, 4]
    df['segment_duration_category'] = pd.cut(df['segment_duration_s'], bins=bins, labels=labels, right=False)
    
    return df