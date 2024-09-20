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
    t_start_window = df[time_column_name].iloc[lower_index]
    df_subset = df.iloc[lower_index:upper_index + 1, df.columns.get_indexer(data_point_level_cols)]
    
    t_start = t_start_window
    t_end = t_start_window + (upper_index - lower_index) / sampling_frequency

    if segment_nr is None:
        l_subset_squeezed = [window_nr + 1, t_start, t_end] + df_subset.values.T.tolist()
    else:
        l_subset_squeezed = [segment_nr, window_nr + 1, t_start, t_end] + df_subset.values.T.tolist()

    return l_subset_squeezed


def tabulate_windows(
        df: pd.DataFrame,
        time_column_name: str,
        data_point_level_cols: list,
        window_length_s: Union[int, float] = 6,
        window_step_size_s: Union[int, float] = 1,
        sampling_frequency: int = 100,
        segment_nr_colname: Union[str, None] = None,
        segment_nr: Union[int, None] = None,
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
    window_length_s: int | float, optional
        The number of seconds a window constitutes (default: 6)
    window_step_size_s: int | float, optional
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
    window_length = int(sampling_frequency * window_length_s) - 1
    window_step_size = int(sampling_frequency * window_step_size_s)

    df = df.reset_index(drop=True)

    if window_step_size <= 0:
        raise ValueError("Step size should be larger than 0.")
    if window_length > df.shape[0]:
        return pd.DataFrame()
    
    n_windows = (len(df) - window_length) // window_step_size + 1
    window_indices = [(i * window_step_size, i * window_step_size + window_length) for i in range(n_windows)]

    l_windows = []
    
    for window_nr, (lower, upper) in enumerate(window_indices):
        l_windows.append(
            create_window(
                df=df,
                time_column_name=time_column_name,
                window_nr=window_nr,
                lower_index=lower,
                upper_index=upper,
                data_point_level_cols=data_point_level_cols,
                segment_nr=segment_nr,
                sampling_frequency=sampling_frequency
            )
        )

    # Construct the dataframe from the windowed data
    if segment_nr is None:
        df_windows = pd.DataFrame(l_windows, columns=['window_nr', 'window_start', 'window_end'] + data_point_level_cols)
    else:
        df_windows = pd.DataFrame(l_windows, columns=[segment_nr_colname, 'window_nr', 'window_start', 'window_end'] + data_point_level_cols)

    return df_windows.reset_index(drop=True)


def create_windows_vectorized(
    df: pd.DataFrame,
    time_column_name: str,
    data_point_level_cols: list,
    window_length_s: Union[int, float],
    window_step_size_s: Union[int, float],
    sampling_frequency: int = 100,
    segment_nr_colname: Union[str, None] = None,
    segment_nr: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Fully vectorized version to create windows across the dataframe.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to window.
    time_column_name: str
        Name of the time column.
    data_point_level_cols: list
        Columns to include in the windowed data.
    window_length_s: int | float
        Duration of each window in seconds.
    window_step_size_s: int | float
        Time step between windows in seconds.
    sampling_frequency: int
        Sampling frequency in Hz.
    segment_nr_colname: str, optional
        Name of the column with segment numbers (optional).
    segment_nr: int, optional
        Segment number (optional).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with windowed data.
    """
    
    window_length = int(window_length_s * sampling_frequency)
    window_step_size = int(window_step_size_s * sampling_frequency)
    
    # Get the number of windows based on the length of the dataframe
    n_windows = (len(df) - window_length) // window_step_size + 1

    # Prepare indices for windows
    indices = np.arange(window_length)[None, :] + np.arange(n_windows)[:, None] * window_step_size

    # Use NumPy advanced indexing to gather window data
    data_windowed = df[data_point_level_cols].values[indices]
    
    # Time columns for window start and end
    t_start = df[time_column_name].values[indices[:, 0]]
    t_end = df[time_column_name].values[indices[:, -1]]

    # Prepare the final DataFrame structure
    if segment_nr_colname is None:
        window_info = np.column_stack([np.arange(1, n_windows + 1), t_start, t_end])
        column_names = ['window_nr', 'window_start', 'window_end']
    else:
        segment_info = np.full((n_windows, 1), segment_nr)
        window_info = np.column_stack([segment_info, np.arange(1, n_windows + 1), t_start, t_end])
        column_names = [segment_nr_colname, 'window_nr', 'window_start', 'window_end']
    
    # Concatenate window info and data
    result = np.column_stack([window_info, data_windowed.reshape(n_windows, -1)])

    # Build column names for the final DataFrame
    final_columns = column_names + data_point_level_cols

    # Return as DataFrame
    return pd.DataFrame(result, columns=final_columns)


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
    array_new_segments = np.where((df[time_colname] - df[time_colname].shift() > minimum_gap_s), 1, 0)
    df['new_segment_cumsum'] = array_new_segments.cumsum()
    df_segments = pd.DataFrame(df.groupby('new_segment_cumsum')[time_colname].count()).reset_index()
    df_segments.columns = [segment_nr_colname, 'length_segment_s']
    df_segments[segment_nr_colname] += 1

    df = df.drop(columns=['new_segment_cumsum'])

    cols_to_append = [segment_nr_colname, 'length_segment_s']

    for col in cols_to_append:
        df[col] = 0

    index_start = 0
    for _, row in df_segments.iterrows():
        len_segment = row['length_segment_s']

        for col in cols_to_append:
            df.loc[index_start:index_start+len_segment-1, col] = row[col]

        index_start += len_segment

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
    # Compute segment lengths and filter out the short segments
    segment_length_bool = df.groupby(segment_nr_colname)[time_colname].apply(lambda x: x.max() - x.min()) > minimum_segment_length_s
    filtered_df = df[df[segment_nr_colname].isin(segment_length_bool[segment_length_bool].index)].copy()

    # Create a new ordered column for segment numbers, starting from 1
    segment_map = {segment_nr: i+1 for i, segment_nr in enumerate(filtered_df[segment_nr_colname].unique())}
    filtered_df[f'{segment_nr_colname}_ordered'] = filtered_df[segment_nr_colname].map(segment_map)

    # Set the new ordered column as the segment column
    filtered_df[f'{segment_nr_colname}_ordered'] = filtered_df[f'{segment_nr_colname}_ordered'].astype(int)

    # Drop the old segment number column and rename the new column to the original segment column name
    filtered_df = filtered_df.drop(columns=[segment_nr_colname])
    filtered_df = filtered_df.rename(columns={f'{segment_nr_colname}_ordered': segment_nr_colname})

    return filtered_df
