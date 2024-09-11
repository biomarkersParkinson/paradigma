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
    t_start_window = df.loc[lower_index, time_column_name]

    df_subset = df.loc[lower_index:upper_index, data_point_level_cols].copy()
    t_start = t_start_window
    t_end = upper_index/sampling_frequency + t_start_window

    if segment_nr is None:
        l_subset_squeezed = [window_nr+1, t_start, t_end] + df_subset.values.T.tolist()
    else:
        l_subset_squeezed = [segment_nr, window_nr+1, t_start, t_end] + df_subset.values.T.tolist()

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
    window_length = sampling_frequency * window_length_s - 1
    window_step_size = sampling_frequency * window_step_size_s

    df = df.reset_index(drop=True)

    if window_step_size <= 0:
        raise Exception("Step size should be larger than 0.")
    if window_length > df.shape[0]:
        return 

    l_windows = []
    n_windows = math.floor(
        (df.shape[0] - window_length) / 
         window_step_size
        ) + 1

    for window_nr in range(n_windows):
        lower = window_nr * window_step_size
        upper = window_nr * window_step_size + window_length
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

    if segment_nr is None:
        df_windows = pd.DataFrame(l_windows, columns=['window_nr', 'window_start', 'window_end'] + data_point_level_cols)
    else:
        df_windows = pd.DataFrame(l_windows, columns=[segment_nr_colname, 'window_nr', 'window_start', 'window_end'] + data_point_level_cols)
            
    return df_windows.reset_index(drop=True)


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
    segment_length_bool = df.groupby(segment_nr_colname)[time_colname].apply(lambda x: x.max() - x.min()) > minimum_segment_length_s

    df = df.loc[df[segment_nr_colname].isin(segment_length_bool.loc[segment_length_bool.values].index)]

    # reorder the segments - starting at 1
    for segment_nr in df[segment_nr_colname].unique():
        df.loc[df[segment_nr_colname]==segment_nr, f'{segment_nr_colname}_ordered'] = np.where(df[segment_nr_colname].unique()==segment_nr)[0][0] + 1

    df[f'{segment_nr_colname}_ordered'] = df[f'{segment_nr_colname}_ordered'].astype(int)

    df = df.drop(columns=[segment_nr_colname])
    df = df.rename(columns={f'{segment_nr_colname}_ordered': segment_nr_colname})

    return df
