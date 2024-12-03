import pandas as pd

def aggregate_segments(
        df: pd.DataFrame,
        time_colname: str,
        segment_nr_colname: str,
        window_step_size_s: float,
        metrics: list,
        aggregates: list,
        quantiles: list=[],
        )-> pd.DataFrame:
    """Extract arm swing aggregations from segments of a dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing windowed arm swing features
    time_colname : str
        Name of the column containing the start time of the window
    segment_nr_colname : str
        Name of the column containing the segment number
    window_step_size_s : float
        Duration of each window in seconds
    metrics : list
        List of metrics to aggregate
    aggregates : list
        List of aggregation functions to apply to the metrics
    quantiles : list
        List of quantiles to calculate

    Returns
    -------
    pd.DataFrame
        Dataframe of segments containing aggregated arm swing features for each segment
    """
    dfs_agg = []
    for metric in metrics:
        df_agg = df.groupby(segment_nr_colname)[metric].agg(aggregates).reset_index().rename(columns={x: f'{metric}_{x}' for x in aggregates})
        df_qs = df.groupby(segment_nr_colname)[metric].quantile(quantiles).reset_index()


        for quantile in quantiles:
            df_agg[f"{metric}_quantile_{int(quantile*100)}"] = df_qs.loc[df_qs[f'level_1']==quantile, metric].reset_index(drop=True) 

        dfs_agg.append(df_agg)

    for j in range(len(dfs_agg)):
        if j == 0:
            df_agg = dfs_agg[j]
        else:
            df_agg = pd.merge(left=df_agg, right=dfs_agg[j], how='left', on=segment_nr_colname)

    df_segments_stats = df.groupby(segment_nr_colname)[time_colname].agg(time='min', segment_duration_s='count')
    df_segments_stats['segment_duration_s'] *= window_step_size_s

    df_agg = pd.merge(left=df_agg, right=df_segments_stats, how='left', on=segment_nr_colname)

    return df_agg
