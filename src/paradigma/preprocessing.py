import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

from paradigma.config import IMUConfig, PPGConfig
from paradigma.segmenting import create_segments, discard_segments
from paradigma.util import invert_watch_side

logger = logging.getLogger(__name__)


def resample_data(
    df: pd.DataFrame,
    time_column: str = "time",
    values_column_names: list[str] | None = None,
    sampling_frequency: int | None = None,
    resampling_frequency: int | None = None,
    tolerance: float | None = None,
    validate_contiguous: bool = True,
    auto_segment: bool = False,
    max_segment_gap_s: float | None = None,
    min_segment_length_s: float | None = None,
) -> pd.DataFrame:
    """
    Unified resampling function with optional auto-segmentation for non-contiguous data.

    This function supports:
    - Automatic frequency detection or explicit specification
    - Contiguity validation with configurable tolerance
    - Automatic segmentation of non-contiguous data
    - Preservation of non-numeric columns

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the sensor data.
    time_column : str, default 'time'
        The name of the column containing the time data.
    values_column_names : List[str], optional
        Column names to resample. If None, auto-detects all numeric columns except time.
    sampling_frequency : int, optional
        Original sampling frequency (Hz). If None, auto-detected from data.
    resampling_frequency : int, optional
        Target sampling frequency in Hz.
    tolerance : float, optional
        Tolerance for contiguity checking (seconds). Defaults to IMUConfig tolerance.
    validate_contiguous : bool, default True
        Whether to validate data contiguity. If False, gaps are silently interpolated.
    auto_segment : bool, default False
        If True, automatically split non-contiguous data into segments and
        process each. Adds 'data_segment_nr' column to output. If False and
        data is non-contiguous with validate_contiguous=True, raises
        ValueError.
    max_segment_gap_s : float, optional
        Maximum gap (seconds) before starting new segment. Used when auto_segment=True.
        Defaults to IMUConfig.max_segment_gap_s (1.5s).
    min_segment_length_s : float, optional
        Minimum segment length (seconds) to keep. Used when auto_segment=True.
        Defaults to IMUConfig.min_segment_length_s (1.5s).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame. If auto_segment=True and multiple segments found,
        includes 'data_segment_nr' column identifying each contiguous data segment.

    Raises
    ------
    ValueError
        - If time array is not strictly increasing
        - If time array is not contiguous and validate_contiguous=True
          and auto_segment=False
        - If no numeric columns found for resampling
        - If all segments are discarded due to min_segment_length_s

    Notes
    -----
    - Uses cubic interpolation for smooth resampling if there are enough points
    - Falls back to linear interpolation if only 2-3 points available
    - Non-numeric columns are preserved (first value copied to all rows)
    - Backwards compatible with both old resample_data signatures

    Examples
    --------
    # Auto-detection mode
    df_resampled = resample_data(df, resampling_frequency=100)

    # Explicit mode
    df_resampled = resample_data(
        df, time_column='time', values_column_names=['acc_x', 'acc_y'],
        sampling_frequency=128, resampling_frequency=100
    )

    # Auto-segmentation mode
    df_segmented = resample_data(
        df, resampling_frequency=100, auto_segment=True,
        max_segment_gap_s=2.0, min_segment_length_s=3.0
    )
    """
    df = df.copy()

    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame")

    # Validate resampling frequency
    if resampling_frequency is None:
        resampling_frequency = 100
        logger.warning("resampling_frequency automatically set to 100 Hz")

    resampling_frequency = float(resampling_frequency)

    # Auto-detect or use provided column names
    if values_column_names is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        values_column_names = [
            col
            for col in numeric_columns
            if col != time_column and col != "data_segment_nr"
        ]
        if not values_column_names:
            raise ValueError("No numeric columns found for resampling")
        logger.debug(f"Auto-detected {len(values_column_names)} columns for resampling")

    # Auto-detect or use provided sampling frequency
    time_abs_array = np.array(df[time_column])
    if sampling_frequency is None:
        time_diff = df[time_column].diff().dropna()
        current_dt = time_diff.median()
        sampling_frequency = 1.0 / current_dt
        logger.debug(f"Auto-detected sampling frequency: {sampling_frequency:.2f} Hz")
    else:
        sampling_frequency = float(sampling_frequency)

    # Ensure time array is strictly increasing
    if not np.all(np.diff(time_abs_array) > 0):
        raise ValueError("Time array is not strictly increasing")

    # Set default tolerance if not provided
    if tolerance is None:
        tolerance = IMUConfig().tolerance

    # Set default segmentation parameters
    if auto_segment:
        if max_segment_gap_s is None:
            max_segment_gap_s = 1.5  # IMUConfig default
        if min_segment_length_s is None:
            min_segment_length_s = 1.5  # IMUConfig default

    # Check contiguity
    expected_interval = 1 / sampling_frequency
    timestamp_diffs = np.diff(time_abs_array)
    is_contiguous = not np.any(np.abs(timestamp_diffs - expected_interval) > tolerance)

    if not is_contiguous:
        if validate_contiguous and not auto_segment:
            raise ValueError(
                "Time array is not contiguous. Consider enabling automatic "
                "segmentation to split and process non-contiguous segments, or "
                "disable contiguity validation to interpolate over gaps."
            )
        elif auto_segment:
            # Split into segments
            logger.info("Non-contiguous data detected. Auto-segmenting...")

            # Create segments based on gaps
            segment_array = create_segments(
                time_array=time_abs_array,
                max_segment_gap_s=max_segment_gap_s,
            )
            df["data_segment_nr"] = segment_array

            # Discard segments that are too short
            df = discard_segments(
                df=df,
                segment_nr_colname="data_segment_nr",
                min_segment_length_s=min_segment_length_s,
                fs=int(sampling_frequency),
                format="timestamps",
            )

            n_segments = df["data_segment_nr"].nunique()
            segment_durations = []
            for seg_nr in df["data_segment_nr"].unique():
                seg_df = df[df["data_segment_nr"] == seg_nr]
                duration = seg_df[time_column].iloc[-1] - seg_df[time_column].iloc[0]
                segment_durations.append(f"{duration:.1f}s")
            logger.info(
                f"Created {n_segments} segments: {', '.join(segment_durations)}"
            )

            # Resample each segment independently
            resampled_segments = []
            for seg_nr in df["data_segment_nr"].unique():
                seg_df = df[df["data_segment_nr"] == seg_nr].copy()
                seg_time = np.array(seg_df[time_column])
                seg_values = np.array(seg_df[values_column_names])

                # Resample this segment
                duration = seg_time[-1] - seg_time[0]
                n_samples = int(np.round(duration * resampling_frequency)) + 1
                t_resampled = np.linspace(seg_time[0], seg_time[-1], n_samples)

                interpolation_kind = "cubic" if len(seg_time) > 3 else "linear"
                interpolator = interp1d(
                    seg_time,
                    seg_values,
                    axis=0,
                    kind=interpolation_kind,
                    fill_value="extrapolate",
                )
                resampled_values = interpolator(t_resampled)

                # Create resampled segment DataFrame
                df_seg_resampled = pd.DataFrame(
                    resampled_values, columns=values_column_names
                )
                df_seg_resampled[time_column] = t_resampled
                df_seg_resampled["data_segment_nr"] = seg_nr

                # Copy non-numeric columns from first row of segment
                for column in seg_df.columns:
                    if (
                        column not in df_seg_resampled.columns
                        and column != "data_segment_nr"
                    ):
                        df_seg_resampled[column] = seg_df[column].iloc[0]

                resampled_segments.append(df_seg_resampled)

            # Concatenate all segments
            df_resampled = pd.concat(resampled_segments, ignore_index=True)

            # Ensure correct column order
            resampled_columns = (
                [time_column] + values_column_names + ["data_segment_nr"]
            )
            other_cols = [
                col for col in df_resampled.columns if col not in resampled_columns
            ]
            df_resampled = df_resampled[resampled_columns + other_cols]

            logger.info(
                f"Resampled: {len(df)} -> {len(df_resampled)} rows at "
                f"{resampling_frequency} Hz"
            )

            return df_resampled

        elif not validate_contiguous:
            logger.warning(
                "Data is not contiguous but validation is disabled. "
                "Interpolating over gaps."
            )

    # Standard resampling for contiguous data (or when validation is disabled)
    values_array = np.array(df[values_column_names])

    # Resample the time data
    t_resampled = np.arange(
        time_abs_array[0], time_abs_array[-1], 1 / resampling_frequency
    )

    # Choose interpolation method
    interpolation_kind = "cubic" if len(time_abs_array) > 3 else "linear"
    interpolator = interp1d(
        time_abs_array,
        values_array,
        axis=0,
        kind=interpolation_kind,
        fill_value="extrapolate",
    )

    # Interpolate
    resampled_values = interpolator(t_resampled)

    # Create resampled DataFrame
    df_resampled = pd.DataFrame(resampled_values, columns=values_column_names)
    df_resampled[time_column] = t_resampled

    # Return with correct column order
    resampled_columns = [time_column] + values_column_names
    df_resampled = df_resampled[resampled_columns]

    logger.info(
        f"Resampled: {len(df)} -> {len(df_resampled)} rows at "
        f"{resampling_frequency} Hz"
    )

    return df_resampled


def butterworth_filter(
    data: np.ndarray,
    order: int,
    cutoff_frequency: float | list[float],
    passband: str,
    sampling_frequency: int,
):
    """
    Applies a Butterworth filter to 1D or 2D sensor data.

    This function applies a low-pass, high-pass, or band-pass Butterworth filter to the
    input data. The filter is designed using the specified order, cutoff frequency,
    and passband type. The function can handle both 1D and 2D data arrays.

    Parameters
    ----------
    data : np.ndarray
        The sensor data to be filtered. Can be 1D (e.g., a single signal) or 2D
        (e.g., multi-axis sensor data).
    order : int
        The order of the Butterworth filter. Higher values result in a steeper roll-off.
    cutoff_frequency : float or list of float
        The cutoff frequency (or frequencies) for the filter. For a low-pass
        or high-pass filter, this is a single float. For a band-pass filter,
        this should be a list of two floats, specifying the lower and upper
        cutoff frequencies.
    passband : str
        The type of passband to apply. Options are:
        - 'hp' : high-pass filter
        - 'lp' : low-pass filter
        - 'band' : band-pass filter
    sampling_frequency : int
        The sampling frequency of the data in Hz. This is used to normalize
        the cutoff frequency.

    Returns
    -------
    np.ndarray
        The filtered sensor data. The shape of the output is the same as the input data.

    Raises
    ------
    ValueError
        If the input data has more than two dimensions, or if an invalid
        passband is specified.

    Notes
    -----
    The function uses `scipy.signal.butter` to design the filter and
    `scipy.signal.sosfiltfilt` to apply it using second-order sections (SOS)
    to improve numerical stability.
    """
    # Design the filter using second-order sections (SOS)
    sos = signal.butter(
        N=order,
        Wn=cutoff_frequency,
        btype=passband,
        analog=False,
        fs=sampling_frequency,
        output="sos",
    )

    # Apply the filter to the data
    if data.ndim == 1:  # 1D data case
        return signal.sosfiltfilt(sos, data)
    elif data.ndim == 2:  # 2D data case
        return signal.sosfiltfilt(sos, data, axis=0)
    else:
        raise ValueError("Data must be either 1D or 2D.")


def preprocess_imu_data(
    df: pd.DataFrame,
    config: IMUConfig,
    sensor: str,
    watch_side: str,
) -> pd.DataFrame:
    """
    Preprocesses IMU data by resampling and applying filters.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing raw accelerometer and/or gyroscope data.
    config : IMUConfig
        Configuration object containing various settings, such as time column
        name, accelerometer and/or gyroscope columns, filter settings, and
        sampling frequency.
    sensor: str
        Name of the sensor data to be preprocessed. Must be one of:
        - "accelerometer": Preprocess accelerometer data only.
        - "gyroscope": Preprocess gyroscope data only.
        - "both": Preprocess both accelerometer and gyroscope data.
    watch_side: str
        The side of the watch where the data was collected. Must be one of:
        - "left": Data was collected from the left wrist.
        - "right": Data was collected from the right wrist.

    Returns
    -------
    pd.DataFrame
        The preprocessed accelerometer and or gyroscope data with the
        following transformations:
        - Resampled data at the specified frequency.
        - Filtered accelerometer data with high-pass and low-pass filtering
          applied.

    Notes
    -----
    - The function applies Butterworth filters to accelerometer data, both
      high-pass and low-pass.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Extract sensor column
    if sensor == "accelerometer":
        values_colnames = config.accelerometer_colnames
    elif sensor == "gyroscope":
        values_colnames = config.gyroscope_colnames
    elif sensor == "both":
        values_colnames = config.accelerometer_colnames + config.gyroscope_colnames
    else:
        raise ("Sensor should be either accelerometer, gyroscope, or both")

    # Check if data needs resampling
    # Skip resampling if already at target frequency or if data has been pre-segmented
    needs_resampling = True
    validate_contiguous = True

    if "data_segment_nr" in df.columns:
        # Data has been pre-segmented, skip contiguity validation
        validate_contiguous = False

    # Check current sampling frequency
    time_diff = df[config.time_colname].diff().dropna()
    current_dt = time_diff.median()
    current_frequency = 1.0 / current_dt

    if abs(current_frequency - config.resampling_frequency) < 0.1:
        needs_resampling = False

    if needs_resampling:
        # Resample the data to the specified frequency
        df = resample_data(
            df=df,
            time_column=config.time_colname,
            values_column_names=values_colnames,
            sampling_frequency=config.sampling_frequency,
            resampling_frequency=config.resampling_frequency,
            tolerance=config.tolerance,
            validate_contiguous=validate_contiguous,
        )

    # Invert the IMU data if the watch was worn on the right wrist
    df = invert_watch_side(df, watch_side, sensor)

    if sensor in ["accelerometer", "both"]:

        # Extract accelerometer data for filtering
        accel_data = df[config.accelerometer_colnames].values

        # Define filter configurations for high-pass and low-pass
        filter_renaming_configs = {
            "hp": {
                "result_columns": config.accelerometer_colnames,
                "replace_original": True,
            },
            "lp": {
                "result_columns": [
                    f"{col}_grav" for col in config.accelerometer_colnames
                ],
                "replace_original": False,
            },
        }

        # Apply filters in a loop
        for passband, filter_config in filter_renaming_configs.items():
            filtered_data = butterworth_filter(
                data=accel_data,
                order=config.filter_order,
                cutoff_frequency=config.lower_cutoff_frequency,
                passband=passband,
                sampling_frequency=config.sampling_frequency,
            )

            # Replace or add new columns based on configuration
            df[filter_config["result_columns"]] = filtered_data

        values_colnames += config.gravity_colnames

    # Preserve data_segment_nr column if it exists (needed for split_by_gaps)
    columns_to_keep = [config.time_colname, *values_colnames]
    if "data_segment_nr" in df.columns:
        columns_to_keep.append("data_segment_nr")

    df = df[columns_to_keep]

    return df


def preprocess_ppg_data(
    df_ppg: pd.DataFrame,
    ppg_config: PPGConfig,
    start_time_ppg: str | None = None,
    df_acc: pd.DataFrame | None = None,
    imu_config: IMUConfig | None = None,
    start_time_imu: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    This function preprocesses PPG and accelerometer data by resampling,
    filtering and aligning the data segments of both sensors (if applicable).
    Aligning is done using the extract_overlapping_segments function which is
    based on the provided start times of the PPG and IMU data and returns
    only the data points where both signals overlap in time. The remaining
    data points are discarded.
    After alignment, the function resamples the data to the specified
    frequency and applies Butterworth filters to both PPG and accelerometer
    data (if applicable).
    The output is two DataFrames: one for the preprocessed PPG data and
    another for the preprocessed accelerometer data (if provided, otherwise
    return is None).

    Parameters
    ----------
    df_ppg : pd.DataFrame
        DataFrame containing PPG data.
    ppg_config : PPGPreprocessingConfig
        Configuration object for PPG preprocessing.
    start_time_ppg : str
        iso8601 formatted start time of the PPG data.
    df_acc : pd.DataFrame
        DataFrame containing accelerometer from IMU data.
    imu_config : IMUPreprocessingConfig
        Configuration object for IMU preprocessing.
    start_time_imu : str
        iso8601 formatted start time of the IMU data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame | None]
        A tuple containing two DataFrames:
        - Preprocessed PPG data with the following transformations:
            - Resampled data at the specified frequency.
            - Filtered PPG data with bandpass filtering applied.
        - Preprocessed accelerometer data (if provided, otherwise return is
          None) with the following transformations:
            - Resampled data at the specified frequency.
            - Filtered accelerometer data with high-pass and low-pass
              filtering applied.

    Notes
    -----
    - If accelerometer data or IMU configuration is not provided, the
      function only preprocesses PPG data.
    - The function applies Butterworth filters to PPG and accelerometer
      (if applicable) data, both high-pass and low-pass.

    """
    # Make copies to avoid SettingWithCopyWarning
    df_ppg = df_ppg.copy()
    if df_acc is not None:
        df_acc = df_acc.copy()

    if df_acc is not None and imu_config is not None:
        # Extract overlapping segments
        df_ppg_overlapping, df_acc_overlapping = extract_overlapping_segments(
            df_ppg=df_ppg,
            df_acc=df_acc,
            time_colname_ppg=ppg_config.time_colname,
            time_colname_imu=imu_config.time_colname,
            start_time_ppg=start_time_ppg,
            start_time_acc=start_time_imu,
        )

        # Resample accelerometer data
        # Skip contiguity validation if data has been pre-segmented
        validate_contiguous_acc = "data_segment_nr" not in df_acc_overlapping.columns
        df_acc_proc = resample_data(
            df=df_acc_overlapping,
            time_column=imu_config.time_colname,
            values_column_names=list(imu_config.d_channels_accelerometer.keys()),
            sampling_frequency=imu_config.sampling_frequency,
            resampling_frequency=imu_config.resampling_frequency,
            tolerance=imu_config.tolerance,
            validate_contiguous=validate_contiguous_acc,
        )

        # Extract accelerometer data for filtering
        accel_data = df_acc_proc[imu_config.accelerometer_colnames].values

        # Define filter configurations for high-pass and low-pass
        filter_renaming_configs = {
            "hp": {
                "result_columns": imu_config.accelerometer_colnames,
                "replace_original": True,
            }
        }

        # Apply filters in a loop
        for passband, filter_config in filter_renaming_configs.items():
            filtered_data = butterworth_filter(
                data=accel_data,
                order=imu_config.filter_order,
                cutoff_frequency=imu_config.lower_cutoff_frequency,
                passband=passband,
                sampling_frequency=imu_config.sampling_frequency,
            )

            # Replace or add new columns based on configuration
            df_acc_proc[filter_config["result_columns"]] = filtered_data

    else:
        df_ppg_overlapping = df_ppg

    # Resample PPG data
    # Skip contiguity validation if data has been pre-segmented
    validate_contiguous_ppg = "data_segment_nr" not in df_ppg_overlapping.columns
    df_ppg_proc = resample_data(
        df=df_ppg_overlapping,
        time_column=ppg_config.time_colname,
        values_column_names=list(ppg_config.d_channels_ppg.keys()),
        sampling_frequency=ppg_config.sampling_frequency,
        resampling_frequency=ppg_config.resampling_frequency,
        tolerance=ppg_config.tolerance,
        validate_contiguous=validate_contiguous_ppg,
    )

    # Extract accelerometer data for filtering
    ppg_data = df_ppg_proc[ppg_config.ppg_colname].values

    # Define filter configurations for high-pass and low-pass
    filter_renaming_configs = {
        "bandpass": {"result_columns": ppg_config.ppg_colname, "replace_original": True}
    }

    # Apply filters in a loop
    for passband, filter_config in filter_renaming_configs.items():
        filtered_data = butterworth_filter(
            data=ppg_data,
            order=ppg_config.filter_order,
            cutoff_frequency=[
                ppg_config.lower_cutoff_frequency,
                ppg_config.upper_cutoff_frequency,
            ],
            passband=passband,
            sampling_frequency=ppg_config.sampling_frequency,
        )

        # Replace or add new columns based on configuration
        df_ppg_proc[filter_config["result_columns"]] = filtered_data

    if df_acc is not None and imu_config is not None:
        return df_ppg_proc, df_acc_proc
    else:
        return df_ppg_proc, None


def extract_overlapping_segments(
    df_ppg: pd.DataFrame,
    df_acc: pd.DataFrame,
    time_colname_ppg: str,
    time_colname_imu: str,
    start_time_ppg: str,
    start_time_acc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract DataFrames with overlapping data segments between accelerometer
    (from the IMU) and PPG datasets based on their timestamps.

    Parameters
    ----------
    df_ppg : pd.DataFrame
        DataFrame containing PPG data.
    df_acc : pd.DataFrame
        DataFrame containing accelerometer data from the IMU.
    time_colname_ppg : str
        The name of the column containing the time data in the PPG dataframe.
    time_colname_imu : str
        The name of the column containing the time data in the IMU dataframe.
    start_time_ppg : str
        iso8601 formatted start time of the PPG data.
    start_time_acc : str
        iso8601 formatted start time of the accelerometer data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames containing the overlapping segments (time and values) of
        PPG and accelerometer data.
    """
    # Convert start times to Unix timestamps
    datetime_ppg_start = datetime.fromisoformat(start_time_ppg.replace("Z", "+00:00"))
    start_unix_ppg = int(datetime_ppg_start.timestamp())
    datetime_acc_start = datetime.fromisoformat(start_time_acc.replace("Z", "+00:00"))
    start_acc_ppg = int(datetime_acc_start.timestamp())

    # Calculate the time in Unix timestamps for each dataset because the
    # timestamps are relative to the start time
    ppg_time = df_ppg[time_colname_ppg] + start_unix_ppg
    acc_time = df_acc[time_colname_imu] + start_acc_ppg

    # Determine the overlapping time interval
    start_time = max(ppg_time.iloc[0], acc_time.iloc[0])
    end_time = min(ppg_time.iloc[-1], acc_time.iloc[-1])

    # Extract indices for overlapping segments
    ppg_start_index = np.searchsorted(ppg_time, start_time, "left")
    ppg_end_index = np.searchsorted(ppg_time, end_time, "right") - 1
    acc_start_index = np.searchsorted(acc_time, start_time, "left")
    acc_end_index = np.searchsorted(acc_time, end_time, "right") - 1

    # Extract overlapping segments from DataFrames
    df_ppg_overlapping = df_ppg.iloc[ppg_start_index : ppg_end_index + 1]
    df_acc_overlapping = df_acc.iloc[acc_start_index : acc_end_index + 1]

    return df_ppg_overlapping, df_acc_overlapping
