"""
Data preparation module for ParaDigMa toolbox.

This module provides functions to prepare raw sensor data for analysis:
- Unit conversion (m/s² to g, rad/s to deg/s)
- Time column formatting
- Column name standardization
- Watch side orientation correction
- Resampling to 100 Hz

Based on data_preparation tutorial.
"""

import logging

import pandas as pd

from paradigma.constants import DataColumns, TimeUnit
from paradigma.preprocessing import resample_data
from paradigma.util import (
    convert_units_accelerometer,
    convert_units_gyroscope,
    transform_time_array,
)

logger = logging.getLogger(__name__)


def standardize_column_names(
    df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Standardize column names to ParaDigMa conventions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column_mapping : dict, optional
        Custom column mapping.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """
    df = df.copy()

    # Apply mapping for existing columns only
    if column_mapping is None:
        return df

    existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_mapping)

    if existing_mapping:
        logger.debug(f"Standardized columns: {existing_mapping}")
    return df


def convert_sensor_units(
    df: pd.DataFrame,
    accelerometer_units: str = "m/s^2",
    gyroscope_units: str = "deg/s",
) -> pd.DataFrame:
    """
    Convert sensor units to ParaDigMa expected format (g for acceleration,
    deg/s for gyroscope).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with sensor data.
    accelerometer_units : str, default 'm/s^2'
        Current units of accelerometer data.
    gyroscope_units : str, default 'deg/s'
        Current units of gyroscope data.

    Returns
    -------
    pd.DataFrame
        DataFrame with converted units.
    """
    df = df.copy()

    # Convert accelerometer units
    accelerometer_columns = [
        col
        for col in [
            DataColumns.ACCELEROMETER_X,
            DataColumns.ACCELEROMETER_Y,
            DataColumns.ACCELEROMETER_Z,
        ]
        if col in df.columns
    ]

    if accelerometer_columns and accelerometer_units != "g":
        logger.debug(f"Converting accelerometer units from {accelerometer_units} to g")
        accelerometer_data = df[accelerometer_columns].values
        df[accelerometer_columns] = convert_units_accelerometer(
            data=accelerometer_data, units=accelerometer_units
        )

    # Convert gyroscope units
    gyroscope_columns = [
        col
        for col in [
            DataColumns.GYROSCOPE_X,
            DataColumns.GYROSCOPE_Y,
            DataColumns.GYROSCOPE_Z,
        ]
        if col in df.columns
    ]

    if gyroscope_columns and gyroscope_units != "deg/s":
        logger.debug(f"Converting gyroscope units from {gyroscope_units} to deg/s")
        gyroscope_data = df[gyroscope_columns].values
        df[gyroscope_columns] = convert_units_gyroscope(
            data=gyroscope_data, units=gyroscope_units
        )

    return df


def prepare_time_column(
    df: pd.DataFrame,
    time_column: str = DataColumns.TIME,
    input_unit_type: TimeUnit = TimeUnit.RELATIVE_S,
    output_unit_type: TimeUnit = TimeUnit.RELATIVE_S,
) -> pd.DataFrame:
    """
    Prepare time column to start from 0 seconds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_column : str, default DataColumns.TIME
        Name of time column.
    input_unit_type : TimeUnit, default TimeUnit.RELATIVE_S
        Input time unit type.
    output_unit_type : TimeUnit, default TimeUnit.RELATIVE_S
        Output time unit type.

    Returns
    -------
    pd.DataFrame
        DataFrame with prepared time column.
    """
    df = df.copy()

    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame")

    logger.debug(f"Preparing time column: {input_unit_type} -> {output_unit_type}")

    df[time_column] = transform_time_array(
        time_array=df[time_column],
        input_unit_type=input_unit_type,
        output_unit_type=output_unit_type,
    )

    return df


def correct_watch_orientation(
    df: pd.DataFrame,
    device_orientation: list[str] | None = None,
    sensor: str = "both",
) -> pd.DataFrame:
    """
    Apply custom device orientation mapping if provided.

    Note: Watch-side inversion is handled separately during preprocessing
    in the pipeline functions (preprocess_imu_data), not during data preparation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with sensor data
    device_orientation : list of str, optional
        Custom orientation correction multipliers for each axis.
        Maps device axes to standard [x, y, z] orientation.
    sensor: str, optional
        Sensor to correct ('accelerometer', 'gyroscope', or 'both').

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected device orientation (if custom mapping provided)
    """
    out = df.copy()

    target_orientation = ["x", "y", "z"]
    valid_axes = ["x", "-x", "y", "-y", "z", "-z"]

    if sensor == "both":
        sensors_to_correct = ["accelerometer", "gyroscope"]
    elif sensor in ["accelerometer", "gyroscope"]:
        sensors_to_correct = [sensor]
    else:
        raise ValueError("Sensor must be 'accelerometer', 'gyroscope', or 'both'")

    if device_orientation is not None:
        if any([axis not in valid_axes for axis in device_orientation]):
            raise ValueError(
                f"Invalid device_orientation values. Must be one of {valid_axes}"
            )
        if len(device_orientation) != 3:
            raise ValueError("device_orientation must have exactly 3 elements")

        if all([device_orientation[x] == target_orientation[x] for x in range(3)]):
            logger.debug(
                "Device orientation matches target orientation, "
                "no correction applied"
            )
        else:
            for sensor_type in sensors_to_correct:
                for target_axis, mapping in zip(["x", "y", "z"], device_orientation):
                    sign = -1 if mapping.startswith("-") else 1
                    source_axis = mapping[-1]

                    out[f"{sensor_type}_{target_axis}"] = (
                        sign * df[f"{sensor_type}_{source_axis}"]
                    )

                    logger.debug(
                        f"Applied custom orientation: {sensor_type} "
                        f"{target_axis} mapped from {mapping}"
                    )

    return out


def validate_prepared_data(df: pd.DataFrame) -> dict[str, bool | str]:
    """
    Validate that data is properly prepared for ParaDigMa analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame

    Returns
    -------
    dict
        Validation results with checks and error messages
    """
    validation = {"valid": True, "errors": [], "warnings": []}

    # Check required columns
    if DataColumns.TIME not in df.columns:
        validation["errors"].append(f"Missing required time column: {DataColumns.TIME}")

    # Check for at least accelerometer or gyroscope data
    accel_cols = [
        DataColumns.ACCELEROMETER_X,
        DataColumns.ACCELEROMETER_Y,
        DataColumns.ACCELEROMETER_Z,
    ]
    gyro_cols = [
        DataColumns.GYROSCOPE_X,
        DataColumns.GYROSCOPE_Y,
        DataColumns.GYROSCOPE_Z,
    ]

    has_accel = all(col in df.columns for col in accel_cols)
    has_gyro = all(col in df.columns for col in gyro_cols)

    if not has_accel and not has_gyro:
        validation["errors"].append("Missing accelerometer and gyroscope data")
    elif not has_accel:
        validation["warnings"].append("Missing accelerometer data")
    elif not has_gyro:
        validation["warnings"].append("Missing gyroscope data")

    # Check time column format
    if DataColumns.TIME in df.columns:
        if df[DataColumns.TIME].iloc[0] != 0:
            validation["warnings"].append("Time column does not start at 0")

        time_diff = df[DataColumns.TIME].diff().dropna()
        if time_diff.std() / time_diff.mean() > 0.1:
            validation["warnings"].append("Time column has irregular sampling")

    # Check for NaN values
    nan_columns = df.columns[df.isnull().any()].tolist()
    if nan_columns:
        validation["warnings"].append(f"Columns with NaN values: {nan_columns}")

    # Check sampling frequency
    if DataColumns.TIME in df.columns and len(df) > 1:
        time_diff = df[DataColumns.TIME].diff().dropna()
        current_dt = time_diff.median()
        current_frequency = 1.0 / current_dt

        if abs(current_frequency - 100.0) > 5.0:
            validation["warnings"].append(
                f"Sampling frequency {current_frequency:.2f} Hz differs from "
                f"expected 100 Hz"
            )

    # Set overall validity
    validation["valid"] = len(validation["errors"]) == 0

    return validation


def prepare_raw_data(
    df: pd.DataFrame,
    accelerometer_units: str = "m/s^2",
    gyroscope_units: str = "deg/s",
    time_input_unit: TimeUnit = TimeUnit.RELATIVE_S,
    resampling_frequency: float = 100.0,
    column_mapping: dict[str, str] | None = None,
    device_orientation: dict[str, int] | None = None,
    validate: bool = True,
    auto_segment: bool = False,
    max_segment_gap_s: float | None = None,
    min_segment_length_s: float | None = None,
) -> pd.DataFrame:
    """
    Complete data preparation pipeline for raw sensor data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sensor data
    accelerometer_units : str, default 'm/s^2'
        Current units of accelerometer data
    gyroscope_units : str, default 'deg/s'
        Current units of gyroscope data
    time_input_unit : TimeUnit, default TimeUnit.RELATIVE_S
        Input time unit type
    resampling_frequency : float, default 100.0
        Target sampling frequency in Hz
    column_mapping : Dict[str, str], optional
        Custom column name mapping
    device_orientation : Dict[str, int], optional
        Custom orientation correction
    validate : bool, default True
        Whether to validate the prepared data
    auto_segment : bool, default False
        If True, automatically split non-contiguous data into segments.
        Adds 'data_segment_nr' column to output.
    max_segment_gap_s : float, optional
        Maximum gap (seconds) before starting new segment. Used when auto_segment=True.
        Defaults to 1.5s.
    min_segment_length_s : float, optional
        Minimum segment length (seconds) to keep. Used when auto_segment=True.
        Defaults to 1.5s.

    Returns
    -------
    pd.DataFrame
        Prepared data ready for ParaDigMa analysis. If auto_segment=True and multiple
        segments found, includes 'data_segment_nr' column.
    """
    logger.info("Starting data preparation pipeline")

    # Step 1: Standardize column names
    logger.info("Step 1: Standardizing column names")
    if column_mapping is None:
        logger.debug("No column mapping provided, using default mapping")
    else:
        df = standardize_column_names(df, column_mapping)

    # Step 2: Convert units
    logger.info("Step 2: Converting sensor units")
    df = convert_sensor_units(df, accelerometer_units, gyroscope_units)

    # Step 3: Prepare time column
    logger.info("Step 3: Preparing time column")
    df = prepare_time_column(df, input_unit_type=time_input_unit)

    # Step 4: Correct device orientation
    logger.info("Step 4: Correcting device orientation")
    df = correct_watch_orientation(df, device_orientation=device_orientation)

    # Step 5: Resample to target frequency
    logger.info(f"Step 5: Resampling to {resampling_frequency} Hz")
    df = resample_data(
        df,
        resampling_frequency=resampling_frequency,
        auto_segment=auto_segment,
        max_segment_gap_s=max_segment_gap_s,
        min_segment_length_s=min_segment_length_s,
        verbose=1 if logger.level <= logging.INFO else 0,
    )

    # Step 6: Validate prepared data
    if validate:
        logger.info("Step 6: Validating prepared data")
        validation = validate_prepared_data(df)

        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(warning)

        if not validation["valid"]:
            for error in validation["errors"]:
                logger.error(error)
            raise ValueError("Data preparation validation failed")

    logger.info(
        f"Data preparation completed: {df.shape[0]} rows, {df.shape[1]} columns"
    )
    return df
