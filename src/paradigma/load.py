"""
Data loading module for ParaDigMa toolbox.

This module provides functions to load sensor data from various formats:
- Raw data: TSDF (.meta/.bin), Empatica (.avro), Axivity (.CWA)
- Prepared data: parquet, pickle, csv

Based on device_specific_data_loading tutorial.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader

from paradigma.util import load_tsdf_dataframe

logger = logging.getLogger(__name__)


def load_tsdf_data(
    data_path: str | Path,
    prefix: str = "IMU",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Load TSDF data from .meta and .bin files.

    Parameters
    ----------
    data_path : str or Path
        Path to directory containing TSDF files.
    prefix : str, default "IMU"
        Prefix for TSDF files (e.g., "IMU_segment0001").

    Returns
    -------
    tuple
        Tuple containing (DataFrame with loaded data, time metadata
        dict, values metadata dict)
    """
    data_path = Path(data_path)
    logger.info(f"Loading TSDF data from {data_path} with prefix '{prefix}'")

    df, time_meta, values_meta = load_tsdf_dataframe(
        path_to_data=data_path, prefix=prefix
    )

    logger.info(f"Loaded TSDF data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, time_meta, values_meta


def load_empatica_data(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load Empatica .avro file.

    Parameters
    ----------
    file_path : str or Path
        Path to .avro file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, time_dt, accelerometer_x/y/z,
        gyroscope_x/y/z (if available).
    """
    file_path = Path(file_path)
    logger.info(f"Loading Empatica data from {file_path}")

    with open(file_path, "rb") as f:
        reader = DataFileReader(f, DatumReader())
        empatica_data = next(reader)

    accel_data = empatica_data["rawData"]["accelerometer"]

    # Check for gyroscope data
    gyro_data = None
    if (
        "gyroscope" in empatica_data["rawData"]
        and len(empatica_data["rawData"]["gyroscope"]["x"]) > 0
    ):
        gyro_data = empatica_data["rawData"]["gyroscope"]
    else:
        raise ValueError("Gyroscope data not found in Empatica file.")

    # Check Avro schema version for conversion
    avro_version = (
        empatica_data["schemaVersion"]["major"],
        empatica_data["schemaVersion"]["minor"],
        empatica_data["schemaVersion"]["patch"],
    )

    # Convert accelerometer data based on schema version
    if avro_version < (6, 5, 0):
        physical_range = (
            accel_data["imuParams"]["physicalMax"]
            - accel_data["imuParams"]["physicalMin"]
        )
        digital_range = (
            accel_data["imuParams"]["digitalMax"]
            - accel_data["imuParams"]["digitalMin"]
        )
        conversion_factor = physical_range / digital_range
    else:
        conversion_factor = accel_data["imuParams"]["conversionFactor"]

    accel_x = [val * conversion_factor for val in accel_data["x"]]
    accel_y = [val * conversion_factor for val in accel_data["y"]]
    accel_z = [val * conversion_factor for val in accel_data["z"]]

    sampling_frequency = accel_data["samplingFrequency"]
    nrows = len(accel_x)

    # Create time arrays
    t_start = accel_data["timestampStart"]
    t_array = [t_start + i * (1e6 / sampling_frequency) for i in range(nrows)]
    t_from_0_array = [(x - t_array[0]) / 1e6 for x in t_array]

    # Build DataFrame
    df_data = {
        "time": t_from_0_array,
        "time_dt": pd.to_datetime(t_array, unit="us"),
        "accelerometer_x": accel_x,
        "accelerometer_y": accel_y,
        "accelerometer_z": accel_z,
    }

    # Add gyroscope data if available
    if gyro_data:
        # Apply same conversion to gyroscope
        gyro_x = [val * conversion_factor for val in gyro_data["x"]]
        gyro_y = [val * conversion_factor for val in gyro_data["y"]]
        gyro_z = [val * conversion_factor for val in gyro_data["z"]]

        df_data.update(
            {
                "gyroscope_x": gyro_x,
                "gyroscope_y": gyro_y,
                "gyroscope_z": gyro_z,
            }
        )

    df = pd.DataFrame(df_data)

    logger.info(f"Loaded Empatica data: {nrows} rows at {sampling_frequency} Hz")
    logger.debug(f"Start time: {pd.to_datetime(t_start, unit='us')}")
    logger.debug(f"Columns: {list(df.columns)}")

    return df


def load_axivity_data(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load Axivity .CWA file.

    Parameters
    ----------
    file_path : str or Path
        Path to .CWA file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, time_dt, accelerometer_x/y/z,
        gyroscope_x/y/z (if available).
    """
    try:
        from openmovement.load import CwaData
    except ImportError:
        raise ImportError(
            "openmovement package required for Axivity data loading. "
            "Install with: pip install git+https://github.com/digitalinteraction/openmovement-python.git@master"
        )

    file_path = Path(file_path)
    logger.info(f"Loading Axivity data from {file_path}")

    with CwaData(
        filename=file_path,
        include_gyro=True,  # Set to False for AX3 devices without gyroscope
        include_temperature=False,
    ) as cwa_data:
        logger.debug(f"Data format info: {cwa_data.data_format}")
        df = cwa_data.get_samples()

    # Set time to start at 0 seconds
    df["time_dt"] = df["time"].copy()
    df["time"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    # Standardize column names
    column_mapping = {}
    if "accel_x" in df.columns:
        column_mapping.update(
            {
                "accel_x": "accelerometer_x",
                "accel_y": "accelerometer_y",
                "accel_z": "accelerometer_z",
            }
        )
    if "gyro_x" in df.columns:
        column_mapping.update(
            {"gyro_x": "gyroscope_x", "gyro_y": "gyroscope_y", "gyro_z": "gyroscope_z"}
        )

    df = df.rename(columns=column_mapping)

    logger.info(f"Loaded Axivity data: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.debug(f"Columns: {list(df.columns)}")

    return df


def load_prepared_data(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load prepared data from various formats (parquet, pickle, csv, json).
    If json, expects TSDF format with corresponding .bin files.

    Parameters
    ----------
    file_path : str or Path
        Path to prepared data file.

    Returns
    -------
    pd.DataFrame
        DataFrame with prepared data.
    """
    file_path = Path(file_path)
    logger.info(f"Loading prepared data from {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file format and load accordingly
    suffix = file_path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in [".pkl", ".pickle"]:
        with open(file_path, "rb") as f:
            df = pickle.load(f)
    elif suffix == ".json":
        # Load TSDF from JSON and corresponding .bin files
        df, _, _ = load_tsdf_dataframe(
            path_to_data=file_path.parent,
            prefix=file_path.stem.replace("_meta", ""),
        )
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported: .parquet, .csv, .pkl, .pickle"
        )

    logger.info(f"Loaded {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.debug(f"Columns: {list(df.columns)}")

    return df


def detect_file_format(file_path: str | Path) -> str:
    """
    Detect the format of a data file based on its extension.

    Parameters
    ----------
    file_path : str or Path
        Path to data file

    Returns
    -------
    str
        File format: 'json', 'empatica', 'axivity', 'prepared'
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        return "tsdf"
    elif suffix == ".avro":
        return "empatica"
    elif suffix == ".cwa":
        return "axivity"
    elif suffix in [".parquet", ".csv", ".pkl", ".pickle"]:
        return "prepared"
    else:
        raise ValueError(f"Unknown file format: {suffix}")


def get_data_file_paths(
    data_path: str | Path,
    file_patterns: list[str] | str | None = None,
) -> list[Path]:
    """
    Get list of data file paths without loading them.

    This function is useful for memory-efficient processing where you want to
    load and process files one at a time instead of loading all at once.

    Parameters
    ----------
    data_path : str or Path
        Path to directory containing data files
    file_patterns : str or list of str, optional
        File extensions to consider (e.g. ["parquet", "csv", "cwa"]).
        If None, all supported formats are considered.

    Returns
    -------
    list of Path
        List of file paths found in the directory
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")

    valid_file_patterns = ["parquet", "csv", "pkl", "pickle", "json", "avro", "cwa"]

    if file_patterns is None:
        file_patterns = valid_file_patterns
    elif isinstance(file_patterns, str):
        file_patterns = [file_patterns]

    # Collect candidate files
    all_files = [
        f
        for f in data_path.iterdir()
        if f.is_file() and f.suffix[1:].lower() in file_patterns
    ]

    logger.info(f"Found {len(all_files)} data files in {data_path}")

    return all_files


def load_single_data_file(
    file_path: str | Path,
) -> tuple[str, pd.DataFrame]:
    """
    Load a single data file with automatic format detection.

    Parameters
    ----------
    file_path : str or Path
        Path to data file

    Returns
    -------
    tuple
        Tuple of (file_key, DataFrame) where file_key is the file name without extension
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        file_format = detect_file_format(file_path)

        if file_format == "tsdf":
            # For TSDF, load based on .meta file and infer prefix
            if file_path.suffix.lower() == ".json":
                prefix = file_path.stem.replace("_meta", "")
                df, _, _ = load_tsdf_data(file_path.parent, prefix)
                return prefix, df

        elif file_format == "empatica":
            df = load_empatica_data(file_path)
            return file_path.stem, df

        elif file_format == "axivity":
            df = load_axivity_data(file_path)
            return file_path.stem, df

        elif file_format == "prepared":
            df = load_prepared_data(file_path)
            prefix = file_path.stem.replace("_meta", "")
            return prefix, df

        else:
            raise ValueError(f"Unknown file format for {file_path}")

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def load_data_files(
    data_path: str | Path,
    file_patterns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all data files from a directory with automatic format detection.

    Note: This function loads all files into memory at once. For large datasets,
    consider using get_data_file_paths() and load_single_data_file() to process
    files one at a time.

    Parameters
    ----------
    data_path : str or Path
        Path to directory containing data files
    file_patterns : str or list of str, optional
        File extensions to consider (e.g. ["parquet", "csv", "cwa"]).
        If None, all supported formats are considered.

    Returns
    -------
    dict
        Dictionary mapping file names (without extension) to DataFrames
    """
    # Get all file paths
    all_files = get_data_file_paths(data_path, file_patterns)

    loaded_files = {}

    # Load each file
    for file_path in all_files:
        try:
            file_key, df = load_single_data_file(file_path)
            loaded_files[file_key] = df
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    if len(loaded_files) == 0:
        logger.warning("No data files were loaded.")
    else:
        logger.info(f"Successfully loaded {len(loaded_files)} files")

    return loaded_files


def save_prepared_data(
    df: pd.DataFrame,
    file_path: str | Path,
    file_format: str = "parquet",
) -> None:
    """
    Save prepared data to file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str or Path
        Path for output file
    file_format : str, default 'parquet'
        Output format: 'parquet', 'csv', 'pickle'
    """
    file_path = Path(file_path)

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "parquet":
        df.to_parquet(file_path, index=False)
    elif file_format == "csv":
        df.to_csv(file_path, index=False)
    elif file_format == "pickle":
        with open(file_path, "wb") as f:
            pickle.dump(df, f)
    else:
        raise ValueError(f"Unsupported file_format: {file_format}")

    logger.info(f"Saved prepared data to {file_path}")
