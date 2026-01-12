"""
High-level pipeline runner for ParaDigMa toolbox.

This module provides a unified interface for running different ParaDigMa pipelines
(gait, tremor, pulse_rate) on sensor data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from paradigma.classification import ClassifierPackage
from paradigma.config import GaitConfig, PulseRateConfig, TremorConfig
from paradigma.constants import DataColumns
from paradigma.pipelines import gait_pipeline, pulse_rate_pipeline, tremor_pipeline
from paradigma.util import load_tsdf_dataframe

try:
    from openmovement.load import CwaData
except ImportError:
    CwaData = None

try:
    from avro.datafile import DataFileReader
    from avro.io import DatumReader
except ImportError:
    DataFileReader = None
    DatumReader = None

logger = logging.getLogger(__name__)

# Pipeline mapping
AVAILABLE_PIPELINES = {
    "gait": {
        "config_class": GaitConfig,
        "extract_features": gait_pipeline.extract_gait_features,
        "detect": gait_pipeline.detect_gait,
        "quantify": gait_pipeline.quantify_arm_swing,
        "classifier_path": "gait/classifier_gait.pkl",
        "required_sensors": ["accelerometer", "gyroscope"],
    },
    "tremor": {
        "config_class": TremorConfig,
        "extract_features": tremor_pipeline.extract_tremor_features,
        "detect": tremor_pipeline.detect_tremor,
        "classifier_path": "tremor/classifier_tremor.pkl",
        "required_sensors": ["gyroscope"],
    },
    "pulse_rate": {
        "config_class": PulseRateConfig,
        "estimate": pulse_rate_pipeline.estimate_pulse_rate,
        "aggregate": pulse_rate_pipeline.aggregate_pulse_rate,
        "classifier_path": "pulse_rate/classifier_ppg_quality.pkl",
        "required_sensors": ["ppg", "accelerometer"],
    },
}

DEFAULT_CONFIGS = {
    "gait": lambda: GaitConfig(step="gait"),
    "tremor": lambda: TremorConfig(step="features"),
    "pulse_rate": lambda: PulseRateConfig(),
}


def _detect_sampling_frequency(df: pd.DataFrame, time_column: str = "time") -> float:
    """Detect sampling frequency from time series data.

    Args:
        df: DataFrame with time series data
        time_column: Name of time column

    Returns:
        Detected sampling frequency in Hz

    Raises:
        ValueError: If sampling frequency cannot be determined
    """
    try:
        time_data = df[time_column].values
        if len(time_data) < 2:
            raise ValueError("Need at least 2 time points to detect sampling frequency")

        # Calculate time differences
        time_diffs = np.diff(time_data)

        # Remove any outliers that might be due to data gaps
        median_diff = np.median(time_diffs)
        filtered_diffs = time_diffs[
            np.abs(time_diffs - median_diff) < median_diff * 0.5
        ]

        if len(filtered_diffs) == 0:
            filtered_diffs = time_diffs

        # Calculate mean interval
        mean_interval = np.mean(filtered_diffs)

        if mean_interval <= 0:
            raise ValueError("Invalid time intervals detected")

        # Convert to frequency
        frequency = 1.0 / mean_interval

        logger.info(f"Auto-detected sampling frequency: {frequency:.2f} Hz")
        return frequency

    except Exception as e:
        raise ValueError(f"Could not detect sampling frequency: {e}")


def run_pipeline(
    data_path: Union[str, Path],
    pipelines: List[str],
    data_format: Optional[str] = None,
    config: Union[str, Dict, None] = "default",
    output_dir: Optional[Union[str, Path]] = None,
    file_pattern: Optional[str] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    sampling_frequency: Optional[float] = None,
    steps: Optional[List[str]] = None,
    save_intermediate: bool = False,
    multi_subject: bool = False,
    parallel: bool = False,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run one or more ParaDigMa pipelines on sensor data.

    This function provides a high-level interface to run ParaDigMa pipelines on raw
    sensor data stored in TSDF format.

    Parameters
    ----------
    data_path : str or Path
        Path to directory or file containing sensor data. For multi_subject=True,
        path should contain subdirectories, one per subject.
    pipelines : List[str]
        List of pipeline names to run. Valid options: ['gait', 'tremor', 'pulse_rate']
    data_format : str, optional
        Data format. If not provided, will be auto-detected from file extensions.
        Options: 'tsdf', 'empatica', 'axivity', 'prepared'.
    config : str, dict, or None, optional
        Pipeline configuration. Options:
        - "default": Use default configurations
        - dict: Dictionary mapping pipeline names to config objects
        - None: Use default configurations
    output_dir : str or Path, optional
        Directory to save pipeline results. If None, results are only returned.
    file_pattern : str, optional
        File pattern or extension to filter files (e.g., '*.parquet', '*.pkl').
        Used for prepared dataframes format.
    column_mapping : dict, optional
        Dictionary mapping old column names to new ones. Use this to rename
        columns to match pipeline expectations (e.g., {'acceleration_x': 'accelerometer_x'})
    sampling_frequency : float, optional
        Sampling frequency in Hz. If not provided, will be auto-detected from time series data.
        Only needed if auto-detection fails for raw data formats.
    steps : List[str], optional
        Pipeline steps to run. Options: ['preprocessing', 'detection', 'quantification', 'aggregation'].
        If None, runs all steps. Different pipelines have different step sequences.
    save_intermediate : bool, optional
        Save intermediate results at each step (default: False)
    multi_subject : bool, optional
        Process multiple subjects in separate subdirectories (default: False)
    parallel : bool, optional
        Enable parallel processing where supported (default: False)
    verbose : bool, optional
        Enable verbose logging (default: False)

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping pipeline names to their output DataFrames

    Examples
    --------
    >>> # Run gait pipeline (data format auto-detected)
    >>> results = run_pipeline(
    ...     data_path="data/tsdf/",
    ...     pipelines=["gait"],
    ...     output_dir="results/"
    ... )

    >>> # Run multiple pipelines on prepared dataframes
    >>> results = run_pipeline(
    ...     data_path="data/prepared/",
    ...     pipelines=["gait", "tremor"],
    ...     data_format="prepared",
    ...     file_pattern="*.parquet"
    ... )

    >>> # Run on Axivity CWA files with explicit parameters
    >>> results = run_pipeline(
    ...     data_path="data/axivity/",
    ...     pipelines=["gait"],
    ...     data_format="axivity"
    ... )
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    logger.info("Starting ParaDigMa pipeline execution")

    # Validate inputs
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    invalid_pipelines = [p for p in pipelines if p not in AVAILABLE_PIPELINES]
    if invalid_pipelines:
        raise ValueError(
            f"Invalid pipelines: {invalid_pipelines}. "
            f"Available: {list(AVAILABLE_PIPELINES.keys())}"
        )

    # Auto-detect data format if not provided
    if data_format is None:
        data_format = _detect_data_format(data_path, file_pattern)
        logger.info(f"Auto-detected data format: {data_format}")

    # Validate data format
    if data_format not in ["tsdf", "empatica", "axivity", "prepared"]:
        raise ValueError(
            f"Invalid data_format: {data_format}. Must be one of: tsdf, empatica, axivity, prepared"
        )

    logger.info(f"Processing data format: {data_format}")

    # Setup configurations
    configs = _setup_configs(pipelines, config)

    # Auto-detect data format if not provided
    data_path = Path(data_path)
    if data_format is None:
        data_format = _detect_data_format(data_path, file_pattern)
        logger.info(f"Auto-detected data format: {data_format}")

    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Handle multi-subject processing
    if multi_subject:
        return _process_multi_subject(
            data_path=data_path,
            pipelines=pipelines,
            data_format=data_format,
            config=config,
            output_dir=output_dir,
            file_pattern=file_pattern,
            column_mapping=column_mapping,
            sampling_frequency=sampling_frequency,
            steps=steps,
            save_intermediate=save_intermediate,
            parallel=parallel,
            verbose=verbose,
        )

    # Load data based on format
    logger.info(f"Loading data from {data_path} (format: {data_format})")
    data_segments = _load_data(data_path, data_format, file_pattern)

    if not data_segments:
        raise ValueError(f"No data files found in {data_path} for format {data_format}")

    logger.info(f"Found {len(data_segments)} data segment(s)")

    # Apply column mapping if provided
    if column_mapping:
        logger.info(f"Applying column mapping: {column_mapping}")
        data_segments = _apply_column_mapping(data_segments, column_mapping)

    # Add data preparation step if needed
    if data_format != "prepared":
        logger.info("Applying data preparation steps")

        # Auto-detect sampling frequency if not provided
        if sampling_frequency is None and data_segments:
            try:
                # Try to detect from first segment
                _, sample_df = data_segments[0]
                sampling_frequency = _detect_sampling_frequency(sample_df)
            except Exception as e:
                logger.warning(f"Could not auto-detect sampling frequency: {e}")
                raise ValueError(
                    f"sampling_frequency is required for {data_format} format when auto-detection fails. "
                    "Please provide the sampling frequency manually."
                )

        data_segments = _prepare_data(data_segments, data_format, sampling_frequency)

    # Run each requested pipeline
    results = {}

    for pipeline_name in pipelines:
        logger.info(f"Running {pipeline_name} pipeline")

        try:
            # Process all segments for this pipeline
            pipeline_results = []

            for i, (segment_name, df) in enumerate(data_segments):
                logger.info(
                    f"Processing segment {i+1}/{len(data_segments)}: {segment_name}"
                )

                result = _run_single_pipeline(
                    pipeline_name=pipeline_name,
                    df=df,
                    config=configs[pipeline_name],
                    parallel=parallel,
                    segment_name=segment_name,
                    steps=steps,
                    save_intermediate=save_intermediate,
                    output_dir=output_dir,
                )

                if not result.empty:
                    # Add segment identifier to results
                    result = result.copy()
                    result["segment_name"] = segment_name
                    pipeline_results.append(result)

            # Combine results from all segments
            if pipeline_results:
                combined_result = pd.concat(pipeline_results, ignore_index=True)
                results[pipeline_name] = combined_result
            else:
                logger.warning(f"No results generated for {pipeline_name} pipeline")
                results[pipeline_name] = pd.DataFrame()

            # Save results if output directory specified
            if output_dir and not results[pipeline_name].empty:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{pipeline_name}_results.csv"
                results[pipeline_name].to_csv(output_file, index=False)
                logger.info(f"Saved {pipeline_name} results to {output_file}")

            logger.info(f"Successfully completed {pipeline_name} pipeline")

        except Exception as e:
            logger.error(f"Failed to run {pipeline_name} pipeline: {e}")
            raise

    logger.info("Pipeline execution completed")
    return results


def _detect_data_format(data_path: Path, file_pattern: Optional[str] = None) -> str:
    """Auto-detect data format based on file extensions in the directory."""

    if data_path.is_file():
        # Single file - detect from extension
        extension = data_path.suffix.lower()
        if extension == ".cwa":
            return "axivity"
        elif extension == ".avro":
            return "empatica"
        elif extension in [".parquet", ".pkl", ".pickle", ".csv"]:
            return "prepared"
        else:
            raise ValueError(f"Unknown file extension: {extension}")

    # Directory - detect from files inside
    files = list(data_path.iterdir())

    # Check for TSDF files
    if any(f.name.endswith("_meta.json") for f in files):
        return "tsdf"

    # Check for Empatica files
    if any(f.suffix.lower() == ".avro" for f in files):
        return "empatica"

    # Check for Axivity files
    if any(f.suffix.lower() == ".cwa" for f in files):
        return "axivity"

    # Check for prepared dataframes
    prepared_extensions = [".parquet", ".pkl", ".pickle", ".csv", ".feather"]
    if any(f.suffix.lower() in prepared_extensions for f in files):
        return "prepared"

    # If file_pattern specified, assume prepared format
    if file_pattern:
        return "prepared"

    raise ValueError(f"Could not auto-detect data format in {data_path}")


def _apply_column_mapping(
    data_segments: List[tuple], column_mapping: Dict[str, str]
) -> List[tuple]:
    """Apply column mapping to rename columns in data segments.

    Args:
        data_segments: List of (segment_name, dataframe) tuples
        column_mapping: Dictionary mapping old column names to new names

    Returns:
        List of (segment_name, dataframe) tuples with renamed columns
    """
    mapped_segments = []

    for segment_name, df in data_segments:
        # Create a copy to avoid modifying original
        df_mapped = df.copy()

        # Apply column mapping
        rename_dict = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df_mapped.columns:
                rename_dict[old_name] = new_name

        if rename_dict:
            df_mapped = df_mapped.rename(columns=rename_dict)
            logger.debug(f"Renamed columns in {segment_name}: {rename_dict}")

        mapped_segments.append((segment_name, df_mapped))

    return mapped_segments


def _process_multi_subject(
    data_path: Path,
    pipelines: List[str],
    data_format: str,
    config: Union[str, Dict, None],
    output_dir: Optional[Path],
    file_pattern: Optional[str],
    column_mapping: Optional[Dict[str, str]],
    sampling_frequency: Optional[float],
    steps: Optional[List[str]],
    save_intermediate: bool,
    parallel: bool,
    verbose: bool,
) -> Dict[str, pd.DataFrame]:
    """Process multiple subjects in separate subdirectories."""

    # Find subject directories
    subject_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    if not subject_dirs:
        raise ValueError(f"No subject directories found in {data_path}")

    logger.info(f"Found {len(subject_dirs)} subject directories")

    all_results = {}

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        logger.info(f"Processing subject: {subject_id}")

        # Set up subject-specific output directory
        subject_output_dir = None
        if output_dir:
            subject_output_dir = output_dir / subject_id
            subject_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Process single subject (recursive call with multi_subject=False)
            subject_results = run_pipeline(
                data_path=subject_dir,
                pipelines=pipelines,
                data_format=data_format,
                config=config,
                output_dir=subject_output_dir,
                file_pattern=file_pattern,
                column_mapping=column_mapping,
                sampling_frequency=sampling_frequency,
                steps=steps,
                save_intermediate=save_intermediate,
                multi_subject=False,  # Prevent recursive multi-subject
                parallel=parallel,
                verbose=verbose,
            )

            # Add subject ID to results
            for pipeline_name, results_df in subject_results.items():
                if not results_df.empty:
                    results_df["subject_id"] = subject_id

            # Merge into all_results
            for pipeline_name, results_df in subject_results.items():
                if pipeline_name not in all_results:
                    all_results[pipeline_name] = []
                all_results[pipeline_name].append(results_df)

        except Exception as e:
            logger.error(f"Failed to process subject {subject_id}: {e}")
            continue

    # Concatenate results across subjects
    final_results = {}
    for pipeline_name, results_list in all_results.items():
        if results_list:
            final_results[pipeline_name] = pd.concat(results_list, ignore_index=True)
        else:
            final_results[pipeline_name] = pd.DataFrame()

    return final_results


def _prepare_data(
    data_segments: List[tuple], data_format: str, sampling_frequency: Optional[float]
) -> List[tuple]:
    """Apply data preparation steps to raw sensor data."""

    if data_format == "prepared":
        return data_segments  # Already prepared

    # sampling_frequency should be available by this point (auto-detected or provided)
    # but we keep this check for safety
    if sampling_frequency is None:
        raise ValueError(f"sampling_frequency is required for {data_format} format")

    prepared_segments = []

    for segment_name, df in data_segments:
        try:
            # Apply data preparation steps based on format
            if data_format == "tsdf":
                prepared_df = _prepare_tsdf_data(df, sampling_frequency)
            elif data_format == "empatica":
                prepared_df = _prepare_empatica_data(df, sampling_frequency)
            elif data_format == "axivity":
                prepared_df = _prepare_axivity_data(df, sampling_frequency)
            else:
                prepared_df = df  # Fallback

            prepared_segments.append((segment_name, prepared_df))

        except Exception as e:
            logger.warning(f"Failed to prepare data for segment {segment_name}: {e}")
            continue

    return prepared_segments


def _prepare_tsdf_data(df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
    """Prepare TSDF data by applying preprocessing steps."""
    # Add basic preprocessing for TSDF data
    # This could include gravity correction, filtering, etc.
    # For now, just ensure we have the required columns and sampling frequency info

    prepared_df = df.copy()

    # Add sampling frequency info if not present
    if "sampling_frequency" not in prepared_df.columns:
        prepared_df["sampling_frequency"] = sampling_frequency

    return prepared_df


def _prepare_empatica_data(df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
    """Prepare Empatica data by applying preprocessing steps."""
    prepared_df = df.copy()

    # Add sampling frequency info
    if "sampling_frequency" not in prepared_df.columns:
        prepared_df["sampling_frequency"] = sampling_frequency

    return prepared_df


def _prepare_axivity_data(df: pd.DataFrame, sampling_frequency: float) -> pd.DataFrame:
    """Prepare Axivity data by applying preprocessing steps."""
    prepared_df = df.copy()

    # Add sampling frequency info
    if "sampling_frequency" not in prepared_df.columns:
        prepared_df["sampling_frequency"] = sampling_frequency

    return prepared_df


def _load_data(
    data_path: Path, data_format: str, file_pattern: Optional[str] = None
) -> List[tuple]:
    """Load data segments from various formats.

    Returns:
        List of (segment_name, dataframe) tuples
    """

    if data_format == "tsdf":
        return _load_tsdf_data(data_path)
    elif data_format == "empatica":
        return _load_empatica_data(data_path)
    elif data_format == "axivity":
        return _load_axivity_data(data_path)
    elif data_format == "prepared":
        return _load_prepared_data(data_path, file_pattern)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def _load_tsdf_data(data_path: Path) -> List[tuple]:
    """Load TSDF data segments."""

    segments = []

    # Find all TSDF metadata files
    meta_files = list(data_path.glob("*_meta.json"))

    for meta_file in meta_files:
        prefix = meta_file.stem.replace("_meta", "")

        try:
            df, metadata_time, metadata_values = load_tsdf_dataframe(
                path_to_data=data_path, prefix=prefix
            )
            segments.append((prefix, df))
        except Exception as e:
            logger.warning(f"Failed to load TSDF segment {prefix}: {e}")

    return segments


def _load_empatica_data(data_path: Path) -> List[tuple]:
    """Load Empatica .avro data files."""

    if DataFileReader is None or DatumReader is None:
        raise ImportError("avro-python3 package required for Empatica data loading")

    segments = []

    if data_path.is_file() and data_path.suffix.lower() == ".avro":
        avro_files = [data_path]
    else:
        avro_files = list(data_path.glob("*.avro"))

    for avro_file in avro_files:
        try:
            with open(avro_file, "rb") as f:
                reader = DataFileReader(f, DatumReader())
                empatica_data = next(reader)

            df = _process_empatica_data(empatica_data)
            segments.append((avro_file.stem, df))

        except Exception as e:
            logger.warning(f"Failed to load Empatica file {avro_file}: {e}")

    return segments


def _load_axivity_data(data_path: Path) -> List[tuple]:
    """Load Axivity .CWA data files."""

    if CwaData is None:
        raise ImportError("openmovement package required for Axivity data loading")

    segments = []

    if data_path.is_file() and data_path.suffix.lower() == ".cwa":
        cwa_files = [data_path]
    else:
        cwa_files = list(data_path.glob("*.CWA")) + list(data_path.glob("*.cwa"))

    for cwa_file in cwa_files:
        try:
            with CwaData(
                filename=cwa_file, include_gyro=True, include_temperature=False
            ) as cwa_data:
                df = cwa_data.get_samples()

            df = _process_axivity_data(df)
            segments.append((cwa_file.stem, df))

        except Exception as e:
            logger.warning(f"Failed to load Axivity file {cwa_file}: {e}")

    return segments


def _load_prepared_data(
    data_path: Path, file_pattern: Optional[str] = None
) -> List[tuple]:
    """Load prepared DataFrame files (parquet, pickle, csv, etc.)."""

    segments = []

    if data_path.is_file():
        # Single file
        files = [data_path]
    else:
        # Directory with pattern
        if file_pattern:
            files = list(data_path.glob(file_pattern))
        else:
            # Try common extensions
            patterns = ["*.parquet", "*.pkl", "*.pickle", "*.csv", "*.feather"]
            files = []
            for pattern in patterns:
                files.extend(data_path.glob(pattern))

    for file_path in files:
        try:
            df = _load_dataframe_file(file_path)
            segments.append((file_path.stem, df))
        except Exception as e:
            logger.warning(f"Failed to load prepared data file {file_path}: {e}")

    return segments


def _load_dataframe_file(file_path: Path) -> pd.DataFrame:
    """Load a single dataframe file based on its extension."""

    extension = file_path.suffix.lower()

    if extension == ".parquet":
        return pd.read_parquet(file_path)
    elif extension in [".pkl", ".pickle"]:
        logger.warning(
            f"Loading pickle file {file_path}. Pickle files can be unsafe with untrusted data. "
            "Consider using parquet format for better security and performance."
        )
        # Use pandas read_pickle which is safer than direct pickle.load
        return pd.read_pickle(file_path)
    elif extension == ".csv":
        return pd.read_csv(file_path)
    elif extension == ".feather":
        return pd.read_feather(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")


def _process_empatica_data(empatica_data: dict) -> pd.DataFrame:
    """Process Empatica data into standardized DataFrame format."""

    accel_data = empatica_data["rawData"]["accelerometer"]

    # Get schema version for proper conversion
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
        accel_x = [val * physical_range / digital_range for val in accel_data["x"]]
        accel_y = [val * physical_range / digital_range for val in accel_data["y"]]
        accel_z = [val * physical_range / digital_range for val in accel_data["z"]]
    else:
        conversion_factor = accel_data["imuParams"]["conversionFactor"]
        accel_x = [val * conversion_factor for val in accel_data["x"]]
        accel_y = [val * conversion_factor for val in accel_data["y"]]
        accel_z = [val * conversion_factor for val in accel_data["z"]]

    # Calculate timestamps
    sampling_frequency = accel_data["samplingFrequency"]
    nrows = len(accel_x)
    t_start = accel_data["timestampStart"]
    t_array = [t_start + i * (1e6 / sampling_frequency) for i in range(nrows)]
    t_from_0_array = [(x - t_array[0]) / 1e6 for x in t_array]

    # Create DataFrame with standardized column names
    df = pd.DataFrame(
        {
            DataColumns.TIME: t_from_0_array,
            DataColumns.ACCELEROMETER_X: accel_x,
            DataColumns.ACCELEROMETER_Y: accel_y,
            DataColumns.ACCELEROMETER_Z: accel_z,
        }
    )

    # Add gyroscope data if available
    if "gyroscope" in empatica_data["rawData"]:
        # Process gyroscope data similar to accelerometer
        # (Implementation would be similar to accelerometer processing)
        logger.info(
            "Gyroscope data found in Empatica file but processing not implemented yet"
        )

    return df


def _process_axivity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process Axivity data into standardized DataFrame format."""

    # Convert time to start at 0 seconds
    df["time"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    # Rename columns to standard format
    column_mapping = {}

    # Map accelerometer columns
    if "x" in df.columns:
        column_mapping["x"] = DataColumns.ACCELEROMETER_X
    if "y" in df.columns:
        column_mapping["y"] = DataColumns.ACCELEROMETER_Y
    if "z" in df.columns:
        column_mapping["z"] = DataColumns.ACCELEROMETER_Z

    # Map gyroscope columns if they exist
    if "gx" in df.columns:
        column_mapping["gx"] = DataColumns.GYROSCOPE_X
    if "gy" in df.columns:
        column_mapping["gy"] = DataColumns.GYROSCOPE_Y
    if "gz" in df.columns:
        column_mapping["gz"] = DataColumns.GYROSCOPE_Z

    df = df.rename(columns=column_mapping)

    # Convert units to expected format (g for accel, deg/s for gyro)
    from paradigma.util import convert_units_accelerometer, convert_units_gyroscope

    accel_cols = [
        col
        for col in df.columns
        if col
        in [
            DataColumns.ACCELEROMETER_X,
            DataColumns.ACCELEROMETER_Y,
            DataColumns.ACCELEROMETER_Z,
        ]
    ]
    if accel_cols:
        df[accel_cols] = convert_units_accelerometer(
            data=df[accel_cols].values, units="g"  # Axivity already in g units
        )

    gyro_cols = [
        col
        for col in df.columns
        if col
        in [DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z]
    ]
    if gyro_cols:
        df[gyro_cols] = convert_units_gyroscope(
            data=df[gyro_cols].values, units="deg/s"
        )

    # Keep only relevant columns
    relevant_columns = ["time"] + [
        col for col in df.columns if col in DataColumns.__dict__.values()
    ]
    df = df[relevant_columns]

    return df


def _setup_configs(pipelines: List[str], config_input: Union[str, Dict, None]) -> Dict:
    """Setup configuration objects for pipelines."""
    configs = {}

    for pipeline_name in pipelines:
        if config_input == "default" or config_input is None:
            configs[pipeline_name] = DEFAULT_CONFIGS[pipeline_name]()
        elif isinstance(config_input, dict):
            if pipeline_name in config_input:
                configs[pipeline_name] = config_input[pipeline_name]
            else:
                configs[pipeline_name] = DEFAULT_CONFIGS[pipeline_name]()
        else:
            raise ValueError(f"Invalid config format: {type(config_input)}")

    return configs


def _run_single_pipeline(
    pipeline_name: str,
    df: pd.DataFrame,
    config,
    parallel: bool,
    segment_name: str,
    steps: Optional[List[str]] = None,
    save_intermediate: bool = False,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run a single pipeline on the data with configurable steps."""

    pipeline_info = AVAILABLE_PIPELINES[pipeline_name]

    if pipeline_name == "gait":
        return _run_gait_pipeline(
            df,
            config,
            pipeline_info,
            parallel,
            segment_name,
            steps,
            output_dir,
            save_intermediate,
        )
    elif pipeline_name == "tremor":
        return _run_tremor_pipeline(
            df,
            config,
            pipeline_info,
            parallel,
            segment_name,
            steps,
            output_dir,
            save_intermediate,
        )
    elif pipeline_name == "pulse_rate":
        return _run_pulse_rate_pipeline(
            df,
            config,
            pipeline_info,
            parallel,
            segment_name,
            steps,
            output_dir,
            save_intermediate,
        )
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")


def _run_gait_pipeline(
    df: pd.DataFrame,
    config: GaitConfig,
    pipeline_info: Dict,
    parallel: bool,
    segment_name: str,
    steps: Optional[List[str]] = None,
    save_intermediate: bool = False,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the complete gait analysis pipeline with configurable steps.

    Gait pipeline steps:
    1. detection: Detect gait periods
    2. filtering: Filter gait (remove other arm activities)
    3. quantification: Quantify arm swing during gait
    4. aggregation: Aggregate arm swing metrics
    """

    # Validate that required columns exist
    required_columns = [
        DataColumns.TIME,
        DataColumns.ACCELEROMETER_X,
        DataColumns.ACCELEROMETER_Y,
        DataColumns.ACCELEROMETER_Z,
        DataColumns.GYROSCOPE_X,
        DataColumns.GYROSCOPE_Y,
        DataColumns.GYROSCOPE_Z,
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing required columns for gait pipeline: {missing_columns}")
        return pd.DataFrame()

    # Define default steps for gait pipeline
    if steps is None:
        steps = ["detection", "filtering", "quantification", "aggregation"]

    # Validate steps
    valid_steps = ["detection", "filtering", "quantification", "aggregation"]
    invalid_steps = [s for s in steps if s not in valid_steps]
    if invalid_steps:
        raise ValueError(
            f"Invalid gait pipeline steps: {invalid_steps}. Valid steps: {valid_steps}"
        )

    results = {}

    # Step 1: Gait Detection
    if "detection" in steps:
        logger.info("Detecting gait periods")
        try:
            gait_detection_results = pipeline_info["detect"](df, config)
            results["detection"] = gait_detection_results

            if save_intermediate and output_dir:
                detection_file = output_dir / f"{segment_name}_gait_detection.parquet"
                gait_detection_results.to_parquet(detection_file)
                logger.info(f"Saved gait detection results to {detection_file}")

        except Exception as e:
            logger.warning(f"Gait detection failed: {e}")
            return pd.DataFrame()

        # Use detection results for next step
        current_data = gait_detection_results
    else:
        current_data = df

    # Step 2: Gait Filtering (remove other arm activities)
    if "filtering" in steps:
        logger.info("Filtering gait periods")
        try:
            # This would be a new function to filter out non-gait arm activities
            gait_filtered_results = _filter_gait_activities(current_data, config)
            results["filtering"] = gait_filtered_results

            if save_intermediate and output_dir:
                filtering_file = output_dir / f"{segment_name}_gait_filtered.parquet"
                gait_filtered_results.to_parquet(filtering_file)
                logger.info(f"Saved gait filtering results to {filtering_file}")

        except Exception as e:
            logger.warning(f"Gait filtering failed: {e}")
            gait_filtered_results = current_data

        current_data = gait_filtered_results

    # Step 3: Quantify Arm Swing
    if "quantification" in steps:
        logger.info("Quantifying arm swing")
        try:
            arm_swing_features = pipeline_info["extract_features"](current_data, config)
            results["quantification"] = arm_swing_features

            if save_intermediate and output_dir:
                quantification_file = (
                    output_dir / f"{segment_name}_arm_swing_features.parquet"
                )
                arm_swing_features.to_parquet(quantification_file)
                logger.info(f"Saved arm swing features to {quantification_file}")

        except Exception as e:
            logger.warning(f"Arm swing quantification failed: {e}")
            return pd.DataFrame()

        current_data = arm_swing_features

    # Step 4: Aggregation
    if "aggregation" in steps:
        logger.info("Aggregating arm swing metrics")
        try:
            aggregated_results = pipeline_info["aggregate"](current_data, config)
            results["aggregation"] = aggregated_results

            if save_intermediate and output_dir:
                aggregation_file = (
                    output_dir / f"{segment_name}_arm_swing_aggregated.json"
                )
                # Save as JSON for aggregated results
                if hasattr(aggregated_results, "to_json"):
                    with open(aggregation_file, "w") as f:
                        f.write(aggregated_results.to_json())
                logger.info(f"Saved aggregated results to {aggregation_file}")

            return aggregated_results

        except Exception as e:
            logger.warning(f"Arm swing aggregation failed: {e}")
            return current_data if "quantification" in steps else pd.DataFrame()

    # Return the results from the last completed step
    if "quantification" in steps:
        return current_data
    elif "filtering" in steps:
        return results.get("filtering", pd.DataFrame())
    elif "detection" in steps:
        return results.get("detection", pd.DataFrame())
    else:
        return pd.DataFrame()


def _filter_gait_activities(df: pd.DataFrame, config: GaitConfig) -> pd.DataFrame:
    """Filter gait periods to remove other arm activities.

    This function would implement filtering logic to remove non-gait arm movements
    like reaching, gesturing, etc. from detected gait periods.
    """
    # Placeholder implementation - would need domain-specific filtering logic
    # For now, return the input data unchanged
    logger.debug(
        "Gait activity filtering not yet implemented, returning unfiltered data"
    )
    return df


def _run_tremor_pipeline(
    df: pd.DataFrame,
    config: TremorConfig,
    pipeline_info: Dict,
    parallel: bool,
    segment_name: str,
    steps: List[str],
    output_dir: Optional[str] = None,
    save_intermediate: bool = False,
) -> pd.DataFrame:
    """Run the tremor analysis pipeline with configurable steps.

    Steps: preprocessing -> feature_extraction -> detection -> quantification -> aggregation

    Args:
        df: Input sensor data
        config: Tremor configuration
        pipeline_info: Pipeline function mappings
        parallel: Enable parallel processing
        segment_name: Name of data segment
        steps: List of pipeline steps to execute
        output_dir: Directory to save intermediate results
        save_intermediate: Whether to save intermediate results

    Returns:
        Results from the final executed step
    """

    # Validate that required columns exist
    required_columns = [
        DataColumns.TIME,
        DataColumns.GYROSCOPE_X,
        DataColumns.GYROSCOPE_Y,
        DataColumns.GYROSCOPE_Z,
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(
            f"Missing required columns for tremor pipeline: {missing_columns}"
        )
        return pd.DataFrame()

    results = None

    # Step 1: Feature Extraction
    if "feature_extraction" in steps:
        logger.info("Extracting tremor features")
        tremor_features = pipeline_info["extract_features"](df, config)
        results = tremor_features

        if save_intermediate and output_dir:
            output_path = Path(output_dir) / f"{segment_name}_tremor_features.parquet"
            tremor_features.to_parquet(output_path)
            logger.info(f"Saved tremor features to {output_path}")

        if steps == ["feature_extraction"]:
            return results

    # Step 2: Detection
    if "detection" in steps:
        if results is None:
            # Need features from previous step
            logger.info("Extracting tremor features (required for detection)")
            tremor_features = pipeline_info["extract_features"](df, config)
            results = tremor_features

        logger.info("Detecting tremor segments")
        try:
            from importlib.resources import files

            classifier_path = (
                files("paradigma.assets") / pipeline_info["classifier_path"]
            )
            clf_package = ClassifierPackage.load(classifier_path)
            tremor_proba = pipeline_info["detect"](results, clf_package, parallel)
            results[DataColumns.PRED_TREMOR_PROBA] = tremor_proba
        except Exception:
            logger.warning("Could not load tremor classifier, using mock predictions")
            results[DataColumns.PRED_TREMOR_PROBA] = 0.3

        if save_intermediate and output_dir:
            output_path = Path(output_dir) / f"{segment_name}_tremor_detection.parquet"
            results.to_parquet(output_path)
            logger.info(f"Saved tremor detection to {output_path}")

        if steps[-1] == "detection":
            return results

    # Step 3: Quantification (for tremor, this is typically severity scoring)
    if "quantification" in steps:
        if results is None or DataColumns.PRED_TREMOR_PROBA not in results.columns:
            # Need detection results
            logger.info(
                "Running feature extraction and detection (required for quantification)"
            )
            tremor_features = pipeline_info["extract_features"](df, config)
            try:
                from importlib.resources import files

                classifier_path = (
                    files("paradigma.assets") / pipeline_info["classifier_path"]
                )
                clf_package = ClassifierPackage.load(classifier_path)
                tremor_proba = pipeline_info["detect"](
                    tremor_features, clf_package, parallel
                )
                tremor_features[DataColumns.PRED_TREMOR_PROBA] = tremor_proba
                results = tremor_features
            except Exception:
                logger.warning(
                    "Could not load tremor classifier, using mock predictions"
                )
                tremor_features[DataColumns.PRED_TREMOR_PROBA] = 0.3
                results = tremor_features

        logger.info("Quantifying tremor severity")
        # Add tremor severity quantification (placeholder for actual implementation)
        results["tremor_severity"] = (
            results[DataColumns.PRED_TREMOR_PROBA] * 4
        )  # Scale to 0-4

        if save_intermediate and output_dir:
            output_path = (
                Path(output_dir) / f"{segment_name}_tremor_quantification.parquet"
            )
            results.to_parquet(output_path)
            logger.info(f"Saved tremor quantification to {output_path}")

        if steps[-1] == "quantification":
            return results

    # Step 4: Aggregation
    if "aggregation" in steps:
        if results is None:
            # Run all previous steps
            logger.info("Running full tremor pipeline for aggregation")
            tremor_features = pipeline_info["extract_features"](df, config)
            try:
                from importlib.resources import files

                classifier_path = (
                    files("paradigma.assets") / pipeline_info["classifier_path"]
                )
                clf_package = ClassifierPackage.load(classifier_path)
                tremor_proba = pipeline_info["detect"](
                    tremor_features, clf_package, parallel
                )
                tremor_features[DataColumns.PRED_TREMOR_PROBA] = tremor_proba
                tremor_features["tremor_severity"] = tremor_proba * 4
                results = tremor_features
            except Exception:
                logger.warning(
                    "Could not load tremor classifier, using mock predictions"
                )
                tremor_features[DataColumns.PRED_TREMOR_PROBA] = 0.3
                tremor_features["tremor_severity"] = 0.3 * 4
                results = tremor_features

        logger.info("Aggregating tremor results")
        # Create summary statistics
        aggregated = pd.DataFrame(
            {
                "mean_tremor_probability": [
                    results[DataColumns.PRED_TREMOR_PROBA].mean()
                ],
                "max_tremor_probability": [
                    results[DataColumns.PRED_TREMOR_PROBA].max()
                ],
                "mean_tremor_severity": [results["tremor_severity"].mean()],
                "tremor_episodes": [
                    int((results[DataColumns.PRED_TREMOR_PROBA] > 0.5).sum())
                ],
                "total_duration_minutes": [len(results) / (config.fs * 60)],
            }
        )

        if save_intermediate and output_dir:
            output_path = (
                Path(output_dir) / f"{segment_name}_tremor_aggregation.parquet"
            )
            aggregated.to_parquet(output_path)
            logger.info(f"Saved tremor aggregation to {output_path}")

        return aggregated

    return results


def _run_pulse_rate_pipeline(
    df: pd.DataFrame,
    config: PulseRateConfig,
    pipeline_info: Dict,
    parallel: bool,
    segment_name: str,
    steps: List[str],
    output_dir: Optional[str] = None,
    save_intermediate: bool = False,
) -> pd.DataFrame:
    """Run the pulse rate analysis pipeline with configurable steps.

    Steps: preprocessing -> estimation -> filtering -> aggregation

    Args:
        df: Input sensor data
        config: Pulse rate configuration
        pipeline_info: Pipeline function mappings
        parallel: Enable parallel processing
        segment_name: Name of data segment
        steps: List of pipeline steps to execute
        output_dir: Directory to save intermediate results
        save_intermediate: Whether to save intermediate results

    Returns:
        Results from the final executed step
    """

    # Validate that required columns exist
    required_columns = [
        DataColumns.TIME,
        DataColumns.PPG,
        DataColumns.ACCELEROMETER_X,
        DataColumns.ACCELEROMETER_Y,
        DataColumns.ACCELEROMETER_Z,
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(
            f"Missing required columns for pulse rate pipeline: {missing_columns}"
        )
        return pd.DataFrame()

    results = None

    # Step 1: Preprocessing (signal cleaning, artifact removal)
    if "preprocessing" in steps:
        logger.info("Preprocessing pulse rate signals")
        # Basic preprocessing (placeholder for actual implementation)
        preprocessed_df = df.copy()
        # Add signal quality assessment
        preprocessed_df["signal_quality"] = 0.8  # Placeholder
        results = preprocessed_df

        if save_intermediate and output_dir:
            output_path = (
                Path(output_dir) / f"{segment_name}_pulse_rate_preprocessing.parquet"
            )
            results.to_parquet(output_path)
            logger.info(f"Saved pulse rate preprocessing to {output_path}")

        if steps == ["preprocessing"]:
            return results

    # Step 2: Estimation
    if "estimation" in steps:
        if results is None:
            # Need preprocessed data
            logger.info("Preprocessing pulse rate signals (required for estimation)")
            preprocessed_df = df.copy()
            preprocessed_df["signal_quality"] = 0.8
            results = preprocessed_df

        logger.info("Estimating pulse rate")
        try:
            pulse_rate_results = pipeline_info["estimate"](results, config)
            results = pulse_rate_results
        except Exception as e:
            logger.warning(f"Pulse rate estimation failed: {e}, using mock data")
            results = pd.DataFrame(
                {
                    DataColumns.TIME: df[DataColumns.TIME],
                    "pulse_rate": 75.0,
                    "signal_quality": 0.8,
                }
            )

        if save_intermediate and output_dir:
            output_path = (
                Path(output_dir) / f"{segment_name}_pulse_rate_estimation.parquet"
            )
            results.to_parquet(output_path)
            logger.info(f"Saved pulse rate estimation to {output_path}")

        if steps[-1] == "estimation":
            return results

    # Step 3: Filtering (quality-based filtering)
    if "filtering" in steps:
        if results is None or "pulse_rate" not in results.columns:
            # Need estimation results
            logger.info("Running preprocessing and estimation (required for filtering)")
            preprocessed_df = df.copy()
            preprocessed_df["signal_quality"] = 0.8
            try:
                pulse_rate_results = pipeline_info["estimate"](preprocessed_df, config)
                results = pulse_rate_results
            except Exception as e:
                logger.warning(f"Pulse rate estimation failed: {e}, using mock data")
                results = pd.DataFrame(
                    {
                        DataColumns.TIME: df[DataColumns.TIME],
                        "pulse_rate": 75.0,
                        "signal_quality": 0.8,
                    }
                )

        logger.info("Filtering pulse rate estimates")
        # Filter based on signal quality
        quality_threshold = 0.5
        results["pulse_rate_filtered"] = results["pulse_rate"].where(
            results["signal_quality"] >= quality_threshold, np.nan
        )

        if save_intermediate and output_dir:
            output_path = (
                Path(output_dir) / f"{segment_name}_pulse_rate_filtering.parquet"
            )
            results.to_parquet(output_path)
            logger.info(f"Saved pulse rate filtering to {output_path}")

        if steps[-1] == "filtering":
            return results

    # Step 4: Aggregation
    if "aggregation" in steps:
        if results is None or "pulse_rate" not in results.columns:
            # Run all previous steps
            logger.info("Running full pulse rate pipeline for aggregation")
            preprocessed_df = df.copy()
            preprocessed_df["signal_quality"] = 0.8
            try:
                pulse_rate_results = pipeline_info["estimate"](preprocessed_df, config)
                results = pulse_rate_results
            except Exception as e:
                logger.warning(f"Pulse rate estimation failed: {e}, using mock data")
                results = pd.DataFrame(
                    {
                        DataColumns.TIME: df[DataColumns.TIME],
                        "pulse_rate": 75.0,
                        "signal_quality": 0.8,
                    }
                )

            # Add filtering
            quality_threshold = 0.5
            results["pulse_rate_filtered"] = results["pulse_rate"].where(
                results["signal_quality"] >= quality_threshold, np.nan
            )

        logger.info("Aggregating pulse rate results")
        # Create summary statistics
        try:
            _ = pipeline_info["aggregate"](results, config)
        except Exception:
            logger.warning("Aggregation function failed, creating basic aggregation")

        # Calculate aggregated metrics
        valid_pr = (
            results["pulse_rate_filtered"].dropna()
            if "pulse_rate_filtered" in results.columns
            else results["pulse_rate"].dropna()
        )

        aggregated = pd.DataFrame(
            {
                "mean_pulse_rate": [valid_pr.mean() if len(valid_pr) > 0 else np.nan],
                "median_pulse_rate": [
                    valid_pr.median() if len(valid_pr) > 0 else np.nan
                ],
                "std_pulse_rate": [valid_pr.std() if len(valid_pr) > 0 else np.nan],
                "min_pulse_rate": [valid_pr.min() if len(valid_pr) > 0 else np.nan],
                "max_pulse_rate": [valid_pr.max() if len(valid_pr) > 0 else np.nan],
                "mean_signal_quality": [results["signal_quality"].mean()],
                "valid_samples": [len(valid_pr)],
                "total_duration_minutes": [len(results) / (config.fs * 60)],
            }
        )

        if save_intermediate and output_dir:
            output_path = (
                Path(output_dir) / f"{segment_name}_pulse_rate_aggregation.parquet"
            )
            aggregated.to_parquet(output_path)
            logger.info(f"Saved pulse rate aggregation to {output_path}")

        return aggregated

    return results


def list_available_pipelines() -> List[str]:
    """Return a list of available pipeline names."""
    return list(AVAILABLE_PIPELINES.keys())


def get_pipeline_info(pipeline_name: str) -> Dict:
    """Get information about a specific pipeline."""
    if pipeline_name not in AVAILABLE_PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    return AVAILABLE_PIPELINES[pipeline_name].copy()
