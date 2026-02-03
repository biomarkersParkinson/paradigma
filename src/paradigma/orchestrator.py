"""
High-level pipeline orchestrator for ParaDigMa toolbox.

This module provides the main entry point for running analysis pipelines:

Main Function
-------------
- run_paradigma(): Complete pipeline from data loading/preparation
  to aggregated results. Main entry point for end-to-end analysis
  supporting multiple pipelines (gait, tremor, pulse_rate).
  Can process raw data from disk or already-prepared DataFrames.

The orchestrator coordinates:
1. Data loading and preparation (unit conversion, resampling, orientation correction)
2. Pipeline execution on single or multiple files (imports from pipeline modules)
3. Result aggregation across files and segments
4. Optional intermediate result storage

Supports multi-file processing with automatic segment numbering and metadata tracking.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from paradigma.config import (
    GaitConfig,
    IMUConfig,
    PPGConfig,
    PulseRateConfig,
    TremorConfig,
)
from paradigma.constants import DataColumns, TimeUnit
from paradigma.load import (
    get_data_file_paths,
    load_single_data_file,
    save_prepared_data,
)
from paradigma.pipelines.gait_pipeline import (
    aggregate_arm_swing_params,
    run_gait_pipeline,
)
from paradigma.pipelines.pulse_rate_pipeline import (
    aggregate_pulse_rate,
    run_pulse_rate_pipeline,
)
from paradigma.pipelines.tremor_pipeline import aggregate_tremor, run_tremor_pipeline
from paradigma.prepare_data import prepare_raw_data

logger = logging.getLogger(__name__)

# Custom logging level for detailed info (between INFO=20 and DEBUG=10)
DETAILED_INFO = 15
logging.addLevelName(DETAILED_INFO, "DETAILED")


def run_paradigma(
    *,
    data_path: str | Path | None = None,
    dfs: pd.DataFrame | list[pd.DataFrame] | dict[str, pd.DataFrame] | None = None,
    save_intermediate: list[str] = [],
    output_dir: str | Path = "./output",
    skip_preparation: bool = False,
    pipelines: list[str] | str | None = None,
    watch_side: str | None = None,
    accelerometer_units: str = "g",
    gyroscope_units: str = "deg/s",
    time_input_unit: TimeUnit = TimeUnit.RELATIVE_S,
    target_frequency: float = 100.0,
    column_mapping: dict[str, str] | None = None,
    device_orientation: list[str] | None = ["x", "y", "z"],
    file_pattern: str | list[str] | None = None,
    aggregates: list[str] | None = None,
    segment_length_bins: list[str] | None = None,
    split_by_gaps: bool = False,
    max_gap_seconds: float | None = None,
    min_segment_seconds: float | None = None,
    imu_config: IMUConfig | None = None,
    ppg_config: PPGConfig | None = None,
    gait_config: GaitConfig | None = None,
    arm_activity_config: GaitConfig | None = None,
    tremor_config: TremorConfig | None = None,
    pulse_rate_config: PulseRateConfig | None = None,
    logging_level: int = logging.INFO,
    custom_logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """
    Complete ParaDigMa analysis pipeline from data loading to aggregated results.

    This is the main entry point for ParaDigMa analysis. It supports
    multiple pipeline types:
    - gait: Arm swing during gait analysis
    - tremor: Tremor detection and quantification
    - pulse_rate: Pulse rate estimation from PPG signals

    The function:
    1. Loads data files from the specified directory or uses provided DataFrame
    2. Prepares raw data if needed (unit conversion, resampling, etc.)
    3. Runs the specified pipeline on each data file
    4. Aggregates results across all data files

    Parameters
    ----------
    data_path : str or Path, optional
        Path to directory containing data files.
    dfs : DataFrame, list of DataFrames, or dict of DataFrames, optional
        Dataframes used as input (bypasses data loading). Can be:
        - Single DataFrame: Will be processed as one file with key 'df_1'.
        - List[DataFrame]: Multiple dataframes assigned IDs as 'df_1', 'df_2', etc.
        - Dict[str, DataFrame]: Keys are file names, values are dataframes.
        Note: The 'file_key' column is only added to quantification results when
        len(dfs) > 1, allowing cleaner output for single-file processing.
        See input_formats guide for details.
    save_intermediate : list of str, default []
        Which intermediate results to store. Valid values:
        - 'preparation': Save prepared data
        - 'preprocessing': Save preprocessed signals
        - 'classification': Save classification results
        - 'quantification': Save quantified measures
        - 'aggregation': Save aggregated results
        If empty, no files are saved (results are only returned).
    output_dir : str or Path, default './output'
        Output directory for all results. Files are only saved if
        save_intermediate is not empty.
    skip_preparation : bool, default False
        Whether data is already prepared. If False, data will be
        prepared (unit conversion, resampling, etc.). If True,
        assumes data is already in the required format.
    pipelines : list of str or str, optional
        Pipelines to run: 'gait', 'tremor', and/or 'pulse_rate'.
        If providing a list, currently only tremor and gait pipelines
        can be run together.
    watch_side : str, optional
        Watch side: 'left' or 'right' (required for gait pipeline).
    accelerometer_units : str, default 'm/s^2'
        Units for accelerometer data.
    gyroscope_units : str, default 'deg/s'
        Units for gyroscope data.
    time_input_unit : TimeUnit, default TimeUnit.RELATIVE_S
        Input time unit type.
    target_frequency : float, default 100.0
        Target sampling frequency for resampling.
    column_mapping : dict, optional
        Custom column name mapping.
    device_orientation : list of str, optional
        Custom device orientation corrections.
    file_pattern : str or list of str, optional
        File pattern(s) to match when loading data (e.g., 'parquet', '*.csv').
    aggregates : list of str, optional
        Aggregation methods for quantification.
    segment_length_bins : list of str, optional
        Duration bins for gait segment aggregation (gait pipeline only).
        Example: ['(0, 10)', '(10, 20)'] for segments 0-10s and 10-20s.
    split_by_gaps : bool, default False
        If True, automatically split non-contiguous data into segments
        during preparation.
        Adds 'data_segment_nr' column to prepared data which is preserved
        through pipeline.
        Useful for handling data with gaps/interruptions.
    max_gap_seconds : float, optional
        Maximum gap (seconds) before starting new segment. Used when split_by_gaps=True.
        Defaults to 1.5s.
    min_segment_seconds : float, optional
        Minimum segment length (seconds) to keep. Used when split_by_gaps=True.
        Defaults to 1.5s.
    imu_config : IMUConfig, optional
        IMU preprocessing configuration.
    ppg_config : PPGConfig, optional
        PPG preprocessing configuration.
    gait_config : GaitConfig, optional
        Gait analysis configuration.
    arm_activity_config : GaitConfig, optional
        Arm activity analysis configuration.
    tremor_config : TremorConfig, optional
        Tremor analysis configuration.
    pulse_rate_config : PulseRateConfig, optional
        Pulse rate analysis configuration.
    logging_level : int, default logging.INFO
        Logging level using standard logging constants:
        - logging.ERROR: Only errors
        - logging.WARNING: Warnings and errors
        - logging.INFO: Basic progress information (default)
        - logging.DEBUG: Detailed debug information
        Can also use DETAILED_INFO (15) for intermediate detail level.
    custom_logger : logging.Logger, optional
        Custom logger instance. If provided, logging_level is ignored.
        Allows full control over logging configuration.

    Returns
    -------
    dict
        Complete analysis results with nested structure for multiple pipelines:
        - 'quantifications': dict with pipeline names as keys and DataFrames as values
        - 'aggregations': dict with pipeline names as keys and result dicts as values
        - 'metadata': dict with pipeline names as keys and metadata dicts as values
        - 'errors': list of dicts tracking any errors that occurred during processing.
          Each error dict contains 'stage', 'error', and optionally 'file' and
          'pipeline'.
          Empty list indicates successful processing of all files.
    """
    if (data_path is None) == (dfs is None):
        raise ValueError("Exactly one of data_path or dfs must be provided")

    if isinstance(pipelines, str):
        pipelines = [pipelines]

    if len(pipelines) > 1 and "pulse_rate" in pipelines:
        raise ValueError(
            "Pulse rate pipeline cannot be run together with other pipelines"
        )

    if any(p not in ["gait", "tremor", "pulse_rate"] for p in pipelines):
        raise ValueError(
            f"At least one unknown pipeline provided: {pipelines}. "
            f"Supported pipelines: 'gait', 'tremor', 'pulse_rate'"
        )

    # Use custom logger if provided, otherwise use module logger
    active_logger = custom_logger if custom_logger is not None else logger

    # Get package logger for configuration (affects all paradigma.* modules)
    package_logger = logging.getLogger("paradigma")

    # Configure package-wide logging level for all paradigma modules
    if custom_logger is None:
        package_logger.setLevel(logging_level)

    if data_path is not None:
        data_path = Path(data_path)
        active_logger.info(f"Applying ParaDigMa pipelines to {data_path}")
    else:
        active_logger.info("Applying ParaDigMa pipelines to provided DataFrame")

    # Convert and create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file - add handler to package logger so ALL paradigma modules
    # log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = output_dir / f"paradigma_run_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    package_logger.addHandler(file_handler)
    active_logger.info(f"Logging to {log_file}")

    # Step 1: Get file paths or convert provided DataFrames
    file_paths = None  # Will hold list of file paths if loading from directory
    dfs_dict = None  # Will hold dict of DataFrames if provided directly

    if data_path is not None:
        active_logger.info("Step 1: Finding data files")
        try:
            file_paths = get_data_file_paths(
                data_path=data_path, file_patterns=file_pattern
            )
        except Exception as e:
            active_logger.error(f"Failed to find data files: {e}")
            raise

        if not file_paths:
            raise ValueError(f"No data files found in {data_path}")
    else:
        active_logger.info("Step 1: Using provided DataFrame(s) as input")

        # Convert provided dfs to dict format
        if isinstance(dfs, list):
            dfs_dict = {f"df_{i}": df for i, df in enumerate(dfs, start=1)}
        elif isinstance(dfs, pd.DataFrame):
            dfs_dict = {"df_1": dfs}
        else:
            dfs_dict = dfs

    # Determine number of files to process
    num_files = len(file_paths) if file_paths else len(dfs_dict)

    # Initialize results storage for each pipeline
    all_results = {
        "quantifications": {p: [] for p in pipelines},
        "aggregations": {p: {} for p in pipelines},
        "metadata": {p: {} for p in pipelines},
        "errors": [],
    }

    # Steps 2-3: Process each file individually
    active_logger.info(f"Steps 2-3: Processing {num_files} files individually")

    # Track maximum gait segment number across files for proper offset
    max_gait_segment_nr = 0

    for i in range(num_files):
        # Load one file at a time
        if file_paths:
            file_path = file_paths[i]
            active_logger.info(f"Processing file {i+1}/{num_files}: {file_path.name}")
            try:
                file_name, df_raw = load_single_data_file(file_path)
            except Exception as e:
                error_msg = f"Failed to load file {file_path.name}: {e}"
                active_logger.error(error_msg)
                all_results["errors"].append(
                    {"file": file_path.name, "stage": "loading", "error": str(e)}
                )
                continue
        else:
            # Using in-memory data
            file_name = list(dfs_dict.keys())[i]
            df_raw = dfs_dict[file_name]
            active_logger.info(f"Processing DataFrame {i+1}/{num_files}: {file_name}")

        try:
            # Step 2: Prepare data (if needed)
            if not skip_preparation:
                active_logger.log(DETAILED_INFO, f"Preparing data for {file_name}")

                prepare_params = {
                    "time_input_unit": time_input_unit,
                    "resampling_frequency": target_frequency,
                    "column_mapping": column_mapping,
                    "auto_segment": split_by_gaps,
                    "max_segment_gap_s": max_gap_seconds,
                    "min_segment_length_s": min_segment_seconds,
                }

                # Add pipeline-specific preparation parameters
                if "gait" in pipelines or "tremor" in pipelines:
                    prepare_params["gyroscope_units"] = gyroscope_units

                if "gait" in pipelines:
                    prepare_params.update(
                        {
                            "accelerometer_units": accelerometer_units,
                            "device_orientation": device_orientation,
                        }
                    )

                df_prepared = prepare_raw_data(df=df_raw, **prepare_params)

                # Save prepared data if requested
                if "preparation" in save_intermediate:
                    prepared_dir = output_dir / "prepared_data"
                    prepared_dir.mkdir(exist_ok=True)
                    save_prepared_data(
                        df_prepared,
                        prepared_dir / f"{file_name}.parquet",
                    )
            else:
                df_prepared = df_raw

            # Release raw data from memory
            del df_raw

            # Step 3: Run each pipeline on this single file
            store_intermediate_per_file = [
                x
                for x in save_intermediate
                if x not in ["aggregation", "quantification"]
            ]

            # Create file-specific output directory
            file_output_dir = output_dir / "individual_files" / file_name

            for pipeline_name in pipelines:
                active_logger.log(
                    DETAILED_INFO, f"Running {pipeline_name} pipeline on {file_name}"
                )

                try:
                    if pipeline_name == "gait":
                        quantification_data, quantification_metadata = (
                            run_gait_pipeline(
                                df_prepared=df_prepared,
                                watch_side=watch_side,
                                imu_config=imu_config,
                                gait_config=gait_config,
                                arm_activity_config=arm_activity_config,
                                store_intermediate=store_intermediate_per_file,
                                output_dir=file_output_dir,
                                segment_number_offset=max_gait_segment_nr,
                                logging_level=logging_level,
                                custom_logger=active_logger,
                            )
                        )

                        if len(quantification_data) > 0:
                            # Add file identifier if processing multiple files
                            quantification_data = quantification_data.copy()
                            if num_files > 1:
                                quantification_data["file_key"] = file_name
                            all_results["quantifications"][pipeline_name].append(
                                quantification_data
                            )

                            # Update max segment number for next file
                            max_gait_segment_nr = int(
                                quantification_data["gait_segment_nr"].max()
                            )

                        # Store metadata and update offset even if no quantifications
                        if (
                            quantification_metadata
                            and "per_segment" in quantification_metadata
                        ):
                            all_results["metadata"][pipeline_name].update(
                                quantification_metadata["per_segment"]
                            )

                            # Update max segment number based on metadata to prevent
                            # overwrites
                            if quantification_metadata["per_segment"]:
                                max_segment_in_metadata = max(
                                    quantification_metadata["per_segment"].keys()
                                )
                                max_gait_segment_nr = max(
                                    max_gait_segment_nr, max_segment_in_metadata
                                )

                    elif pipeline_name == "tremor":
                        quantification_data = run_tremor_pipeline(
                            df_prepared=df_prepared,
                            store_intermediate=store_intermediate_per_file,
                            output_dir=file_output_dir,
                            tremor_config=tremor_config,
                            imu_config=imu_config,
                            logging_level=logging_level,
                            custom_logger=active_logger,
                        )

                        if len(quantification_data) > 0:
                            quantification_data = quantification_data.copy()
                            if num_files > 1:
                                quantification_data["file_key"] = file_name
                            all_results["quantifications"][pipeline_name].append(
                                quantification_data
                            )

                    elif pipeline_name == "pulse_rate":
                        quantification_data = run_pulse_rate_pipeline(
                            df_ppg_prepared=df_prepared,
                            store_intermediate=store_intermediate_per_file,
                            output_dir=file_output_dir,
                            pulse_rate_config=pulse_rate_config,
                            ppg_config=ppg_config,
                            logging_level=logging_level,
                            custom_logger=active_logger,
                        )

                        if len(quantification_data) > 0:
                            quantification_data = quantification_data.copy()
                            if num_files > 1:
                                quantification_data["file_key"] = file_name
                            all_results["quantifications"][pipeline_name].append(
                                quantification_data
                            )

                except Exception as e:
                    error_msg = (
                        f"Failed to run {pipeline_name} pipeline on {file_name}: {e}"
                    )
                    active_logger.error(error_msg)
                    all_results["errors"].append(
                        {
                            "file": file_name,
                            "pipeline": pipeline_name,
                            "stage": "pipeline_execution",
                            "error": str(e),
                        }
                    )
                    continue

            # Release prepared data from memory
            del df_prepared

        except Exception as e:
            error_msg = f"Failed to process file {file_name}: {e}"
            active_logger.error(error_msg)
            all_results["errors"].append(
                {"file": file_name, "stage": "preparation", "error": str(e)}
            )
            continue

    # Step 4: Combine quantifications from all files
    active_logger.info("Step 4: Combining quantifications from all files")

    for pipeline_name in pipelines:
        # Concatenate all quantifications for this pipeline
        if all_results["quantifications"][pipeline_name]:
            combined_quantified = pd.concat(
                all_results["quantifications"][pipeline_name], ignore_index=True
            )

            num_files_processed = len(all_results["quantifications"][pipeline_name])
            all_results["quantifications"][pipeline_name] = combined_quantified

            active_logger.info(
                f"{pipeline_name.capitalize()}: Combined "
                f"{len(combined_quantified)} windows from "
                f"{num_files_processed} files"
            )

            # Step 5: Perform aggregation on combined results FROM ALL FILES
            try:
                if pipeline_name == "gait" and all_results["metadata"][pipeline_name]:
                    active_logger.info(
                        "Step 5: Aggregating gait results across ALL files"
                    )

                    if segment_length_bins is None:
                        gait_segment_categories = [
                            (0, 10),
                            (10, 20),
                            (20, np.inf),
                            (0, np.inf),
                        ]
                    else:
                        gait_segment_categories = segment_length_bins

                    if aggregates is None:
                        agg_methods = ["median", "95p", "cov"]
                    else:
                        agg_methods = aggregates

                    aggregations = aggregate_arm_swing_params(
                        df_arm_swing_params=combined_quantified,
                        segment_meta=all_results["metadata"][pipeline_name],
                        segment_cats=gait_segment_categories,
                        aggregates=agg_methods,
                    )
                    all_results["aggregations"][pipeline_name] = aggregations
                    active_logger.info(
                        f"Aggregation completed across "
                        f"{len(gait_segment_categories)} gait segment categories"
                    )

                elif pipeline_name == "tremor":
                    active_logger.info(
                        "Step 5: Aggregating tremor results across ALL files"
                    )

                    # Work on a copy for tremor aggregation
                    tremor_data_for_aggregation = combined_quantified.copy()

                    # Need to add datetime column for aggregate_tremor
                    if (
                        "time_dt" not in tremor_data_for_aggregation.columns
                        and "time" in tremor_data_for_aggregation.columns
                    ):
                        tremor_data_for_aggregation["time_dt"] = pd.to_datetime(
                            tremor_data_for_aggregation["time"], unit="s"
                        )

                    if tremor_config is None:
                        tremor_config = TremorConfig()

                    aggregation_output = aggregate_tremor(
                        tremor_data_for_aggregation, tremor_config
                    )
                    all_results["aggregations"][pipeline_name] = aggregation_output[
                        "aggregated_tremor_measures"
                    ]
                    all_results["metadata"][pipeline_name] = aggregation_output[
                        "metadata"
                    ]
                    active_logger.info("Tremor aggregation completed")

                elif pipeline_name == "pulse_rate":
                    active_logger.info(
                        "Step 5: Aggregating pulse rate results across ALL files"
                    )

                    pulse_rate_values = (
                        combined_quantified[DataColumns.PULSE_RATE].dropna().values
                    )

                    if len(pulse_rate_values) > 0:
                        aggregation_output = aggregate_pulse_rate(
                            pr_values=pulse_rate_values,
                            aggregates=aggregates if aggregates else ["mode", "99p"],
                        )
                        all_results["aggregations"][pipeline_name] = aggregation_output[
                            "pr_aggregates"
                        ]
                        all_results["metadata"][pipeline_name] = aggregation_output[
                            "metadata"
                        ]
                        active_logger.info(
                            f"Pulse rate aggregation completed with "
                            f"{len(pulse_rate_values)} valid estimates"
                        )
                    else:
                        active_logger.warning(
                            "No valid pulse rate estimates found for aggregation"
                        )

            except Exception as e:
                error_msg = f"Failed to aggregate {pipeline_name} results: {e}"
                active_logger.error(error_msg)
                all_results["errors"].append(
                    {"pipeline": pipeline_name, "stage": "aggregation", "error": str(e)}
                )
                all_results["aggregations"][pipeline_name] = {}

        else:
            # No quantifications found for this pipeline
            all_results["quantifications"][pipeline_name] = pd.DataFrame()
            active_logger.warning(f"No quantified {pipeline_name} results found")

    # Save combined quantifications if requested
    if "quantification" in save_intermediate:
        for pipeline_name in pipelines:
            if not all_results["quantifications"][pipeline_name].empty:
                quant_file = output_dir / f"quantifications_{pipeline_name}.parquet"
                save_prepared_data(
                    all_results["quantifications"][pipeline_name],
                    quant_file,
                )

    # Save aggregations if requested
    if "aggregation" in save_intermediate:
        for pipeline_name in pipelines:
            if all_results["aggregations"][pipeline_name]:
                agg_file = output_dir / f"aggregations_{pipeline_name}.json"
                with open(agg_file, "w") as f:
                    json.dump(all_results["aggregations"][pipeline_name], f, indent=2)
                active_logger.info(f"Saved aggregations to {agg_file}")

    if all_results["errors"]:
        active_logger.warning(
            f"ParaDigMa analysis completed with {len(all_results['errors'])} error(s)"
        )
    else:
        active_logger.info(
            "ParaDigMa analysis completed successfully for all pipelines"
        )

    # Log final summary for all pipelines
    for pipeline_name in pipelines:
        quant_df = all_results["quantifications"][pipeline_name]
        if not quant_df.empty and "file_key" in quant_df.columns:
            successful_files = np.unique(quant_df["file_key"].values)
            active_logger.log(
                DETAILED_INFO,
                f"{pipeline_name.capitalize()}: Files successfully "
                f"processed: {successful_files}",
            )
        elif not quant_df.empty:
            active_logger.log(
                DETAILED_INFO,
                f"{pipeline_name.capitalize()}: Single file processed " f"successfully",
            )
        else:
            active_logger.log(
                DETAILED_INFO, f"{pipeline_name.capitalize()}: No successful results"
            )

    # Close file handler to release log file - remove from package logger
    package_logger = logging.getLogger("paradigma")
    for handler in package_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            package_logger.removeHandler(handler)

    return all_results
