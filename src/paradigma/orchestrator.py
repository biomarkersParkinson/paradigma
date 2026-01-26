"""
High-level pipeline orchestrator for ParaDigMa toolbox.

This module provides the main entry point for running analysis pipelines:

Main Functions
--------------
- run_paradigma(): Complete pipeline from data loading/preparation to aggregated results.
  Main entry point for end-to-end analysis supporting multiple pipelines (gait, tremor, pulse_rate).

- run_pipeline_batch(): Runs a specific pipeline on multiple data files and aggregates results.
  Used internally by run_paradigma() for processing batches of prepared data.

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
from typing import Dict, List

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
from paradigma.load import load_data_files, save_prepared_data
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


def run_pipeline_batch(
    pipeline_name: str,
    dfs: pd.DataFrame | List[pd.DataFrame] | Dict[str, pd.DataFrame],
    output_dir: str | Path = "./output",
    watch_side: str | None = None,
    store_intermediate: List[str] | None = [],
    aggregates: List[str] | None = None,
    imu_config: IMUConfig | None = None,
    ppg_config: PPGConfig | None = None,
    gait_config: GaitConfig | None = None,
    arm_activity_config: GaitConfig | None = None,
    tremor_config: TremorConfig | None = None,
    pulse_rate_config: PulseRateConfig | None = None,
    gait_segment_categories: List[str] | None = None,
    verbose: int = 1,
) -> Dict[str, pd.DataFrame | Dict]:
    """
    Run specific pipeline on multiple data files and aggregate results.

    Parameters
    ----------
    dfs : DataFrame, list of DataFrames, or dict of DataFrames
        Data files to process. Can be:
        - Single DataFrame: Will be processed as one file with key 'df_1'.
        - List[DataFrame]: Multiple dataframes assigned IDs as 'df_1', 'df_2', etc.
        - Dict[str, DataFrame]: Keys are file names, values are dataframes.
        Note: The 'file_key' column is only added to quantification results when
        len(dfs) > 1, allowing cleaner output for single-file processing.
        See input_formats guide for details.
    pipeline_name: str
        Name of the pipeline to run: 'gait', 'tremor', or 'pulse_rate'.
    watch_side : str, optional
        Watch side: 'left' or 'right' (required for gait pipeline).
    store_intermediate : list of str, default []
        Steps of which output is to be stored. Valid values:
        ['preparation', 'preprocessing', 'classification', 'quantification', 'aggregation']
        If empty, no files are saved (results are only returned).
    output_dir : str or Path, optional
        Output directory for results (required only if store_intermediate is not empty).
    aggregates : list of str, optional
        Aggregation methods for quantification.
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
    gait_segment_categories : list of str, optional
        Categories for gait segment aggregation.
    verbose : int, default 1
        Logging verbose level.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'quantifications': Combined quantified results from all files (DataFrame)
        - 'metadata': Combined metadata from all files (dict)
        - 'aggregations': Aggregated results across all files (dict)
    """

    # Validate pipeline-specific requirements
    if pipeline_name == "gait" and watch_side not in ["left", "right"]:
        logger.error(
            "watch_side must be specified as 'left' or 'right' for gait pipeline"
        )
        raise ValueError(
            "watch_side must be specified as 'left' or 'right' for gait pipeline"
        )

    # Convert single DataFrame or list of DataFrames to dict if needed
    if isinstance(dfs, pd.DataFrame):
        dfs = {"df_1": dfs}
    elif isinstance(dfs, list):
        dfs = {f"df_{i}": df for i, df in enumerate(dfs, start=1)}

    # Convert and validate output directory
    output_dir = Path(output_dir)
    if len(store_intermediate) > 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results storage
    all_quantifications = []
    all_metadata = {}

    # Process each file with the appropriate pipeline
    for i, (file_key, df) in enumerate(dfs.items(), start=1):
        if len(dfs) > 1:
            logger.info(f"Processing file: {file_key} ({i}/{len(dfs)})")
        else:
            logger.info("Processing single provided DataFrame")

        try:
            # Set up output directory for this file (only if storing intermediate results)
            file_output_dir = output_dir / "individual_files" / file_key

            # Run the appropriate pipeline based on pipeline_name
            if pipeline_name == "gait":
                # Calculate gait segment number offset to avoid conflicts when concatenating multiple gait segments
                if len(all_quantifications) == 0:
                    segment_nr_offset = 0
                else:
                    # Find maximum gait segment number from all previously processed data files
                    segment_nr_offset = int(
                        max(
                            [
                                df["gait_segment_nr"].max()
                                for df in all_quantifications
                                if len(df) > 0
                            ]
                        )
                    )

                quantification_data, quantification_metadata = run_gait_pipeline(
                    df_prepared=df,
                    watch_side=watch_side,
                    imu_config=imu_config,
                    gait_config=gait_config,
                    arm_activity_config=arm_activity_config,
                    store_intermediate=store_intermediate,
                    output_dir=file_output_dir,
                    segment_number_offset=segment_nr_offset,
                    verbose=verbose,
                )

                # Store metadata in the combined metadata dictionary
                if quantification_metadata and "per_segment" in quantification_metadata:
                    for seg_id, meta in quantification_metadata["per_segment"].items():
                        all_metadata[seg_id] = meta

            elif pipeline_name == "tremor":
                quantification_data = run_tremor_pipeline(
                    df_prepared=df,
                    store_intermediate=store_intermediate,
                    output_dir=file_output_dir,
                    tremor_config=tremor_config,
                    imu_config=imu_config,
                    verbose=verbose,
                )
            elif pipeline_name == "pulse_rate":
                quantification_data = run_pulse_rate_pipeline(
                    df_ppg_prepared=df,
                    store_intermediate=store_intermediate,
                    output_dir=file_output_dir,
                    pulse_rate_config=pulse_rate_config,
                    ppg_config=ppg_config,
                    verbose=verbose,
                )
            else:
                raise ValueError(
                    f"Unknown pipeline: {pipeline_name}. Supported pipelines: 'gait', 'tremor', 'pulse_rate'"
                )

            if len(quantification_data) > 0:
                # Add file identifier to quantified results
                quantification_data = (
                    quantification_data.copy()
                )  # Ensure we're working with a copy
                if len(dfs) > 1:
                    quantification_data["file_key"] = file_key
                all_quantifications.append(quantification_data)
            else:
                logger.warning(
                    f"Data file {file_key}: No quantified data returned from {pipeline_name} pipeline"
                )

        except Exception as e:
            logger.error(f"Failed to process data file {file_key}: {e}")

    # Combine all quantified files
    if all_quantifications:
        combined_quantified = pd.concat(all_quantifications, ignore_index=True)
        logger.info(
            f"Combined results: {len(combined_quantified)} windows from {len(all_quantifications)} data files"
        )
    else:
        combined_quantified = pd.DataFrame()
        logger.warning(f"No quantified {pipeline_name} found in any data file")

    # Aggregate results across all data files
    aggregations = {}
    if (
        len(combined_quantified) > 0
        and pipeline_name == "gait"
        and len(all_metadata) > 0
    ):
        logger.info("Aggregating gait results across all data files")
        try:
            # Use default gait segment categories if none provided
            if gait_segment_categories is None:
                gait_segment_categories = [(0, 10), (10, 20), (20, np.inf), (0, np.inf)]

            # Use default aggregates if none provided
            if aggregates is None:
                aggregates = ["median", "95p", "cov"]

            aggregations = aggregate_arm_swing_params(
                df_arm_swing_params=combined_quantified,
                segment_meta=all_metadata,
                segment_cats=gait_segment_categories,
                aggregates=aggregates,
            )
            logger.info(
                f"Aggregation completed across {len(gait_segment_categories)} gait segment length categories"
            )
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            aggregations = {}
    elif len(combined_quantified) > 0 and pipeline_name == "tremor":
        logger.info("Aggregating tremor results across all data files")
        try:
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

            # Use default tremor config for aggregation
            if tremor_config is None:
                tremor_config = TremorConfig()

            aggregation_output_tremor = aggregate_tremor(
                tremor_data_for_aggregation, tremor_config
            )
            aggregations = aggregation_output_tremor["aggregated_tremor_measures"]
            all_metadata = aggregation_output_tremor["metadata"]
            logger.info("Tremor aggregation completed")
        except Exception as e:
            logger.error(f"Tremor aggregation failed: {e}")
    elif len(combined_quantified) > 0 and pipeline_name == "pulse_rate":
        logger.info("Aggregating pulse rate results across all data files")
        try:
            # Extract pulse rate values for aggregation
            pulse_rate_values = (
                combined_quantified[DataColumns.PULSE_RATE].dropna().values
            )

            if len(pulse_rate_values) > 0:
                aggregation_output_pr = aggregate_pulse_rate(
                    pr_values=pulse_rate_values,
                    aggregates=aggregates if aggregates else ["mode", "99p"],
                )
                aggregations = aggregation_output_pr["pr_aggregates"]
                all_metadata = aggregation_output_pr["metadata"]
                logger.info(
                    f"Pulse rate aggregation completed with {len(pulse_rate_values)} valid estimates"
                )
            else:
                logger.warning("No valid pulse rate estimates found for aggregation")
                aggregations = {}
        except Exception as e:
            logger.error(f"Pulse rate aggregation failed: {e}")
            aggregations = {}

    # Save combined results (only if store_intermediate is not empty)
    # Save combined quantified data
    if len(combined_quantified) > 0 and "quantification" in store_intermediate:
        save_prepared_data(
            combined_quantified,
            output_dir / f"quantifications_{pipeline_name}.parquet",
            verbose=verbose,
        )

    # Save aggregated results
    if aggregations and "aggregation" in store_intermediate:
        with open(output_dir / f"aggregations_{pipeline_name}.json", "w") as f:
            json.dump(aggregations, f, indent=2)

    if store_intermediate:
        logger.info(f"Results saved to {output_dir}")

    return {
        "quantifications": combined_quantified,
        "aggregations": aggregations,
        "metadata": all_metadata,
    }


def run_paradigma(
    output_dir: str | Path = "./output",
    data_path: str | Path | None = None,
    dfs: pd.DataFrame | List[pd.DataFrame] | Dict[str, pd.DataFrame] | None = None,
    skip_preparation: bool = False,
    pipelines: List[str] | str | None = None,
    watch_side: str | None = None,
    accelerometer_units: str = "g",
    gyroscope_units: str = "deg/s",
    time_input_unit: TimeUnit = TimeUnit.RELATIVE_S,
    target_frequency: float = 100.0,
    column_mapping: Dict[str, str] | None = None,
    device_orientation: List[str] | None = ["x", "y", "z"],
    save_intermediate: List[str] = [],
    file_pattern: str | List[str] | None = None,
    aggregates: List[str] | None = None,
    segment_length_bins: List[str] | None = None,
    split_by_gaps: bool = False,
    max_gap_seconds: float | None = None,
    min_segment_seconds: float | None = None,
    imu_config: IMUConfig | None = None,
    ppg_config: PPGConfig | None = None,
    gait_config: GaitConfig | None = None,
    tremor_config: TremorConfig | None = None,
    pulse_rate_config: PulseRateConfig | None = None,
    verbose: int = 1,
) -> Dict[str, pd.DataFrame | Dict]:
    """
    Complete ParaDigMa analysis pipeline from data loading to aggregated results.

    This is the main entry point for ParaDigMa analysis. It supports multiple pipeline types:
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
    skip_preparation : bool, default False
        Whether data is already prepared. If False, data will be prepared (unit conversion,
        resampling, etc.). If True, assumes data is already in the required format.
    pipelines : list of str or str, optional
        Pipelines to run: 'gait', 'tremor', and/or 'pulse_rate'. If providing a list, currently
        only tremor and gait pipelines can be run together.
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
    output_dir : str or Path, default './output'
        Output directory for all results. Files are only saved if save_intermediate is not empty.
    save_intermediate : list of str, default []
        Which intermediate results to store. Valid values:
        - 'preparation': Save prepared data
        - 'preprocessing': Save preprocessed signals
        - 'classification': Save classification results
        - 'quantification': Save quantified measures
        - 'aggregation': Save aggregated results
        If empty, no files are saved (results are only returned).
    file_pattern : str or list of str, optional
        File pattern(s) to match when loading data (e.g., 'parquet', '*.csv').
    aggregates : list of str, optional
        Aggregation methods for quantification.
    segment_length_bins : list of str, optional
        Duration bins for gait segment aggregation (gait pipeline only).
        Example: ['(0, 10)', '(10, 20)'] for segments 0-10s and 10-20s.
    split_by_gaps : bool, default False
        If True, automatically split non-contiguous data into segments during preparation.
        Adds 'data_segment_nr' column to prepared data which is preserved through pipeline.
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
    tremor_config : TremorConfig, optional
        Tremor analysis configuration.
    pulse_rate_config : PulseRateConfig, optional
        Pulse rate analysis configuration.
    verbose : int, default 1
        Logging verbose level:
        - 0: Only errors and warnings
        - 1: Basic info (default)
        - 2: Detailed info with progress
        - 3: Debug level with all details

    Returns
    -------
    dict
        Complete analysis results with nested structure for multiple pipelines:
        - 'quantifications': dict with pipeline names as keys and DataFrames as values
        - 'aggregations': dict with pipeline names as keys and result dicts as values
        - 'metadata': dict with pipeline names as keys and metadata dicts as values
    """
    if (data_path is None and dfs is None) or (
        data_path is not None and dfs is not None
    ):
        raise ValueError("Either data_path or dfs must be provided, but not both")

    if isinstance(pipelines, str):
        pipelines = [pipelines]

    if len(pipelines) > 1 and "pulse_rate" in pipelines:
        raise ValueError(
            "Pulse rate pipeline cannot be run together with other pipelines"
        )

    if "gait" in pipelines and watch_side not in ["left", "right"]:
        raise ValueError(
            "watch_side must be specified as 'left' or 'right' for gait pipeline"
        )

    if any(p not in ["gait", "tremor", "pulse_rate"] for p in pipelines):
        raise ValueError(
            f"At least one unknown pipeline provided: {pipelines}. "
            f"Supported pipelines: 'gait', 'tremor', 'pulse_rate'"
        )

    # Set logging level based on verbose
    if verbose == 0:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose >= 2:
        logger.setLevel(logging.DEBUG)

    if data_path is not None:
        data_path = Path(data_path)
        if verbose >= 1:
            logger.info(f"Applying ParaDigMa pipelines to {data_path}")
    else:
        if verbose >= 1:
            logger.info("Applying ParaDigMa pipelines to provided DataFrame")

    # Convert and create output directory
    output_dir = Path(output_dir)
    if len(save_intermediate) > 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = output_dir / f"paradigma_run_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    # Step 1: Load data files
    if data_path is not None:
        if verbose >= 1:
            logger.info("Step 1: Loading data files")
        try:
            dfs = load_data_files(
                data_path=data_path, file_patterns=file_pattern, verbose=verbose
            )
            if verbose >= 1:
                logger.info(f"Loaded {len(dfs)} data files")
        except Exception as e:
            logger.error(f"Failed to load data files: {e}")
            raise

        if not dfs:
            raise ValueError(f"No data files found in {data_path}")
    else:
        if verbose >= 1:
            logger.info("Step 1: Using provided DataFrame(s) as input")

    if isinstance(dfs, list):
        # Convert list to dict with unique dataframe IDs
        dfs = {f"df_{i}": df for i, df in enumerate(dfs, start=1)}
    elif isinstance(dfs, pd.DataFrame):
        # Convert single DataFrame to dict with default name
        dfs = {"df_1": dfs}

    # Step 2: Prepare data if needed
    if not skip_preparation:
        if verbose >= 1:
            logger.info("Step 2: Preparing raw data")

        prepare_params = {
            "time_input_unit": time_input_unit,
            "resampling_frequency": target_frequency,
            "column_mapping": column_mapping,
            "auto_segment": split_by_gaps,
            "max_segment_gap_s": max_gap_seconds,
            "min_segment_length_s": min_segment_seconds,
            "verbose": verbose,
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

        prepared_files = {}
        for file_name, df_raw in dfs.items():
            if verbose >= 2:
                logger.info(f"Preparing data for {file_name}")
            try:
                df_prepared = prepare_raw_data(
                    df=df_raw, watch_side=watch_side, **prepare_params
                )
                prepared_files[file_name] = df_prepared

                # Save prepared data
                if output_dir and "preparation" in save_intermediate:
                    prepared_dir = output_dir / "prepared_data"
                    prepared_dir.mkdir(exist_ok=True)
                    save_prepared_data(
                        df_prepared,
                        prepared_dir / f"{file_name}.parquet",
                        verbose=verbose,
                    )

            except Exception as e:
                logger.error(f"Failed to prepare data for {file_name}: {e}")
                # Continue with other files

        dfs = prepared_files
        if verbose >= 1:
            logger.info(f"Successfully prepared {len(dfs)} data files")
    else:
        if verbose >= 1:
            logger.info("Step 2: Data already prepared, skipping preparation")

    # Step 3: Run pipelines on all data files
    if verbose >= 1:
        logger.info(f"Step 3: Running pipelines {pipelines} on {len(dfs)} data files")

    # Initialize combined results structure
    all_results = {
        "quantifications": {},
        "aggregations": {},
        "metadata": {},
    }

    # Run each pipeline
    for pipeline_name in pipelines:
        if verbose >= 1:
            logger.info(f"Running {pipeline_name} pipeline")

        try:
            pipeline_results = run_pipeline_batch(
                dfs=dfs,
                pipeline_name=pipeline_name,
                watch_side=watch_side,
                imu_config=imu_config,
                ppg_config=ppg_config,
                gait_config=gait_config,
                tremor_config=tremor_config,
                pulse_rate_config=pulse_rate_config,
                store_intermediate=save_intermediate,
                output_dir=output_dir,
                gait_segment_categories=segment_length_bins,
                aggregates=aggregates,
                verbose=verbose,
            )

            # Store results for this pipeline
            all_results["quantifications"][pipeline_name] = pipeline_results[
                "quantifications"
            ]
            all_results["aggregations"][pipeline_name] = pipeline_results[
                "aggregations"
            ]
            all_results["metadata"][pipeline_name] = pipeline_results["metadata"]

            if verbose >= 1:
                logger.info(f"{pipeline_name.capitalize()} pipeline completed")

        except Exception as e:
            logger.error(f"Failed to run {pipeline_name} pipeline: {e}")
            # Store empty results for failed pipeline
            all_results["quantifications"][pipeline_name] = pd.DataFrame()
            all_results["aggregations"][pipeline_name] = {}
            all_results["metadata"][pipeline_name] = {}

    if verbose >= 1:
        logger.info("ParaDigMa analysis completed for all pipelines")

    # Log final summary for all pipelines
    if verbose >= 2:
        for pipeline_name in pipelines:
            try:
                quant_df = all_results["quantifications"][pipeline_name]
                if not quant_df.empty and "file_key" in quant_df.columns:
                    successful_files = np.unique(quant_df["file_key"].values)
                    logger.info(
                        f"{pipeline_name.capitalize()}: Files successfully processed: {successful_files}"
                    )
                elif not quant_df.empty:
                    logger.info(
                        f"{pipeline_name.capitalize()}: Single file processed successfully"
                    )
                else:
                    logger.info(f"{pipeline_name.capitalize()}: No successful results")
            except Exception as e:
                logger.error(f"Error logging summary for {pipeline_name}: {e}")

    return all_results
