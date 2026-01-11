#!/usr/bin/env python3
"""
Example script demonstrating the ParaDigMa pipeline runner.

This script shows how to use the high-level run_pipeline() function
to process sensor data through ParaDigMa pipelines.
"""


from paradigma import run_pipeline
from paradigma.config import GaitConfig, TremorConfig


def main():
    """Run example pipelines on sample data."""

    # Example 1: Basic usage with TSDF data (auto-detected)
    print("Example 1: Running gait pipeline on TSDF data (auto-detected)")
    try:
        results = run_pipeline(
            data_path="example_data/verily/imu",  # TSDF data path
            pipelines=["gait"],
            config="default",
            output_dir="results/example1_tsdf",
            verbose=True,
        )
        print(f"Gait pipeline completed. Results shape: {results['gait'].shape}")
    except Exception as e:
        print(f"Example 1 failed: {e}")

    print("\\n" + "=" * 50 + "\\n")

    # Example 2: Prepared dataframes with custom configuration
    print("Example 2: Running multiple pipelines on prepared dataframes")
    try:
        # Custom configuration objects
        gait_config = GaitConfig(step="gait")
        gait_config.window_length_s = 2.0  # Customize window length

        tremor_config = TremorConfig(step="features")
        tremor_config.window_length_s = 1.5  # Different window for tremor

        custom_config = {"gait": gait_config, "tremor": tremor_config}

        results = run_pipeline(
            data_path="data/prepared",  # Prepared dataframes directory
            pipelines=["gait", "tremor"],
            config=custom_config,
            data_format="prepared",
            file_pattern="*.parquet",  # Look for parquet files
            output_dir="results/example2_prepared",
            parallel=True,  # Enable parallel processing
            verbose=True,
        )

        print("Multi-pipeline execution completed:")
        for pipeline_name, result in results.items():
            print(f"  {pipeline_name}: {len(result)} rows")

    except Exception as e:
        print(f"Example 2 failed: {e}")

    print("\\n" + "=" * 50 + "\\n")

    # Example 3: Axivity CWA file processing
    print("Example 3: Processing Axivity CWA files")
    try:
        results = run_pipeline(
            data_path="data/axivity/device001.cwa",  # Single CWA file
            pipelines=["gait"],
            data_format="axivity",
            output_dir="results/example3_axivity",
            verbose=True,
        )

        print(f"Axivity processing completed: {list(results.keys())}")

    except Exception as e:
        print(f"Example 3 failed: {e}")

    print("\\n" + "=" * 50 + "\\n")

    # Example 4: Empatica AVRO file processing
    print("Example 4: Processing Empatica AVRO files")
    try:
        results = run_pipeline(
            data_path="data/empatica/",  # Directory with .avro files
            pipelines=["gait"],
            data_format="empatica",
            output_dir="results/example4_empatica",
            verbose=True,
        )

        print(f"Empatica processing completed: {list(results.keys())}")

    except Exception as e:
        print(f"Example 4 failed: {e}")

    print("\\n" + "=" * 50 + "\\n")

    # Example 5: Auto-detection and programmatic pipeline selection
    print("Example 5: Auto-detection and programmatic pipeline selection")
    try:
        from paradigma import list_available_pipelines

        available = list_available_pipelines()
        print(f"Available pipelines: {available}")

        # Auto-detect format and run appropriate pipelines
        results = run_pipeline(
            data_path="example_data/verily/imu",
            pipelines=["gait"],  # Start with just gait
            output_dir="results/example5_auto",
            verbose=True,  # Format will be auto-detected
        )

        print(f"Completed pipeline execution for: {list(results.keys())}")

    except Exception as e:
        print(f"Example 5 failed: {e}")

    print("\\n" + "=" * 50 + "\\n")

    # Example 6: Column mapping for different naming conventions
    print("Example 6: Using column mapping for TSDF data with different column names")
    try:
        # Define column mapping for TSDF data with non-standard naming
        column_mapping = {
            "acceleration_x": "accelerometer_x",
            "acceleration_y": "accelerometer_y",
            "acceleration_z": "accelerometer_z",
            "rotation_x": "gyroscope_x",
            "rotation_y": "gyroscope_y",
            "rotation_z": "gyroscope_z",
        }

        results = run_pipeline(
            data_path="data/tsdf_nonstandard_names/",
            pipelines=["tremor"],  # Tremor is more forgiving than gait
            column_mapping=column_mapping,
            verbose=True,
        )

        print(
            f"Successfully mapped columns and processed {len(results['tremor'])} rows"
        )

    except Exception as e:
        print(f"Example 6 failed (expected for demo data): {e}")


if __name__ == "__main__":
    main()
