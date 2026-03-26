"""Test script to verify datetime implementation across different data formats."""

import json
import tempfile
from pathlib import Path

from paradigma.load import load_single_data_file
from paradigma.pipelines.gait_pipeline import run_gait_pipeline

# Define example data paths
example_data_dir = Path("example_data")
verily_dir = example_data_dir / "verily"
empatica_file = example_data_dir / "empatica" / "test_data.avro"
axivity_file = example_data_dir / "axivity" / "test_data.CWA"


def test_verily_tsdf():
    """Test TSDF format (Verily data with start_iso8601)."""
    print("\n" + "=" * 60)
    print("TEST 1: VERILY/TSDF FORMAT")
    print("=" * 60)

    meta_file = verily_dir / "segment0001_meta.json"

    try:
        file_key, df, start_dt = load_single_data_file(meta_file)
        print(f"✓ Loaded TSDF data: {len(df)} rows")
        print(f"✓ Extracted start_dt: {start_dt}")
        print(f"✓ Columns: {list(df.columns)}")

        # Run gait pipeline with required parameters
        with tempfile.TemporaryDirectory() as tmpdir:
            results, metadata_dict = run_gait_pipeline(
                df, watch_side="left", output_dir=tmpdir, start_dt=start_dt
            )

        # Check metadata
        if metadata_dict and "Arm_Swing_Parameters" in metadata_dict:
            metadata = metadata_dict["Arm_Swing_Parameters"]
            print("\n✓ Pipeline completed successfully")
            print("Metadata structure:")
            print(json.dumps(metadata, indent=2, default=str))

            # Verify datetime fields are present
            if "combined" in metadata:
                if "start_dt" in metadata["combined"]:
                    print("\n✓ SUCCESS: DateTime fields present in metadata!")
                else:
                    print(
                        "\n✗ WARNING: DateTime fields missing from 'combined' section"
                    )
        else:
            print("✗ Pipeline returned no results")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")


def test_empatica():
    """Test Empatica format (with time_dt column)."""
    print("\n" + "=" * 60)
    print("TEST 2: EMPATICA FORMAT")
    print("=" * 60)

    if not empatica_file.exists():
        print(f"✗ File not found: {empatica_file}")
        return

    try:
        file_key, df, start_dt = load_single_data_file(empatica_file)
        print(f"✓ Loaded Empatica data: {len(df)} rows")
        print(f"✓ Extracted start_dt: {start_dt}")
        print(f"✓ Columns: {list(df.columns)}")

        # Run gait pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            results, metadata_dict = run_gait_pipeline(
                df, watch_side="left", output_dir=tmpdir, start_dt=start_dt
            )

        if metadata_dict and "Arm_Swing_Parameters" in metadata_dict:
            metadata = metadata_dict["Arm_Swing_Parameters"]
            print("\n✓ Pipeline completed successfully")
            print("Metadata structure:")
            print(json.dumps(metadata, indent=2, default=str))

            # Verify datetime fields are present
            if "combined" in metadata:
                if "start_dt" in metadata["combined"]:
                    print("\n✓ SUCCESS: DateTime fields present in metadata!")
                else:
                    print(
                        "\n✗ WARNING: DateTime fields missing from 'combined' section"
                    )
        else:
            print("✗ Pipeline returned no results")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")


def test_axivity():
    """Test Axivity format (CWA file with time_dt)."""
    print("\n" + "=" * 60)
    print("TEST 3: AXIVITY FORMAT")
    print("=" * 60)

    if not axivity_file.exists():
        print(f"✗ File not found: {axivity_file}")
        return

    try:
        file_key, df, start_dt = load_single_data_file(axivity_file)
        print(f"✓ Loaded Axivity data: {len(df)} rows")
        print(f"✓ Extracted start_dt: {start_dt}")
        print(f"✓ Columns: {list(df.columns)}")

        # Run gait pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            results, metadata_dict = run_gait_pipeline(
                df, watch_side="left", output_dir=tmpdir, start_dt=start_dt
            )

        if metadata_dict and "Arm_Swing_Parameters" in metadata_dict:
            metadata = metadata_dict["Arm_Swing_Parameters"]
            print("\n✓ Pipeline completed successfully")
            print("Metadata structure:")
            print(json.dumps(metadata, indent=2, default=str))

            # Verify datetime fields are present
            if "combined" in metadata:
                if "start_dt" in metadata["combined"]:
                    print("\n✓ SUCCESS: DateTime fields present in metadata!")
                else:
                    print(
                        "\n✗ WARNING: DateTime fields missing from 'combined' section"
                    )
        else:
            print("✗ Pipeline returned no results")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")


def test_dataframe_only():
    """Test DataFrame-only case (no datetime)."""
    print("\n" + "=" * 60)
    print("TEST 4: DATAFRAME ONLY (NO DATETIME)")
    print("=" * 60)

    # Create a simple test DataFrame
    import numpy as np
    import pandas as pd

    # Generate sample accelerometer data
    sampling_rate = 100  # Hz
    duration = 10  # seconds
    time = np.arange(0, duration, 1 / sampling_rate)

    df = pd.DataFrame(
        {
            "time": time,
            "accelerometer_x": np.sin(2 * np.pi * 0.5 * time)
            + np.random.normal(0, 0.01, len(time)),
            "accelerometer_y": np.cos(2 * np.pi * 0.5 * time)
            + np.random.normal(0, 0.01, len(time)),
            "accelerometer_z": np.ones(len(time))
            + np.random.normal(0, 0.01, len(time)),
            "gyroscope_x": np.random.normal(0, 0.1, len(time)),
            "gyroscope_y": np.random.normal(0, 0.1, len(time)),
            "gyroscope_z": np.random.normal(0, 0.1, len(time)),
        }
    )

    try:
        # Run pipeline without datetime (start_dt=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            results, metadata_dict = run_gait_pipeline(
                df, watch_side="left", output_dir=tmpdir, start_dt=None
            )

        if metadata_dict and "Arm_Swing_Parameters" in metadata_dict:
            metadata = metadata_dict["Arm_Swing_Parameters"]
            print("✓ Pipeline completed successfully (no datetime)")
            print("Metadata structure:")
            print(json.dumps(metadata, indent=2, default=str))

            # Verify datetime fields are NOT present
            if "combined" in metadata:
                if "start_dt" not in metadata["combined"]:
                    print("\n✓ SUCCESS: DateTime fields correctly absent!")
                    # Check that relative time fields are present
                    if "start_s" in metadata.get("per_segment", {}).get("1", {}):
                        print(
                            "✓ Relative time fields (start_s/end_s) present as expected"
                        )
                else:
                    print(
                        "\n✗ ERROR: DateTime fields should not be present without start_dt!"
                    )
        else:
            print("✗ Pipeline returned no results")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# DATETIME IMPLEMENTATION TEST SUITE")
    print("#" * 60)

    # Test all formats
    test_verily_tsdf()
    test_empatica()
    test_axivity()
    test_dataframe_only()

    print("\n" + "#" * 60)
    print("# TEST SUITE COMPLETE")
    print("#" * 60)
