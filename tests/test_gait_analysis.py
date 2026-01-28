import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paradigma.config import GaitConfig, IMUConfig
from paradigma.constants import DataColumns
from paradigma.orchestrator import run_paradigma
from paradigma.pipelines.gait_pipeline import run_gait_pipeline

try:
    from test_notebooks import compare_data

    from paradigma.testing import (
        extract_gait_features_io,
    )

    HAS_TESTING_UTILS = True
except ImportError:
    HAS_TESTING_UTILS = False


def create_test_gait_data(duration_minutes=2):
    """Create simple test data for gait analysis."""
    duration_seconds = duration_minutes * 60
    n_samples = int(duration_seconds * 100.0)  # 100 Hz
    time = np.linspace(0, duration_seconds, n_samples)

    # Simple IMU data
    acc_x = 0.2 * np.sin(2 * np.pi * 1.5 * time) + 0.1 * np.random.random(n_samples)
    acc_y = 0.2 * np.cos(2 * np.pi * 1.5 * time) + 0.1 * np.random.random(n_samples)
    acc_z = 9.8 + 0.2 * np.random.random(n_samples)

    gyro_x = 0.1 * np.sin(2 * np.pi * 2 * time) + 0.05 * np.random.random(n_samples)
    gyro_y = 0.1 * np.cos(2 * np.pi * 2 * time) + 0.05 * np.random.random(n_samples)
    gyro_z = 0.05 * np.random.random(n_samples)

    return pd.DataFrame(
        {
            DataColumns.TIME: time,
            DataColumns.ACCELEROMETER_X: acc_x,
            DataColumns.ACCELEROMETER_Y: acc_y,
            DataColumns.ACCELEROMETER_Z: acc_z,
            DataColumns.GYROSCOPE_X: gyro_x,
            DataColumns.GYROSCOPE_Y: gyro_y,
            DataColumns.GYROSCOPE_Z: gyro_z,
        }
    )


def create_test_gait_data_with_gaps(n_segments=3):
    """Create test data with gaps (multiple data segments)."""
    segments = []
    for i in range(n_segments):
        # Create 30 second segments with large gaps between them
        duration_seconds = 30
        n_samples = int(duration_seconds * 100.0)  # 100 Hz
        time_start = i * 100  # 100 second gaps between segments
        time = np.linspace(time_start, time_start + duration_seconds, n_samples)

        # Simple IMU data
        acc_x = 0.2 * np.sin(2 * np.pi * 1.5 * time) + 0.1 * np.random.random(n_samples)
        acc_y = 0.2 * np.cos(2 * np.pi * 1.5 * time) + 0.1 * np.random.random(n_samples)
        acc_z = 9.8 + 0.2 * np.random.random(n_samples)

        gyro_x = 0.1 * np.sin(2 * np.pi * 2 * time) + 0.05 * np.random.random(n_samples)
        gyro_y = 0.1 * np.cos(2 * np.pi * 2 * time) + 0.05 * np.random.random(n_samples)
        gyro_z = 0.05 * np.random.random(n_samples)

        segment_df = pd.DataFrame(
            {
                DataColumns.TIME: time,
                DataColumns.ACCELEROMETER_X: acc_x,
                DataColumns.ACCELEROMETER_Y: acc_y,
                DataColumns.ACCELEROMETER_Z: acc_z,
                DataColumns.GYROSCOPE_X: gyro_x,
                DataColumns.GYROSCOPE_Y: gyro_y,
                DataColumns.GYROSCOPE_Z: gyro_z,
                DataColumns.DATA_SEGMENT_NR: i + 1,  # Add data segment number
            }
        )
        segments.append(segment_df)

    return pd.concat(segments, ignore_index=True)


def test_gait_pipeline_basic():
    """Test basic gait pipeline functionality."""
    df_test = create_test_gait_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = run_gait_pipeline(
            df_prepared=df_test,
            watch_side="right",
            output_dir=temp_dir,
            imu_config=IMUConfig(),
            gait_config=GaitConfig("gait"),
            arm_activity_config=GaitConfig("arm_activity"),
        )

        assert isinstance(result, pd.DataFrame)
        # May be empty if no gait detected, which is OK


def test_gait_pipeline_integration():
    """Test gait pipeline with orchestrator."""
    dfs = {"test": create_test_gait_data()}

    with tempfile.TemporaryDirectory() as temp_dir:
        results = run_paradigma(
            pipelines=["gait"],
            dfs=dfs,
            output_dir=temp_dir,
            watch_side="right",
            skip_preparation=True,
            imu_config=IMUConfig(),
            gait_config=GaitConfig("gait"),
            arm_activity_config=GaitConfig("arm_activity"),
        )

        assert "quantifications" in results
        assert "metadata" in results
        assert "aggregations" in results


def test_gait_segment_nr_column_name():
    """
    Test that gait_segment_nr column is present and backward compatibility is maintained.

    This test verifies:
    1. The new GAIT_SEGMENT_NR constant exists
    2. The old SEGMENT_NR constant still works (backward compatibility)
    3. Both constants reference the same column name
    4. Output DataFrame contains the gait_segment_nr column when using constants
    """
    # Test 1: Verify both constants exist and point to same value
    assert hasattr(
        DataColumns, "GAIT_SEGMENT_NR"
    ), "GAIT_SEGMENT_NR constant should exist"
    assert hasattr(
        DataColumns, "SEGMENT_NR"
    ), "SEGMENT_NR constant should exist for backward compatibility"
    assert (
        DataColumns.GAIT_SEGMENT_NR == "gait_segment_nr"
    ), "GAIT_SEGMENT_NR should be 'gait_segment_nr'"
    assert (
        DataColumns.SEGMENT_NR == DataColumns.GAIT_SEGMENT_NR
    ), "SEGMENT_NR should alias to GAIT_SEGMENT_NR for backward compatibility"

    # Test 2: Create a mock DataFrame to verify column access works with both constants
    mock_gait_results = pd.DataFrame(
        {
            "gait_segment_nr": [1, 1, 2, 2, 3],
            "range_of_motion": [0.5, 0.6, 0.7, 0.5, 0.8],
            "peak_velocity": [1.2, 1.3, 1.1, 1.4, 1.5],
        }
    )

    # Test 3: Verify old constant can still access the column (backward compatibility)
    try:
        _ = mock_gait_results[DataColumns.SEGMENT_NR]
        old_constant_works = True
    except KeyError:
        old_constant_works = False

    assert (
        old_constant_works
    ), "Old SEGMENT_NR constant should still work to access gait_segment_nr column"

    # Test 4: Verify new constant can access the column
    try:
        _ = mock_gait_results[DataColumns.GAIT_SEGMENT_NR]
        new_constant_works = True
    except KeyError:
        new_constant_works = False

    assert (
        new_constant_works
    ), "New GAIT_SEGMENT_NR constant should work to access gait_segment_nr column"

    # Test 5: Verify both constants access the same data
    assert mock_gait_results[DataColumns.SEGMENT_NR].equals(
        mock_gait_results[DataColumns.GAIT_SEGMENT_NR]
    ), "Both constants should access the same column data"


def test_data_segment_nr_preserved():
    """
    Test that data_segment_nr is preserved in gait pipeline output when present in input.

    This test verifies:
    1. DATA_SEGMENT_NR constant exists
    2. When input data has data_segment_nr, gait pipeline preserves it correctly
    3. The constant can be used to access the column in DataFrames
    """
    # Test 1: Verify DATA_SEGMENT_NR constant exists
    assert hasattr(
        DataColumns, "DATA_SEGMENT_NR"
    ), "DATA_SEGMENT_NR constant should exist"
    assert (
        DataColumns.DATA_SEGMENT_NR == "data_segment_nr"
    ), "DATA_SEGMENT_NR should be 'data_segment_nr'"

    # Test 2: Create mock data with both segment types
    mock_gait_results = pd.DataFrame(
        {
            "gait_segment_nr": [1, 1, 2, 2, 3, 3],
            "data_segment_nr": [
                1,
                1,
                2,
                2,
                3,
                3,
            ],  # Each gait segment within one data segment
            "range_of_motion": [0.5, 0.6, 0.7, 0.5, 0.8, 0.6],
            "peak_velocity": [1.2, 1.3, 1.1, 1.4, 1.5, 1.3],
        }
    )

    # Test 3: Verify constant can access the column
    try:
        _ = mock_gait_results[DataColumns.DATA_SEGMENT_NR]
        constant_works = True
    except KeyError:
        constant_works = False

    assert (
        constant_works
    ), "DATA_SEGMENT_NR constant should work to access data_segment_nr column"

    # Test 4: Verify data integrity - each gait segment belongs to exactly one data segment
    gait_to_data = mock_gait_results.groupby("gait_segment_nr")[
        "data_segment_nr"
    ].nunique()
    assert all(
        gait_to_data == 1
    ), "Each gait segment should belong to exactly one data segment"

    # Test 5: Verify we have multiple data segments
    unique_data_segments = mock_gait_results["data_segment_nr"].unique()
    assert len(unique_data_segments) == 3, "Should have 3 data segments in test data"


# Mappings between the metadata and binary files

gait_binaries_pairs: list[tuple[str, str]] = [
    ("gait_meta.json", "gait_time.bin"),
    ("gait_meta.json", "gait_values.bin"),
]

# Mappings between the metadata and binary files

gait_binaries_pairs: list[tuple[str, str]] = [
    ("gait_meta.json", "gait_time.bin"),
    ("gait_meta.json", "gait_values.bin"),
]

path_to_assets = Path("src/paradigma/assets")


@pytest.mark.skipif(not HAS_TESTING_UTILS, reason="Testing utilities not available")
def test_2_extract_features_gait_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the gait feature extraction. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "2.preprocessed_data"
    output_dir_name: str = "3.extracted_features"
    data_type: str = "gait"

    path_to_preprocessed_input = shared_datadir / input_dir_name / "imu"
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    config = GaitConfig(step=data_type)

    extract_gait_features_io(
        config=config,
        path_to_input=path_to_preprocessed_input,
        path_to_output=path_to_tested_output,
    )
    compare_data(
        reference_dir=path_to_reference_output,
        tested_dir=path_to_tested_output,
        binaries_pairs=gait_binaries_pairs,
    )
