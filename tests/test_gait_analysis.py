import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paradigma.config import GaitConfig, IMUConfig
from paradigma.constants import DataColumns
from paradigma.orchestrator import run_pipeline_batch
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
        results = run_pipeline_batch(
            pipeline_name="gait",
            dfs=dfs,
            output_dir=temp_dir,
            watch_side="right",
            imu_config=IMUConfig(),
            gait_config=GaitConfig("gait"),
            arm_activity_config=GaitConfig("arm_activity"),
        )

        assert "quantifications" in results
        assert "metadata" in results
        assert "aggregations" in results


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
