import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paradigma.config import IMUConfig, TremorConfig
from paradigma.constants import DataColumns
from paradigma.orchestrator import run_paradigma
from paradigma.pipelines.tremor_pipeline import run_tremor_pipeline

try:
    from test_notebooks import compare_data

    from paradigma.testing import detect_tremor_io, extract_tremor_features_io

    HAS_TESTING_UTILS = True
except ImportError:
    HAS_TESTING_UTILS = False


def create_test_tremor_data(duration_minutes=2):
    """Create simple test data for tremor analysis."""
    duration_seconds = duration_minutes * 60
    n_samples = int(duration_seconds * 100.0)  # 100 Hz
    time = np.linspace(0, duration_seconds, n_samples)

    # Simple gyroscope data
    gyro_x = 0.1 * np.sin(2 * np.pi * 5 * time) + 0.05 * np.random.random(n_samples)
    gyro_y = 0.1 * np.cos(2 * np.pi * 5 * time) + 0.05 * np.random.random(n_samples)
    gyro_z = 0.05 * np.random.random(n_samples)

    return pd.DataFrame(
        {
            DataColumns.TIME: time,
            DataColumns.GYROSCOPE_X: gyro_x,
            DataColumns.GYROSCOPE_Y: gyro_y,
            DataColumns.GYROSCOPE_Z: gyro_z,
        }
    )


def test_tremor_pipeline_basic():
    """Test basic tremor pipeline functionality."""
    df_test = create_test_tremor_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = run_tremor_pipeline(
            df_prepared=df_test,
            output_dir=temp_dir,
            tremor_config=TremorConfig(),
            imu_config=IMUConfig(),
        )

        assert isinstance(result, pd.DataFrame)
        assert DataColumns.TIME in result.columns


def test_tremor_pipeline_integration():
    """Test tremor pipeline with orchestrator."""
    dfs = {"test": create_test_tremor_data()}

    with tempfile.TemporaryDirectory() as temp_dir:
        results = run_paradigma(
            pipelines=["tremor"],
            dfs=dfs,
            output_dir=temp_dir,
            skip_preparation=True,
            tremor_config=TremorConfig(),
            imu_config=IMUConfig(),
        )

        assert "quantifications" in results
        assert "metadata" in results
        assert "aggregations" in results


tremor_binaries_pairs: list[tuple[str, str]] = [
    ("tremor_meta.json", "tremor_time.bin"),
    ("tremor_meta.json", "tremor_values.bin"),
]

path_to_assets = Path("src/paradigma/assets")


@pytest.mark.skipif(not HAS_TESTING_UTILS, reason="Testing utilities not available")
def test_2_extract_features_tremor_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the tremor feature extraction. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "2.preprocessed_data"
    output_dir_name: str = "3.extracted_features"
    data_type: str = "tremor"

    path_to_imu_input = shared_datadir / input_dir_name / "imu"
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    config = TremorConfig("features")
    extract_tremor_features_io(path_to_imu_input, path_to_tested_output, config)
    compare_data(path_to_reference_output, path_to_tested_output, tremor_binaries_pairs)


@pytest.mark.skipif(not HAS_TESTING_UTILS, reason="Testing utilities not available")
def test_3_tremor_detection_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the tremor detection. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "3.extracted_features"
    output_dir_name: str = "4.predictions"
    data_type: str = "tremor"
    classifier_package_filename: str = "tremor_detection_clf_package.pkl"

    # Temporary path to store the output of the notebook
    path_to_feature_input = shared_datadir / input_dir_name / data_type
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    full_path_to_classifier_package = path_to_assets / classifier_package_filename

    config = TremorConfig("classification")
    detect_tremor_io(
        path_to_feature_input,
        path_to_tested_output,
        full_path_to_classifier_package,
        config,
    )
    compare_data(path_to_reference_output, path_to_tested_output, tremor_binaries_pairs)
