import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paradigma.config import IMUConfig, PPGConfig, PulseRateConfig
from paradigma.constants import DataColumns
from paradigma.orchestrator import run_paradigma
from paradigma.pipelines.pulse_rate_pipeline import run_pulse_rate_pipeline

try:
    from test_notebooks import compare_data

    from paradigma.testing import preprocess_ppg_data_io

    HAS_TESTING_UTILS = True
except ImportError:
    HAS_TESTING_UTILS = False


def create_test_pulse_rate_data(duration_minutes=2):
    """Create simple test data for pulse rate analysis."""
    duration_seconds = duration_minutes * 60
    n_samples = int(duration_seconds * 50.0)  # 50 Hz
    time = np.linspace(0, duration_seconds, n_samples)

    # Simple PPG signal
    heart_rate_hz = 70.0 / 60.0  # 70 BPM
    ppg_signal = np.sin(2 * np.pi * heart_rate_hz * time) + 0.1 * np.random.random(
        n_samples
    )

    # Simple accelerometer data
    acc_x = 0.1 * np.random.random(n_samples)
    acc_y = 0.1 * np.random.random(n_samples)
    acc_z = 9.8 + 0.1 * np.random.random(n_samples)

    return pd.DataFrame(
        {
            DataColumns.TIME: time,
            DataColumns.PPG: ppg_signal,
            DataColumns.ACCELEROMETER_X: acc_x,
            DataColumns.ACCELEROMETER_Y: acc_y,
            DataColumns.ACCELEROMETER_Z: acc_z,
        }
    )


def test_pulse_rate_pipeline_basic():
    """Test basic pulse rate pipeline functionality."""
    df_test = create_test_pulse_rate_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = run_pulse_rate_pipeline(
            df_ppg_prepared=df_test,
            output_dir=temp_dir,
            pulse_rate_config=PulseRateConfig(),
            ppg_config=PPGConfig(),
        )

        assert isinstance(result, pd.DataFrame)
        assert DataColumns.TIME in result.columns


def test_pulse_rate_pipeline_integration():
    """Test pulse rate pipeline with orchestrator."""
    dfs = {"test": create_test_pulse_rate_data()}

    with tempfile.TemporaryDirectory() as temp_dir:
        results = run_paradigma(
            pipelines=["pulse_rate"],
            dfs=dfs,
            output_dir=temp_dir,
            skip_preparation=True,
            pulse_rate_config=PulseRateConfig(),
            ppg_config=PPGConfig(),
        )

        assert "quantifications" in results
        assert "metadata" in results
        assert "aggregations" in results


@pytest.mark.skipif(not HAS_TESTING_UTILS, reason="Testing utilities not available")
def compare_ppg_preprocessing(
    shared_datadir: Path, binaries_pairs: list[tuple[str, str]]
):
    """
    This function is used to evaluate the output of the PPG pipeline preprocessing
    function. It evaluates it by comparing the output to a reference output.

    Parameters
    ----------
    shared_datadir : Path
        The path to the shared data directory.
    binaries_pairs : list[tuple[str, str]]
            The list of pairs of metadata and binary files to compare.
    """
    input_dir_name: str = "1.prepared_data"
    output_dir_name: str = "2.preprocessed_data"

    path_to_imu_input = shared_datadir / input_dir_name / "imu"
    path_to_ppg_input = shared_datadir / input_dir_name / "ppg"
    path_to_reference_output = shared_datadir / output_dir_name / "ppg"
    path_to_tested_output = path_to_reference_output / "test-output"

    ppg_config = PPGConfig()
    imu_config = IMUConfig()

    preprocess_ppg_data_io(
        path_to_ppg_input,
        path_to_imu_input,
        path_to_tested_output,
        ppg_config,
        imu_config,
    )
    compare_data(path_to_reference_output, path_to_tested_output, binaries_pairs)
