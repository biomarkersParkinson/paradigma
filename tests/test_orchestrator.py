"""
Tests for orchestrator run_steps and return_intermediate parameters.

These tests verify that:
1. run_steps parameter correctly controls which pipeline steps execute
2. Parameter validation rejects invalid combinations
3. Steps execute in correct order
4. Error handling works correctly
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from paradigma.constants import DataColumns
from paradigma.orchestrator import run_paradigma


def create_test_imu_data(duration_seconds=60, n_files=1):
    """Create simple test IMU data for gait analysis."""
    n_samples = int(duration_seconds * 100.0)  # 100 Hz
    time = np.linspace(0, duration_seconds, n_samples)

    # Simple IMU data
    acc_x = 0.2 * np.sin(2 * np.pi * 1.5 * time) + 0.1 * np.random.random(n_samples)
    acc_y = 0.2 * np.cos(2 * np.pi * 1.5 * time) + 0.1 * np.random.random(n_samples)
    acc_z = 9.8 + 0.2 * np.random.random(n_samples)

    gyro_x = 0.1 * np.sin(2 * np.pi * 2 * time) + 0.05 * np.random.random(n_samples)
    gyro_y = 0.1 * np.cos(2 * np.pi * 2 * time) + 0.05 * np.random.random(n_samples)
    gyro_z = 0.05 * np.random.random(n_samples)

    df = pd.DataFrame(
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

    if n_files == 1:
        return df
    else:
        return {f"file_{i}": df.copy() for i in range(1, n_files + 1)}


class TestRunStepsValidation:
    """Test run_steps parameter validation in orchestrator."""

    def test_run_steps_all_string(self):
        """Test that run_steps='all' executes all steps."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_paradigma(
                dfs=df,
                watch_side="left",
                pipelines="gait",
                output_dir=temp_dir,
                skip_preparation=True,
                run_steps="all",
            )

            # Should have quantifications and aggregations
            assert "quantifications" in result
            assert "aggregations" in result
            assert "errors" in result
            # Should complete without errors (data might not have gait, but processing
            # should work)
            assert isinstance(result["errors"], list)

    def test_run_steps_full_pipeline(self):
        """Test full pipeline: preprocessing, classification, quantification."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_paradigma(
                dfs=df,
                watch_side="left",
                pipelines="gait",
                output_dir=temp_dir,
                skip_preparation=True,
                run_steps=["preprocessing", "classification", "quantification"],
            )

            # Should have quantifications
            assert "quantifications" in result
            assert "gait" in result["quantifications"]

    def test_run_steps_invalid_skips_intermediate(self):
        """Test that skipping intermediate steps raises error."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Cannot skip 'classification' and go straight to 'quantification'
            with pytest.raises(ValueError, match="requires.*but they are not"):
                run_paradigma(
                    dfs=df,
                    watch_side="left",
                    pipelines="gait",
                    output_dir=temp_dir,
                    skip_preparation=True,
                    run_steps=[
                        "preprocessing",
                        "quantification",
                    ],  # Missing 'classification'
                )

    def test_run_steps_invalid_order(self):
        """Test that steps must be in correct order."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Steps must be in order
            with pytest.raises(ValueError, match="Steps must be in order"):
                run_paradigma(
                    dfs=df,
                    watch_side="left",
                    pipelines="gait",
                    output_dir=temp_dir,
                    skip_preparation=True,
                    run_steps=["quantification", "preprocessing"],
                )

    def test_run_steps_invalid_step_name(self):
        """Test that invalid step names raise error."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Invalid steps in run_steps"):
                run_paradigma(
                    dfs=df,
                    watch_side="left",
                    pipelines="gait",
                    output_dir=temp_dir,
                    skip_preparation=True,
                    run_steps=["preprocessing", "invalid_step"],
                )

    def test_run_steps_invalid_string(self):
        """Test that invalid run_steps string is rejected."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Invalid run_steps value"):
                run_paradigma(
                    dfs=df,
                    watch_side="left",
                    pipelines="gait",
                    output_dir=temp_dir,
                    skip_preparation=True,
                    run_steps="invalid_string",
                )


class TestReturnIntermediateValidation:
    """Test return_intermediate parameter validation."""

    def test_return_intermediate_subset_of_run_steps(self):
        """Test that return_intermediate must be subset of run_steps."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Cannot return 'quantification' if not in run_steps
            with pytest.raises(
                ValueError, match="return_intermediate steps.*are not in run_steps"
            ):
                run_paradigma(
                    dfs=df,
                    watch_side="left",
                    pipelines="gait",
                    output_dir=temp_dir,
                    skip_preparation=True,
                    run_steps=["preprocessing"],
                    return_intermediate=["quantification"],  # Not in run_steps
                )

    def test_return_intermediate_valid_subset(self):
        """Test that return_intermediate works when subset of run_steps."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            # This should work - preprocessing is in run_steps
            result = run_paradigma(
                dfs=df,
                watch_side="left",
                pipelines="gait",
                output_dir=temp_dir,
                skip_preparation=True,
                run_steps=["preprocessing", "classification", "quantification"],
                return_intermediate=["quantification"],  # Valid subset
            )

            assert "errors" in result
            assert result["errors"] == []


class TestSaveIntermediateValidation:
    """Test save_intermediate parameter validation."""

    def test_save_intermediate_subset_of_run_steps(self):
        """Test that save_intermediate must be subset of run_steps."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Cannot save 'quantification' if not in run_steps
            with pytest.raises(
                ValueError, match="save_intermediate steps.*are not in run_steps"
            ):
                run_paradigma(
                    dfs=df,
                    watch_side="left",
                    pipelines="gait",
                    output_dir=temp_dir,
                    skip_preparation=True,
                    run_steps=["preprocessing"],
                    save_intermediate=["quantification"],  # Not in run_steps
                )


class TestParallelProcessingWorkflow:
    """Test parallel processing workflows using run_steps."""

    def test_full_pipeline_execution(self):
        """Test full pipeline execution with all steps."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_paradigma(
                dfs=df,
                watch_side="left",
                pipelines="gait",
                output_dir=temp_dir,
                skip_preparation=True,
                run_steps=["preprocessing", "classification", "quantification"],
            )

            # Should complete successfully
            assert "quantifications" in result
            assert "aggregations" in result
            assert isinstance(result["errors"], list)


class TestErrorHandling:
    """Test error handling with run_steps."""

    def test_return_intermediate_validation_error(self):
        """Test that return_intermediate validation works."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(
                ValueError, match="return_intermediate steps.*are not in run_steps"
            ):
                run_paradigma(
                    dfs=df,
                    watch_side="left",
                    pipelines="gait",
                    output_dir=temp_dir,
                    skip_preparation=True,
                    run_steps=["preprocessing"],
                    return_intermediate=["quantification"],  # Not in run_steps
                )


class TestSubsetData:
    """Test subset_data parameter behavior in orchestrator."""

    def test_subset_data_callable_applied_before_pipeline(self, monkeypatch):
        """Test that callable subset_data is applied before pipeline execution."""
        df = create_test_imu_data(duration_seconds=30)
        observed_lengths = []

        def mock_tremor_pipeline(df_prepared, **kwargs):
            observed_lengths.append(len(df_prepared))
            return {"quantification": pd.DataFrame({DataColumns.TIME: [0.0]})}

        monkeypatch.setattr(
            "paradigma.orchestrator.run_tremor_pipeline", mock_tremor_pipeline
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_paradigma(
                dfs=df,
                pipelines="tremor",
                output_dir=temp_dir,
                skip_preparation=True,
                run_steps=["preprocessing", "classification", "quantification"],
                subset_data=lambda d: d.iloc[:123],
            )

        assert observed_lengths == [123]

    def test_subset_data_per_pipeline_dict(self, monkeypatch):
        """Test that per-pipeline subset functions are applied independently."""
        df = create_test_imu_data(duration_seconds=30)
        observed_lengths = {}

        def mock_gait_pipeline(df_prepared, **kwargs):
            observed_lengths["gait"] = len(df_prepared)
            return {
                "quantification": {
                    "filtered": pd.DataFrame(),
                    "unfiltered": pd.DataFrame(),
                },
                "metadata": {
                    "filtered": {"per_segment": {}},
                    "unfiltered": {"per_segment": {}},
                },
            }

        def mock_tremor_pipeline(df_prepared, **kwargs):
            observed_lengths["tremor"] = len(df_prepared)
            return {"quantification": pd.DataFrame()}

        monkeypatch.setattr(
            "paradigma.orchestrator.run_gait_pipeline", mock_gait_pipeline
        )
        monkeypatch.setattr(
            "paradigma.orchestrator.run_tremor_pipeline", mock_tremor_pipeline
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_paradigma(
                dfs=df,
                watch_side="left",
                pipelines=["gait", "tremor"],
                output_dir=temp_dir,
                skip_preparation=True,
                run_steps=["preprocessing", "classification", "quantification"],
                subset_data={
                    "gait": lambda d: d.iloc[:100],
                    "tremor": lambda d: d.iloc[:200],
                },
            )

        assert observed_lengths["gait"] == 100
        assert observed_lengths["tremor"] == 200

    def test_subset_data_invalid_pipeline_key_raises(self):
        """Test that invalid subset_data pipeline key raises error."""
        df = create_test_imu_data(duration_seconds=30)

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(
                ValueError, match="subset_data has unknown or inactive pipeline keys"
            ):
                run_paradigma(
                    dfs=df,
                    pipelines="tremor",
                    output_dir=temp_dir,
                    skip_preparation=True,
                    run_steps=["preprocessing"],
                    subset_data={"gait": lambda d: d},
                )
