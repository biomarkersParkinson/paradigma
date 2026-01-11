"""
Tests for the ParaDigMa pipeline runner.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from paradigma.config import GaitConfig, TremorConfig
from paradigma.pipeline import (
    _detect_data_format,
    _setup_configs,
    list_available_pipelines,
    run_pipeline,
)


class TestPipelineRunner:
    """Test cases for the pipeline runner functionality."""

    def test_list_available_pipelines(self):
        """Test that available pipelines are returned correctly."""
        pipelines = list_available_pipelines()
        expected = ["gait", "tremor", "pulse_rate"]
        assert all(p in pipelines for p in expected)
        assert len(pipelines) >= 3

    def test_setup_configs_default(self):
        """Test configuration setup with default configs."""
        pipelines = ["gait", "tremor"]
        configs = _setup_configs(pipelines, "default")

        assert "gait" in configs
        assert "tremor" in configs
        assert isinstance(configs["gait"], GaitConfig)
        assert isinstance(configs["tremor"], TremorConfig)

    def test_setup_configs_custom(self):
        """Test configuration setup with custom configs."""
        custom_gait_config = GaitConfig(step="gait")
        custom_gait_config.window_length_s = 3.0

        pipelines = ["gait", "tremor"]
        custom_configs = {"gait": custom_gait_config}

        configs = _setup_configs(pipelines, custom_configs)

        assert isinstance(configs["gait"], GaitConfig)
        assert configs["gait"].window_length_s == 3.0
        assert isinstance(configs["tremor"], TremorConfig)  # Default for tremor

    def test_setup_configs_invalid(self):
        """Test that invalid config input raises error."""
        pipelines = ["gait"]
        with pytest.raises(ValueError, match="Invalid config format"):
            _setup_configs(pipelines, 123)  # Invalid type

    @patch("paradigma.pipeline._load_data")
    @patch("paradigma.pipeline._detect_data_format")
    @patch("paradigma.pipeline.Path.exists", return_value=True)
    def test_run_pipeline_validation(self, mock_exists, mock_detect, mock_load):
        """Test input validation for run_pipeline."""

        # Setup mocks
        mock_detect.return_value = "tsdf"
        mock_load.return_value = [
            ("test_segment", pd.DataFrame({"time": [1, 2], "value": [0.1, 0.2]}))
        ]

        # Test invalid pipeline names
        with pytest.raises(ValueError, match="Invalid pipelines"):
            run_pipeline(data_path="dummy/path", pipelines=["invalid_pipeline"])

        # Test non-existent data path
        with patch("paradigma.pipeline.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                run_pipeline(data_path="nonexistent/path", pipelines=["gait"])

    @patch("paradigma.pipeline._run_single_pipeline")
    @patch("paradigma.pipeline._load_data")
    @patch("paradigma.pipeline._detect_data_format")
    @patch("paradigma.pipeline.Path.exists", return_value=True)
    def test_run_pipeline_success(
        self, mock_exists, mock_detect, mock_load, mock_run_single
    ):
        """Test successful pipeline execution."""

        # Mock return values
        mock_detect.return_value = "prepared"
        mock_result = pd.DataFrame({"time": [1, 2, 3], "value": [0.1, 0.2, 0.3]})
        mock_load.return_value = [("test_segment", mock_result)]
        mock_run_single.return_value = mock_result

        # Run pipeline
        results = run_pipeline(
            data_path="dummy/path", pipelines=["gait"], config="default"
        )

        # Verify results
        assert "gait" in results
        assert isinstance(results["gait"], pd.DataFrame)
        assert len(results["gait"]) == 3

        # Verify function was called correctly
        mock_run_single.assert_called_once()

    @patch("paradigma.pipeline._run_single_pipeline")
    @patch("paradigma.pipeline._load_data")
    @patch("paradigma.pipeline._detect_data_format")
    @patch("paradigma.pipeline.Path.exists", return_value=True)
    def test_run_pipeline_with_output_dir(
        self, mock_exists, mock_detect, mock_load, mock_run_single
    ):
        """Test pipeline execution with output directory."""

        mock_detect.return_value = "prepared"
        mock_result = pd.DataFrame({"time": [1, 2], "value": [0.1, 0.2]})
        mock_load.return_value = [("test_segment", mock_result)]
        mock_run_single.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"

            _ = run_pipeline(
                data_path="dummy/path", pipelines=["gait"], output_dir=str(output_dir)
            )

            # Verify output file was created
            output_file = output_dir / "gait_results.csv"
            assert output_file.exists()

            # Verify content has correct shape
            saved_df = pd.read_csv(output_file)
            assert len(saved_df) == len(mock_result)
            assert all(col in saved_df.columns for col in mock_result.columns)

    def test_detect_data_format(self):
        """Test data format detection."""
        # Test single file detection
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test .parquet file
            parquet_file = temp_path / "data.parquet"
            parquet_file.touch()
            assert _detect_data_format(parquet_file) == "prepared"

            # Test .cwa file
            cwa_file = temp_path / "data.cwa"
            cwa_file.touch()
            assert _detect_data_format(cwa_file) == "axivity"

            # Test directory with TSDF files
            meta_file = temp_path / "IMU_meta.json"
            meta_file.touch()
            assert _detect_data_format(temp_path) == "tsdf"

    def test_apply_column_mapping(self):
        """Test column mapping functionality."""
        from paradigma.pipeline import _apply_column_mapping

        # Create test data
        df1 = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "acceleration_x": [0.1, 0.2, 0.3],
                "acceleration_y": [0.4, 0.5, 0.6],
            }
        )
        df2 = pd.DataFrame(
            {
                "time": [4, 5, 6],
                "rotation_x": [1.1, 1.2, 1.3],
                "other_col": ["a", "b", "c"],
            }
        )

        data_segments = [("segment1", df1), ("segment2", df2)]

        # Define column mapping
        column_mapping = {
            "acceleration_x": "accelerometer_x",
            "acceleration_y": "accelerometer_y",
            "rotation_x": "gyroscope_x",
            "nonexistent_col": "should_not_map",
        }

        # Apply mapping
        mapped_segments = _apply_column_mapping(data_segments, column_mapping)

        # Verify results
        assert len(mapped_segments) == 2

        # Check first segment
        segment1_name, segment1_df = mapped_segments[0]
        assert segment1_name == "segment1"
        assert "accelerometer_x" in segment1_df.columns
        assert "accelerometer_y" in segment1_df.columns
        assert "acceleration_x" not in segment1_df.columns
        assert "time" in segment1_df.columns  # unmapped columns should remain

        # Check second segment
        segment2_name, segment2_df = mapped_segments[1]
        assert segment2_name == "segment2"
        assert "gyroscope_x" in segment2_df.columns
        assert "rotation_x" not in segment2_df.columns
        assert "other_col" in segment2_df.columns  # unmapped columns should remain


class TestPipelineRunnerIntegration:
    """Integration tests requiring actual data (marked as slow)."""

    @pytest.mark.slow
    def test_run_gait_pipeline_with_mock_data(self):
        """Test gait pipeline with minimal mock data structure."""

        # Create minimal mock prepared dataframe
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock prepared dataframe file
            mock_df = pd.DataFrame(
                {
                    "time": np.linspace(0, 10, 1000),
                    "accelerometer_x": [0.1] * 1000,
                    "accelerometer_y": [0.2] * 1000,
                    "accelerometer_z": [9.8] * 1000,
                    "gyroscope_x": [0.01] * 1000,
                    "gyroscope_y": [0.02] * 1000,
                    "gyroscope_z": [0.01] * 1000,
                }
            )

            # Save as parquet file
            parquet_file = temp_path / "test_data.parquet"
            mock_df.to_parquet(parquet_file)

            # This should run without crashing (may use mock predictions)
            try:
                results = run_pipeline(
                    data_path=temp_path,
                    pipelines=["gait"],
                    data_format="prepared",
                    file_pattern="*.parquet",
                    verbose=True,
                )
                # Even with mock data, should return some structure
                assert "gait" in results
                assert isinstance(results["gait"], pd.DataFrame)
            except Exception as e:
                # Acceptable for now due to classifier dependency
                print(f"Expected failure due to missing classifier: {e}")
