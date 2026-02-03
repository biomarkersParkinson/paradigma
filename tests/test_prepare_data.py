"""
Minimal tests for data preparation functionality including auto-segmentation.
"""

import numpy as np
import pandas as pd
import pytest

from paradigma.constants import DataColumns, TimeUnit
from paradigma.prepare_data import prepare_raw_data


class TestPrepareRawData:
    """Minimal tests for data preparation pipeline."""

    def test_prepare_contiguous_data(self):
        """Test complete data preparation pipeline with contiguous data."""
        # Create synthetic contiguous IMU data at 100 Hz
        time = np.arange(0, 5, 0.01)  # 5 seconds
        df_raw = pd.DataFrame(
            {
                DataColumns.TIME: time,
                DataColumns.ACCELEROMETER_X: np.sin(2 * np.pi * time)
                + 1.0,  # Simulated acceleration with gravity
                DataColumns.ACCELEROMETER_Y: np.cos(2 * np.pi * time),
                DataColumns.ACCELEROMETER_Z: np.ones_like(time)
                + 0.1 * np.random.randn(len(time)),
                DataColumns.GYROSCOPE_X: 0.1 * np.sin(2 * np.pi * 2 * time),
                DataColumns.GYROSCOPE_Y: 0.1 * np.cos(2 * np.pi * 2 * time),
                DataColumns.GYROSCOPE_Z: 0.05 * np.random.randn(len(time)),
            }
        )

        # Prepare data
        df_prepared = prepare_raw_data(
            df=df_raw,
            accelerometer_units="g",
            gyroscope_units="deg/s",
            time_input_unit=TimeUnit.RELATIVE_S,
            resampling_frequency=100.0,
            validate=True,
        )

        # Verify output
        # Allow small difference due to resampling edge effects
        assert (
            abs(len(df_prepared) - len(df_raw)) <= 1
        )  # Should be approximately same length
        assert DataColumns.TIME in df_prepared.columns
        assert DataColumns.ACCELEROMETER_X in df_prepared.columns
        assert DataColumns.GYROSCOPE_X in df_prepared.columns
        assert df_prepared[DataColumns.TIME].iloc[0] == pytest.approx(0.0)
        assert (
            "data_segment_nr" not in df_prepared.columns
        )  # No segmentation for contiguous data

    def test_prepare_non_contiguous_data_with_auto_segment(self):
        """Test data preparation with non-contiguous data and auto-segmentation."""
        # Create 3 segments with gaps (simulating interrupted recording)
        time_seg1 = np.arange(0, 2, 0.01)  # 2 seconds
        time_seg2 = np.arange(5, 7, 0.01)  # Gap of 3 seconds, then 2 seconds
        time_seg3 = np.arange(10, 12, 0.01)  # Gap of 3 seconds, then 2 seconds

        time_all = np.concatenate([time_seg1, time_seg2, time_seg3])

        df_raw = pd.DataFrame(
            {
                DataColumns.TIME: time_all,
                DataColumns.ACCELEROMETER_X: np.sin(2 * np.pi * time_all) + 1.0,
                DataColumns.ACCELEROMETER_Y: np.cos(2 * np.pi * time_all),
                DataColumns.ACCELEROMETER_Z: np.ones_like(time_all),
                DataColumns.GYROSCOPE_X: 0.1 * np.sin(2 * np.pi * 2 * time_all),
                DataColumns.GYROSCOPE_Y: 0.1 * np.cos(2 * np.pi * 2 * time_all),
                DataColumns.GYROSCOPE_Z: np.zeros_like(time_all),
            }
        )

        # Prepare with auto-segmentation
        df_prepared = prepare_raw_data(
            df=df_raw,
            accelerometer_units="g",
            gyroscope_units="deg/s",
            time_input_unit=TimeUnit.RELATIVE_S,
            resampling_frequency=100.0,
            auto_segment=True,
            max_segment_gap_s=2.0,  # Gaps > 2s trigger segmentation
            min_segment_length_s=1.0,
            validate=False,  # Skip validation since we have segments
        )

        # Verify segmentation occurred
        assert "data_segment_nr" in df_prepared.columns
        assert df_prepared["data_segment_nr"].nunique() == 3  # Should have 3 segments
        assert set(df_prepared["data_segment_nr"].unique()) == {1, 2, 3}

        # Verify each segment has roughly 200 samples (2 seconds at 100 Hz)
        for seg_nr in [1, 2, 3]:
            seg_data = df_prepared[df_prepared["data_segment_nr"] == seg_nr]
            assert 190 <= len(seg_data) <= 210  # Allow some tolerance

        # Verify all expected columns are present
        assert DataColumns.TIME in df_prepared.columns
        assert DataColumns.ACCELEROMETER_X in df_prepared.columns
        assert DataColumns.GYROSCOPE_X in df_prepared.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
