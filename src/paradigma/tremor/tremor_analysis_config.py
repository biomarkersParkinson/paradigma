from typing import Dict, List

from paradigma.constants import DataColumns, DataUnits

class IMUConfig:
    """
    Base class for Tremor feature extraction and Tremor detection configurations, based on the IMU data (accelerometer, gyroscope).
    """
    def __init__(self):

        self.time_colname = DataColumns.TIME

        self.l_accelerometer_cols: List[str] = [
            DataColumns.ACCELEROMETER_X,
            DataColumns.ACCELEROMETER_Y,
            DataColumns.ACCELEROMETER_Z,
        ]
        self.l_gyroscope_cols: List[str] = [
            DataColumns.GYROSCOPE_X,
            DataColumns.GYROSCOPE_Y,
            DataColumns.GYROSCOPE_Z,
        ]

        self.l_gravity_cols: List[str] = [
            DataColumns.GRAV_ACCELEROMETER_X,
            DataColumns.GRAV_ACCELEROMETER_Y,
            DataColumns.GRAV_ACCELEROMETER_Z,
        ]

    def set_sensor(self, sensor: str) -> None:
        """Sets the sensor and derived filenames"""
        self.sensor: str = sensor
        self.set_filenames(sensor)

    def set_filenames(self, prefix: str) -> None:
        """Sets the filenames based on the prefix,
        
        Parameters
        ----------
        prefix : str
            The prefix for the filenames.
        """
        self.meta_filename = f"{prefix}_meta.json"
        self.time_filename = f"{prefix}_time.bin"
        self.values_filename = f"{prefix}_samples.bin"

    def set_filenames_values(self, prefix: str) -> None:
        """Sets the filenames based on the prefix,
        
        Parameters
        ----------
        prefix : str
            The prefix for the filenames.
        """
        self.meta_filename = f"{prefix}_meta.json"
        self.time_filename = f"{prefix}_time.bin"
        self.values_filename = f"{prefix}_values.bin"


class TremorFeatureExtractionConfig (IMUConfig):

    def __init__(self) -> None:
        super().__init__()

        self.set_sensor("gyroscope")
        self.set_sampling_frequency(100)

        self.window_length_s: int = 4
        self.window_step_size_s: int = 4
        self.single_value_cols: List[str] = None
        self.list_value_cols: List[str] = (self.l_gyroscope_cols)

        self.window_type: str = "hann"
        self.verbose: int = 0

        # cepstral coefficients
        self.cc_low_frequency: int = 0
        self.cc_high_frequency: int = 25
        self.n_dct_filters_cc: int = 20
        self.n_coefficients_cc: int = 12



    def set_sampling_frequency(self, sampling_frequency: int) -> None:
        """Sets the sampling frequency and derived variables"""
        self.sampling_frequency: int = sampling_frequency
        self.spectrum_low_frequency: int = 0  # Hz
        self.spectrum_high_frequency: int = int(self.sampling_frequency / 2)  # Hz
        self.filter_length: int = self.spectrum_high_frequency - 1
