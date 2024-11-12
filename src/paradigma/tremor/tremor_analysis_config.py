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
        self.sampling_frequency: int=100
        self.window_length_s: int = 4
        self.window_step_size_s: int = 4
        self.single_value_cols: List[str] = None
        self.list_value_cols: List[str] = (self.l_gyroscope_cols)

        # power spectral density
        self.window_type = 'hann'
        self.overlap: int = 0.8
        self.segment_length_s_psd: int = 2
        self.spectral_resolution_psd: int = 0.25

        # cepstral coefficients
        self.segment_length_s_mfcc: int = 2
        self.mfcc_low_frequency: int = 0
        self.mfcc_high_frequency: int = 25
        self.n_dct_filters_mfcc: int = 15
        self.n_coefficients_mfcc: int = 12

        self.d_channels_values: Dict[str, str] = {}
        for mfcc_coef in range(1, self.n_coefficients_mfcc + 1):
            self.d_channels_values[f"mfcc_{mfcc_coef}"] = "unitless"