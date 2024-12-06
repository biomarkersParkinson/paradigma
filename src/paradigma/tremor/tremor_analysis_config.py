from typing import Dict, List

from paradigma.constants import DataColumns

class IMUConfig:
    """
    Base class for Tremor feature extraction and Tremor detection configurations, based on the IMU data (accelerometer, gyroscope).
    """
    def __init__(self):

        self.time_colname = DataColumns.TIME

        self.accelerometer_cols: List[str] = [
            DataColumns.ACCELEROMETER_X,
            DataColumns.ACCELEROMETER_Y,
            DataColumns.ACCELEROMETER_Z,
        ]
        self.gyroscope_cols: List[str] = [
            DataColumns.GYROSCOPE_X,
            DataColumns.GYROSCOPE_Y,
            DataColumns.GYROSCOPE_Z,
        ]

        self.gravity_cols: List[str] = [
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
        self.sampling_frequency: int = 100
        self.window_length_s: float = 4
        self.window_step_size_s: float = 4
        self.single_value_cols: List[str] = None
        self.list_value_cols: List[str] = self.gyroscope_cols

        # power spectral density
        self.window_type = 'hann'
        self.overlap_fraction: float = 0.8
        self.segment_length_s_psd: float = 3
        self.spectral_resolution_psd: float = 0.25
        self.fmin_peak: float = 1
        self.fmax_peak: float = 25
        self.fmin_low_power: float = 0.5
        self.fmax_low_power: float = 3
        self.fmin_tremor_power: float = 3
        self.fmax_tremor_power: float = 7

        # cepstral coefficients
        self.segment_length_s_mfcc: float = 2
        self.fmin_mfcc: float = 0
        self.fmax_mfcc: float = 25
        self.n_dct_filters_mfcc: int = 15
        self.n_coefficients_mfcc: int = 12

        self.d_channels_values: Dict[str, str] = {}
        for mfcc_coef in range(1, self.n_coefficients_mfcc + 1):
            self.d_channels_values[f"mfcc_{mfcc_coef}"] = "unitless"
        self.d_channels_values["freq_peak"] = "Hz"
        self.d_channels_values["low_freq_power"] = "(deg/s)^2"
        self.d_channels_values["tremor_power"] = "(deg/s)^2"
        

class TremorDetectionConfig(IMUConfig):

    def __init__(self) -> None:
        super().__init__()
        self.coefficients_file_name = "tremor_detection_coefficients.txt"
        self.thresholds_file_name = "tremor_detection_threshold.txt"
        self.mean_scaling_file_name = "tremor_detection_mean_scaling.txt"
        self.std_scaling_file_name = "tremor_detection_std_scaling.txt"

        self.fmin_peak: float = 3
        self.fmax_peak: float = 7
        self.movement_threshold: float = 50

        self.d_channels_values = {
        "pred_tremor_proba": "probability",
        "pred_tremor_logreg": "boolean",
        "pred_tremor_checked": "boolean"
        }

        self.set_filenames_values("tremor")


class TremorQuantificationConfig(IMUConfig):
    def __init__(self) -> None:
        super().__init__()

        self.window_length_s: float = 4
        self.valid_day_threshold_hr: float = 0 # change to 10 later!
        self.daytime_hours_lower_bound: float = 8
        self.daytime_hours_upper_bound: float = 22
        self.movement_threshold: float = 50
        self.percentile_tremor_power: float = 0.9
        
        self.set_filenames_values("tremor")