from typing import Dict, List

from paradigma.constants import DataColumns, DataUnits



class IMUConfig:
    """
    Base class for Gait feature extraction and Gait detection configurations, based on the IMU data (accelerometer, gyroscope).
    """
    def __init__(self):

        self.time_colname = DataColumns.TIME
        self.segment_nr_colname = DataColumns.SEGMENT_NR

        self.axes = ["x", "y", "z"]

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

        self.sampling_frequency = 100

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


class GaitFeatureExtractionConfig(IMUConfig):

    def __init__(self) -> None:
        super().__init__()
        self.set_sensor("accelerometer")
        self.set_sampling_frequency(self.sampling_frequency)
        self.initialize_column_names()

        self.window_type: str = "hann"
        self.verbose: int = 0

        self.window_length_s: int = 6
        self.window_step_length_s: int = 1
        self.max_segment_gap_s = 1.5
        self.min_segment_length_s = 1.5

        # cepstral coefficients
        self.mfcc_low_frequency: int = 0
        self.mfcc_high_frequency: int = 25
        self.mfcc_n_dct_filters: int = 15
        self.mfcc_n_coefficients: int = 12

        self.d_frequency_bandwidths: Dict[str, List[float]] = {
            "power_below_gait": [0.3, 0.7],
            "power_gait": [0.7, 3.5],
            "power_tremor": [3.5, 8],
            "power_above_tremor": [8, 25],
        }

        # TODO: generate this dictionary using object attributes (self.X) and parameters (e.g., n_dct_filters for cc)
        self.d_channels_values: Dict[str, str] = {
            f"{self.sensor}_std_norm": DataUnits.GRAVITY,
            f"{self.sensor}_x_grav_mean": DataUnits.GRAVITY,
            f"{self.sensor}_y_grav_mean": DataUnits.GRAVITY,
            f"{self.sensor}_z_grav_mean": DataUnits.GRAVITY,
            f"{self.sensor}_x_grav_std": DataUnits.GRAVITY,
            f"{self.sensor}_y_grav_std": DataUnits.GRAVITY,
            f"{self.sensor}_z_grav_std": DataUnits.GRAVITY,
            f"{self.sensor}_x_power_below_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_y_power_below_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_z_power_below_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_x_power_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_y_power_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_z_power_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_x_power_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_y_power_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_z_power_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_x_power_above_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_y_power_above_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_z_power_above_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            f"{self.sensor}_x_dominant_frequency": DataUnits.FREQUENCY,
            f"{self.sensor}_y_dominant_frequency": DataUnits.FREQUENCY,
            f"{self.sensor}_z_dominant_frequency": DataUnits.FREQUENCY,
        }

        for mfcc_coef in range(1, self.mfcc_n_coefficients + 1):
            self.d_channels_values[f"{self.sensor}_mfcc_{mfcc_coef}"] = "g"

    def set_sampling_frequency(self, sampling_frequency: int) -> None:
        """Sets the sampling frequency and derived variables"""
        self.sampling_frequency: int = sampling_frequency
        self.spectrum_low_frequency: int = 0  # Hz
        self.spectrum_high_frequency: int = int(self.sampling_frequency / 2)  # Hz
        self.filter_length: int = self.spectrum_high_frequency - 1

    def initialize_column_names(
        self
    ) -> None:

        self.pred_gait_proba_colname=DataColumns.PRED_GAIT_PROBA
        self.pred_gait_colname=DataColumns.PRED_GAIT


class GaitDetectionConfig(IMUConfig):

    def __init__(self) -> None:
        super().__init__()
        self.classifier_file_name = "gait_detection_classifier.pkl"
        self.thresholds_file_name = "gait_detection_threshold.txt"

        self.set_filenames_values("gait")



class ArmActivityFeatureExtractionConfig(IMUConfig):

    def initialize_window_length_fields(self, window_length_s: int) -> None:
        self.window_length_s = window_length_s
        self.window_overlap_s = window_length_s * 0.75
        self.window_step_length_s = window_length_s - self.window_overlap_s
        self.max_segment_gap_s = 1.5
        self.min_segment_length_s = 1.5

    def initialize_sampling_frequency_fields(self, sampling_frequency: int) -> None:
        self.sampling_frequency = sampling_frequency

        # computing power
        self.power_band_low_frequency = 0.2
        self.power_band_high_frequency = 3
        self.spectrum_low_frequency = 0
        self.spectrum_high_frequency = int(sampling_frequency / 2)

        self.d_frequency_bandwidths = {
            "power_below_gait": [0.3, 0.7],
            "power_gait": [0.7, 3.5],
            "power_tremor": [3.5, 8],
            "power_above_tremor": [8, 25],
        }

        # cepstral coefficients
        self.mfcc_low_frequency = 0
        self.mfcc_high_frequency = 25
        self.mfcc_n_dct_filters: int = 15
        self.mfcc_n_coefficients: int = 12

    def initialize_column_names(
        self
    ) -> None:

        self.pred_gait_proba_colname=DataColumns.PRED_GAIT_PROBA
        self.pred_gait_colname=DataColumns.PRED_GAIT
        self.angle_colname=DataColumns.ANGLE
        self.velocity_colname=DataColumns.VELOCITY
        self.segment_nr_colname=DataColumns.SEGMENT_NR

    def __init__(self) -> None:
        super().__init__()
        # general
        self.sensor = "IMU"
        self.units = "degrees"

        # windowing
        self.window_type = "hann"
        self.initialize_window_length_fields(3)
        self.initialize_sampling_frequency_fields(self.sampling_frequency)
        self.initialize_column_names()

        # dominant frequency of first harmonic
        self.angle_fmin = 0.5
        self.angle_fmax = 1.5

        sensor = 'accelerometer'

        self.d_channels_values = {
            "range_of_motion": "deg",
            f"forward_peak_{self.velocity_colname}_mean": DataUnits.ROTATION,
            f"forward_peak_{self.velocity_colname}_std": DataUnits.ROTATION,
            f"backward_peak_{self.velocity_colname}_mean": DataUnits.ROTATION,
            f"backward_peak_{self.velocity_colname}_std": DataUnits.ROTATION,
            f"{sensor}_std_norm": DataUnits.GRAVITY,
            f"{sensor}_x_grav_mean": DataUnits.GRAVITY,
            f"{sensor}_x_grav_std": DataUnits.GRAVITY,
            f"{sensor}_y_grav_mean": DataUnits.GRAVITY,
            f"{sensor}_y_grav_std": DataUnits.GRAVITY,
            f"{sensor}_z_grav_mean": DataUnits.GRAVITY,
            f"{sensor}_z_grav_std": DataUnits.GRAVITY,
            f"{sensor}_x_power_below_gait": "X",
            f"{sensor}_x_power_gait": "X",
            f"{sensor}_x_power_tremor": "X",
            f"{sensor}_x_power_above_tremor": "X",
            f"{sensor}_x_dominant_frequency": DataUnits.FREQUENCY,
            f"{sensor}_y_power_below_gait": "X",
            f"{sensor}_y_power_gait": "X",
            f"{sensor}_y_power_tremor": "X",
            f"{sensor}_y_power_above_tremor": "X",
            f"{sensor}_y_dominant_frequency": DataUnits.FREQUENCY,
            f"{sensor}_z_power_below_gait": "X",
            f"{sensor}_z_power_gait": "X",
            f"{sensor}_z_power_tremor": "X",
            f"{sensor}_z_power_above_tremor": "X",
            f"{sensor}_z_dominant_frequency": DataUnits.FREQUENCY,
            f"{self.angle_colname}_dominant_frequency": DataUnits.FREQUENCY,
        }

        for sensor in ["accelerometer", "gyroscope"]:
            for mfcc_coef in range(1, self.mfcc_n_coefficients + 1):
                self.d_channels_values[f"{sensor}_mfcc_{mfcc_coef}"] = DataUnits.GRAVITY


class FilteringGaitConfig(IMUConfig):

    def initialize_column_names(
        self
    ) -> None:
        
        self.angle_colname=DataColumns.ANGLE
        self.velocity_colname=DataColumns.VELOCITY

    def __init__(self) -> None:
        super().__init__()
        self.classifier_file_name = "gait_filtering_classifier.pkl"

        self.set_filenames_values("arm_activity")
        self.initialize_column_names()



class ArmSwingQuantificationConfig(IMUConfig):

    def __init__(self) -> None:
        super().__init__()
        self.set_filenames_values("arm_activity")

        self.angle_colname = DataColumns.ANGLE
        self.velocity_colname = DataColumns.VELOCITY
        self.pred_other_arm_activity_proba_colname = DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA
        self.pred_other_arm_activity_colname = DataColumns.PRED_NO_OTHER_ARM_ACTIVITY

        self.window_length_s = 3
        self.window_step_length_s = 0.25 * self.window_length_s
        self.max_segment_gap_s = 1.5
        self.min_segment_length_s = 3
