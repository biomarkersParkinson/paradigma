from typing import Dict, List

from paradigma.constants import DataColumns, DataUnits



class IMUConfig:
    """
    Base class for Gait feature extraction and Gait detection configurations, based on the IMU data (accelerometer, gyroscope).
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


class GaitFeatureExtractionConfig (IMUConfig):

    def __init__(self) -> None:
        super().__init__()
        self.set_sensor("accelerometer")
        self.set_sampling_frequency(100)

        self.window_type: str = "hann"
        self.verbose: int = 0

        self.window_length_s: int = 6
        self.window_step_size_s: int = 1

        # cepstral coefficients
        self.cc_low_frequency: int = 0
        self.cc_high_frequency: int = 25
        self.n_dct_filters_cc: int = 20
        self.n_coefficients_cc: int = 12

        self.d_frequency_bandwidths: Dict[str, List[float]] = {
            "power_below_gait": [0.3, 0.7],
            "power_gait": [0.7, 3.5],
            "power_tremor": [3.5, 8],
            "power_above_tremor": [8, self.sampling_frequency],
        }


        self.l_window_level_cols: List[str] = [
            "id",
            "window_nr",
            "window_start",
            "window_end",
        ]
        self.l_data_point_level_cols: List[str] = (
            self.l_accelerometer_cols + self.l_gravity_cols
        )

        # TODO: generate this dictionary using object attributes (self.X) and parameters (e.g., n_dct_filters for cc)
        self.d_channels_values: Dict[str, str] = {
            f"grav_{self.sensor}_x_mean": DataUnits.GRAVITY,
            f"grav_{self.sensor}_y_mean": DataUnits.GRAVITY,
            f"grav_{self.sensor}_z_mean": DataUnits.GRAVITY,
            f"grav_{self.sensor}_x_std": DataUnits.GRAVITY,
            f"grav_{self.sensor}_y_std": DataUnits.GRAVITY,
            f"grav_{self.sensor}_z_std": DataUnits.GRAVITY,
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
            "std_norm_acc": DataUnits.GRAVITY,
        }

        for cc_coef in range(1, self.n_coefficients_cc + 1):
            self.d_channels_values[f"cc_{cc_coef}_{self.sensor}"] = "g"

    def set_sampling_frequency(self, sampling_frequency: int) -> None:
        """Sets the sampling frequency and derived variables"""
        self.sampling_frequency: int = sampling_frequency
        self.spectrum_low_frequency: int = 0  # Hz
        self.spectrum_high_frequency: int = int(self.sampling_frequency / 2)  # Hz
        self.filter_length: int = self.spectrum_high_frequency - 1


class GaitDetectionConfig(IMUConfig):

    def __init__(self) -> None:
        super().__init__()
        self.classifier_file_name = "gd_classifier.pkl"
        self.thresholds_file_name = "gd_threshold.txt"

        self.set_filenames_values("gait")



class ArmSwingFeatureExtractionConfig(IMUConfig):

    def initialize_window_length_fields(self, window_length_s: int) -> None:
        self.window_length_s = window_length_s
        self.window_overlap_s = window_length_s * 0.75
        self.window_step_size_s = window_length_s - self.window_overlap_s

    def initialize_sampling_frequency_fields(self, sampling_frequency: int) -> None:
        self.sampling_frequency = sampling_frequency

        # computing power
        self.power_band_low_frequency = 0.3
        self.power_band_high_frequency = 3
        self.spectrum_low_frequency = 0
        self.spectrum_high_frequency = int(sampling_frequency / 2)

        self.d_frequency_bandwidths = {
            "power_below_gait": [0.3, 0.7],
            "power_gait": [0.7, 3.5],
            "power_tremor": [3.5, 8],
            "power_above_tremor": [8, sampling_frequency],
        }

        # cepstral coefficients
        self.cc_low_frequency = 0
        self.cc_high_frequency = 25
        self.n_dct_filters_cc: int = 20
        self.n_coefficients_cc: int = 12

    def initialize_column_names(
        self
    ) -> None:

        self.pred_gait_colname=DataColumns.PRED_GAIT
        self.angle_smooth_colname: str = DataColumns.ANGLE_SMOOTH
        self.angle_colname=DataColumns.ANGLE
        self.velocity_colname=DataColumns.VELOCITY
        self.segment_nr_colname=DataColumns.SEGMENT_NR


        self.l_data_point_level_cols: List[str] = (
            self.l_accelerometer_cols
            + self.l_gyroscope_cols
            + self.l_gravity_cols
            + [self.angle_smooth_colname, self.velocity_colname]
        )

    def __init__(self) -> None:
        super().__init__()
        # general
        self.sensor = "IMU"
        self.units = "degrees"

        # windowing
        self.window_type = "hann"
        self.initialize_window_length_fields(3)

        self.initialize_sampling_frequency_fields(100)

        self.initialize_column_names()

        self.d_channels_values = {
            "angle_perc_power": "proportion",
            "range_of_motion": "deg",
            "forward_peak_ang_vel_mean": DataUnits.ROTATION,
            "forward_peak_ang_vel_std": DataUnits.ROTATION,
            "backward_peak_ang_vel_mean": DataUnits.ROTATION,
            "backward_peak_ang_vel_std": DataUnits.ROTATION,
            "std_norm_acc": DataUnits.GRAVITY,
            "grav_accelerometer_x_mean": DataUnits.GRAVITY,
            "grav_accelerometer_x_std": DataUnits.GRAVITY,
            "grav_accelerometer_y_mean": DataUnits.GRAVITY,
            "grav_accelerometer_y_std": DataUnits.GRAVITY,
            "grav_accelerometer_z_mean": DataUnits.GRAVITY,
            "grav_accelerometer_z_std": DataUnits.GRAVITY,
            "accelerometer_x_power_below_gait": "X",
            "accelerometer_x_power_gait": "X",
            "accelerometer_x_power_tremor": "X",
            "accelerometer_x_power_above_tremor": "X",
            "accelerometer_x_dominant_frequency": DataUnits.FREQUENCY,
            "accelerometer_y_power_below_gait": "X",
            "accelerometer_y_power_gait": "X",
            "accelerometer_y_power_tremor": "X",
            "accelerometer_y_power_above_tremor": "X",
            "accelerometer_y_dominant_frequency": DataUnits.FREQUENCY,
            "accelerometer_z_power_below_gait": "X",
            "accelerometer_z_power_gait": "X",
            "accelerometer_z_power_tremor": "X",
            "accelerometer_z_power_above_tremor": "X",
            "accelerometer_z_dominant_frequency": DataUnits.FREQUENCY,
            "angle_dominant_frequency": DataUnits.FREQUENCY,
        }

        for sensor in ["accelerometer", "gyroscope"]:
            for cc_coef in range(1, self.n_coefficients_cc + 1):
                self.d_channels_values[f"cc_{cc_coef}_{sensor}"] = DataUnits.GRAVITY


class ArmSwingDetectionConfig(IMUConfig):

    def __init__(self) -> None:
        super().__init__()
        self.classifier_file_name = "asd_classifier.pkl"

        self.set_filenames_values("arm_swing")



class ArmSwingQuantificationConfig(IMUConfig):

    def __init__(self) -> None:
        super().__init__()
        self.set_filenames_values("arm_swing")

        self.pred_arm_swing_colname = DataColumns.PRED_ARM_SWING

        self.window_length_s = 3
        self.window_step_size = 0.75
        self.segment_gap_s = 3
        self.min_segment_length_s = 3
