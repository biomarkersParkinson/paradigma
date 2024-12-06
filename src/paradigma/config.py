from typing import Dict, List
from paradigma.constants import DataColumns, DataUnits

class BaseConfig:

    def __init__(self) -> None:

        self.meta_filename = ''
        self.values_filename = ''
        self.time_filename = ''

        self.time_colname = DataColumns.TIME
        self.segment_nr_colname = DataColumns.SEGMENT_NR

    def set_sensor(self, sensor: str) -> None:
        """Sets the sensor and derived filenames"""
        self.sensor: str = sensor
        self.set_filenames(sensor)

    def set_filenames(self, prefix: str) -> None:
        """Sets the filenames based on the prefix. This method is duplicated from `gaits_analysis_config.py`.
        
        Parameters
        ----------
        prefix : str
            The prefix for the filenames.
        """
        self.meta_filename = f"{prefix}_meta.json"
        self.time_filename = f"{prefix}_time.bin"
        self.values_filename = f"{prefix}_values.bin"

    
# Signal preprocessing configs
class IMUConfig(BaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self.set_filenames('IMU')
        self.acceleration_units = DataUnits.ACCELERATION
        self.rotation_units = DataUnits.ROTATION

        self.side_watch = 'right'

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

        self.d_channels_accelerometer = {
            DataColumns.ACCELEROMETER_X: self.acceleration_units,
            DataColumns.ACCELEROMETER_Y: self.acceleration_units,
            DataColumns.ACCELEROMETER_Z: self.acceleration_units,
        }
        self.d_channels_gyroscope = {
            DataColumns.GYROSCOPE_X: self.rotation_units,
            DataColumns.GYROSCOPE_Y: self.rotation_units,
            DataColumns.GYROSCOPE_Z: self.rotation_units,
        }
        self.d_channels_imu = {**self.d_channels_accelerometer, **self.d_channels_gyroscope}

        self.sampling_frequency = 100
        self.lower_cutoff_frequency = 0.2
        self.upper_cutoff_frequency = 3.5
        self.filter_order = 4


class PPGConfig(BaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self.set_filenames('PPG')

        self.ppg_colname = DataColumns.PPG

        self.sampling_frequency = 30
        self.lower_cutoff_frequency = 0.4
        self.upper_cutoff_frequency = 3.5
        self.filter_order = 4

        self.d_channels_ppg = {
            DataColumns.PPG: DataUnits.NONE
        }


# Domain base configs
class GaitBaseConfig(IMUConfig):

    def __init__(self) -> None:
        super().__init__()

        self.set_sensor('accelerometer')

        self.pred_gait_proba_colname = DataColumns.PRED_GAIT_PROBA
        self.pred_gait_colname=DataColumns.PRED_GAIT
        self.pred_no_other_arm_activity_proba_colname = DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA
        self.pred_no_other_arm_activity_colname = DataColumns.PRED_NO_OTHER_ARM_ACTIVITY
        self.angle_colname=DataColumns.ANGLE
        self.velocity_colname=DataColumns.VELOCITY

        self.window_type: str = "hann"
        self.max_segment_gap_s = 1.5
        self.min_segment_length_s = 1.5

        self.spectrum_low_frequency: int = 0
        self.spectrum_high_frequency: int = int(self.sampling_frequency / 2)

        # feature parameters
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


class TremorBaseConfig(IMUConfig):

    def __init__(self) -> None:
        super().__init__()

        self.set_sensor('gyroscope')

        self.window_type = 'hann'
        self.window_length_s: float = 4
        self.window_step_size_s: float = 4

        self.fmin_tremor_power: float = 3
        self.fmax_tremor_power: float = 7

        self.movement_threshold: float = 50


class HeartRateBaseConfig(PPGConfig):
    def __init__(self) -> None:
        super().__init__()

        self.window_length_s: int = 6
        self.window_step_size_s: int = 1
        self.segment_gap_s = 1.5


# Domain feature extraction configs
class GaitFeatureExtractionConfig(GaitBaseConfig):

    def __init__(self) -> None:
        super().__init__()

        # segmenting
        self.window_length_s: float = 6
        self.window_step_length_s: float = 1

        # channels
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
            self.d_channels_values[f"{self.sensor}_mfcc_{mfcc_coef}"] = DataUnits.GRAVITY


class ArmActivityFeatureExtractionConfig(GaitBaseConfig):
    def __init__(self) -> None:
        super().__init__()

        # segmenting
        self.window_length_s: float = 3
        self.window_step_length_s: float = self.window_length_s * 0.25

        # dominant frequency of first harmonic of arm swing
        self.angle_fmin: float = 0.5
        self.angle_fmax: float = 1.5

        # channels
        self.d_channels_values = {
            "range_of_motion": "deg",
            f"forward_peak_{self.velocity_colname}_mean": DataUnits.ROTATION,
            f"forward_peak_{self.velocity_colname}_std": DataUnits.ROTATION,
            f"backward_peak_{self.velocity_colname}_mean": DataUnits.ROTATION,
            f"backward_peak_{self.velocity_colname}_std": DataUnits.ROTATION,
            f"{self.sensor}_std_norm": DataUnits.GRAVITY,
            f"{self.sensor}_x_grav_mean": DataUnits.GRAVITY,
            f"{self.sensor}_x_grav_std": DataUnits.GRAVITY,
            f"{self.sensor}_y_grav_mean": DataUnits.GRAVITY,
            f"{self.sensor}_y_grav_std": DataUnits.GRAVITY,
            f"{self.sensor}_z_grav_mean": DataUnits.GRAVITY,
            f"{self.sensor}_z_grav_std": DataUnits.GRAVITY,
            f"{self.sensor}_x_power_below_gait": "X",
            f"{self.sensor}_x_power_gait": "X",
            f"{self.sensor}_x_power_tremor": "X",
            f"{self.sensor}_x_power_above_tremor": "X",
            f"{self.sensor}_x_dominant_frequency": DataUnits.FREQUENCY,
            f"{self.sensor}_y_power_below_gait": "X",
            f"{self.sensor}_y_power_gait": "X",
            f"{self.sensor}_y_power_tremor": "X",
            f"{self.sensor}_y_power_above_tremor": "X",
            f"{self.sensor}_y_dominant_frequency": DataUnits.FREQUENCY,
            f"{self.sensor}_z_power_below_gait": "X",
            f"{self.sensor}_z_power_gait": "X",
            f"{self.sensor}_z_power_tremor": "X",
            f"{self.sensor}_z_power_above_tremor": "X",
            f"{self.sensor}_z_dominant_frequency": DataUnits.FREQUENCY,
            f"{self.angle_colname}_dominant_frequency": DataUnits.FREQUENCY,
        }

        for sensor in ["accelerometer", "gyroscope"]:
            for mfcc_coef in range(1, self.mfcc_n_coefficients + 1):
                self.d_channels_values[f"{sensor}_mfcc_{mfcc_coef}"] = DataUnits.GRAVITY


class TremorFeatureExtractionConfig(TremorBaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self.single_value_cols: List[str] = None
        self.list_value_cols: List[str] = self.gyroscope_cols

        # power spectral density
        self.overlap_fraction: float = 0.8
        self.segment_length_s_psd: float = 3
        self.spectral_resolution_psd: float = 0.25
        self.fmin_peak: float = 1
        self.fmax_peak: float = 25
        self.fmin_low_power: float = 0.5
        self.fmax_low_power: float = 3

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


class SignalQualityFeatureExtractionConfig(HeartRateBaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self.window_length_welch = 3 * self.sampling_frequency
        self.overlap_welch_window = self.window_length_welch // 2

        self.freq_band_physio = [0.75, 3] # Hz
        self.bandwidth = 0.2   # Hz

        config_imu = IMUConfig()
        self.sampling_frequency_imu = config_imu.sampling_frequency

        self.single_value_cols: List[str] = None
        self.list_value_cols: List[str] = [
            self.ppg_colname
        ]


# Classification
class GaitDetectionConfig(GaitBaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self.classifier_file_name = "gait_detection_classifier.pkl"
        self.thresholds_file_name = "gait_detection_threshold.txt"

        self.set_filenames('gait')


class FilteringGaitConfig(GaitBaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self.classifier_file_name = "gait_filtering_classifier.pkl"
        self.thresholds_file_name = "gait_filtering_threshold.txt"

        self.set_filenames('arm_activity')


class TremorDetectionConfig(TremorBaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self.coefficients_file_name = "tremor_detection_coefficients.txt"
        self.thresholds_file_name = "tremor_detection_threshold.txt"
        self.mean_scaling_file_name = "tremor_detection_mean_scaling.txt"
        self.std_scaling_file_name = "tremor_detection_std_scaling.txt"

        self.fmin_peak: float = self.fmin_tremor_power
        self.fmax_peak: float = self.fmax_tremor_power

        self.d_channels_values = {
            DataColumns.PRED_TREMOR_PROBA: "probability",
            DataColumns.PRED_TREMOR_LOGREG: "boolean",
            DataColumns.PRED_TREMOR_CHECKED: "boolean"
        }

        self.set_filenames('tremor')


class SignalQualityClassificationConfig(HeartRateBaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self.classifier_file_name = "ppg_quality_classifier.pkl"
        self.thresholds_file_name = "ppg_acc_quality_threshold.txt"

        self.set_filenames('gait')


# Quantification
class ArmSwingQuantificationConfig(GaitBaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self.min_segment_length_s = 3


class TremorQuantificationConfig(TremorBaseConfig):
    def __init__(self) -> None:
        super().__init__()

        self.valid_day_threshold_hr: float = 0 # change to 10 later!
        self.daytime_hours_lower_bound: float = 8
        self.daytime_hours_upper_bound: float = 22
        self.percentile_tremor_power: float = 0.9

        self.set_filenames('tremor')
        

class HeartRateExtractionConfig(HeartRateBaseConfig):

    def __init__(self, min_window_length: float = 10) -> None:
        super().__init__()

         # Parameters for HR analysis
        self.window_length_s: int = 6
        self.window_step_size_s: int = 1
        self.min_hr_samples = min_window_length * self.sampling_frequency
        self.threshold_sqa = 0.5

        # Heart rate estimation parameters
        hr_est_length = 2
        self.hr_est_samples = hr_est_length * self.sampling_frequency

        # Time-frequency distribution parameters
        self.tfd_length = 10
        self.kern_type = 'sep'
        win_type_doppler = 'hamm'
        win_type_lag = 'hamm'
        win_length_doppler = 1
        win_length_lag = 8
        doppler_samples = self.sampling_frequency * win_length_doppler
        lag_samples = win_length_lag * self.sampling_frequency
        self.kern_params = [
            {'doppler_samples': doppler_samples, 'win_type_doppler': win_type_doppler}, 
            {'lag_samples': lag_samples, 'win_type_lag': win_type_lag}
        ]
            