from typing import Dict, List
from paradigma.constants import DataColumns, DataUnits
import numpy as np

class BaseConfig:
    def __init__(self) -> None:
        self.meta_filename = ''
        self.values_filename = ''
        self.time_filename = ''

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

class IMUConfig(BaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self.set_filenames('IMU')

        self.acceleration_units = DataUnits.ACCELERATION
        self.rotation_units = DataUnits.ROTATION

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
class GaitConfig(IMUConfig):

    def __init__(self, step) -> None:
        super().__init__()

        self.set_sensor('accelerometer')

        # ----------
        # Segmenting
        # ----------
        self.max_segment_gap_s = 1.5
        self.min_segment_length_s = 1.5

        if step == 'gait':
            self.window_length_s: float = 6
            self.window_step_length_s: float = 1
        else:
            self.window_length_s: float = 3
            self.window_step_length_s: float = self.window_length_s * 0.25

        # -----------------
        # Feature extraction
        # -----------------
        self.window_type: str = "hann"
        self.spectrum_low_frequency: int = 0
        self.spectrum_high_frequency: int = int(self.sampling_frequency / 2)

        # Power in specified frequency bands
        self.d_frequency_bandwidths: Dict[str, List[float]] = {
            "power_below_gait": [0.2, 0.7],
            "power_gait": [0.7, 3.5],
            "power_tremor": [3.5, 8],
            "power_above_tremor": [8, 25],
        }

        # Mel frequency cepstral coefficients
        self.mfcc_low_frequency: int = 0
        self.mfcc_high_frequency: int = 25
        self.mfcc_n_dct_filters: int = 15
        self.mfcc_n_coefficients: int = 12

        # Dominant frequency of first harmonic of arm swing
        self.angle_fmin: float = 0.5
        self.angle_fmax: float = 1.5

        # -----------------
        # TSDF data storage
        # -----------------
        self.d_channels_values: Dict[str, str] = {
            "accelerometer_std_norm": DataUnits.GRAVITY,
            "accelerometer_x_grav_mean": DataUnits.GRAVITY,
            "accelerometer_y_grav_mean": DataUnits.GRAVITY,
            "accelerometer_z_grav_mean": DataUnits.GRAVITY,
            "accelerometer_x_grav_std": DataUnits.GRAVITY,
            "accelerometer_y_grav_std": DataUnits.GRAVITY,
            "accelerometer_z_grav_std": DataUnits.GRAVITY,
            "accelerometer_x_power_below_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_y_power_below_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_z_power_below_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_x_power_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_y_power_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_z_power_gait": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_x_power_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_y_power_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_z_power_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_x_power_above_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_y_power_above_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_z_power_above_tremor": DataUnits.POWER_SPECTRAL_DENSITY,
            "accelerometer_x_dominant_frequency": DataUnits.FREQUENCY,
            "accelerometer_y_dominant_frequency": DataUnits.FREQUENCY,
            "accelerometer_z_dominant_frequency": DataUnits.FREQUENCY,
        }

        for mfcc_coef in range(1, self.mfcc_n_coefficients + 1):
            self.d_channels_values[f"accelerometer_mfcc_{mfcc_coef}"] = DataUnits.GRAVITY

        if step == 'arm_activity':
            for mfcc_coef in range(1, self.mfcc_n_coefficients + 1):
                self.d_channels_values[f"gyroscope_mfcc_{mfcc_coef}"] = DataUnits.GRAVITY


class TremorConfig(IMUConfig):

    def __init__(self, step: str | None = None) -> None:
        """
        Parameters
        ----------
        step : str (optional)
            The step of the tremor pipeline. Can be 'features' or 'classification'.
        """
        super().__init__()

        self.set_sensor('gyroscope')

        # ----------
        # Segmenting
        # ----------
        self.window_length_s: float = 4
        self.window_step_length_s: float = 4

        # -----------------
        # Feature extraction
        # -----------------
        self.window_type = 'hann'
        self.overlap_fraction: float = 0.8
        self.segment_length_psd_s: float = 3
        self.segment_length_spectrogram_s: float = 2
        self.spectral_resolution: float = 0.25
        
        # PSD based features
        self.fmin_peak_search: float = 1
        self.fmax_peak_search: float = 25
        self.fmin_below_rest_tremor: float = 0.5
        self.fmax_below_rest_tremor: float = 3
        self.fmin_rest_tremor: float = 3
        self.fmax_rest_tremor: float = 7

        # Mel frequency cepstral coefficients
        self.fmin_mfcc: float = 0
        self.fmax_mfcc: float = 25
        self.n_dct_filters_mfcc: int = 15
        self.n_coefficients_mfcc: int = 12

        # --------------
        # Classification
        # --------------
        self.movement_threshold: float = 50

        # -----------
        # Aggregation
        # -----------
        self.aggregates_tremor_power: List[str] = ['mode', 'median', '90p']

        # -----------------
        # TSDF data storage
        # -----------------
        if step == 'features':
            self.d_channels_values: Dict[str, str] = {}
            for mfcc_coef in range(1, self.n_coefficients_mfcc + 1):
                self.d_channels_values[f"mfcc_{mfcc_coef}"] = "unitless"

            self.d_channels_values["freq_peak"] = "Hz"
            self.d_channels_values["below_tremor_power"] = "(deg/s)^2"
            self.d_channels_values["tremor_power"] = "(deg/s)^2"
        elif step == 'classification':
            self.d_channels_values = {
                DataColumns.PRED_TREMOR_PROBA: "probability",
                DataColumns.PRED_TREMOR_LOGREG: "boolean",
                DataColumns.PRED_TREMOR_CHECKED: "boolean",
                DataColumns.PRED_ARM_AT_REST: "boolean"
            }

        
class PulseRateConfig(PPGConfig):
    def __init__(self, sensor: str = 'ppg', min_window_length_s: int = 30) -> None:
        super().__init__()

        # ----------
        # Segmenting
        # ----------
        self.window_length_s: int = 6
        self.window_step_length_s: int = 1
        self.window_overlap_s = self.window_length_s - self.window_step_length_s

        self.accelerometer_cols = IMUConfig().accelerometer_cols

        # -----------------------
        # Signal quality analysis
        # -----------------------
        self.freq_band_physio = [0.75, 3] # Hz
        self.bandwidth = 0.2   # Hz      
        self.freq_bin_resolution = 0.05 # Hz

        # ---------------------
        # Pulse rate estimation
        # ---------------------
        self.set_tfd_length(min_window_length_s)  # Set tfd length to default of 30 seconds
        self.threshold_sqa = 0.5
        self.threshold_sqa_accelerometer = 0.10

        pr_est_length = 2  # pulse rate estimation length in seconds
        self.pr_est_samples = pr_est_length * self.sampling_frequency

        # Time-frequency distribution parameters
        self.kern_type = 'sep'
        win_type_doppler = 'hamm'
        win_type_lag = 'hamm'
        win_length_doppler = 8
        win_length_lag = 1
        doppler_samples = self.sampling_frequency * win_length_doppler
        lag_samples = win_length_lag * self.sampling_frequency
        self.kern_params = {
            'doppler': {
                'win_length': doppler_samples,
                'win_type': win_type_doppler,
            },
            'lag': {
                'win_length': lag_samples,
                'win_type': win_type_lag,
            }
        }

        self.set_sensor(sensor)

    def set_tfd_length(self, tfd_length: int):
        self.tfd_length = tfd_length
        self.min_pr_samples = int(round(self.tfd_length * self.sampling_frequency))

    def set_sensor(self, sensor):
        self.sensor = sensor 

        if sensor not in ['ppg', 'imu']:
            raise ValueError(f"Invalid sensor type: {sensor}")
        
        if sensor == 'imu':
            self.sampling_frequency = IMUConfig().sampling_frequency
        else:
            self.sampling_frequency = PPGConfig().sampling_frequency

        self.window_length_welch = 3 * self.sampling_frequency
        self.overlap_welch_window = self.window_length_welch // 2
        self.nfft = len(np.arange(0, self.sampling_frequency/2, self.freq_bin_resolution))*2
