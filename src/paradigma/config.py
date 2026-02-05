import warnings
from dataclasses import asdict

import numpy as np

from paradigma.constants import DataColumns, DataUnits


class BaseConfig:
    def __init__(self) -> None:
        self.meta_filename = ""
        self.values_filename = ""
        self.time_filename = ""

    def set_sensor(self, sensor: str) -> None:
        """Sets the sensor and derived filenames"""
        self.sensor: str = sensor
        self.set_filenames(sensor)

    def set_filenames(self, prefix: str) -> None:
        """Sets the filenames based on the prefix.

        This method is duplicated from `gaits_analysis_config.py`.

        Parameters
        ----------
        prefix : str
            The prefix for the filenames.
        """
        self.meta_filename = f"{prefix}_meta.json"
        self.time_filename = f"{prefix}_time.bin"
        self.values_filename = f"{prefix}_values.bin"


class IMUConfig(BaseConfig):
    """
    IMU configuration that uses DataColumns() to dynamically map available channels.
    Works even if only accelerometer or only gyroscope data is present.
    """

    def __init__(self, column_mapping: dict[str, str] | None = None) -> None:
        super().__init__()
        self.set_filenames("IMU")

        self.acceleration_units = DataUnits.ACCELERATION
        self.rotation_units = DataUnits.ROTATION
        self.axes = ["x", "y", "z"]

        # Generate a default mapping or override with user-provided mapping
        default_mapping = asdict(DataColumns())
        self.column_mapping = {**default_mapping, **(column_mapping or {})}

        self.time_colname = self.column_mapping["TIME"]

        self.accelerometer_colnames: list[str] = []
        self.gyroscope_colnames: list[str] = []
        self.gravity_colnames: list[str] = []

        self.d_channels_accelerometer: dict[str, str] = {}
        self.d_channels_gyroscope: dict[str, str] = {}

        accel_keys = ["ACCELEROMETER_X", "ACCELEROMETER_Y", "ACCELEROMETER_Z"]
        grav_keys = [
            "GRAV_ACCELEROMETER_X",
            "GRAV_ACCELEROMETER_Y",
            "GRAV_ACCELEROMETER_Z",
        ]
        gyro_keys = ["GYROSCOPE_X", "GYROSCOPE_Y", "GYROSCOPE_Z"]

        if all(k in self.column_mapping for k in accel_keys):
            self.accelerometer_colnames = [self.column_mapping[k] for k in accel_keys]

            if all(k in self.column_mapping for k in grav_keys):
                self.gravity_colnames = [self.column_mapping[k] for k in grav_keys]

            self.d_channels_accelerometer = {
                c: self.acceleration_units for c in self.accelerometer_colnames
            }

        if all(k in self.column_mapping for k in gyro_keys):
            self.gyroscope_colnames = [self.column_mapping[k] for k in gyro_keys]

            self.d_channels_gyroscope = {
                c: self.rotation_units for c in self.gyroscope_colnames
            }

        self.d_channels_imu: dict[str, str] = {
            **self.d_channels_accelerometer,
            **self.d_channels_gyroscope,
        }

        self.sampling_frequency = 100
        self.resampling_frequency = 100
        self.tolerance = 3 * 1 / self.sampling_frequency
        self.lower_cutoff_frequency = 0.2
        self.upper_cutoff_frequency = 3.5
        self.filter_order = 4

        # Segmentation parameters for handling non-contiguous data
        self.max_segment_gap_s = 1.5
        self.min_segment_length_s = 1.5


class PPGConfig(BaseConfig):

    def __init__(self, column_mapping: dict[str, str] | None = None) -> None:
        super().__init__()

        self.set_filenames("PPG")

        # Generate a default mapping or override with user-provided mapping
        default_mapping = asdict(DataColumns())
        self.column_mapping = {**default_mapping, **(column_mapping or {})}

        self.time_colname = self.column_mapping["TIME"]
        self.ppg_colname = self.column_mapping["PPG"]

        self.sampling_frequency = 30
        self.resampling_frequency = 30
        self.tolerance = 3 * 1 / self.sampling_frequency
        self.lower_cutoff_frequency = 0.4
        self.upper_cutoff_frequency = 3.5
        self.filter_order = 4

        self.d_channels_ppg = {self.ppg_colname: DataUnits.NONE}


# Domain base configs
class GaitConfig(IMUConfig):

    def __init__(self, step, column_mapping: dict[str, str] | None = None) -> None:
        # Pass column_mapping through to IMUConfig
        super().__init__(column_mapping=column_mapping)

        self.set_sensor("accelerometer")

        # ----------
        # Segmenting
        # ----------
        self.max_segment_gap_s = 1.5
        self.min_segment_length_s = 1.5

        if step == "gait":
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
        self.d_frequency_bandwidths: dict[str, list[float]] = {
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
        self.d_channels_values: dict[str, str] = {
            "accelerometer_std_norm": DataUnits.GRAVITY,
            "accelerometer_x_grav_mean": DataUnits.GRAVITY,
            "accelerometer_y_grav_mean": DataUnits.GRAVITY,
            "accelerometer_z_grav_mean": DataUnits.GRAVITY,
            "accelerometer_x_grav_std": DataUnits.GRAVITY,
            "accelerometer_y_grav_std": DataUnits.GRAVITY,
            "accelerometer_z_grav_std": DataUnits.GRAVITY,
            "accelerometer_x_power_below_gait": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_y_power_below_gait": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_z_power_below_gait": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_x_power_gait": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_y_power_gait": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_z_power_gait": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_x_power_tremor": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_y_power_tremor": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_z_power_tremor": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_x_power_above_tremor": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_y_power_above_tremor": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_z_power_above_tremor": DataUnits.POWER_SPECTRAL_DENSITY_ACC,
            "accelerometer_x_dominant_frequency": DataUnits.FREQUENCY,
            "accelerometer_y_dominant_frequency": DataUnits.FREQUENCY,
            "accelerometer_z_dominant_frequency": DataUnits.FREQUENCY,
        }

        for mfcc_coef in range(1, self.mfcc_n_coefficients + 1):
            self.d_channels_values[f"accelerometer_mfcc_{mfcc_coef}"] = (
                DataUnits.GRAVITY
            )

        if step == "arm_activity":
            for mfcc_coef in range(1, self.mfcc_n_coefficients + 1):
                self.d_channels_values[f"gyroscope_mfcc_{mfcc_coef}"] = (
                    DataUnits.GRAVITY
                )


class TremorConfig(IMUConfig):

    def __init__(self, step: str | None = None) -> None:
        """
        Parameters
        ----------
        step : str (optional)
            The step of the tremor pipeline. Can be 'features' or 'classification'.
        """
        super().__init__()

        self.set_sensor("gyroscope")

        # ----------
        # Segmenting
        # ----------
        self.window_length_s: float = 4
        self.window_step_length_s: float = 4

        # -----------------
        # Feature extraction
        # -----------------
        self.window_type = "hann"
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
        self.aggregates_tremor_power: list[str] = ["mode_binned", "median", "90p"]
        self.evaluation_points_tremor_power: np.ndarray = np.linspace(0, 6, 301)

        # -----------------
        # TSDF data storage
        # -----------------
        if step == "features":
            self.d_channels_values: dict[str, str] = {}
            for mfcc_coef in range(1, self.n_coefficients_mfcc + 1):
                self.d_channels_values[f"mfcc_{mfcc_coef}"] = DataUnits.NONE

            self.d_channels_values[DataColumns.FREQ_PEAK] = DataUnits.FREQUENCY
            self.d_channels_values[DataColumns.BELOW_TREMOR_POWER] = (
                DataUnits.POWER_ROTATION
            )
            self.d_channels_values[DataColumns.TREMOR_POWER] = DataUnits.POWER_ROTATION

        elif step == "classification":
            self.d_channels_values = {
                DataColumns.PRED_TREMOR_PROBA: "probability",
                DataColumns.PRED_TREMOR_LOGREG: "boolean",
                DataColumns.PRED_TREMOR_CHECKED: "boolean",
                DataColumns.PRED_ARM_AT_REST: "boolean",
            }


class PulseRateConfig(PPGConfig):
    def __init__(
        self,
        sensor: str = "ppg",
        ppg_sampling_frequency: int = 30,
        imu_sampling_frequency: int | None = None,
        min_window_length_s: int = 30,
        accelerometer_colnames: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.ppg_sampling_frequency = ppg_sampling_frequency

        if sensor == "imu":
            if imu_sampling_frequency is not None:
                self.imu_sampling_frequency = imu_sampling_frequency
            else:
                self.imu_sampling_frequency = IMUConfig().sampling_frequency
                warnings.warn(
                    f"imu_sampling_frequency not provided, using default "
                    f"of {self.imu_sampling_frequency} Hz"
                )

        # Windowing parameters
        self.window_length_s: int = 6
        self.window_step_length_s: int = 1
        self.window_overlap_s = self.window_length_s - self.window_step_length_s

        self.accelerometer_colnames = accelerometer_colnames

        # Signal quality analysis parameters
        self.freq_band_physio = [0.75, 3]  # Hz
        self.bandwidth = 0.2  # Hz
        self.freq_bin_resolution = 0.05  # Hz

        # Pulse rate estimation parameters
        self.threshold_sqa = 0.5
        self.threshold_sqa_accelerometer = 0.10

        # Set initial sensor and update sampling-dependent params
        self.set_sensor(sensor, min_window_length_s)

    def set_sensor(self, sensor: str, min_window_length_s: int | None = None) -> None:
        """Sets the active sensor and recomputes sampling-dependent parameters."""
        if sensor not in ["ppg", "imu"]:
            raise ValueError(f"Invalid sensor type: {sensor}")
        self.sensor = sensor

        # Decide which frequency to use
        self.sampling_frequency = (
            self.imu_sampling_frequency
            if sensor == "imu"
            else self.ppg_sampling_frequency
        )

        # Update all frequency-dependent parameters
        if min_window_length_s is not None:
            self._update_sampling_dependent_params(min_window_length_s)
        else:
            # Reuse previous tfd_length if it exists, else fallback to 30
            self._update_sampling_dependent_params(getattr(self, "tfd_length", 30))

    def _update_sampling_dependent_params(self, tfd_length: int):
        """Compute attributes that depend on sampling frequency."""

        # --- PPG-dependent parameters ---
        self.tfd_length = tfd_length
        self.min_pr_samples = int(round(self.tfd_length * self.ppg_sampling_frequency))

        pr_est_length = 2  # pulse rate estimation length in seconds
        self.pr_est_samples = pr_est_length * self.ppg_sampling_frequency

        # Time-frequency distribution parameters
        win_type_doppler = "hamm"
        win_type_lag = "hamm"
        win_length_doppler = 8
        win_length_lag = 1
        doppler_samples = self.ppg_sampling_frequency * win_length_doppler
        lag_samples = win_length_lag * self.ppg_sampling_frequency
        self.kern_type = "sep"
        self.kern_params = {
            "doppler": {"win_length": doppler_samples, "win_type": win_type_doppler},
            "lag": {"win_length": lag_samples, "win_type": win_type_lag},
        }

        # --- Welch / FFT parameters based on current sensor frequency ---
        self.window_length_welch = 3 * self.sampling_frequency
        self.overlap_welch_window = self.window_length_welch // 2
        self.nfft = (
            len(np.arange(0, self.sampling_frequency / 2, self.freq_bin_resolution)) * 2
        )
