from typing import Dict, List
from paradigma.constants import DataColumns, DataUnits

class IMUconfig:
    """
    Base class for Gait feature extraction and Gait detection configurations, based on the IMU data (accelerometer, gyroscope).
    """
    def __init__(self):

        self.time_colname = DataColumns.TIME
        self.segment_nr_colname = DataColumns.SEGMENT_NR

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

class PPGconfig:
    """
    Base class for signal quality feature extraction and heart rate estimation configurations, based on the PPG data.
    """
    def __init__(self):

        self.time_colname = DataColumns.TIME
        self.segment_nr_colname = DataColumns.SEGMENT_NR
        self.ppg_colname = DataColumns.PPG
        self.sampling_frequency = 30

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

class SignalQualityFeatureExtractionConfig(PPGconfig):

    def __init__(self) -> None:
        super().__init__()
        self.set_sensor("PPG")

        self.window_length_s: int = 6
        self.window_step_size_s: int = 1
        self.segment_gap_s = 1.5
        self.window_length_welch = 3*self.sampling_frequency
        self.overlap_welch_window = self.window_length_welch // 2

        self.freq_band_physio = [0.75, 3] # Hz
        self.bandwidth = 0.2   # Hz

        config_imu = IMUconfig()
        self.sampling_frequency_imu = config_imu.sampling_frequency

        self.single_value_cols: List[str] = None
        self.list_value_cols: List[str] = [
            self.ppg_colname
        ]

    def set_sampling_frequency(self, sampling_frequency: int) -> None:
        """Sets the sampling frequency and derived variables"""
        self.sampling_frequency: int = sampling_frequency
        self.spectrum_low_frequency: int = 0  # Hz
        self.spectrum_high_frequency: int = int(self.sampling_frequency / 2)  # Hz
        self.filter_length: int = self.spectrum_high_frequency - 1

class SignalQualityClassificationConfig(PPGconfig):

    def __init__(self) -> None:
        super().__init__()
        self.classifier_file_name = "ppg_quality_classifier.pkl"
        self.thresholds_file_name = "ppg_acc_quality_threshold.txt"

        self.set_filenames_values("ppg")

class HeartRateExtractionConfig(PPGconfig):

    def __init__(self) -> None:
        super().__init__()
         # Parameters for HR analysis
        self.sqa_window_length_s: int = 6
        self.sqa_window_overlap_s: int = 5
        self.sqa_window_step_size_s: int = 1
        min_window_length = 10
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
        self.kern_params = [{'doppler_samples': doppler_samples, 'win_type_doppler': win_type_doppler}, 
                    {'lag_samples': lag_samples, 'win_type_lag': win_type_lag}]
        
        self.kern_params = {
            'doppler': {
                'samples': doppler_samples,
                'win_type': win_type_doppler,
            },
            'lag': {
                'samples': lag_samples,
                'win_type': win_type_lag,
            }
        }

        self.kern_params = {
            'samples': {
                'doppler': doppler_samples,
                'lag': lag_samples,
            },
            'win_type': {
                'doppler': win_type_doppler,
                'lag': win_type_lag,
            }
        }
            