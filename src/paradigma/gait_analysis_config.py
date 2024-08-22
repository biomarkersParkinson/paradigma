from typing import Dict, List

from paradigma.constants import DataColumns


class GaitFeatureExtractionConfig:

    def __init__(self) -> None:
        self.set_sensor('accelerometer')
        self.set_sampling_frequency(100)

        self.window_type: str = 'hann'
        self.verbose: int = 0
        
        self.window_length_s: int = 6
        self.window_step_size_s: int = 1

        # cepstral coefficients
        self.cc_low_frequency = 0
        self.cc_high_frequency = 25
        self.n_dct_filters_cc: int = 20
        self.n_coefficients_cc: int = 12
        
        self.d_frequency_bandwidths: Dict[str, List[float]] = {
            'power_below_gait': [0.3, 0.7],
            'power_gait': [0.7, 3.5],
            'power_tremor': [3.5, 8],
            'power_above_tremor': [8, self.sampling_frequency]
        }

        self.time_colname = 'time'
        
        self.l_accelerometer_cols: List[str] = [
            DataColumns.ACCELEROMETER_X, 
            DataColumns.ACCELEROMETER_Y, 
            DataColumns.ACCELEROMETER_Z
        ]

        self.l_gravity_cols: List[str] = [f'grav_{x}' for x in self.l_accelerometer_cols]
        self.l_window_level_cols: List[str] = ['id', 'window_nr', 'window_start', 'window_end']
        self.l_data_point_level_cols: List[str] = self.l_accelerometer_cols + self.l_gravity_cols
        
        # TODO: generate this dictionary using object attributes (self.X) and parameters (e.g., n_dct_filters for cc)
        self.d_channels_values: Dict[str, str] = {
            f'grav_{self.sensor}_x_mean': 'g',
            f'grav_{self.sensor}_y_mean': 'g',
            f'grav_{self.sensor}_z_mean': 'g',
            f'grav_{self.sensor}_x_std': 'g',
            f'grav_{self.sensor}_y_std': 'g',
            f'grav_{self.sensor}_z_std': 'g',
            f'{self.sensor}_x_power_below_gait': 'g^2/Hz',
            f'{self.sensor}_y_power_below_gait': 'g^2/Hz',
            f'{self.sensor}_z_power_below_gait': 'g^2/Hz',
            f'{self.sensor}_x_power_gait': 'g^2/Hz',
            f'{self.sensor}_y_power_gait': 'g^2/Hz',
            f'{self.sensor}_z_power_gait': 'g^2/Hz',
            f'{self.sensor}_x_power_tremor': 'g^2/Hz',
            f'{self.sensor}_y_power_tremor': 'g^2/Hz',
            f'{self.sensor}_z_power_tremor': 'g^2/Hz',
            f'{self.sensor}_x_power_above_tremor': 'g^2/Hz',
            f'{self.sensor}_y_power_above_tremor': 'g^2/Hz',
            f'{self.sensor}_z_power_above_tremor': 'g^2/Hz',
            f'{self.sensor}_x_dominant_frequency': 'Hz',
            f'{self.sensor}_y_dominant_frequency': 'Hz',
            f'{self.sensor}_z_dominant_frequency': 'Hz',
            f'std_norm_acc': 'g',
        }

        for cc_coef in range(1, self.n_coefficients_cc+1):
            self.d_channels_values[f'cc_{cc_coef}_{self.sensor}'] = 'g'

    # TODO: move to higher level config class (duplicate in armswing feature extraction)
    def set_sensor(self, sensor: str) -> None:
        """ Sets the sensor and derived filenames """
        self.sensor: str = sensor
        self.meta_filename: str = f'{self.sensor}_meta.json'
        self.values_filename: str = f'{self.sensor}_samples.bin'
        self.time_filename: str = f'{self.sensor}_time.bin'

    def set_sampling_frequency(self, sampling_frequency: int) -> None:
        """ Sets the sampling frequency and derived variables """
        self.sampling_frequency: int = sampling_frequency
        self.spectrum_low_frequency: int = 0  # Hz
        self.spectrum_high_frequency: int = int(self.sampling_frequency / 2)  # Hz
        self.filter_length: int = self.spectrum_high_frequency - 1


class GaitDetectionConfig:

    def __init__(self) -> None:
        self.classifier_file_name = 'gd_classifier.pkl'
        self.thresholds_file_name = 'gd_threshold.txt'

        self.meta_filename = 'gait_meta.json'
        self.time_filename = 'gait_time.bin'
        self.values_filename = 'gait_values.bin'

        self.l_accel_cols = [DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]

        self.time_colname = 'time'


class ArmSwingFeatureExtractionConfig:

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
            'power_below_gait': [0.3, 0.7],
            'power_gait': [0.7, 3.5],
            'power_tremor': [3.5, 8],
            'power_above_tremor': [8, sampling_frequency]
        }

        # cepstral coefficients
        self.cc_low_frequency = 0
        self.cc_high_frequency = 25
        self.n_dct_filters_cc: int = 20
        self.n_coefficients_cc: int = 12

    def initialize_column_names(self, time_colname='time', pred_gait_colname='pred_gait',
                         angle_smooth_colname='angle_smooth', angle_colname='angle',
                         velocity_colname='velocity', segment_nr_colname='segment_nr') -> None:
        self.time_colname = time_colname
        self.pred_gait_colname = pred_gait_colname
        self.angle_smooth_colname = angle_smooth_colname
        self.angle_colname = angle_colname
        self.velocity_colname = velocity_colname
        self.segment_nr_colname = segment_nr_colname

        self.l_accelerometer_cols: List[str] = [
            DataColumns.ACCELEROMETER_X, 
            DataColumns.ACCELEROMETER_Y, 
            DataColumns.ACCELEROMETER_Z
        ]

        self.l_gyroscope_cols: List[str] = [
            DataColumns.GYROSCOPE_X, 
            DataColumns.GYROSCOPE_Y, 
            DataColumns.GYROSCOPE_Z
        ]
        
        self.l_gravity_cols: List[str] = [f'grav_{x}' for x in self.l_accelerometer_cols]

        self.l_data_point_level_cols = self.l_accelerometer_cols + self.l_gyroscope_cols + self.l_gravity_cols + [
            angle_smooth_colname, velocity_colname
        ]

    def __init__(self) -> None:
        # general
        self.sensor = 'IMU'
        self.units = 'degrees'

        # windowing
        self.window_type = 'hann'
        self.initialize_window_length_fields(3)

        self.initialize_sampling_frequency_fields(100)

        self.initialize_column_names()

        self.d_channels_values = {
            'angle_perc_power': 'proportion',
            'range_of_motion': 'deg',
            'forward_peak_ang_vel_mean': 'deg/s',
            'forward_peak_ang_vel_std': 'deg/s',
            'backward_peak_ang_vel_mean': 'deg/s',
            'backward_peak_ang_vel_std': 'deg/s',
            'std_norm_acc': 'g',
            'grav_accelerometer_x_mean': 'g',
            'grav_accelerometer_x_std': 'g',
            'grav_accelerometer_y_mean': 'g',
            'grav_accelerometer_y_std': 'g',
            'grav_accelerometer_z_mean': 'g',
            'grav_accelerometer_z_std': 'g',
            'accelerometer_x_power_below_gait': 'X', 
            'accelerometer_x_power_gait': 'X',
            'accelerometer_x_power_tremor': 'X',
            'accelerometer_x_power_above_tremor': 'X',
            'accelerometer_x_dominant_frequency': 'Hz',
            'accelerometer_y_power_below_gait': 'X',
            'accelerometer_y_power_gait': 'X',
            'accelerometer_y_power_tremor': 'X',
            'accelerometer_y_power_above_tremor': 'X',
            'accelerometer_y_dominant_frequency': 'Hz',
            'accelerometer_z_power_below_gait': 'X',
            'accelerometer_z_power_gait': 'X',
            'accelerometer_z_power_tremor': 'X',
            'accelerometer_z_power_above_tremor': 'X',
            'accelerometer_z_dominant_frequency': 'Hz',
            'angle_dominant_frequency': 'Hz',
        }

        for sensor in ['accelerometer', 'gyroscope']:
            for cc_coef in range(1, self.n_coefficients_cc+1):
                self.d_channels_values[f'cc_{cc_coef}_{sensor}'] = 'g'


    # TODO: move to higher level config class (duplicate in gait feature extraction)
    def set_sensor(self, sensor: str) -> None:
        """ Sets the sensor and derived filenames """
        self.sensor: str = sensor
        self.meta_filename: str = f'{self.sensor}_meta.json'
        self.values_filename: str = f'{self.sensor}_samples.bin'
        self.time_filename: str = f'{self.sensor}_time.bin'


class ArmSwingDetectionConfig:

    def __init__(self) -> None:
        self.classifier_file_name = 'asd_classifier.pkl'

        self.meta_filename = 'arm_swing_meta.json'
        self.time_filename = 'arm_swing_time.bin'
        self.values_filename = 'arm_swing_values.bin'

        self.l_accel_cols = [DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]
        self.l_gyro_cols = [DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z]


class ArmSwingQuantificationConfig:

    def __init__(self) -> None:
        self.meta_filename = 'arm_swing_meta.json'
        self.time_filename = 'arm_swing_time.bin'
        self.values_filename = 'arm_swing_values.bin'

        self.pred_arm_swing_colname = 'pred_arm_swing'

        self.window_length_s = 3
        self.window_step_size = 0.75
        self.segment_gap_s = 3
        self.min_segment_length_s = 3
