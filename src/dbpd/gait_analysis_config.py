from typing import Dict, List, Tuple
from dbpd import DataColumns


class GaitFeatureExtractionConfig:

    def __init__(self) -> None:
        self.set_sensor('accelerometer')
        self.set_sampling_frequency(100)

        self.window_type: str = 'hann'
        self.verbose: int = 0
        
        self.window_length_s: int = 6
        self.window_step_size_s: int = 1

        self.n_dct_filters: int = 16
        
        self.d_frequency_bandwidths: Dict[str, List[float]] = {
            'power_below_gait': [0.3, 0.7],
            'power_gait': [0.7, 3.5],
            'power_tremor': [3.5, 8],
            'power_above_tremor': [8, self.sampling_frequency]
        }
        
        self.l_accelerometer_cols: List[str] = [
            DataColumns.ACCELEROMETER_X, 
            DataColumns.ACCELEROMETER_Y, 
            DataColumns.ACCELEROMETER_Z
        ]
        self.l_gravity_cols: List[str] = [f'grav_{x}' for x in self.l_accelerometer_cols]
        self.l_window_level_cols: List[str] = ['id', 'window_nr', 'window_start', 'window_end']
        self.l_data_point_level_cols: List[str] = self.l_accelerometer_cols + self.l_gravity_cols
        
        self.d_channels_values: Dict[str, str] = {
            'grav_accelerometer_x_mean': 'g',
            'grav_accelerometer_y_mean': 'g',
            'grav_accelerometer_z_mean': 'g',
            'grav_accelerometer_x_std': 'g',
            'grav_accelerometer_y_std': 'g',
            'grav_accelerometer_z_std': 'g',
            'accelerometer_x_power_below_gait': 'X',
            'accelerometer_y_power_below_gait': 'X',
            'accelerometer_z_power_below_gait': 'X',
            'accelerometer_x_power_gait': 'X',
            'accelerometer_y_power_gait': 'X',
            'accelerometer_z_power_gait': 'X',
            'accelerometer_x_power_tremor': 'X',
            'accelerometer_y_power_tremor': 'X',
            'accelerometer_z_power_tremor': 'X',
            'accelerometer_x_power_above_tremor': 'X',
            'accelerometer_y_power_above_tremor': 'X',
            'accelerometer_z_power_above_tremor': 'X',
            'accelerometer_x_dominant_frequency': 'Hz',
            'accelerometer_y_dominant_frequency': 'Hz',
            'accelerometer_z_dominant_frequency': 'Hz',
            'std_norm_acc': 'X',
            'cc_1_acc': 'X',
            'cc_2_acc': 'X',
            'cc_3_acc': 'X',
            'cc_4_acc': 'X',
            'cc_5_acc': 'X',
            'cc_6_acc': 'X',
            'cc_7_acc': 'X',
            'cc_8_acc': 'X',
            'cc_9_acc': 'X',
            'cc_10_acc': 'X',
            'cc_11_acc': 'X',
            'cc_12_acc': 'X',
            'cc_13_acc': 'X',
            'cc_14_acc': 'X',
            'cc_15_acc': 'X',
            'cc_16_acc': 'X'
        }

    def set_sensor(self, sensor: str) -> None:
        """ Sets the sensor and derived filenames """
        self.sensor: str = sensor
        self.meta_filename: str = f'{self.sensor}_meta.json'
        self.values_filename: str = f'{self.sensor}_samples.bin'
        self.time_filename: str = f'{self.sensor}_time.bin'

    def set_sampling_frequency(self, sampling_frequency: int) -> None:
        """ Sets the sampling frequency and derived variables """
        self.sampling_frequency: int = sampling_frequency
        self.low_frequency: int = 0  # Hz
        self.high_frequency: int = int(self.sampling_frequency / 2)  # Hz
        self.filter_length: int = self.high_frequency - 1
