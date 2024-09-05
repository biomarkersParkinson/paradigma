from paradigma.constants import DataColumns


class BasePreprocessingConfig:

    def __init__(self) -> None:

        self.meta_filename = ''
        self.values_filename = ''
        self.time_filename = ''

        self.acceleration_units = 'm/s^2'
        self.rotation_units = 'deg/s'

        self.time_colname = DataColumns.TIME

        # participant information
        self.side_watch = 'right'

        # filtering
        self.sampling_frequency = 100
        self.lower_cutoff_frequency = 0.2
        self.upper_cutoff_frequency = 3.5
        self.filter_order = 4


class IMUPreprocessingConfig(BasePreprocessingConfig):

    def __init__(self) -> None:
        super().__init__()

        self.meta_filename = 'IMU_meta.json'
        self.values_filename = 'IMU_samples.bin'
        self.time_filename = 'IMU_time.bin'

        self.acceleration_units = 'm/s^2'
        self.rotation_units = 'deg/s'

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

class PPGPreprocessingConfig(BasePreprocessingConfig):

    def __init__(self) -> None:
        super().__init__()

        self.meta_filename = 'PPG_meta.json'
        self.values_filename = 'PPG_samples.bin'
        self.time_filename = 'PPG_time.bin'

        self.d_channels_ppg = {
            DataColumns.PPG: 'none'
        }

        self.sampling_frequency = 30
