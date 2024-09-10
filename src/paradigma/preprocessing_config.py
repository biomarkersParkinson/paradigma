from paradigma.constants import DataColumns, DataUnits
from paradigma.gait_analysis_config import IMUConfig

class BasePreprocessingConfig:

    def __init__(self) -> None:

        self.meta_filename = ''
        self.values_filename = ''
        self.time_filename = ''

        self.acceleration_units = DataUnits.ACCELERATION
        self.rotation_units = DataUnits.ROTATION

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

        IMUConfig.set_filenames(self, 'IMU')
        self.acceleration_units = DataUnits.ACCELERATION
        self.rotation_units = DataUnits.ROTATION

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

        IMUConfig.set_filenames(self, 'PPG')
        self.d_channels_ppg = {
            DataColumns.PPG: DataUnits.NONE
        }

        self.sampling_frequency = 30
