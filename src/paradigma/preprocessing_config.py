from paradigma.constants import DataColumns, DataUnits
from paradigma.gait.gait_analysis_config import IMUConfig

class BasePreprocessingConfig:

    def __init__(self) -> None:

        self.meta_filename = ''
        self.values_filename = ''
        self.time_filename = ''

        self.time_colname = DataColumns.TIME

    def set_filenames(self, prefix: str) -> None:
        """Sets the filenames based on the prefix. This method is duplicated from `gaits_analysis_config.py`.
        
        Parameters
        ----------
        prefix : str
            The prefix for the filenames.
        """
        self.meta_filename = f"{prefix}_meta.json"
        self.time_filename = f"{prefix}_time.bin"
        self.values_filename = f"{prefix}_samples.bin"

class IMUPreprocessingConfig(BasePreprocessingConfig):

    def __init__(self) -> None:
        super().__init__()

        self.set_filenames('IMU')
        self.acceleration_units = DataUnits.ACCELERATION
        self.rotation_units = DataUnits.ROTATION

        self.accelerometer_cols = [DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]
        self.gyroscope_cols = [DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z]

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
        
        # participant information
        self.side_watch = 'right'

        # filtering
        self.sampling_frequency = 100
        self.lower_cutoff_frequency = 0.2
        self.upper_cutoff_frequency = 3.5
        self.filter_order = 4

class GyroPreprocessingConfig(BasePreprocessingConfig):

    def __init__(self) -> None:
        super().__init__()

        self.set_filenames('IMU')

        self.rotation_units = DataUnits.ROTATION
        self.gyroscope_cols = [DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z]

        self.d_channels_gyroscope = {
            DataColumns.GYROSCOPE_X: self.rotation_units,
            DataColumns.GYROSCOPE_Y: self.rotation_units,
            DataColumns.GYROSCOPE_Z: self.rotation_units,
        }

        self.sampling_frequency = 100

class PPGPreprocessingConfig(BasePreprocessingConfig):

    def __init__(self) -> None:
        super().__init__()

        self.set_filenames('PPG')
        self.d_channels_ppg = {
            DataColumns.PPG: DataUnits.NONE
        }

        # filtering
        self.sampling_frequency = 30
        self.lower_cutoff_frequency = 0.4
        self.upper_cutoff_frequency = 3.5
        self.filter_order = 4
        
