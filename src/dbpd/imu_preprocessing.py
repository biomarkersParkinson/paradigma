import math
import numpy as np
import pandas as pd

from datetime import datetime
from scipy import signal
from scipy.interpolate import CubicSpline


class PreprocessingPipeline():
    def __init__(self,
                 df_sensors: pd.DataFrame,
                 time_column: str,
                 sampling_frequency: int,
                 resampling_frequency: int,
                 verbose: int):
        
        self.verbose = verbose
        self.df_sensors = df_sensors
        self.time_column = time_column
        self.sampling_frequency = sampling_frequency
        self.resampling_frequency = resampling_frequency


    def transform_time_array(self, scale_factor, do_convert_to_abs_time):
        """ Optionally transforms the time array to absolute time and scales the values
        """
        if do_convert_to_abs_time:
            return np.cumsum(np.double(self.df_sensors[self.time_column])) / scale_factor
        return self.df_sensors[self.time_column] / 1000.0


    def resample_data(self, scale_factors):

        l_imu_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'rotation_x', 'rotation_y', 'rotation_z']

        # scale data
        scaled_values = self.df_sensors[l_imu_cols] * scale_factors

        # resample
        t_resampled = np.arange(0, self.df_sensors[self.time_column][-1:].values[0], 1/self.resampling_frequency)

        # create dataframe
        df = pd.DataFrame(t_resampled, columns=[self.time_column])

        # interpolate IMU
        for sensor_col in l_imu_cols:
            cs = CubicSpline(self.df_sensors[self.time_column], scaled_values[sensor_col])
            df[sensor_col] = cs(df[self.time_column])

        return df


    def butterworth_filter(
            self,
            sensor_col: str,
            order: int,
            cutoff_frequency: float,
            passband: str,
            ):
        """Applies the Butterworth filter to a single sensor column
        
        Parameters
        ----------
        sensor_column: pd.Series
            A single column containing sensor data in float format
        frequency: int
            The sampling frequency of sensor_column in Hz
        order: int
            The exponential order of the filter
        cutoff_frequency: float
            The frequency at which the gain drops to 1/sqrt(2) that of the passband
        passband: str
            Type of passband: ['hp' or 'lp']
        verbose: bool
            The verbosity of the output
            
        Returns
        -------
        sensor_column_filtered: pd.Series
            The origin sensor column filtered applying a Butterworth filter"""

        sos = signal.butter(N=order, Wn=cutoff_frequency, btype=passband, analog=False, fs=self.resampling_frequency, output='sos')
        return signal.sosfilt(sos, self.df_sensors[sensor_col])
    

    def create_window(self,
                      df: pd.DataFrame,
                      window_nr: int,
                      lower_index: int,
                      upper_index: int,
                      data_point_level_cols: list
                      ):
        """Transforms (a subset of) a dataframe into a single row

        Parameters
        ----------
        df_sensor: pd.DataFrame
            The original dataframe to be windowed
        subject: str
            The identification of the participant
        window_nr: int
            The identification of the window
        window_length: int
            The number of samples a window constitutes
        lower_index: int
            The dataframe index of the first sample to be windowed
        upper_index: int
            The dataframe index of the final sample to be windowed
        data_point_level_cols: list
            The columns in sensor_df that are to be kept as individual datapoints in a list instead of aggregates
        verbose: bool
            The verbosity of the output

        Returns
        -------
        l_subset_squeezed: list
            Rows corresponding to single windows
        """
        df_subset = df.loc[lower_index:upper_index, data_point_level_cols].copy()
        l_subset_squeezed = [self.id, window_nr+1, lower_index, upper_index] + df_subset.values.T.tolist()

        return l_subset_squeezed
    

    def tabulate_windows(self,
                        data_point_level_cols: list,
                        ):
        """Compiles multiple windows into a single dataframe

        Parameters
        ----------
        df_sensor: pd.DataFrame
            The original dataframe to be windowed
        subject: str
            The identification of the participant
        window_length: int
            The number of samples a window constitutes
        step_size: int
            The number of samples between the start of the previous and the start of the next window
        data_point_level_cols: list
            The columns in sensor_df that are to be kept as individual datapoints in a list instead of aggregates
        verbose: bool
            The verbosity of the output

        Returns
        -------
        df_windowed: pd.DataFrame
            Dataframe with each row corresponding to an individual window
        """

        self.df_sensors = self.df_sensors.reset_index(drop=True)

        if self.window_step_size <= 0:
            raise Exception("Step size should be larger than 0.")
        if self.window_length > self.df_sensors.shape[0]:
            return 
        
        # self.df_sensors = self.df_sensors.reset_index(drop=True)

        # self.df_sensors['segment_nr'] = self.create_segments(self.df_sensors, TIME_COLUMN)

        l_windows = []
        n_windows = math.floor(
            (self.df_sensors.shape[0] - self.window_length) / 
            self.window_step_size
            ) + 1

        for window_nr in range(n_windows):
            lower = window_nr * self.window_step_size
            upper = window_nr * self.window_step_size + self.window_length - 1
            l_windows.append(self.create_window(self.df_sensors, window_nr, lower, upper, data_point_level_cols))

        df_windows = pd.DataFrame(l_windows, columns=['id', 'window_nr', 'window_start', 'window_end'] + data_point_level_cols)
                
        return df_windows.reset_index(drop=True)