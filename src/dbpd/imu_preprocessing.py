import math
import numpy as np
import pandas as pd

from datetime import datetime
from scipy import signal, fft
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
        l_subset_squeezed = [window_nr+1, lower_index, upper_index] + df_subset.values.T.tolist()

        return l_subset_squeezed
    

    def tabulate_windows(self,
                         window_step_size: int,
                         window_length: int,
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

        if window_step_size <= 0:
            raise Exception("Step size should be larger than 0.")
        if window_length > self.df_sensors.shape[0]:
            return 

        l_windows = []
        n_windows = math.floor(
            (self.df_sensors.shape[0] - window_length) / 
            window_step_size
            ) + 1

        for window_nr in range(n_windows):
            lower = window_nr * window_step_size
            upper = window_nr * window_step_size + window_length - 1
            l_windows.append(self.create_window(self.df_sensors, window_nr, lower, upper, data_point_level_cols))

        df_windows = pd.DataFrame(l_windows, columns=['window_nr', 'window_start', 'window_end'] + data_point_level_cols)
                
        return df_windows.reset_index(drop=True)
    

    def generate_statistics(self,
                            sensor_col: str,
                            statistic: str
                            ):
        if statistic == 'mean':
            return self.df_windows.apply(lambda x: np.mean(x[sensor_col]), axis=1)
        elif statistic == 'std':
            return self.df_windows.apply(lambda x: np.std(x[sensor_col]), axis=1)
        elif statistic == 'max':
            return self.df_windows.apply(lambda x: np.max(x[sensor_col]), axis=1)
        elif statistic == 'min':
            return self.df_windows.apply(lambda x: np.min(x[sensor_col]), axis=1)
        

    def generate_std_norm(
                          self,
                          cols: list,
                         ):
        return self.df_windows.apply(
            lambda x: np.std(np.sqrt(sum(
            [np.array([y**2 for y in x[col]]) for col in cols]
            ))), axis=1)
    

    def compute_fft(self,
                    values: list,
                    window_type: str,
                    ):

        w = signal.get_window(window_type, len(values), fftbins=False)
        yf = 2*fft.fft(values*w)[:int(len(values)/2+1)]
        xf = fft.fftfreq(len(values), 1/self.resampling_frequency)[:int(len(values)/2+1)]

        return yf, xf
    

    def signal_to_ffts(self,
                       sensor_col: str,
                       window_type: str,
                       ):
        l_values_total = []
        l_freqs_total = []
        for _, row in self.df_windows.iterrows():
            l_values, l_freqs = self.compute_fft(
                values=row[sensor_col],
                window_type=window_type)
            l_values_total.append(l_values)
            l_freqs_total.append(l_freqs)

        return l_freqs_total, l_values_total
    

    def compute_power_in_bandwidth(self,
                                   sensor_col,
                                   window_type: str,
                                   fmin: int,
                                   fmax: int,
                                   ):
        fxx, pxx = signal.periodogram(sensor_col, fs=self.resampling_frequency, window=window_type)
        ind_min = np.argmax(fxx > fmin) - 1
        ind_max = np.argmax(fxx > fmax) - 1
        return np.log10(np.trapz(pxx[ind_min:ind_max], fxx[ind_min:ind_max]))
    

    def get_dominant_frequency(
            self,
            signal_ffts: pd.Series,
            signal_freqs: pd.Series,
            fmin: int,
            fmax: int
            ):
        
        valid_indices = np.where((signal_freqs>fmin) & (signal_freqs<fmax))
        signal_freqs_adjusted = signal_freqs[valid_indices]
        signal_ffts_adjusted = signal_ffts[valid_indices]

        idx = np.argmax(np.abs(signal_ffts_adjusted))
        return np.abs(signal_freqs_adjusted[idx])
    

    def compute_power(self,
                      fft_cols: list
                      ):
        df = self.df_windows.copy()
        for col in fft_cols:
            df['{}_power'.format(col)] = df[col].apply(lambda x: np.square(np.abs(x)))

        return df.apply(lambda x: sum([np.array([y for y in x[col+'_power']]) for col in fft_cols]), axis=1)
    

    def generate_cepstral_coefficients(
            self,
            window_length: int,
            total_power_col: str,
            low_frequency: int,
            high_frequency: int,
            filter_length: int,
            n_dct_filters: int,
            ):
        
        # compute filter points
        freqs = np.linspace(low_frequency, high_frequency, num=filter_length+2)
        filter_points = np.floor((window_length + 1) / self.resampling_frequency * freqs).astype(int)  

        # construct filterbank
        filters = np.zeros((len(filter_points)-2, int(window_length/2+1)))
        for j in range(len(filter_points)-2):
            filters[j, filter_points[j] : filter_points[j+1]] = np.linspace(0, 1, filter_points[j+1] - filter_points[j])
            filters[j, filter_points[j+1] : filter_points[j+2]] = np.linspace(1, 0, filter_points[j+2] - filter_points[j+1])

        # filter signal
        power_filtered = self.df_windows[total_power_col].apply(lambda x: np.dot(filters, x))
        log_power_filtered = power_filtered.apply(lambda x: 10.0 * np.log10(x))

        # generate cepstral coefficients
        dct_filters = np.empty((n_dct_filters, filter_length))
        dct_filters[0, :] = 1.0 / np.sqrt(filter_length)

        samples = np.arange(1, 2 * filter_length, 2) * np.pi / (2.0 * filter_length)

        for i in range(1, n_dct_filters):
            dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_length)

        cepstral_coefs = log_power_filtered.apply(lambda x: np.dot(dct_filters, x))

        return pd.DataFrame(np.vstack(cepstral_coefs), columns=['cc_{}'.format(j+1) for j in range(n_dct_filters)])