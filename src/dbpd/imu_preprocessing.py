from datetime import datetime
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline


class PreprocessingPipeline:
    def __init__(
        self,
        df_sensors: pd.DataFrame,
        time_column: str,
        sampling_frequency: int,
        resampling_frequency: int,
        verbose: int,
    ):
        self.verbose = verbose
        self.df_sensors = df_sensors
        self.time_column = time_column
        self.sampling_frequency = sampling_frequency
        self.resampling_frequency = resampling_frequency

    def transform_time_array(
        self, time_array, scale_factor, do_convert_to_abs_time: bool
    ) -> np.ndarray:
        """Transforms the time array to absolute time (when required) and scales the values.
        TODO: This function outputs relative time, and not absolute time as far as I (Vedran) can see. This should be fixed.
        :param time_array: The time array to transform
        :param scale_factor: The scale factor to apply to the time array
        :param do_convert_to_abs_time: Whether to convert the time array to absolute time
        :return: The transformed time array.
        """
        if do_convert_to_abs_time:
            return np.cumsum(np.double(time_array)) / scale_factor
        return time_array / 1000.0

    def resample_data(self, time_abs_array, values_unscaled, scale_factors):
        """Resamples the data to the resampling frequency. The data is scaled before resampling.
        :param time_abs_array: The absolute time array.
        :param values_unscaled: The values to resample.
        :param scale_factors: The scale factors to apply to the values.
        :return: The resampled data.
        """

        # scale data
        scaled_values = values_unscaled * scale_factors

        # resample
        t_resampled = np.arange(0, time_abs_array[-1], 1 / self.resampling_frequency)

        # create dataframe
        df = pd.DataFrame(t_resampled, columns=[self.time_column])

        # interpolate IMU
        for j, sensor_col in enumerate(
            [
                "acceleration_x",
                "acceleration_y",
                "acceleration_z",
                "rotation_x",
                "rotation_y",
                "rotation_z",
            ]
        ):
            cs = CubicSpline(time_abs_array, scaled_values.T[j])
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

        sos = signal.butter(
            N=order,
            Wn=cutoff_frequency,
            btype=passband,
            analog=False,
            fs=self.resampling_frequency,
            output="sos",
        )
        return signal.sosfilt(sos, self.df_sensors[sensor_col])
