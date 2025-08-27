from dataclasses import dataclass


@dataclass(frozen=True)
class DataColumns():
    """
    Class containing the data channels in `tsdf`.
    """
    ACCELEROMETER_X : str = "accelerometer_x"
    ACCELEROMETER_Y : str = "accelerometer_y"
    ACCELEROMETER_Z : str = "accelerometer_z"
    GYROSCOPE_X : str = "gyroscope_x"
    GYROSCOPE_Y : str = "gyroscope_y"
    GYROSCOPE_Z : str = "gyroscope_z"
    PPG : str = "green"
    TIME : str = "time"
    SEGMENT_NR : str = "segment_nr"
    SEGMENT_CAT: str = "segment_category"

    # Gait 
    GRAV_ACCELEROMETER_X : str = "accelerometer_x_grav"
    GRAV_ACCELEROMETER_Y : str = "accelerometer_y_grav"
    GRAV_ACCELEROMETER_Z : str = "accelerometer_z_grav"
    PRED_GAIT_PROBA: str = "pred_gait_proba"
    PRED_GAIT : str = "pred_gait"
    PRED_NO_OTHER_ARM_ACTIVITY_PROBA: str = "pred_no_other_arm_activity_proba"
    PRED_NO_OTHER_ARM_ACTIVITY : str = "pred_no_other_arm_activity"
    ANGLE : str = "angle"
    VELOCITY : str = "velocity"
    DOMINANT_FREQUENCY: str = "dominant_frequency"
    RANGE_OF_MOTION: str = "range_of_motion"
    PEAK_VELOCITY: str = "peak_velocity"

    # The following are used in tremor analysis
    PRED_TREMOR_PROBA: str = "pred_tremor_proba"
    PRED_TREMOR_LOGREG : str = "pred_tremor_logreg"
    PRED_TREMOR_CHECKED : str = "pred_tremor_checked"
    PRED_ARM_AT_REST: str = "pred_arm_at_rest"

    # Constants for PPG features
    VARIANCE: str = "variance"
    MEAN: str = "mean"
    MEDIAN: str = "median"
    KURTOSIS: str = "kurtosis"
    SKEWNESS: str = "skewness"
    DOMINANT_FREQUENCY: str = "dominant_frequency"
    RELATIVE_POWER: str = "relative_power"
    SPECTRAL_ENTROPY: str = "spectral_entropy"
    SIGNAL_NOISE_RATIO: str = "signal_noise_ratio"
    SECOND_HIGHEST_PEAK: str = "second_highest_peak"
    POWER_RATIO: str = "power_ratio"

    # Constants for PPG SQA feature using accerometer data
    ACC_POWER_RATIO: str = "acc_power_ratio"

    # Constants for SQA classification
    PRED_SQA_PROBA: str = "pred_sqa_proba"
    PRED_SQA_ACC_LABEL: str = "pred_sqa_acc_label"
    PRED_SQA: str = "pred_sqa"

    # Constants for pulse rate
    PULSE_RATE: str = "pulse_rate"
    
@dataclass(frozen=True)
class DataUnits():
    """
    Class containing the data channel unit types in `tsdf`.
    """
    ACCELERATION: str = "m/s^2"
    """ The acceleration is in m/s^2. """
    
    ROTATION: str = "deg/s"
    """ The rotation is in degrees per second. """
    
    GRAVITY: str = "g"
    """ The acceleration due to gravity is in g. """
    
    POWER_SPECTRAL_DENSITY: str = "g^2/Hz"
    """ The power spectral density is in g^2/Hz. """
    
    FREQUENCY: str = "Hz"
    """ The frequency is in Hz. """

    NONE: str = "none"
    """ The data channel has no unit. """
    

@dataclass(frozen=True)
class TimeUnit():
    """
    Class containing the `time` channel unit types in `tsdf`.
    """
    RELATIVE_MS : str = "relative_ms"
    """ The time is relative to the start time in milliseconds. """
    RELATIVE_S : str = "relative_s"
    """ The time is relative to the start time in seconds. """
    ABSOLUTE_MS : str = "absolute_ms"
    """ The time is absolute in milliseconds. """
    ABSOLUTE_S : str = "absolute_s"
    """ The time is absolute in seconds. """
    DIFFERENCE_MS : str = "difference_ms"
    """ The time is the difference between consecutive samples in milliseconds. """
    DIFFERENCE_S : str = "difference_s"
    """ The time is the difference between consecutive samples in seconds. """

UNIX_TICKS_MS: int = 1000
