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

    # The following are used in gait analysis
    GRAV_ACCELEROMETER_X : str = "grav_accelerometer_x"
    GRAV_ACCELEROMETER_Y : str = "grav_accelerometer_y"
    GRAV_ACCELEROMETER_Z : str = "grav_accelerometer_z"
    PRED_GAIT : str = "pred_gait"
    PRED_ARM_SWING : str = "pred_arm_swing"
    ANGLE : str = "angle"
    ANGLE_SMOOTH : str = "angle_smooth"
    VELOCITY : str = "velocity"
    SEGMENT_NR : str = "segment_nr"
    
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
    ABSOLUTE_MS : str = "absolute_ms"
    """ The time is absolute in milliseconds. """
    DIFFERENCE_MS : str = "difference_ms"
    """ The time is the difference between consecutive samples in milliseconds. """

UNIX_TICKS_MS: int = 1000
