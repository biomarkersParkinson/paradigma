from dataclasses import dataclass


@dataclass(frozen=True)
class DataColumns():
    """
    Enum for the data channels in tsdf.
    """
    ACCELEROMETER_X = "accelerometer_x"
    ACCELEROMETER_Y = "accelerometer_y"
    ACCELEROMETER_Z = "accelerometer_z"
    GYROSCOPE_X = "gyroscope_x"
    GYROSCOPE_Y = "gyroscope_y"
    GYROSCOPE_Z = "gyroscope_z"
    PPG = "green"
    TIME = "time"

    # The following are used in gait analysis
    GRAV_ACCELEROMETER_X = "grav_accelerometer_x"
    GRAV_ACCELEROMETER_Y = "grav_accelerometer_y"
    GRAV_ACCELEROMETER_Z = "grav_accelerometer_z"
    PRED_GAIT = "pred_gait"
    ANGLE = "angle"
    ANGLE_SMOOTH = "angle_smooth"
    VELOCITY = "velocity"
    SEGMENT_NR = "segment_nr"
    
@dataclass(frozen=True)
class DataUnits():
    """
    Enum for the data channel unit types in tsdf.
    """
    ACCELERATION = "m/s^2"
    """ The acceleration is in m/s^2. """
    ROTATION = "deg/s"
    """ The rotation is in degrees per second. """

@dataclass(frozen=True)
class TimeUnit():
    """
    Enum for the `time` channel unit types in tsdf.
    """
    RELATIVE_MS = "relative_ms"
    """ The time is relative to the start time in milliseconds. """
    ABSOLUTE_MS = "absolute_ms"
    """ The time is absolute in milliseconds. """
    DIFFERENCE_MS = "difference_ms"
    """ The time is the difference between consecutive samples in milliseconds. """

UNIX_TICKS_MS: int = 1000
