class DataColumns:
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

class DataUnits:
    """
    Enum for the data channel unit types in tsdf.
    """
    ACCELERATION = "m/s^2"
    """ The acceleration is in m/s^2. """
    ROTATION = "deg/s"
    """ The rotation is in degrees per second. """
    

class TimeUnit:
    """
    Enum for the `time` channel unit types in tsdf.
    """
    relative_ms = "relative_ms"
    """ The time is relative to the start time in milliseconds. """
    absolute_ms = "absolute_ms"
    """ The time is absolute in milliseconds. """
    difference_ms = "difference_ms"
    """ The time is the difference between consecutive samples in milliseconds. """

UNIX_TICKS_MS: int = 1000
