# read version from installed package
from importlib.metadata import version

__version__ = version("paradigma")

from .imu_preprocessing import *

__all__ = ["PreprocessingPipelineConfig"]
