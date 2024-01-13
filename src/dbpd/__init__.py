# read version from installed package
from importlib.metadata import version

__version__ = version("dbpd")

from .imu_preprocessing import PreprocessingPipelineConfig

__all__ = ["PreprocessingPipelineConfig"]
