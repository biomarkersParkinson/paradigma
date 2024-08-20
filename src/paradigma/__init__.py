# read version from installed package
from importlib.metadata import version

__version__ = version("paradigma")

from .constants import DataColumns, TimeUnit, UNIX_TICKS_MS

from .preprocessing_config import IMUPreprocessingConfig, PPGPreprocessingConfig
from .imu_preprocessing import preprocess_imu_data, transform_time_array, resample_data, butterworth_filter

from .gait_analysis_config import GaitFeatureExtractionConfig, GaitDetectionConfig, \
    ArmSwingFeatureExtractionConfig, ArmSwingDetectionConfig, ArmSwingQuantificationConfig
from .gait_analysis import extract_gait_features, detect_gait, extract_arm_swing_features, \
    detect_arm_swing, quantify_arm_swing, aggregate_weekly_arm_swing


__all__ = [
    "DataColumns",
    "TimeUnit",
    "UNIX_TICKS_MS",
    "IMUPreprocessingConfig",
    "PPGPreprocessingConfig",
    "preprocess_imu_data",
    "transform_time_array",
    "resample_data",
    "butterworth_filter",
    "GaitFeatureExtractionConfig",
    "GaitDetectionConfig",
    "ArmSwingFeatureExtractionConfig",
    "ArmSwingDetectionConfig",
    "ArmSwingQuantificationConfig",
    "extract_gait_features",
    "detect_gait",
    "extract_arm_swing_features",
    "detect_arm_swing",
    "quantify_arm_swing",
    "aggregate_weekly_arm_swing",
]
