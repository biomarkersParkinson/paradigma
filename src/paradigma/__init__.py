"""
ParaDigMa: Parkinson Digital Biomarker Analysis Toolbox
"""

# read version from installed package
from importlib.metadata import version

__version__ = version("paradigma")

# Import data loading and preparation functions
from .load import (
    load_data_files,
    load_prepared_data,
)

# Import main pipeline functions for easy access
from .orchestrator import run_paradigma
from .pipelines.gait_pipeline import run_gait_pipeline
from .prepare_data import prepare_raw_data

__all__ = [
    "run_paradigma",
    "run_gait_pipeline",
    "run_tremor_pipeline",
    "run_pulse_rate_pipeline",
    "load_data_files",
    "load_prepared_data",
    "prepare_raw_data",
]
