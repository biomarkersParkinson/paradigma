"""
ParaDigMa: Parkinson Digital Biomarker Analysis Toolbox
"""

# read version from installed package
from importlib.metadata import version

__version__ = version("paradigma")

# Import main pipeline functions for easy access
from .orchestrator import run_paradigma

__all__ = [
    "run_paradigma",
]
