# read version from installed package
from importlib.metadata import version

__version__ = version("paradigma")

# Import main pipeline runner function
from paradigma.pipeline import list_available_pipelines, run_pipeline

__all__ = ["run_pipeline", "list_available_pipelines"]
