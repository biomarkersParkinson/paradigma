import os
from typing import Tuple
import numpy as np
import pandas as pd
import papermill as pm

import tsdf

# Step names

# Tolerance for the np.allclose function
tolerance: float = 1e-8
abs_tol: float = 1e-10

notebooks_dir: str = "docs/notebooks/gait"


def test_imu_preprocessing_outputs(shared_datadir):
    """
    This function is used to evaluate the output of the preprocessing function. It evaluates it by comparing the output to a reference output.
    """
    # Notebook step
    step_dir: str = "2.preprocessed_data"

    # pairs of metadata and binary files that are used in the tests
    binaries_pairs: list[tuple[str, str]] = [
        ("gyroscope_meta.json", "gyroscope_samples.bin"),
        # ("gyroscope_meta.json", "gyroscope_time.bin"),
    ]
    compare_data(shared_datadir, step_dir, binaries_pairs)



def compare_data(shared_datadir, step_dir: str, biaries_pairs: list[tuple[str, str]]):
    """
    This function is used to evaluate the output of a notebook. It evaluates it by comparing the output to a reference output.

    Parameters
    ----------
    shared_datadir : str
        The path to the shared data directory.
    biaries_pairs : list[tuple[str, str]]
        The list of pairs of metadata and binary files to compare.
    """
    for metadata, binary in biaries_pairs:

        # load the reference data
        reference_metadata = tsdf.load_metadata_from_path(
            shared_datadir / step_dir / "tremor" / metadata
        )
        ref_metadata_samples = reference_metadata[binary]
        ref_data = tsdf.load_ndarray_from_binary(ref_metadata_samples)
        # load the generated data
        original_metadata = tsdf.load_metadata_from_path(
            shared_datadir / step_dir / "gait" / metadata
        )
        original_metadata_samples = original_metadata[binary]
        original_data = tsdf.load_ndarray_from_binary(original_metadata_samples)

        print(original_data.shape, ref_data.shape)
        # Check if the data is the same
        assert np.allclose(original_data, ref_data, tolerance, abs_tol)
