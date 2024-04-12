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

def test_imu_preprocessed(shared_datadir):
    """
    The initial test to check if the preprocessing function works as expected. It checks the output dimensions and the type of the output.
    """
    metadata_dict = tsdf.load_metadata_from_path(
        shared_datadir / "1.sensor_data/ppg/PPG_meta.json"
    )

    # Retrieve the metadata object we want, using the name of the binary as key
    metadata_samples = metadata_dict["PPG_time.bin"]
    data = tsdf.load_ndarray_from_binary(metadata_samples)
    assert data.shape == (64775,)

def test_imu_preprocessing_outputs(shared_datadir):
    """
    This function is used to evaluate the output of the preprocessing function. It evaluates it by comparing the output to a reference output.
    """
    # Notebook step
    step_dir: str = "2.preprocessed_data"

    input_path = os.path.join(shared_datadir, '1.sensor_data/imu')
    new_output_path = os.path.join(shared_datadir, "tmp", step_dir)

    execute_notebook(shared_datadir, "1.preprocess_imu", input_path, new_output_path)

    # pairs of metadata and binary files that are used in the tests
    binaries_pairs: list[tuple[str, str]] = [
        ("acceleration_meta.json", "acceleration_samples.bin"),
        ("acceleration_meta.json", "acceleration_time.bin"),
        # ("gyroscope_meta.json", "gyroscope_samples.bin"),
        # ("gyroscope_meta.json", "gyroscope_time.bin"),
    ]
    compare_data(shared_datadir, step_dir, binaries_pairs)


def test_extract_features_gait_output(shared_datadir):
    """
    This function is used to evaluate the output of the gait feature extraction. It evaluates it by comparing the output to a reference output.
    """

    # Notebook step
    step_dir: str = "3.extracted_features"

    # Temporary path to store the output of the notebook
    input_path = os.path.join(shared_datadir, "2.preprocessed_data/gait")
    new_output_path = os.path.join(shared_datadir, "tmp", step_dir)

    execute_notebook(shared_datadir, "2.extract_features_gait", input_path, new_output_path)

    binaries_pairs: list[tuple[str, str]] = [
        ("gait_meta.json", "gait_time.bin"),
        ("gait_meta.json", "gait_values.bin"),
    ]
    compare_data(shared_datadir, step_dir, binaries_pairs)


def test_extract_features_arm_swing_output(shared_datadir):
    """
    This function is used to evaluate the output of the arm swing feature extraction. It evaluates it by comparing the output to a reference output.
    """

    # Notebook step
    step_dir: str = "3.extracted_features"

    # Temporary path to store the output of the notebook
    input_path = os.path.join(shared_datadir, "2.preprocessed_data/gait")
    new_output_path = os.path.join(shared_datadir, "tmp", step_dir)

    execute_notebook(shared_datadir, "4.extract_features_arm_swing", input_path, new_output_path)
    binaries_pairs: list[tuple[str, str]] = [
        ("arm_swing_meta.json", "arm_swing_values.bin"),
        ("arm_swing_meta.json", "arm_swing_time.bin"),
    ]
    compare_data(shared_datadir, step_dir, binaries_pairs)


def execute_notebook(shared_datadir, notebook_name:str, input_dir: str, output_dir: str):
    notebook_path = f"{notebooks_dir}/{notebook_name}.ipynb"
    # compute shared_datadir / "tmp" / step_dir / metadata
    notebook_output = f"{shared_datadir}/tmp/{notebook_name}.ipynb"

    pm.execute_notebook(
        notebook_path,
        notebook_output,
        parameters=dict(input_path = input_dir,output_path = output_dir),
    )
    

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
            shared_datadir / step_dir / "gait" / metadata
        )
        ref_metadata_samples = reference_metadata[binary]
        ref_data = tsdf.load_ndarray_from_binary(ref_metadata_samples)
        # load the generated data
        original_metadata = tsdf.load_metadata_from_path(
            shared_datadir / "tmp" / step_dir / metadata
        )
        original_metadata_samples = original_metadata[binary]
        original_data = tsdf.load_ndarray_from_binary(original_metadata_samples)

        print(original_data.shape, ref_data.shape)
        # Check if the data is the same
        assert np.allclose(original_data, ref_data, tolerance, abs_tol)
