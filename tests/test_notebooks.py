import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import papermill as pm

import tsdf

# Step names

# Tolerance for the np.allclose function
tolerance: float = 1e-8
abs_tol: float = 1e-10

# Path to the notebooks
notebooks_dir: str = "docs/notebooks/gait"


def create_tmp_folder_if_not_exists(base_path: str):
    """
    This function checks if a folder exists at the specified path and creates it if it does not exist.

    Parameters
    ----------
    base_path : str
        The path to the base directory where the temporary folder should be created.

    Returns
    -------
    str
        The path to the temporary folder.
    """
    tmp_folder_name = "tmp"
    tmp_folder_path = os.path.join(base_path, tmp_folder_name)
    if not os.path.exists(tmp_folder_path):
        os.makedirs(tmp_folder_path)
    return tmp_folder_path


def test_1_imu_preprocessing_outputs(shared_datadir):
    """
    This function is used to evaluate the output of the preprocessing function. It evaluates it by comparing the output to a reference output.
    """
    # Notebook step
    notebook_input_dir_name: str = "1.sensor_data"
    notebook_output_dir_name: str = "2.preprocessed_data"
    data_type: str = "imu"
    notebook_name: str = "1.preprocess_imu"

    input_path = os.path.join(shared_datadir, notebook_input_dir_name, data_type)
    tmp_output_folder = create_tmp_folder_if_not_exists(shared_datadir)
    output_path = os.path.join(tmp_output_folder, notebook_output_dir_name)

    execute_notebook(shared_datadir, notebook_name, input_path, output_path)

    # pairs of metadata and binary files that are used in the tests
    binaries_pairs: list[tuple[str, str]] = [
        ("accelerometer_meta.json", "accelerometer_samples.bin"),
        ("accelerometer_meta.json", "accelerometer_time.bin"),
        ("gyroscope_meta.json", "gyroscope_samples.bin"),
        ("gyroscope_meta.json", "gyroscope_time.bin"),
    ]
    compare_data(shared_datadir, notebook_output_dir_name, binaries_pairs)


def test_2_extract_features_gait_output(shared_datadir):
    """
    This function is used to evaluate the output of the gait feature extraction. It evaluates it by comparing the output to a reference output.
    """

    # Notebook step
    notebook_input_dir_name: str = "2.preprocessed_data"
    notebook_output_dir_name: str = "3.extracted_features"
    data_type: str = "gait"
    notebook_name: str = "2.extract_features_gait"

    # Temporary path to store the output of the notebook
    input_path = os.path.join(shared_datadir, notebook_input_dir_name, data_type)
    tmp_output_folder = create_tmp_folder_if_not_exists(shared_datadir)
    output_path = os.path.join(tmp_output_folder, notebook_output_dir_name)

    execute_notebook(shared_datadir, notebook_name, input_path, output_path)

    binaries_pairs: list[tuple[str, str]] = [
        ("gait_meta.json", "gait_time.bin"),
        ("gait_meta.json", "gait_values.bin"),
    ]
    compare_data(shared_datadir, notebook_output_dir_name, binaries_pairs)


def test_3_gait_detection_output(shared_datadir):
    """
    This function is used to evaluate the output of the gait detection. It evaluates it by comparing the output to a reference output.
    """

    # Notebook step
    notebook_input_dir_name: str = "3.extracted_features"
    notebook_output_dir_name: str = "4.predictions"
    data_type: str = "gait"
    notebook_name: str = "3.gait_detection"

    # Temporary path to store the output of the notebook

    input_path = os.path.join(shared_datadir, notebook_input_dir_name, data_type)
    tmp_output_folder = create_tmp_folder_if_not_exists(shared_datadir)
    output_path = os.path.join(tmp_output_folder, notebook_output_dir_name)

    execute_notebook(shared_datadir, notebook_name, input_path, output_path)

    binaries_pairs: list[tuple[str, str]] = [
        ("gait_meta.json", "gait_time.bin"),
        ("gait_meta.json", "gait_values.bin"),
    ]
    compare_data(shared_datadir, notebook_output_dir_name, binaries_pairs)


def test_4_extract_features_arm_swing_output(shared_datadir):
    """
    This function is used to evaluate the output of the arm swing feature extraction. It evaluates it by comparing the output to a reference output.
    """

    # Notebook step
    notebook_input_dir_name: str = "2.preprocessed_data"
    notebook_output_dir_name: str = "3.extracted_features"
    data_type: str = "gait"
    notebook_name: str = "4.extract_features_arm_swing"

    # Temporary path to store the output of the notebook
    input_path = os.path.join(shared_datadir, notebook_input_dir_name, data_type)
    tmp_output_folder = create_tmp_folder_if_not_exists(shared_datadir)
    output_path = os.path.join(tmp_output_folder, notebook_output_dir_name)

    execute_notebook(shared_datadir, notebook_name, input_path, output_path)
    binaries_pairs: list[tuple[str, str]] = [
        ("arm_swing_meta.json", "arm_swing_values.bin"),
        ("arm_swing_meta.json", "arm_swing_time.bin"),
    ]
    compare_data(shared_datadir, notebook_output_dir_name, binaries_pairs)


def test_5_arm_swing_detection_output(shared_datadir):
    """
    This function is used to evaluate the output of the gait detection. It evaluates it by comparing the output to a reference output.
    """

    # Notebook info
    notebook_input_dir_name: str = "3.extracted_features"
    notebook_output_dir_name: str = "4.predictions"
    data_type: str = "gait"
    notebook_name: str = "5.arm_swing_detection"

    # Temporary path to store the output of the notebook

    input_path = os.path.join(shared_datadir, notebook_input_dir_name, data_type)
    tmp_output_folder = create_tmp_folder_if_not_exists(shared_datadir)
    output_path = os.path.join(tmp_output_folder, notebook_output_dir_name)

    execute_notebook(shared_datadir, notebook_name, input_path, output_path)

    binaries_pairs: list[tuple[str, str]] = [
        ("arm_swing_meta.json", "arm_swing_values.bin"),
        ("arm_swing_meta.json", "arm_swing_time.bin"),
    ]
    compare_data(shared_datadir, notebook_output_dir_name, binaries_pairs)


def test_6_arm_swing_quantification_output(shared_datadir):
    """
    This function is used to evaluate the output of the arm swing quantification. It evaluates it by comparing the output to a reference output.
    """
    shared_datadir = Path("/Users/vedran/git/biomarkers_repo/dbpd-toolbox/tests/data")

    # Notebook step
    notebook_input_dir_name: str = "3.extracted_features"
    notebook_output_dir_name: str = "5.quantification"
    data_type: str = "gait"
    notebook_name: str = "6.arm_swing_quantification"

    # Temporary path to store the output of the notebook
    input_path = os.path.join(shared_datadir, notebook_input_dir_name, data_type)
    tmp_output_folder = create_tmp_folder_if_not_exists(shared_datadir)
    output_path = os.path.join(tmp_output_folder, notebook_output_dir_name)

    execute_notebook(shared_datadir, notebook_name, input_path, output_path)
    binaries_pairs: list[tuple[str, str]] = [
        ("arm_swing_meta.json", "arm_swing_values.bin"),
        ("arm_swing_meta.json", "arm_swing_time.bin"),
    ]
    compare_data(shared_datadir, notebook_output_dir_name, binaries_pairs)


def execute_notebook(
    shared_datadir, notebook_name: str, input_dir: str, output_dir: str
):
    """
    This function is used to execute a notebook.

    Parameters
    ----------
    shared_datadir : str
        The path to the shared data directory.
    notebook_name : str
        The name of the notebook to execute.
    input_dir : str
        The path to the input directory.
    output_dir : str
        The path to the output directory.
    """
    notebook_path = f"{notebooks_dir}/{notebook_name}.ipynb"
    # compute shared_datadir / "tmp" / notebook_output_dir_name / metadata
    notebook_output = f"{shared_datadir}/tmp/{notebook_name}.ipynb"

    path_to_data = f"{shared_datadir}"
    pm.execute_notebook(
        notebook_path,
        notebook_output,
        parameters=dict(
            input_path=input_dir, output_path=output_dir, path_to_data=path_to_data
        ),
    )


def compare_data(
    shared_datadir, notebook_output_dir_name: str, biaries_pairs: list[tuple[str, str]]
):
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
            shared_datadir / notebook_output_dir_name / "gait" / metadata
        )
        ref_metadata_samples = reference_metadata[binary]
        ref_data = tsdf.load_ndarray_from_binary(ref_metadata_samples)
        # load the generated data
        original_metadata = tsdf.load_metadata_from_path(
            shared_datadir / "tmp" / notebook_output_dir_name / metadata
        )
        original_metadata_samples = original_metadata[binary]
        original_data = tsdf.load_ndarray_from_binary(original_metadata_samples)

        print(original_data.shape, ref_data.shape)
        # Check if the data is the same
        assert np.allclose(original_data, ref_data, tolerance, abs_tol)
