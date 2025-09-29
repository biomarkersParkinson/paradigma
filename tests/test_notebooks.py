from pathlib import Path

import numpy as np
import papermill as pm
import tsdf

arm_swing_binaries_pairs: list[tuple[str, str]] = [
    ("arm_activity_meta.json", "arm_activity_values.bin"),
    ("arm_activity_meta.json", "arm_activity_time.bin"),
]

# Tolerance for the np.allclose function
tolerance: float = 1e-8
abs_tol: float = 1e-10

# Path to the notebooks
notebooks_dir: str = "tests/notebooks/gait"


def execute_notebook(name: str, parameters: dict[str, str]):
    """
    This function is used to execute a notebook.

    Parameters
    ----------
    shared_datadir : Path
        The path to the shared data directory.
    name : str
        The name of the notebook to execute.
    parameters : dict[str, str]
        The parameters to pass to the notebook.
    """
    path = f"{notebooks_dir}/{name}.ipynb"
    # compute shared_datadir / "tmp" / output_dir_name / metadata
    output = None

    pm.execute_notebook(
        path,
        output,
        parameters=parameters,
    )


def compare_data(
    reference_dir: Path, tested_dir: Path, binaries_pairs: list[tuple[str, str]]
):
    """
    This function is used to evaluate the output of a notebook. It evaluates it by comparing the output to a reference output.

    Parameters
    ----------
    reference_dir : Path
        The path to the reference data.
    tested_dir : Path
        The name of the output directory.
    binaries_pairs : list[tuple[str, str]]
        The list of pairs of metadata and binary files to compare.
    """
    for metadata, binary in binaries_pairs:

        # load the reference data
        reference_metadata = tsdf.load_metadata_from_path(reference_dir / metadata)
        ref_metadata_values = reference_metadata[binary]
        ref_data = tsdf.load_ndarray_from_binary(ref_metadata_values)
        # load the generated data

        tested_metadata = tsdf.load_metadata_from_path(tested_dir / metadata)
        tested_metadata_values = tested_metadata[binary]
        tested_data = tsdf.load_ndarray_from_binary(tested_metadata_values)

        print(tested_data.shape, ref_data.shape)
        # Check if the data is the same
        assert np.allclose(tested_data, ref_data, tolerance, abs_tol)
