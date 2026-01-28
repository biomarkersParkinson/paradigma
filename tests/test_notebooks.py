from pathlib import Path

import numpy as np
import papermill as pm
import tsdf

arm_swing_binaries_pairs: list[tuple[str, str]] = [
    ("arm_activity_meta.json", "arm_activity_values.bin"),
    ("arm_activity_meta.json", "arm_activity_time.bin"),
]

# Tolerance for the np.allclose function
# Relaxed tolerances for preprocessing operations which involve filtering and resampling
tolerance: float = 1e-5  # Relative tolerance
abs_tol: float = 1e-6  # Absolute tolerance

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

        # Allow small differences in length due to edge effects in resampling/filtering
        # Compare shapes allowing up to 0.5% difference in length
        shape_diff_pct = abs(len(tested_data) - len(ref_data)) / len(ref_data) * 100
        assert (
            shape_diff_pct < 0.5
        ), f"Shape difference too large: {shape_diff_pct:.2f}% (tested: {tested_data.shape}, ref: {ref_data.shape})"

        # Compare overlapping data
        min_len = min(len(tested_data), len(ref_data))
        assert np.allclose(
            tested_data[:min_len], ref_data[:min_len], tolerance, abs_tol
        )
