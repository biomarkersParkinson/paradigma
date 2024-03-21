from typing import Tuple
import numpy as np
import pandas as pd
import tsdf

# Processing step
step_dir : str = "2.preprocessed_data"

# pairs of constants that are used in the tests
tsdf_binaries_pairs : list[tuple[str, str]] = [
    ("acceleration_meta.json", "acceleration_samples.bin"),
    ("acceleration_meta.json", "acceleration_time.bin"),
    ("rotation_meta.json", "rotation_samples.bin"),
    ("rotation_meta.json", "rotation_time.bin"),
]
# Tolerance for the np.allclose function
tolerance : float = 1e-8
abs_tol : float = 1e-10


def test_imu_preprocessed(shared_datadir):
    """
    The initial test to check if the preprocessing function works as expected. It checks the output dimensions and the type of the output.
    """
    metadata_dict = tsdf.load_metadata_from_path(shared_datadir / '1.sensor_data/PPG_meta.json')

    # Retrieve the metadata object we want, using the name of the binary as key
    metadata_samples = metadata_dict["PPG_time.bin"]
    data = tsdf.load_ndarray_from_binary(metadata_samples)
    assert data.shape == (64775,)


def test_imu_preprocessing_outputs(shared_datadir):
    """
    This function is used to evaluate the output of the preprocessing function. It evaluates it by comparing the output to a reference output.
    """
    for metadata, binary in tsdf_binaries_pairs:
        # load the original data
        original_metadata = tsdf.load_metadata_from_path(shared_datadir / 'tmp' / step_dir / metadata )
        original_metadata_samples = original_metadata[binary]
        original_data = tsdf.load_ndarray_from_binary(original_metadata_samples)
        # load the reference data
        reference_metadata = tsdf.load_metadata_from_path(shared_datadir / step_dir / metadata )
        ref_metadata_samples = reference_metadata[binary]
        ref_data = tsdf.load_ndarray_from_binary(ref_metadata_samples)

        print(original_data.shape, ref_data.shape)
        # Check if the data is the same
        assert np.allclose(original_data, ref_data, tolerance, abs_tol)




