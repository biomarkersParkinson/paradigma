from pathlib import Path

from paradigma.testing import preprocess_imu_data_io
from paradigma.config import IMUConfig
from test_notebooks import compare_data


# Mappings between the metadata and binary files
accelerometer_binaries_pairs: list[tuple[str, str]] = [
        ("accelerometer_meta.json", "accelerometer_values.bin"),
        ("accelerometer_meta.json", "accelerometer_time.bin"),
    ]
gyroscope_binaries_pairs: list[tuple[str, str]] = [
        ("gyroscope_meta.json", "gyroscope_values.bin"),
        ("gyroscope_meta.json", "gyroscope_time.bin"),
    ]
imu_binaries_pairs: list[tuple[str, str]] = accelerometer_binaries_pairs + gyroscope_binaries_pairs


def test_1_imu_preprocessing_outputs(shared_datadir: Path):
    """
    This function is used to evaluate the output of the preprocessing function. It evaluates it by comparing the output to a reference output.
    """
    input_dir_name: str = "1.prepared_data"
    output_dir_name: str = "2.preprocessed_data"

    path_to_imu_input = shared_datadir / input_dir_name / "imu"
    path_to_reference_output = shared_datadir / output_dir_name / "imu"
    path_to_tested_output = path_to_reference_output / "test-output"

    config = IMUConfig()
    preprocess_imu_data_io(path_to_imu_input, path_to_tested_output, config, sensor='both', watch_side='left')
    compare_data(path_to_reference_output, path_to_tested_output, imu_binaries_pairs)