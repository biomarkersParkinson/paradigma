from pathlib import Path

from paradigma.imu_preprocessing import preprocess_imu_data_io
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
    input_dir_name: str = "1.sensor_data"
    output_dir_name: str = "2.preprocessed_data"

    input_path = shared_datadir / input_dir_name / "imu"
    reference_output_path = shared_datadir / output_dir_name / "imu"
    tested_output_path = reference_output_path / "test-output"

    config = IMUConfig()
    preprocess_imu_data_io(input_path, tested_output_path, config)
    compare_data(reference_output_path, tested_output_path, imu_binaries_pairs)