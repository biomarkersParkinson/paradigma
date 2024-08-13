from pathlib import Path

from dbpd.preprocessing_config import PPGPreprocessingConfig, IMUPreprocessingConfig
from dbpd.ppg_preprocessing import *
from dbpd.imu_preprocessing import *
from dbpd.gait_analysis import *
from dbpd.gait_analysis_config import *
from test_notebooks import compare_data


# Mappings between the metadata and binary files

accelerometer_binaries_pairs: list[tuple[str, str]] = [
        ("accelerometer_meta.json", "accelerometer_samples.bin"),
        ("accelerometer_meta.json", "accelerometer_time.bin"),
]
accelerometer_filt_binaries_pairs: list[tuple[str, str]] = [
        ("accelerometer_filt_meta.json", "accelerometer_filt_samples.bin"),
        ("accelerometer_filt_meta.json", "accelerometer_filt_time.bin"),
]
ppg_binaries_pairs: list[tuple[str, str]] = [
        ("PPG_meta.json", "PPG_samples.bin"),
        ("PPG_meta.json", "PPG_time.bin"),
]
all_binaries_pairs: list[tuple[str, str]] = accelerometer_binaries_pairs + ppg_binaries_pairs


def test_accel_preprocessing(shared_datadir: Path):
    """
    This function is used to evaluate the output of the PPG pipeline preprocessing function. It evaluates it by comparing the acceleration data output to the PPG reference output.
    """
    input_dir_name: str = "1.sensor_data"
    output_dir_name: str = "2.preprocessed_data"

    input_path_imu = shared_datadir / input_dir_name / "imu"
    input_path_ppg = shared_datadir / input_dir_name / "ppg"
    reference_output_path = shared_datadir / output_dir_name / "ppg"
    tested_output_path = reference_output_path / "test-output"

    ppg_config = PPGPreprocessingConfig()
    imu_config = IMUPreprocessingConfig()
    metadatas_ppg, metadatas_imu = scan_and_sync_segments(input_path_ppg, input_path_imu)
    preprocess_ppg_data(metadatas_ppg[0], metadatas_imu[0], tested_output_path, ppg_config, imu_config)
    compare_data(reference_output_path, tested_output_path, accelerometer_binaries_pairs)


def test_accel_preprocessing_to_gait(shared_datadir: Path):
    """
    This function is used to evaluate the output of the PPG pipeline preprocessing function. It evaluates it by comparing the acceleration data output to the Gate pipeline reference output.
    """
    input_dir_name: str = "1.sensor_data"
    output_dir_name: str = "2.preprocessed_data"

    input_path_imu = shared_datadir / input_dir_name / "imu"
    input_path_ppg = shared_datadir / input_dir_name / "ppg"
    reference_output_path = shared_datadir / output_dir_name / "gait"
    tested_output_path = reference_output_path / "test-output"

    ppg_config = PPGPreprocessingConfig()
    imu_config = IMUPreprocessingConfig()
    metadatas_ppg, metadatas_imu = scan_and_sync_segments(input_path_ppg, input_path_imu)
    preprocess_ppg_data(metadatas_ppg[0], metadatas_imu[0], tested_output_path, ppg_config, imu_config)
    compare_data(reference_output_path, tested_output_path, accelerometer_binaries_pairs)