from pathlib import Path

from dbpd.preprocessing_config import PPGPreprocessingConfig, IMUPreprocessingConfig
from dbpd.ppg_preprocessing import *
from dbpd.imu_preprocessing import *
from dbpd.gait_analysis import *
from dbpd.gait_analysis_config import *
from test_notebooks import compare_data


# Mappings between the metadata and binary files

acceleration_binaries_pairs: list[tuple[str, str]] = [
        ("acceleration_meta.json", "acceleration_samples.bin"),
        ("acceleration_meta.json", "acceleration_time.bin"),
]
acceleration_filt_binaries_pairs: list[tuple[str, str]] = [
        ("acceleration_filt_meta.json", "acceleration_filt_samples.bin"),
        ("acceleration_filt_meta.json", "acceleration_filt_time.bin"),
]
ppg_binaries_pairs: list[tuple[str, str]] = [
        ("PPG_meta.json", "PPG_samples.bin"),
        ("PPG_meta.json", "PPG_time.bin"),
]
imu_binaries_pairs: list[tuple[str, str]] = acceleration_binaries_pairs + ppg_binaries_pairs


def test_1_imu_preprocessing(shared_datadir: Path):
    """
    This function is used to evaluate the output of the preprocessing function. It evaluates it by comparing the output to a reference output.
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
    compare_data(reference_output_path, tested_output_path, imu_binaries_pairs)
