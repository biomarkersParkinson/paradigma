from pathlib import Path

from paradigma.heart_rate.heart_rate_analysis import extract_signal_quality_features
from paradigma.config import SignalQualityFeatureExtractionConfig
from paradigma.preprocessing_config import PPGPreprocessingConfig, IMUPreprocessingConfig
from paradigma.ppg_preprocessing import preprocess_ppg_data, scan_and_sync_segments
from test_notebooks import compare_data


# Mappings between the metadata and binary files

accelerometer_preproc_pairs: list[tuple[str, str]] = [
    ("accelerometer_meta.json", "accelerometer_samples.bin"),
    ("accelerometer_meta.json", "accelerometer_time.bin"),
]
ppg_preproc_pairs: list[tuple[str, str]] = [
    ("PPG_meta.json", "PPG_samples.bin"),
    ("PPG_meta.json", "PPG_time.bin"),
]
ppg_features_pairs: list[tuple[str, str]] = [
    ("features_ppg_meta.json", "features_ppg_samples.bin"),
    ("features_ppg_meta.json", "features_ppg_time.bin"),
]

accelerometer_features_pairs: list[tuple[str, str]] = [
    ("features_acc_meta.json", "features_acc_samples.bin"),
    ("features_acc_meta.json", "features_acc_time.bin"),
]


def compare_ppg_preprocessing(
    shared_datadir: Path, binaries_pairs: list[tuple[str, str]]
):
    """
    This function is used to evaluate the output of the PPG pipeline preprocessing function. It evaluates it by comparing the output to a reference output.

    Parameters
    ----------
    shared_datadir : Path
        The path to the shared data directory.
    binaries_pairs : list[tuple[str, str]]
            The list of pairs of metadata and binary files to compare.
    """
    input_dir_name: str = "1.sensor_data"
    output_dir_name: str = "2.preprocessed_data"

    input_path_imu = shared_datadir / input_dir_name / "imu"
    input_path_ppg = shared_datadir / input_dir_name / "ppg"
    reference_output_path = shared_datadir / output_dir_name / "ppg"
    tested_output_path = reference_output_path / "test-output"

    ppg_config = PPGPreprocessingConfig()
    imu_config = IMUPreprocessingConfig()
    metadatas_ppg, metadatas_imu = scan_and_sync_segments(
        input_path_ppg, input_path_imu
    )
    preprocess_ppg_data(
        metadatas_ppg[0], metadatas_imu[0], tested_output_path, ppg_config, imu_config
    )
    compare_data(reference_output_path, tested_output_path, binaries_pairs)


# def test_accelerometer_preprocessing(shared_datadir: Path):
#     """
#     This function is used to evaluate the output of the PPG pipeline preprocessing function. It evaluates it by comparing the accelerometer data output to the PPG reference output.
#     """
#     compare_ppg_preprocessing(shared_datadir, accelerometer_preproc_pairs)


# def test_ppg_preprocessing(shared_datadir: Path):
#     """
#     This function is used to evaluate the output of the PPG pipeline preprocessing function. It evaluates it by comparing the PPG data output to the PPG reference output.
#     """
#     compare_ppg_preprocessing(shared_datadir, ppg_preproc_pairs)


# def test_accelerometer_feature_extraction(shared_datadir: Path):
#     """
#     This function is used to evaluate the output of the feature extraction function. It evaluates it by comparing the output to a reference output.
#     """
#     input_dir_name: str = "2.preprocessed_data"
#     output_dir_name: str = "3.extracted_features"
#     classifier_path = "src/paradigma/ppg/classifier/LR_PPG_quality.pkl"

#     input_path = shared_datadir / input_dir_name / "ppg"
#     reference_output_path = shared_datadir / output_dir_name / "ppg"
#     tested_output_path = reference_output_path / "test-output"

#     config = HeartRateFeatureExtractionConfig()
#     extract_signal_quality_features(
#         input_path, classifier_path, tested_output_path, config
#     )
#     compare_data(reference_output_path, tested_output_path, accelerometer_features_pairs)
