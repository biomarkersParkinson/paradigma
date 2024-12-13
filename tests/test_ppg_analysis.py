from pathlib import Path

from paradigma.heart_rate.heart_rate_analysis import extract_signal_quality_features
from paradigma.config import PPGConfig, IMUConfig, SignalQualityFeatureExtractionConfig
from paradigma.preprocessing import preprocess_ppg_data, scan_and_sync_segments
from test_notebooks import compare_data


# Mappings between the metadata and binary files

accelerometer_preproc_pairs: list[tuple[str, str]] = [
    ("accelerometer_meta.json", "accelerometer_values.bin"),
    ("accelerometer_meta.json", "accelerometer_time.bin"),
]
ppg_preproc_pairs: list[tuple[str, str]] = [
    ("PPG_meta.json", "PPG_values.bin"),
    ("PPG_meta.json", "PPG_time.bin"),
]
ppg_features_pairs: list[tuple[str, str]] = [
    ("features_ppg_meta.json", "features_ppg_values.bin"),
    ("features_ppg_meta.json", "features_ppg_time.bin"),
]

accelerometer_features_pairs: list[tuple[str, str]] = [
    ("features_acc_meta.json", "features_acc_values.bin"),
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
    input_dir_name: str = "1.prepared_data"
    output_dir_name: str = "2.preprocessed_data"

    path_to_imu_input = shared_datadir / input_dir_name / "imu"
    path_to_ppg_input = shared_datadir / input_dir_name / "ppg"
    path_to_reference_output = shared_datadir / output_dir_name / "ppg"
    path_to_tested_output = path_to_reference_output / "test-output"

    ppg_config = PPGConfig()
    imu_config = IMUConfig()
    metadatas_ppg, metadatas_imu = scan_and_sync_segments(
        path_to_ppg_input, path_to_imu_input
    )
    preprocess_ppg_data(
        metadatas_ppg[0], metadatas_imu[0], path_to_tested_output, ppg_config, imu_config
    )
    compare_data(path_to_reference_output, path_to_tested_output, binaries_pairs)


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
