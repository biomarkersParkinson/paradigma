from pathlib import Path

from dbpd.imu_preprocessing import *
from dbpd.gait_analysis import *
from dbpd.gait_analysis_config import *
from test_notebooks import compare_data


# Mappings between the metadata and binary files

gait_binaries_pairs: list[tuple[str, str]] = [
        ("gait_meta.json", "gait_time.bin"),
        ("gait_meta.json", "gait_values.bin"),
    ]

arm_swing_binaries_pairs: list[tuple[str, str]] = [
        ("arm_swing_meta.json", "arm_swing_values.bin"),
        ("arm_swing_meta.json", "arm_swing_time.bin"),
    ]

accelerometer_binaries_pairs: list[tuple[str, str]] = [
        ("accelerometer_meta.json", "accelerometer_samples.bin"),
        ("accelerometer_meta.json", "accelerometer_time.bin"),
    ]
gyroscope_binaries_pairs: list[tuple[str, str]] = [
        ("gyroscope_meta.json", "gyroscope_samples.bin"),
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
    reference_output_path = shared_datadir / output_dir_name / "gait"
    tested_output_path = reference_output_path / "test-output"

    config = PreprocessingConfig()
    preprocess_imu_data(input_path, tested_output_path, config)
    compare_data(reference_output_path, tested_output_path, imu_binaries_pairs)


def test_2_extract_features_gait_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the gait feature extraction. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "2.preprocessed_data"
    output_dir_name: str = "3.extracted_features"
    data_type: str = "gait"

    input_path = shared_datadir / input_dir_name / data_type
    reference_output_path = shared_datadir / output_dir_name / data_type
    tested_output_path = reference_output_path / "test-output"

    config = GaitFeatureExtractionConfig()
    extract_gait_features(input_path, tested_output_path, config)
    compare_data(reference_output_path, tested_output_path, gait_binaries_pairs)


def test_3_gait_detection_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the gait detection. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "3.extracted_features"
    output_dir_name: str = "4.predictions"
    data_type: str = "gait"

    # Temporary path to store the output of the notebook
    path_to_classifier_input = shared_datadir / '0.classifiers' / 'gait'
    input_path = shared_datadir / input_dir_name / data_type
    reference_output_path = shared_datadir / output_dir_name / data_type
    tested_output_path = reference_output_path / "test-output"

    config = GaitDetectionConfig()
    detect_gait(input_path, tested_output_path, path_to_classifier_input, config)
    compare_data(reference_output_path, tested_output_path, gait_binaries_pairs)


def test_4_extract_features_arm_swing_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the arm swing feature extraction. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "2.preprocessed_data"
    output_dir_name: str = "3.extracted_features"
    data_type: str = "gait"

    # Temporary path to store the output of the notebook
    input_path = shared_datadir / input_dir_name / data_type
    reference_output_path = shared_datadir / output_dir_name / data_type
    tested_output_path = reference_output_path / "test-output"

    config = ArmSwingFeatureExtractionConfig()
    extract_arm_swing_features(input_path, tested_output_path, config)
    compare_data(reference_output_path, tested_output_path, arm_swing_binaries_pairs)


def test_5_arm_swing_detection_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the gait detection. It evaluates it by comparing the output to a reference output.
    """

    # Notebook info
    input_dir_name: str = "3.extracted_features"
    output_dir_name: str = "4.predictions"
    data_type: str = "gait"

    # Temporary path to store the output of the notebook
    path_to_classifier_input = shared_datadir / '0.classifiers' / 'gait'
    input_path = shared_datadir / input_dir_name / data_type
    reference_output_path = shared_datadir / output_dir_name / data_type
    tested_output_path = reference_output_path / "test-output"

    config = ArmSwingDetectionConfig()
    detect_arm_swing(input_path, tested_output_path, path_to_classifier_input, config)
    compare_data(reference_output_path, tested_output_path, arm_swing_binaries_pairs)


def test_6_arm_swing_quantification_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the arm swing quantification. It evaluates it by comparing the output to a reference output.
    """

    feature_input_dir_name: str = "3.extracted_features"
    prediction_input_dir_name: str = "4.predictions"
    output_dir_name: str = "5.quantification"
    data_type: str = "gait"

    # Temporary path to store the output of the notebook
    path_to_feature_input = shared_datadir / feature_input_dir_name / data_type
    path_to_prediction_input = shared_datadir / prediction_input_dir_name / data_type
    reference_output_path = shared_datadir / output_dir_name / data_type
    tested_output_path = reference_output_path / "test-output"

    config = ArmSwingQuantificationConfig()
    quantify_arm_swing(path_to_feature_input, path_to_prediction_input, tested_output_path, config)
    compare_data(reference_output_path, tested_output_path, arm_swing_binaries_pairs)
