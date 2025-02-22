import pandas as pd
from pathlib import Path

from paradigma.testing import filter_gait_io, detect_gait_io, extract_arm_activity_features_io, extract_gait_features_io
from paradigma.config import GaitConfig
from test_notebooks import compare_data


# Mappings between the metadata and binary files

gait_binaries_pairs: list[tuple[str, str]] = [
        ("gait_meta.json", "gait_time.bin"),
        ("gait_meta.json", "gait_values.bin"),
    ]

arm_activity_binaries_pairs: list[tuple[str, str]] = [
        ("arm_activity_meta.json", "arm_activity_values.bin"),
        ("arm_activity_meta.json", "arm_activity_time.bin"),
    ]

arm_swing_binaries_pairs: list[tuple[str, str]] = [
        ("arm_swing_meta.json", "arm_swing_values.bin"),
        ("arm_swing_meta.json", "arm_swing_time.bin"),
    ]

accelerometer_binaries_pairs: list[tuple[str, str]] = [
        ("accelerometer_meta.json", "accelerometer_values.bin"),
        ("accelerometer_meta.json", "accelerometer_time.bin"),
    ]
gyroscope_binaries_pairs: list[tuple[str, str]] = [
        ("gyroscope_meta.json", "gyroscope_values.bin"),
        ("gyroscope_meta.json", "gyroscope_time.bin"),
    ]
imu_binaries_pairs: list[tuple[str, str]] = accelerometer_binaries_pairs + gyroscope_binaries_pairs

path_to_assets = Path('src/paradigma/assets')


def test_2_extract_features_gait_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the gait feature extraction. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "2.preprocessed_data"
    output_dir_name: str = "3.extracted_features"
    data_type: str = "gait"

    path_to_preprocessed_input = shared_datadir / input_dir_name / "imu"
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    config = GaitConfig(step=data_type)

    extract_gait_features_io(
        config=config, 
        path_to_input=path_to_preprocessed_input, 
        path_to_output=path_to_tested_output
    )
    compare_data(
        reference_dir=path_to_reference_output, 
        tested_dir=path_to_tested_output, 
        binaries_pairs=gait_binaries_pairs
    )


def test_3_gait_detection_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the gait detection. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "3.extracted_features"
    output_dir_name: str = "4.predictions"
    data_type: str = "gait"
    classifier_package_filename: str = "gait_detection_clf_package.pkl"

    # Temporary path to store the output of the notebook
    path_to_feature_input = shared_datadir / input_dir_name / data_type
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    full_path_to_classifier_package = path_to_assets / classifier_package_filename

    config = GaitConfig(step=data_type)
    detect_gait_io(
        config=config, 
        path_to_input=path_to_feature_input, 
        path_to_output=path_to_tested_output, 
        full_path_to_classifier_package=full_path_to_classifier_package, 
    )
    compare_data(
        reference_dir=path_to_reference_output, 
        tested_dir=path_to_tested_output, 
        binaries_pairs=gait_binaries_pairs
    )


def test_4_extract_features_arm_activity_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the arm activity feature extraction. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "2.preprocessed_data"
    output_dir_name: str = "3.extracted_features"
    data_type: str = "gait"
    classifier_package_filename: str = "gait_detection_clf_package.pkl"

    # Temporary path to store the output of the notebook
    path_to_timestamp_input = shared_datadir / input_dir_name / "imu"
    path_to_prediction_input = shared_datadir / "4.predictions" / data_type
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    full_path_to_classifier_package = path_to_assets / classifier_package_filename

    config = GaitConfig(step='arm_activity')
    extract_arm_activity_features_io(
        config=config, 
        path_to_timestamp_input=path_to_timestamp_input, 
        path_to_prediction_input=path_to_prediction_input, 
        full_path_to_classifier_package=full_path_to_classifier_package, 
        path_to_output=path_to_tested_output
    )
    compare_data(
        reference_dir=path_to_reference_output, 
        tested_dir=path_to_tested_output, 
        binaries_pairs=arm_activity_binaries_pairs
    )


def test_5_arm_swing_detection_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the gait filtering. It evaluates it by comparing the output to a reference output.
    """

    # Notebook info
    input_dir_name: str = "3.extracted_features"
    output_dir_name: str = "4.predictions"
    data_type: str = "gait"
    classifier_package_filename: str = "gait_filtering_clf_package.pkl"

    # Temporary path to store the output of the notebook
    path_to_prediction_input = shared_datadir / input_dir_name / data_type
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    full_path_to_classifier_package = path_to_assets / classifier_package_filename

    config = GaitConfig(step='arm_activity')
    filter_gait_io(
        config=config, 
        path_to_input=path_to_prediction_input, 
        path_to_output=path_to_tested_output, 
        full_path_to_classifier_package=full_path_to_classifier_package, 
    )
    compare_data(
        reference_dir=path_to_reference_output, 
        tested_dir=path_to_tested_output, 
        binaries_pairs=arm_activity_binaries_pairs
    )
