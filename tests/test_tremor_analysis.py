from pathlib import Path

from paradigma.testing import extract_tremor_features_io, detect_tremor_io
from paradigma.config import TremorConfig

from test_notebooks import compare_data

tremor_binaries_pairs: list[tuple[str, str]] = [
        ("tremor_meta.json", "tremor_time.bin"),
        ("tremor_meta.json", "tremor_values.bin"),
    ]

path_to_assets = Path('src/paradigma/assets')

def test_2_extract_features_tremor_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the tremor feature extraction. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "2.preprocessed_data"
    output_dir_name: str = "3.extracted_features"
    data_type: str = "tremor"

    path_to_imu_input = shared_datadir / input_dir_name / "imu"
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    config = TremorConfig('features')
    extract_tremor_features_io(path_to_imu_input, path_to_tested_output, config)
    compare_data(path_to_reference_output, path_to_tested_output, tremor_binaries_pairs)

def test_3_tremor_detection_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the tremor detection. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "3.extracted_features"
    output_dir_name: str = "4.predictions"
    data_type: str = "tremor"
    classifier_package_filename: str = "tremor_detection_clf_package.pkl"

    # Temporary path to store the output of the notebook
    path_to_feature_input = shared_datadir / input_dir_name / data_type
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    full_path_to_classifier_package = path_to_assets / classifier_package_filename

    config = TremorConfig('classification')
    detect_tremor_io(path_to_feature_input, path_to_tested_output, full_path_to_classifier_package, config)
    compare_data(path_to_reference_output, path_to_tested_output, tremor_binaries_pairs)
