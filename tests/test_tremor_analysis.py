from pathlib import Path

from paradigma.tremor.tremor_analysis import extract_tremor_features_io, detect_tremor_io, quantify_tremor_io
from paradigma.config import TremorFeatureExtractionConfig, TremorDetectionConfig, TremorQuantificationConfig
from test_notebooks import compare_data


tremor_binaries_pairs: list[tuple[str, str]] = [
        ("tremor_meta.json", "tremor_time.bin"),
        ("tremor_meta.json", "tremor_values.bin"),
    ]

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

    config = TremorFeatureExtractionConfig()
    extract_tremor_features_io(path_to_imu_input, path_to_tested_output, config)
    compare_data(path_to_reference_output, path_to_tested_output, tremor_binaries_pairs)

def test_3_tremor_detection_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the tremor detection. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "3.extracted_features"
    output_dir_name: str = "4.predictions"
    data_type: str = "tremor"

    # Temporary path to store the output of the notebook
    path_to_classifier_input = shared_datadir / '0.classification' / data_type
    path_to_feature_input = shared_datadir / input_dir_name / data_type
    path_to_reference_output = shared_datadir / output_dir_name / data_type
    path_to_tested_output = path_to_reference_output / "test-output"

    config = TremorDetectionConfig()
    detect_tremor_io(path_to_feature_input, path_to_tested_output, path_to_classifier_input, config)
    compare_data(path_to_reference_output, path_to_tested_output, tremor_binaries_pairs)
