from pathlib import Path

from paradigma.tremor.tremor_analysis import extract_tremor_features_io, detect_tremor_io
from paradigma.tremor.tremor_analysis_config import TremorFeatureExtractionConfig, TremorDetectionConfig
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

    input_path = shared_datadir / input_dir_name / "imu"
    reference_output_path = shared_datadir / output_dir_name / data_type
    tested_output_path = reference_output_path / "test-output"

    config = TremorFeatureExtractionConfig()
    extract_tremor_features_io(input_path, tested_output_path, config)
    compare_data(reference_output_path, tested_output_path, tremor_binaries_pairs)

def test_3_tremor_detection_output(shared_datadir: Path):
    """
    This function is used to evaluate the output of the gait detection. It evaluates it by comparing the output to a reference output.
    """

    input_dir_name: str = "3.extracted_features"
    output_dir_name: str = "4.predictions"
    data_type: str = "tremor"

    # Temporary path to store the output of the notebook
    path_to_classifier_input = shared_datadir / '0.classification' / 'tremor'
    input_path = shared_datadir / input_dir_name / data_type
    reference_output_path = shared_datadir / output_dir_name / data_type
    tested_output_path = reference_output_path / "test-output"

    config = TremorDetectionConfig()
    detect_tremor_io(input_path, tested_output_path, path_to_classifier_input, config)
    compare_data(reference_output_path, tested_output_path, tremor_binaries_pairs)
