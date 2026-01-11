# ParaDigMa Copilot Instructions

## Overview
ParaDigMa is a Python toolbox for extracting Parkinson's disease digital markers from real-life wrist sensor data. It processes accelerometer, gyroscope, and photoplethysmography (PPG) signals through three scientifically validated pipelines: arm swing during gait, tremor detection, and pulse rate analysis.

## Architecture

### Core Processing Pattern
All pipelines follow a consistent 5-step pattern:
1. **Preprocessing** - Signal preparation and filtering
2. **Feature extraction** - Windowed temporal/spectral features
3. **Classification** - ML-based segment detection using `ClassifierPackage`
4. **Quantification** - Extract specific measures from detected segments
5. **Aggregation** - Weekly/daily measure aggregation

### Key Components
- **Pipeline Runner**: `src/paradigma/pipeline.py` - High-level interface for running pipelines
- **CLI**: `src/paradigma/cli.py` - Command-line interface for pipeline execution
- **Pipelines**: `src/paradigma/pipelines/` - Three main processing pipelines
- **Config System**: `config.py`, `constants.py` - Flexible column mapping via `DataColumns`
- **Feature Extraction**: `feature_extraction.py` - Temporal/spectral domain features
- **Classification**: `classification.py` - `ClassifierPackage` with classifier + threshold + scaler
- **Data Format**: TSDF (Time Series Data Format) via `tsdf` dependency

## Development Patterns

### Configuration
- Use `DataColumns` dataclass for consistent column naming across pipelines
- Config classes inherit from `BaseConfig` and support flexible sensor column mapping
- Example: `IMUConfig(column_mapping={"ACCELEROMETER_X": "custom_accel_x"})`

### Pipeline Development
```python
# Standard pipeline structure
def pipeline_function(df: pd.DataFrame, config: SomeConfig) -> pd.DataFrame:
    # 1. Extract features using windowing
    features = extract_features(df, config)
    # 2. Apply classification
    predictions = clf_package.predict(features)
    # 3. Quantify measures from segments
    measures = quantify_measures(df, predictions, config)
    return measures
```

### Testing & Validation
- Use `pytest` with `poetry run pytest`
- Test data in `tests/data/` with reference outputs for regression testing
- Notebook tests via `papermill` in CI pipeline
- Pre-commit hooks for formatting/linting

### TSDF Data Handling
- Load data: `load_tsdf_dataframe(path_to_data, prefix="IMU_segment0001")`
- Data structure: separate time, values, and metadata files
- Column mapping handled through config objects, not hardcoded

## Critical Workflows

### Setup & Testing
```bash
# Install with Poetry
poetry install
poetry run pytest --maxfail=1 --disable-warnings -q

# Pre-commit validation
poetry run pre-commit run --all-files --show-diff-on-failure
```

### High-Level Pipeline Execution
```python
# Use the pipeline runner for production workflows
from paradigma import run_pipeline
results = run_pipeline(
    data_path="data/tsdf/",
    pipelines=["gait", "tremor"],
    config="default",
    output_dir="results/"
)

# CLI usage
poetry run paradigma run data/ --pipelines gait --output results/ --verbose
```

### Pipeline Usage Pattern
1. Load TSDF data using `paradigma.util.load_tsdf_dataframe`
2. Create appropriate config object (GaitConfig, TremorConfig, etc.)
3. Call pipeline functions in sequence: preprocess → extract features → classify → quantify
4. Use `ClassifierPackage` from `importlib.resources` for validated models

## Project-Specific Conventions

### Naming Patterns
- Pipeline files: `{domain}_pipeline.py` (gait_pipeline.py, tremor_pipeline.py)
- Config classes: `{Domain}Config` with inheritance from `BaseConfig`
- Constants: Use `DataColumns` dataclass attributes, not magic strings
- TSDF files: `{prefix}_{type}.{ext}` (e.g., "IMU_segment0001_values.bin")

### Scientific Validation Requirements
- All pipelines validated on Parkinson@Home and PPP datasets
- Maintain compatibility with Verily Study Watch and Gait-up Physilog 4
- Document sensor requirements (sampling rates, ranges) in docstrings
- Reference validation papers in pipeline documentation

### Dependencies & Versioning
- Poetry for dependency management (pyproject.toml)
- Pin `tsdf` to main branch: `git = "https://github.com/biomarkersParkinson/tsdf.git"`
- Python 3.11+ required
- Keep scikit-learn compatibility for model loading

### Configuration & Entry Points
- Config classes require specific parameters: `GaitConfig(step="gait")`, `TremorConfig(step="features")`
- Use `run_pipeline()` function for high-level execution or individual pipeline functions for granular control
- CLI available via `paradigma` command after installing with Poetry
- Console script defined in pyproject.toml: `paradigma = "paradigma.cli:main"`

When extending pipelines, follow the established preprocessing → feature extraction → classification → quantification pattern. Always validate against test data and maintain TSDF compatibility. Use the pipeline runner for production workflows and notebooks for research/exploration.
