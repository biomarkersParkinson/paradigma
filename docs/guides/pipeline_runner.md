# ParaDigMa Pipeline Runner

The ParaDigMa pipeline runner provides a high-level interface for running ParaDigMa pipelines on sensor data, eliminating the need to work with Jupyter notebooks for basic pipeline execution.

## Quick Start

### Python API

```python
from paradigma import run_pipeline

# Run on TSDF data (auto-detected)
results = run_pipeline(
    data_path="path/to/tsdf/data",
    pipelines=["gait"],
    output_dir="results/"
)

# Run on Axivity CWA files
results = run_pipeline(
    data_path="path/to/axivity/data",
    pipelines=["gait"],
    data_format="axivity",
    output_dir="results/"
)

# Run on Empatica AVRO files
results = run_pipeline(
    data_path="path/to/empatica/data",
    pipelines=["gait"],
    data_format="empatica",
    output_dir="results/"
)

# Run on prepared dataframes (parquet files)
results = run_pipeline(
    data_path="path/to/prepared/data",
    pipelines=["gait", "tremor"],
    data_format="prepared",
    file_pattern="*.parquet",
    output_dir="results/"
)
```

### Command Line Interface

```bash
# Install ParaDigMa
pip install paradigma

# Run gait pipeline on TSDF data (auto-detected)
paradigma run path/to/tsdf/data --pipelines gait --output results/

# Run on Axivity CWA files with verbose output
paradigma run path/to/axivity/data --pipelines gait --data-format axivity --verbose --output results/

# Run on Empatica AVRO files
paradigma run path/to/empatica/data --pipelines gait tremor --data-format empatica --output results/

# Run on prepared dataframes (parquet files)
paradigma run path/to/prepared/data --pipelines gait tremor --data-format prepared --file-pattern "*.parquet" --output results/

# List available pipelines
paradigma list-pipelines
```

## Features

- **Unified Interface**: Single function for all ParaDigMa pipelines
- **Multiple Data Formats**: Support TSDF, Empatica (.avro), Axivity (.cwa), and prepared DataFrames
- **Auto-Detection**: Automatically detect data format from file extensions
- **Flexible Configuration**: Use default configs or provide custom configurations
- **CLI Support**: Run pipelines from command line without Python scripting
- **Output Management**: Automatically save results to CSV files
- **Error Handling**: Comprehensive error messages and validation
- **Parallel Processing**: Optional parallel execution for supported operations

## Available Pipelines

- **gait**: Arm swing during gait analysis
- **tremor**: Tremor detection and quantification
- **pulse_rate**: Pulse rate estimation from PPG signals

## Configuration

### Default Configuration

Use `config="default"` to use standard pipeline configurations:

```python
results = run_pipeline(
    data_path="data/",
    pipelines=["gait"],
    config="default"
)
```

### Custom Configuration

Provide custom configuration objects for fine-tuned control:

```python
from paradigma.config import GaitConfig, TremorConfig

# Create custom configurations
gait_config = GaitConfig(step="gait")
gait_config.window_length_s = 2.0  # Custom window length

tremor_config = TremorConfig(step="features")
tremor_config.sampling_frequency = 100  # Custom sampling rate

# Use custom configs
results = run_pipeline(
    data_path="data/",
    pipelines=["gait", "tremor"],
    config={
        "gait": gait_config,
        "tremor": tremor_config
    }
)
```

### JSON Configuration File (CLI)

For CLI usage, provide configuration via JSON file:

```json
{
    "gait": {
        "window_length_s": 2.0,
        "sampling_frequency": 100
    },
    "tremor": {
        "window_length_s": 1.5,
        "sampling_frequency": 100
    }
}
```

```bash
paradigma run data/ --pipelines gait tremor --config config.json
```

## Column Mapping

The pipeline runner supports flexible column mapping to handle different naming conventions across datasets. This is especially useful when your data uses different column names than what the pipelines expect.

### Common Naming Mismatches

Different data sources often use varying column names:

| Pipeline Expects | Common Alternatives |
|------------------|-------------------|
| `accelerometer_x` | `acceleration_x`, `acc_x`, `accel_x` |
| `accelerometer_y` | `acceleration_y`, `acc_y`, `accel_y` |
| `accelerometer_z` | `acceleration_z`, `acc_z`, `accel_z` |
| `gyroscope_x` | `rotation_x`, `gyro_x`, `angular_velocity_x` |
| `gyroscope_y` | `rotation_y`, `gyro_y`, `angular_velocity_y` |
| `gyroscope_z` | `rotation_z`, `gyro_z`, `angular_velocity_z` |

### Python API

```python
from paradigma.pipeline import run_pipeline

# Define column mapping for TSDF data with different naming
column_mapping = {
    'acceleration_x': 'accelerometer_x',
    'acceleration_y': 'accelerometer_y',
    'acceleration_z': 'accelerometer_z',
    'rotation_x': 'gyroscope_x',
    'rotation_y': 'gyroscope_y',
    'rotation_z': 'gyroscope_z'
}

results = run_pipeline(
    data_path="data/tsdf/",
    pipelines=["gait"],
    column_mapping=column_mapping
)
```

### Command Line Interface

#### JSON String
```bash
paradigma run data/tsdf/ --pipelines gait \\
  --column-mapping '{"acceleration_x": "accelerometer_x", "rotation_x": "gyroscope_x"}'
```

#### JSON File
Create a mapping file `column_mapping.json`:
```json
{
    "acceleration_x": "accelerometer_x",
    "acceleration_y": "accelerometer_y",
    "acceleration_z": "accelerometer_z",
    "rotation_x": "gyroscope_x",
    "rotation_y": "gyroscope_y",
    "rotation_z": "gyroscope_z"
}
```

Then use it:
```bash
paradigma run data/tsdf/ --pipelines gait --column-mapping column_mapping.json
```

## Pipeline Steps

Each pipeline consists of multiple steps that can be run individually or in sequence:

### Gait Pipeline Steps

1. **detection**: Detect gait periods from sensor data
2. **filtering**: Filter gait periods (remove other arm activities)
3. **quantification**: Quantify arm swing during filtered gait
4. **aggregation**: Aggregate arm swing metrics

### Tremor Pipeline Steps

1. **preprocessing**: Preprocess gyroscope signals and filter noise
2. **feature_extraction**: Extract tremor-specific features from gyroscope data
3. **detection**: Detect tremor episodes using trained classifier
4. **quantification**: Quantify tremor severity scores
5. **aggregation**: Aggregate tremor metrics and statistics

### Pulse Rate Pipeline Steps

1. **preprocessing**: Preprocess PPG and acceleration signals
2. **estimation**: Estimate pulse rate from PPG signal
3. **filtering**: Filter estimates based on signal quality
4. **aggregation**: Aggregate pulse rate statistics

### Running Specific Steps

```python
# Run only detection and quantification
results = run_pipeline(
    data_path="data/",
    pipelines=["gait"],
    data_format="prepared",
    steps=["detection", "quantification"]
)

# Run all steps (default)
results = run_pipeline(
    data_path="data/",
    pipelines=["gait"],
    data_format="prepared"
    # steps=None runs all steps
)
```

## Multi-Subject Processing

Process multiple subjects organized in separate subdirectories:

```python
# Directory structure:
# data/
#   ├── subject_001/
#   │   ├── data_file1.parquet
#   │   └── data_file2.parquet
#   ├── subject_002/
#   │   └── data_file.parquet
#   └── subject_003/
#       └── data_file.parquet

results = run_pipeline(
    data_path="data/",
    pipelines=["gait"],
    data_format="prepared",
    multi_subject=True,
    output_dir="results/"  # Creates subject subdirectories
)
```

### CLI Multi-Subject

```bash
paradigma run data/ --pipelines gait --data-format prepared --multi-subject --output results/
```

## Intermediate Results

Save outputs at each pipeline step for detailed analysis:

```python
results = run_pipeline(
    data_path="data/",
    pipelines=["gait"],
    data_format="prepared",
    save_intermediate=True,
    output_dir="results/"
)
```

### Output Files by Step

**Gait Pipeline:**
- `{segment}_gait_detection.parquet` - Detected gait periods
- `{segment}_gait_filtered.parquet` - Filtered gait periods
- `{segment}_arm_swing_features.parquet` - Arm swing features
- `{segment}_arm_swing_aggregated.json` - Aggregated metrics

**Tremor Pipeline:**
- `{segment}_tremor_features.parquet` - Tremor features extracted from gyroscope data
- `{segment}_tremor_detection.parquet` - Tremor detection results with probabilities
- `{segment}_tremor_quantification.parquet` - Tremor severity quantification
- `{segment}_tremor_aggregation.parquet` - Aggregated tremor metrics

**Pulse Rate Pipeline:**
- `{segment}_pulse_rate_preprocessing.parquet` - Preprocessed signals with quality metrics
- `{segment}_pulse_rate_estimation.parquet` - Pulse rate estimates
- `{segment}_pulse_rate_filtering.parquet` - Quality-filtered pulse rate estimates
- `{segment}_pulse_rate_aggregation.parquet` - Aggregated pulse rate statistics

## Default Configuration Settings

ParaDigMa uses sensible default configurations for each pipeline:

### Default Settings

- **Sampling Frequency**: Automatically detected for TSDF format, required for raw formats
- **Pipeline Steps**: All steps enabled by default (full pipeline execution)
- **File Pattern**: `"*.tsdf"` for TSDF format, `"*.parquet"` for prepared format, `"*.avro"` for Empatica, `"*.cwa"` for Axivity
- **Parallel Processing**: Enabled by default for supported operations
- **Multi-Subject**: Disabled by default (single subject/dataset processing)
- **Save Intermediate**: Disabled by default (only final results saved)

### Configuration Override

```python
# Override default configurations
from paradigma.config import GaitConfig

custom_config = GaitConfig(
    window_length=3.0,  # 3-second windows instead of default
    overlap=0.5,        # 50% overlap instead of default
    min_gait_duration=10.0  # Minimum 10s gait bouts
)

results = run_pipeline(
    data_path="data/",
    pipelines=["gait"],
    data_format="prepared",
    config=custom_config  # Use custom configuration
)

- **Gait Config**: See [`GaitConfig`](../../src/paradigma/config.py) class
- **Tremor Config**: See [`TremorConfig`](../../src/paradigma/config.py) class
- **Pulse Rate Config**: See [`PulseRateConfig`](../../src/paradigma/config.py) class

### Custom Configuration

```python
from paradigma.config import GaitConfig

# Customize configuration
gait_config = GaitConfig(step="gait")
gait_config.window_length_s = 5.0  # 5-second windows
gait_config.overlap = 0.5  # 50% overlap

results = run_pipeline(
    data_path="data/",
    pipelines=["gait"],
    data_format="prepared",
    config={"gait": gait_config}
)
```

### Supported Data Formats

The pipeline runner supports multiple data formats:

#### 1. TSDF (Time Series Data Format)
- **File Extensions**: `.bin`, `.json` (metadata)
- **Structure**: Separate time, values, and metadata files
- **Detection**: Presence of `*_meta.json` files
- **Usage**: `data_format="tsdf"` or auto-detected

#### 2. Empatica
- **File Extensions**: `.avro`
- **Structure**: Apache Avro format with nested sensor data
- **Detection**: Presence of `.avro` files
- **Usage**: `data_format="empatica"` or auto-detected
- **Requirements**: `avro-python3` package

#### 3. Axivity
- **File Extensions**: `.cwa`, `.CWA`
- **Structure**: Continuous Wave Accelerometry format
- **Detection**: Presence of `.cwa/.CWA` files
- **Usage**: `data_format="axivity"` or auto-detected
- **Requirements**: `openmovement` package

#### 4. Prepared DataFrames
- **File Extensions**: `.parquet`, `.pkl`, `.pickle`, `.csv`, `.feather`
- **Structure**: DataFrames with standardized ParaDigMa column names
- **Detection**: Presence of supported file extensions
- **Usage**: `data_format="prepared"` with optional `file_pattern`
- **Column Requirements**: Must follow `DataColumns` naming convention

### Data Preparation Requirements

For **prepared DataFrames**, data must follow the structure from `data_preparation.ipynb`:

- **Time Column**: `time` in seconds relative to start (starting at 0)
- **Accelerometer**: `accelerometer_x/y/z` in g units
- **Gyroscope**: `gyroscope_x/y/z` in deg/s units
- **PPG**: `green` for photoplethysmography signals
- **Coordinate System**: Must follow ParaDigMa coordinate system conventions

## Output

### Return Values

The `run_pipeline()` function returns a dictionary mapping pipeline names to result DataFrames:

```python
results = run_pipeline(...)
# results = {
#     "gait": pd.DataFrame(...),      # Gait analysis results
#     "tremor": pd.DataFrame(...)     # Tremor detection results
# }
```

### File Outputs

When `output_dir` is specified, results are automatically saved as CSV files:

- `gait_results.csv` - Gait analysis quantification
- `tremor_results.csv` - Tremor detection features
- `pulse_rate_results.csv` - Pulse rate estimates

## Error Handling

The pipeline runner provides comprehensive error handling:

```python
try:
    results = run_pipeline(...)
except FileNotFoundError:
    print("Data path not found")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"Pipeline execution failed: {e}")
```

## Advanced Usage

### Programmatic Pipeline Selection

```python
from paradigma import list_available_pipelines

# Get all available pipelines
available = list_available_pipelines()
print(f"Available: {available}")

# Run all pipelines
results = run_pipeline(
    data_path="data/",
    pipelines=available,
    config="default"
)
```

### Parallel Processing

Enable parallel processing for supported operations:

```python
results = run_pipeline(
    data_path="data/",
    pipelines=["gait"],
    parallel=True  # Enable parallel processing
)
```

### Verbose Output

Enable detailed logging for debugging:

```python
results = run_pipeline(
    data_path="data/",
    pipelines=["gait"],
    verbose=True  # Enable verbose logging
)
```

## Integration with Existing Workflows

The pipeline runner is designed to complement, not replace, the existing notebook-based workflows:

- **Exploration**: Use notebooks for data exploration and method development
- **Production**: Use pipeline runner for automated processing and production workflows
- **Batch Processing**: Process multiple datasets efficiently with the CLI
- **Integration**: Embed in larger data processing pipelines

## Limitations and Future Development

Current limitations:

- Processes single TSDF segments (multi-segment support planned)
- Requires pre-installed classifier models
- Limited configuration validation

Planned features:

- Multi-segment processing
- Configuration file validation
- Progress bars for long operations
- Integration with cloud storage
- Docker support for containerized execution

## Examples

See `examples/run_pipeline_example.py` for comprehensive usage examples demonstrating different configuration options and use cases.
