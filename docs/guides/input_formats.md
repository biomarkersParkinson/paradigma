# Data Input Formats Guide

ParaDigMa's `run_paradigma()` function supports multiple flexible input formats for providing data to the analysis pipeline.

## Prerequisites

Before using ParaDigMa, ensure your data meets the requirements:
- **Sensor requirements**: See [Sensor Requirements](sensor_requirements.md)
- **Device compatibility**: See [Supported Devices](supported_devices.md)
- **Data format**: Pandas DataFrame with required columns (see below)

## Input Format Options

The `dfs` parameter accepts three input formats:

### 1. Single DataFrame

Use when you have a single prepared DataFrame to analyze:

```python
import pandas as pd
from paradigma.orchestrator import run_paradigma

# Load your data
df = pd.read_parquet('data.parquet')

# Process with a single DataFrame
results = run_paradigma(
    dfs=df,  # Single DataFrame
    pipelines=['gait'],
    watch_side='right',  # Required for gait pipeline
    save_intermediate=['aggregation']  # Saves to ./output by default
)
```

The DataFrame is automatically assigned the identifier `'df_1'` internally.

### 2. List of DataFrames

Use when you have multiple DataFrames that should be automatically assigned sequential IDs:

```python
# Load multiple data segments
df1 = pd.read_parquet('morning_session.parquet')
df2 = pd.read_parquet('afternoon_session.parquet')
df3 = pd.read_parquet('evening_session.parquet')

# Process as list - automatically assigned to 'df_1', 'df_2', 'df_3'
results = run_paradigma(
    dfs=[df1, df2, df3],
    pipelines=['gait'],
    watch_side='right',
    save_intermediate=['quantification', 'aggregation']
)
```

**Benefits:**
- Automatic segment ID assignment
- Each DataFrame processed independently before aggregation
- Aggregation performed across all input DataFrames

### 3. Dictionary of DataFrames

Use when you need custom identifiers for your data segments:

```python
# Create dictionary with custom segment identifiers
dfs = {
    'patient_001_morning': pd.read_parquet('session1.parquet'),
    'patient_001_evening': pd.read_parquet('session2.parquet'),
    'patient_002_morning': pd.read_parquet('session3.parquet'),
}

# Process with custom segment identifiers
results = run_paradigma(
    dfs=dfs,
    pipelines=['gait'],
    watch_side='right',
    save_intermediate=[]  # No files saved - results only in memory
)
```

**Benefits:**
- Custom segment identifiers in output
- Improved traceability of data sources
- Useful for multi-patient or multi-session datasets

## Loading Data from Disk

To automatically load data files from a directory:

```python
from paradigma.orchestrator import run_paradigma

# Load all files from a directory
results = run_paradigma(
    data_path='./data/patient_001/',
    pipelines=['gait'],
    watch_side='right',
    file_pattern='*.parquet',  # Optional: filter by pattern
    save_intermediate=['aggregation']
)
```

**Supported file formats:**
- Pandas: `.parquet`, `.csv`, `.pkl`, `.pickle`
- TSDF: `.meta` + `.bin` pairs
- Device-specific: `.avro` (Empatica), `.cwa` (Axivity)

See [Supported Devices](supported_devices.md) for device-specific loading examples.

## Required DataFrame Columns

Your DataFrame must contain the following columns depending on the pipeline:

### For Gait and Tremor Pipelines

```python
# Required columns
df.columns = ['time', 'x', 'y', 'z', 'gyro_x', 'gyro_y', 'gyro_z']
```

- `time`: Timestamp (float seconds or datetime)
- `x`, `y`, `z`: Accelerometer data
- `gyro_x`, `gyro_y`, `gyro_z`: Gyroscope data

### For Pulse Rate Pipeline

```python
# Required columns
df.columns = ['time', 'ppg']  # Accelerometer optional
```

- `time`: Timestamp (float seconds or datetime)
- `ppg`: PPG/BVP signal

### Custom Column Names

If your data uses different column names, use `column_mapping`:

```python
results = run_paradigma(
    dfs=df,
    pipelines=['gait'],
    watch_side='left',
    column_mapping={
        'timestamp': 'time',
        'acc_x': 'x',
        'acc_y': 'y',
        'acc_z': 'z',
        'gyr_x': 'gyro_x',
        'gyr_y': 'gyro_y',
        'gyr_z': 'gyro_z'
    }
)
```

## Data Preparation Parameters

If your data needs preparation (unit conversion, resampling, etc.), ParaDigMa can handle it automatically:

```python
results = run_paradigma(
    dfs=df_raw,
    pipelines=['gait'],
    watch_side='left',
    skip_preparation=False,  # Default: perform preparation

    # Unit conversion
    accelerometer_units='m/s^2',  # Auto-converts to 'g'
    gyroscope_units='rad/s',      # Auto-converts to 'deg/s'

    # Resampling
    target_frequency=100.0,

    # Time handling
    time_input_unit='relative_s',  # Or 'absolute_datetime'

    # Orientation correction
    device_orientation=['x', 'y', 'z'],

    # Segmentation for non-contiguous data
    split_by_gaps=True,
    max_gap_seconds=1.5,
    min_segment_seconds=1.5,
)
```

If your data is already prepared (correct units, sampling rate, column names), skip preparation:

```python
results = run_paradigma(
    dfs=df_prepared,
    pipelines=['gait', 'tremor'],
    watch_side='left',
    skip_preparation=True
)
```

## Output Control

### Output Directory

```python
results = run_paradigma(
    dfs=df,
    pipelines=['gait'],
    watch_side='left',
    output_dir='./results',  # Custom output directory (default: './output')
)
```

### Saving Intermediate Results

Control which intermediate steps are saved to disk:

```python
results = run_paradigma(
    dfs=df,
    pipelines=['gait'],
    watch_side='left',
    save_intermediate=[
        'preparation',      # Prepared data
        'preprocessing',    # Preprocessed data
        'classification',   # Gait/tremor bout classifications
        'quantification',   # Segment-level measures
        'aggregation'       # Aggregated measures
    ]
)
```

To keep results only in memory without saving files:

```python
results = run_paradigma(
    dfs=df,
    pipelines=['gait'],
    watch_side='left',
    save_intermediate=[]  # No files saved
)
```

## Results Structure

Regardless of input format, results are returned in the same structure:

```python
results = {
    'quantifications': {
        'gait': pd.DataFrame,    # Segment-level gait measures
        'tremor': pd.DataFrame,  # Segment-level tremor measures
    },
    'aggregations': {
        'gait': dict,            # Time-period aggregated gait measures
        'tremor': dict,          # Time-period aggregated tremor measures
    },
    'metadata': dict             # Analysis metadata
}
```

### File Key Column

When processing multiple files, the `quantifications` DataFrame includes a `file_key` column:

- **Single DataFrame input**: No `file_key` column
- **List input (2+ files)**: `'df_1'`, `'df_2'`, etc.
- **Dict input (2+ files)**: Custom keys you provided

This preserves traceability while keeping single-file results concise.

## Best Practices

1. **Single DataFrame**: Use for single files or pre-aggregated data
2. **List of DataFrames**: Use when you don't need specific naming
3. **Dictionary of DataFrames**: Use when segment identifiers are important for traceability
4. **Check `file_key` column**: Trace results back to input segments in multi-file processing
5. **Skip preparation**: Set `skip_preparation=True` if data is already standardized
6. **Save selectively**: Only save intermediate results you need to reduce disk usage

## See Also

- [Sensor Requirements](sensor_requirements.md) - What sensor specs are needed
- [Supported Devices](supported_devices.md) - Device-specific loading examples
- [Data Preparation Tutorial](../tutorials/data_preparation.html) - Step-by-step preparation guide
