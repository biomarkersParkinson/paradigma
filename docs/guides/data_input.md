# Data Input Formats Guide

ParaDigMa's `run_paradigma()` function supports multiple flexible input formats for providing data to the analysis pipeline.

## Output Control

Before diving into input formats, it's important to understand how to control output:

- **`output_dir`**: Defaults to `"./output"`. You can specify a custom directory.
- **`store_intermediate`**: Controls which intermediate results are saved to disk. Valid options: `['preparation', 'preprocessing', 'classification', 'quantification', 'aggregation']`
- **No storage**: If `store_intermediate=[]` (empty list), no files are saved - results are only returned in memory.

## Input Format Options

The `dfs` parameter accepts three input formats:

### 1. Single DataFrame

Use this when you have a single prepared DataFrame to analyze:

```python
import pandas as pd
from paradigma.orchestrator import run_paradigma

# Load your data
df = pd.read_parquet('data.parquet')

# Process with a single DataFrame
results = run_paradigma(
    dfs=df,  # Single DataFrame
    pipeline_names='gait',
    watch_side='right',
    store_intermediate=['aggregation']  # Uses default ./output directory
)
```

The DataFrame will be automatically assigned the segment identifier `'data'` internally.

### 2. List of DataFrames

Use this when you have multiple DataFrames that should be automatically assigned unique segment IDs. Each DataFrame is assigned a sequential segment ID:

```python
import pandas as pd
from paradigma.orchestrator import run_paradigma

# Load multiple data segments
df1 = pd.read_parquet('morning_session.parquet')
df2 = pd.read_parquet('afternoon_session.parquet')
df3 = pd.read_parquet('evening_session.parquet')

# Process as a list - automatically assigned to 'segment_1', 'segment_2', 'segment_3'
results = run_paradigma(
    dfs=[df1, df2, df3],  # List of DataFrames
    pipeline_names='gait',
    output_dir='./results',  # Custom output directory
    watch_side='right',
    store_intermediate=['quantification', 'aggregation']
)
```

**Benefits:**
- Automatic segment ID assignment
- Clean, concise syntax for multiple inputs
- Each DataFrame is processed independently before aggregation

### 3. Dictionary of DataFrames

Use this when you need custom identifiers for your data segments (e.g., file names, session types):

```python
import pandas as pd
from paradigma.orchestrator import run_paradigma

# Create a dictionary with custom segment identifiers
dfs = {
    'patient_001_morning': pd.read_parquet('session1.parquet'),
    'patient_001_evening': pd.read_parquet('session2.parquet'),
    'patient_002_morning': pd.read_parquet('session3.parquet'),
}

# Process with custom segment identifiers
results = run_paradigma(
    dfs=dfs,  # Dictionary with custom keys
    watch_side='right',
    store_intermediate=[]  # No files saved - results only in memorylts',
    watch_side='right'
)
```

**Benefits:**
- Custom segment identifiers in output
- Improved traceability of data sources
- Useful for multi-patient or multi-session datasets

## Loading Data from Disk

To load data files from disk automatically:

```python
from paradigma.orchestrator import run_paradigma

# Load all files from a directory
results = run_paradigma(
    data_path='./data/patient_001/',  # Directory containing data files
    pipeline_names='gait',
    watch_side='right',
    file_patterns='parquet',  # Optional: specify file pattern
    store_intermediate=['aggregation']  # Uses default ./output directory
)
```

This uses the `load_data_files()` function which automatically returns a dictionary of DataFrames with file names as keys.

## Mixed Input Scenarios

### Combining loaded data with additional DataFrames

```python
from paradigma.load import load_data_files
from paradigma.orchestrator import run_paradigma
import pandas as pd

# Load some files
loaded_dfs = load_data_files('./data/')

# Add additional DataFrames
loaded_dfs['custom_session'] = pd.read_parquet('custom_data.parquet')

# Process combined data
results = run_paradigma(
    dfs=loaded_dfs,
    pipeline_names='gait',
    output_dir='./results',  # Custom output directory
    watch_side='right',
    store_intermediate=['quantification']
)
```

## Results Structure

Regardless of input format, results are aggregated and returned in the same structure:

```python
results = {
    'quantifications': pd.DataFrame,   # Segment-level results
    'aggregations': dict,              # Time-period aggregated results
    'metadata': dict                   # Analysis metadata
}
```

The `quantifications` DataFrame includes a `file_key` column that preserves:
- For single DataFrame input: `'data'`
- For list input: `'segment_0'`, `'segment_1'`, etc.
- For dict input: The custom keys provided

## Best Practices

1. **Use list input** when you have multiple data segments with no specific naming requirements
2. **Use dict input** when segment identifiers are important for traceability
3. **Use single DataFrame** for processing single files or pre-aggregated data
4. **Check output `file_key`** column to trace results back to input segments
