# ParaDigMa Pipeline Orchestrator Tutorial

This tutorial demonstrates how to use the **pipeline orchestrator** `run_paradigma()`, which serves as the main entry point for running ParaDigMa analysis pipelines. The orchestrator coordinates multiple analysis steps and can process different formats of sensor data.

## Overview

The `run_paradigma()` function is called an _orchestrator_ because it coordinates multiple analysis steps depending on the user input. It can process:

- **Gait analysis**: Arm swing quantification from IMU data
- **Tremor analysis**: Tremor detection and quantification from gyroscope data
- **Pulse rate estimation**: Pulse rate analysis from PPG data

### Key Features

- **Multi-pipeline support**: Run multiple analyses simultaneously
- **Flexible data input**: Works with both prepared and raw sensor data
- **Multiple data formats**: Supports Verily, Axivity, Empatica, and custom formats
- **Robust processing**: Automatic data preparation and error handling

### Data Requirements

The orchestrator accepts either:
1. **Prepared data**: Prepared according to the [Data preparation tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/data_preparation.html)
2. **Raw data**: Automatically processed (note: this feature has a limited scope)

Let's explore different usage scenarios with examples.

## Import required modules


```python
import json
from pathlib import Path

from paradigma.constants import TimeUnit
from paradigma.load import load_data_files
from paradigma.orchestrator import run_paradigma
```

## 1. Single pipeline with prepared data

Let's start with a simple example using prepared PPG data for pulse rate analysis.

The function `load_data_files` attempts to load data of any or multiple of the following formats:
'parquet', 'csv', 'pkl', 'pickle', 'json', 'avro', 'cwa'. You can load the data in your preferred
ways, but note that the output should be of format `Dict[str, pd.DataFrame]`:
```python
{
    'file_1': df_1,
    'file_2': df_2,
    ...,
    'file_n': df_n
}
```

Alternatively, you can provide:
- A **single DataFrame**: Will be processed with key `'df_1'`
- A **list of DataFrames**: Each will get keys like `'df_1'`, `'df_2'`, etc.

This means ParaDigMa can run multiple files in sequence. This is useful when you have multiple files
spanning a week, and you want aggregations to be computed across all files.


```python
path_to_ppg_data = Path('../../example_data/verily/ppg')

dfs_ppg = load_data_files(
    data_path=path_to_ppg_data,
    file_patterns='json',
    verbose=0
)

print(f"Loaded {len(dfs_ppg)} PPG files:")
for filename in dfs_ppg.keys():
    df = dfs_ppg[filename]
    print(f"  - {filename}: {len(df)} samples, {len(df.columns)} columns")

print(f"\nFirst 5 rows of {list(dfs_ppg.keys())[0]}:")
dfs_ppg[list(dfs_ppg.keys())[0]].head()
```

    Loaded 2 PPG files:
      - PPG_segment0001: 1029375 samples, 2 columns
      - PPG_segment0002: 2214450 samples, 2 columns

    First 5 rows of PPG_segment0001:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>green</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>262316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0334</td>
      <td>262320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0668</td>
      <td>262446</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1002</td>
      <td>262770</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1336</td>
      <td>262623</td>
    </tr>
  </tbody>
</table>
</div>



### Output Control

When running ParaDigMa, you can control where results are saved and what intermediate results to store:

**Output Directory:**
- Default: `output_dir` defaults to `"./output"`
- Custom: Specify your own path like `output_dir="./my_results"`
- No storage: Files are only saved if `save_intermediate` is not empty

**Store Intermediate Results:**

The `save_intermediate` parameter accepts a list of strings:
```python
save_intermediate=['preprocessing', 'quantification', 'aggregation']
```

Valid options are:
- `'preparation'`: Save prepared data
- `'preprocessing'`: Save preprocessed signals
- `'classification'`: Save classification results
- `'quantification'`: Save quantified measures
- `'aggregation'`: Save aggregated results

If `save_intermediate=[]` (empty list), **no files are saved** - results are only returned in memory.

Also, set the correct units of the `time` column. For all options, please check [the API reference](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/constants/index.html#paradigma.constants.TimeUnit).


```python
pipeline = 'pulse_rate'

# Example 1: Using default output directory with storage
results_single_pipeline = run_paradigma(
    dfs=dfs_ppg,
    pipelines=pipeline,
    skip_preparation=True,
    time_input_unit=TimeUnit.RELATIVE_S,
    save_intermediate=['quantification', 'aggregation'],  # Files saved to ./output
    verbose=1
)

print(results_single_pipeline['metadata'][pipeline])
print(results_single_pipeline['aggregations'][pipeline])
results_single_pipeline['quantifications'][pipeline].head()
```

    INFO: Applying ParaDigMa pipelines to provided DataFrame


    INFO: Logging to output\paradigma_run_20260129_0919.log


    INFO: Step 1: Using provided DataFrame(s) as input


    INFO: Steps 2-3: Processing 2 files individually


    INFO: Processing DataFrame 1/2: PPG_segment0001


    INFO: Step 1: Preprocessing PPG and accelerometer data


    INFO: Step 2: Extracting signal quality features


    Resampled: 1029375 -> 1030189 rows at 30.0 Hz


    INFO: Step 3: Signal quality classification


    INFO: Step 4: Pulse rate estimation


    INFO: Step 5: Quantifying pulse rate


    INFO: Pulse rate analysis completed: 830 valid pulse rate estimates from 830 total windows


    INFO: Processing DataFrame 2/2: PPG_segment0002


    INFO: Step 1: Preprocessing PPG and accelerometer data


    INFO: Step 2: Extracting signal quality features


    Resampled: 2214450 -> 2216689 rows at 30.0 Hz


    INFO: Step 3: Signal quality classification


    INFO: Step 4: Pulse rate estimation


    INFO: Step 5: Quantifying pulse rate


    INFO: Pulse rate analysis completed: 7854 valid pulse rate estimates from 7854 total windows


    INFO: Step 4: Combining quantifications from all files


    INFO: Pulse_rate: Combined 8684 windows from 2 files


    INFO: Step 5: Aggregating pulse rate results across ALL files


    INFO: Pulse rate aggregation completed with 8684 valid estimates


    INFO: Saved prepared data to output\quantifications_pulse_rate.parquet


    INFO: Saved aggregations to output\aggregations_pulse_rate.json


    INFO: ParaDigMa analysis completed for all pipelines


    {'nr_pr_est': 8684}
    {'mode_pulse_rate': np.float64(63.59175662414131), '99p_pulse_rate': np.float64(85.77263444520081)}





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>pulse_rate</th>
      <th>file_key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47.0</td>
      <td>80.372915</td>
      <td>PPG_segment0001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.0</td>
      <td>79.769382</td>
      <td>PPG_segment0001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51.0</td>
      <td>79.136408</td>
      <td>PPG_segment0001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53.0</td>
      <td>78.606477</td>
      <td>PPG_segment0001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.0</td>
      <td>77.870461</td>
      <td>PPG_segment0001</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Example 2: No file storage - results only in memory
results_no_storage = run_paradigma(
    dfs=dfs_ppg,
    pipelines=pipeline,
    skip_preparation=True,
    time_input_unit=TimeUnit.RELATIVE_S,
    save_intermediate=[],  # No files saved
    verbose=1
)

print("Results returned without file storage:")
print(f"  Quantifications: {len(results_no_storage['quantifications'][pipeline])} rows")
print(f"  Aggregations: {results_no_storage['aggregations'][pipeline]}")
```

    INFO: Applying ParaDigMa pipelines to provided DataFrame


    INFO: Logging to output\paradigma_run_20260129_0922.log


    INFO: Step 1: Using provided DataFrame(s) as input


    INFO: Steps 2-3: Processing 2 files individually


    INFO: Processing DataFrame 1/2: PPG_segment0001


    INFO: Step 1: Preprocessing PPG and accelerometer data


    INFO: Step 2: Extracting signal quality features


    Resampled: 1029375 -> 1030189 rows at 30.0 Hz


    INFO: Step 3: Signal quality classification


    INFO: Step 4: Pulse rate estimation


    INFO: Step 5: Quantifying pulse rate


    INFO: Pulse rate analysis completed: 830 valid pulse rate estimates from 830 total windows


    INFO: Processing DataFrame 2/2: PPG_segment0002


    INFO: Step 1: Preprocessing PPG and accelerometer data


    INFO: Step 2: Extracting signal quality features


    Resampled: 2214450 -> 2216689 rows at 30.0 Hz


    INFO: Step 3: Signal quality classification


    INFO: Step 4: Pulse rate estimation


    INFO: Step 5: Quantifying pulse rate


    INFO: Pulse rate analysis completed: 7854 valid pulse rate estimates from 7854 total windows


    INFO: Step 4: Combining quantifications from all files


    INFO: Pulse_rate: Combined 8684 windows from 2 files


    INFO: Step 5: Aggregating pulse rate results across ALL files


    INFO: Pulse rate aggregation completed with 8684 valid estimates


    INFO: ParaDigMa analysis completed for all pipelines


    Results returned without file storage:
      Quantifications: 8684 rows
      Aggregations: {'mode_pulse_rate': np.float64(63.59175662414131), '99p_pulse_rate': np.float64(85.77263444520081)}


### Example: No File Storage

If you only want to work with results in memory without saving any files, use an empty `save_intermediate` list:

Note that `run_paradigma` currently does not accept accelerometer data as a supplement to the pulse
rate pipeline for signal quality analysis. If you want to do these analyses, please check out the
[Pulse rate analysis](https://biomarkersparkinson.github.io/paradigma/tutorials/_static/pulse_rate_analysis.html)
tutorial for more info.

## 2. Multi-pipeline with prepared data

One of the key features of the orchestrator is the ability to run multiple analysis pipelines simultaneously on the same data. This is more efficient than running them separately.

### Results Structure

The multi-pipeline orchestrator returns a nested structure that organizes results by pipeline:

```python
{
    'quantifications': {
        'gait': DataFrame,      # Gait segment-level quantifications
        'tremor': DataFrame     # Tremor window-level quantifications
    },
    'aggregations': {
        'gait': {...},         # Aggregated gait metrics
        'tremor': {...}        # Aggregated tremor metrics
    },
    'metadata': {
        'gait': {...},         # Gait analysis metadata
        'tremor': {...}        # Tremor analysis metadata
    }
}
```


```python
# Load prepared IMU data
path_to_imu_data = Path('../../example_data/verily/imu')

dfs_imu = load_data_files(
    data_path=path_to_imu_data,
    file_patterns='json',
    verbose=0
)

print(f"Loaded {len(dfs_imu)} IMU files:")
for filename in dfs_imu.keys():
    df = dfs_imu[filename]
    print(f"  - {filename}: {len(df)} samples, {len(df.columns)} columns")

print(f"\nFirst 5 rows of {list(dfs_imu.keys())[0]}:")
dfs_imu[list(dfs_imu.keys())[0]].head()
```

    Loaded 2 IMU files:
      - IMU_segment0001: 3455331 samples, 7 columns
      - IMU_segment0002: 7434685 samples, 7 columns

    First 5 rows of IMU_segment0001:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>accelerometer_x</th>
      <th>accelerometer_y</th>
      <th>accelerometer_z</th>
      <th>gyroscope_x</th>
      <th>gyroscope_y</th>
      <th>gyroscope_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-0.474641</td>
      <td>-0.379426</td>
      <td>0.770335</td>
      <td>0.000000</td>
      <td>1.402439</td>
      <td>0.243902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.009933</td>
      <td>-0.472727</td>
      <td>-0.378947</td>
      <td>0.765072</td>
      <td>0.426829</td>
      <td>0.670732</td>
      <td>-0.121951</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.019867</td>
      <td>-0.471770</td>
      <td>-0.375598</td>
      <td>0.766986</td>
      <td>1.158537</td>
      <td>-0.060976</td>
      <td>-0.304878</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.029800</td>
      <td>-0.472727</td>
      <td>-0.375598</td>
      <td>0.770335</td>
      <td>1.158537</td>
      <td>-0.548780</td>
      <td>-0.548780</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.039733</td>
      <td>-0.475120</td>
      <td>-0.379426</td>
      <td>0.772249</td>
      <td>0.670732</td>
      <td>-0.609756</td>
      <td>-0.731707</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Run gait and tremor analysis on the prepared data
# Using custom output directory
results_multi_pipeline = run_paradigma(
    output_dir=Path('./output_multi'),
    dfs=dfs_imu,                        # Pre-loaded data
    skip_preparation=True,              # Data is already prepared
    pipelines=['gait', 'tremor'],       # Multiple pipelines (list format)
    watch_side='left',                  # Required for gait analysis
    save_intermediate=['quantification'],  # Store quantifications only
    verbose=1
)
```

    INFO: Applying ParaDigMa pipelines to provided DataFrame


    INFO: Logging to output_multi\paradigma_run_20260129_0925.log


    INFO: Step 1: Using provided DataFrame(s) as input


    INFO: Steps 2-3: Processing 2 files individually


    INFO: Processing DataFrame 1/2: IMU_segment0001


    INFO: Step 1: Preprocessing IMU data


    Resampled: 3455331 -> 3433961 rows at 100.0 Hz


    INFO: Step 2: Extracting gait features


    INFO: Step 3: Detecting gait


    INFO: Step 4: Extracting arm activity features


    INFO: Step 5: Filtering gait


    INFO: Step 6: Quantifying arm swing


    INFO: Gait analysis pipeline completed. Found 1228 windows of gait without other arm activities.


    INFO: Step 1: Preprocessing gyroscope data


    INFO: Step 2: Extracting tremor features


    Resampled: 3455331 -> 3433961 rows at 100.0 Hz


    INFO: Step 3: Detecting tremor


    INFO: Step 4: Quantifying tremor


    INFO: Tremor analysis completed: 728 tremor windows detected from 8584 total windows


    INFO: Processing DataFrame 2/2: IMU_segment0002


    INFO: Step 1: Preprocessing IMU data


    Resampled: 7434685 -> 7388945 rows at 100.0 Hz


    INFO: Step 2: Extracting gait features


    INFO: Step 3: Detecting gait


    INFO: Step 4: Extracting arm activity features


    INFO: Step 5: Filtering gait


    INFO: Step 6: Quantifying arm swing


    INFO: Gait analysis pipeline completed. Found 4071 windows of gait without other arm activities.


    INFO: Step 1: Preprocessing gyroscope data


    INFO: Step 2: Extracting tremor features


    Resampled: 7434685 -> 7388945 rows at 100.0 Hz


    INFO: Step 3: Detecting tremor


    INFO: Step 4: Quantifying tremor


    INFO: Tremor analysis completed: 1794 tremor windows detected from 18472 total windows


    INFO: Step 4: Combining quantifications from all files


    INFO: Gait: Combined 5299 windows from 2 files


    INFO: Step 5: Aggregating gait results across ALL files


    INFO: Aggregation completed across 4 gait segment categories


    INFO: Tremor: Combined 27056 windows from 2 files


    INFO: Step 5: Aggregating tremor results across ALL files


    INFO: Tremor aggregation completed


    INFO: Saved prepared data to output_multi\quantifications_gait.parquet


    INFO: Saved prepared data to output_multi\quantifications_tremor.parquet


    INFO: ParaDigMa analysis completed for all pipelines



```python
# Explore the results structure
print("Detailed Results Analysis:")

# Gait results
arm_swing_quantified = results_multi_pipeline['quantifications']['gait']
arm_swing_aggregates = results_multi_pipeline['aggregations']['gait']
arm_swing_meta = results_multi_pipeline['metadata']['gait']
print(f"\nArm swing quantification ({len(arm_swing_quantified)} windows):")
print(f"   Columns: {list(arm_swing_quantified.columns[:5])}... ({len(arm_swing_quantified.columns)} total)")
print(f"   Files: {arm_swing_quantified['file_key'].unique()}")

print(f"\nArm swing aggregation ({len(arm_swing_aggregates)} time ranges):")
print(f"   Gait segment categories: {list(arm_swing_aggregates.keys())}")
print(f"   Aggregates: {list(arm_swing_aggregates['0_10'].keys())}")
print(f"   Metadata first gait segment: {arm_swing_meta[1]}")

# Tremor results
tremor_quantified = results_multi_pipeline['quantifications']['tremor']
tremor_aggregates = results_multi_pipeline['aggregations']['tremor']
tremor_meta = results_multi_pipeline['metadata']['tremor']
print(f"\nTremor quantification ({len(tremor_quantified)} windows):")
print(f"   Columns: {list(tremor_quantified.columns[:5])}... ({len(tremor_quantified.columns)} total)")
print(f"   Files: {tremor_quantified['file_key'].unique()}")

print(f"\nTremor aggregation ({len(tremor_aggregates)} time ranges):")
print(f"   Aggregates: {list(tremor_aggregates.keys())}")
print(f"   Metadata first tremor segment: {tremor_meta}")
```

    Detailed Results Analysis:

    Arm swing quantification (5299 windows):
       Columns: ['gait_segment_nr', 'range_of_motion', 'peak_velocity', 'file_key']... (4 total)
       Files: ['IMU_segment0001' 'IMU_segment0002']

    Arm swing aggregation (4 time ranges):
       Gait segment categories: ['0_10', '10_20', '20_inf', '0_inf']
       Aggregates: ['duration_s', 'median_range_of_motion', '95p_range_of_motion', 'median_cov_range_of_motion', 'mean_cov_range_of_motion', 'median_peak_velocity', '95p_peak_velocity', 'median_cov_peak_velocity', 'mean_cov_peak_velocity']
       Metadata first gait segment: {'start_time_s': 2221.75, 'end_time_s': 2230.74, 'duration_unfiltered_segment_s': 12.75, 'duration_filtered_segment_s': 9.0}

    Tremor quantification (27056 windows):
       Columns: ['time', 'pred_arm_at_rest', 'pred_tremor_checked', 'tremor_power', 'file_key']... (5 total)
       Files: ['IMU_segment0001' 'IMU_segment0002']

    Tremor aggregation (4 time ranges):
       Aggregates: ['perc_windows_tremor', 'median_tremor_power', 'modal_tremor_power', '90p_tremor_power']
       Metadata first tremor segment: {'nr_valid_days': 1, 'nr_windows_total': 27056, 'nr_windows_rest': 18766}


## 3. Raw Data Processing

The orchestrator can also process raw sensor data automatically. This includes data preparation steps like format standardization, unit conversion, and orientation correction. Note that this feature has been developed on limited data examples, and therefore may not function as expected on newly observed data.

### Column Mapping for Custom Data Formats

If your raw data uses different column names than ParaDigMa's standard naming convention, use the `column_mapping` parameter to map your column names to the expected ones.

**Standard ParaDigMa column names:**
- **Required for all pipelines:**
  - `time`: Timestamp column

- **For IMU pipelines (gait, tremor):**
  - `accelerometer_x`, `accelerometer_y`, `accelerometer_z`: Accelerometer axes
  - `gyroscope_x`, `gyroscope_y`, `gyroscope_z`: Gyroscope axes

- **For PPG pipeline (pulse_rate):**
  - `ppg`: PPG signal

**Example mapping:**
```python
column_mapping = {
    'timestamp': 'time',                      # Your 'timestamp' column → ParaDigMa 'time' column
    'acceleration_x': 'accelerometer_x',      # Your 'acceleration' columns → ParaDigMa 'accelerometer' columns'
    'acceleration_y': 'accelerometer_y',
    'acceleration_z': 'accelerometer_z',
    'rotation_x': 'gyroscope_x',              # Your 'rotation' columns → ParaDigMa 'gyroscope' columns
    'rotation_y': 'gyroscope_y',
    'rotation_z': 'gyroscope_z',
}
```


```python
path_to_raw_data = Path('../../example_data/axivity')

device_orientation = ["-x", "-y", "z"]      # Sensor was worn upside-down
pipeline = 'gait'

# Working with raw data - this requires data preparation
# Using custom output directory
results_end_to_end = run_paradigma(
    output_dir=Path('./output_raw'),
    data_path=path_to_raw_data,             # Point to data folder
    skip_preparation=False,                 # ParaDigMa will prepare the data
    pipelines=pipeline,
    watch_side="left",
    time_input_unit=TimeUnit.RELATIVE_S,    # Specify time unit for raw data
    accelerometer_units='g',
    gyroscope_units='deg/s',
    target_frequency=100.0,
    device_orientation=device_orientation,
    save_intermediate=['aggregation'],      # Only save aggregations
    verbose=1,
)

print(f"\nMetadata:\n{json.dumps(results_end_to_end['metadata'][pipeline][1], indent=2)}")
print(f"\nAggregations:\n{json.dumps(results_end_to_end['aggregations'][pipeline], indent=2)}")
print("\nQuantifications (first 5 rows; each row represents a single arm swing):")
results_end_to_end['quantifications'][pipeline].head()
```

    INFO: Applying ParaDigMa pipelines to ..\..\example_data\axivity


    INFO: Logging to output_raw\paradigma_run_20260129_0925.log


    INFO: Step 1: Finding data files


    INFO: Found 1 data files in ..\..\example_data\axivity


    INFO: Steps 2-3: Processing 1 files individually


    INFO: Processing file 1/1: test_data.CWA


    INFO: Loading Axivity data from ..\..\example_data\axivity\test_data.CWA


    INFO: Loaded Axivity data: 36400 rows, 8 columns


    INFO: Starting data preparation pipeline


    INFO: Step 1: Standardizing column names


    INFO: Step 2: Converting sensor units


    INFO: Step 3: Preparing time column


    INFO: Step 4: Correcting orientation for left wrist


    INFO: Step 5: Resampling to 100.0 Hz


    INFO: Step 6: Validating prepared data


    INFO: Data preparation completed: 36433 rows, 7 columns


    INFO: Step 1: Preprocessing IMU data


    INFO: Step 2: Extracting gait features


    Resampled: 36400 -> 36433 rows at 100.0 Hz


    INFO: Step 3: Detecting gait


    INFO: Step 4: Extracting arm activity features


    INFO: Step 5: Filtering gait


    INFO: Step 6: Quantifying arm swing


    INFO: Gait analysis pipeline completed. Found 27 windows of gait without other arm activities.


    INFO: Step 4: Combining quantifications from all files


    INFO: Gait: Combined 27 windows from 1 files


    INFO: Step 5: Aggregating gait results across ALL files


    INFO: Aggregation completed across 4 gait segment categories


    INFO: Saved aggregations to output_raw\aggregations_gait.json


    INFO: ParaDigMa analysis completed for all pipelines



    Metadata:
    {
      "start_time_s": 124.5,
      "end_time_s": 127.49,
      "duration_unfiltered_segment_s": 124.5,
      "duration_filtered_segment_s": 3.0
    }

    Aggregations:
    {
      "0_10": {
        "duration_s": 0
      },
      "10_20": {
        "duration_s": 0
      },
      "20_inf": {
        "duration_s": 18.0,
        "median_range_of_motion": 7.182233339196239,
        "95p_range_of_motion": 27.529007915195255,
        "median_cov_range_of_motion": 0.19564530259481105,
        "mean_cov_range_of_motion": 0.2725453668861871,
        "median_peak_velocity": 52.92434205521389,
        "95p_peak_velocity": 258.93016146092725,
        "median_cov_peak_velocity": 0.23137490496592453,
        "mean_cov_peak_velocity": 0.2872492141424207
      },
      "0_inf": {
        "duration_s": 18.0,
        "median_range_of_motion": 7.182233339196239,
        "95p_range_of_motion": 27.529007915195255,
        "median_cov_range_of_motion": 0.19564530259481105,
        "mean_cov_range_of_motion": 0.2725453668861871,
        "median_peak_velocity": 52.92434205521389,
        "95p_peak_velocity": 258.93016146092725,
        "median_cov_peak_velocity": 0.23137490496592453,
        "mean_cov_peak_velocity": 0.2872492141424207
      }
    }

    Quantifications (first 5 rows; each row represents a single arm swing):





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gait_segment_nr</th>
      <th>range_of_motion</th>
      <th>peak_velocity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>17.182270</td>
      <td>136.030218</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22.832489</td>
      <td>181.524445</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>23.181810</td>
      <td>283.032602</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>29.694767</td>
      <td>253.460914</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>29.392093</td>
      <td>221.580715</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Auto-Segmentation for Non-Contiguous Data

When working with sensor data, you may encounter gaps or interruptions in the recording (e.g., battery died, device removed, multiple recording sessions). The orchestrator can automatically detect these gaps and split the data into contiguous segments for processing.

### When to Use Auto-Segmentation

Use `split_by_gaps=True` when:
- Your data has recording interruptions or gaps
- You're getting "Time array is not contiguous" errors
- You want to process multiple recording sessions in one file
- Data spans multiple days with breaks

### Understanding Data Segments vs Gait Segments

Important distinction:
- **Data segments (`data_segment_nr`)**: Contiguous recording chunks separated by temporal gaps
  - Created during data preparation
  - Example: 4 segments if recording had 3 interruptions

- **Gait segments (`gait_segment_nr`)**: Detected gait bouts within the data
  - Created during gait pipeline analysis
  - Example: 52 gait bouts detected across all data segments
  - Only applicable to gait analysis

The orchestrator will:
1. Detect gaps larger than `max_gap_seconds` (default: 1.5 seconds)
2. Split data into contiguous data segments
3. Discard segments shorter than `min_segment_seconds` (default: 1.5 seconds)
4. Add a `data_segment_nr` column to track which recording chunk each data point belongs to
5. Process each data segment independently through the pipeline
6. Combine results with `gait_segment_nr` for detected gait bouts (gait pipeline only)

### Example: Gait-up Physilog Data with Gaps

This example uses data from a Gait-up Physilog 4 device with 3 large gaps (up to ~20 minutes). The data is already in Parquet format with standard column names, but timestamps are non-contiguous.


```python
# Load Gait-up Physilog data with non-contiguous timestamps
path_to_physilog_data = Path('../../example_data/gait_up_physilog')

dfs_physilog = load_data_files(
    data_path=path_to_physilog_data,
    file_patterns='parquet',
    verbose=0
)

print(f"Loaded {len(dfs_physilog)} Gait-up Physilog file(s):")
for filename, df in dfs_physilog.items():
    print(f"  - {filename}: {len(df)} samples, {len(df.columns)} columns")
    print(f"    Time range: {df['time'].min():.1f}s to {df['time'].max():.1f}s")
    print(f"    Duration: {(df['time'].max() - df['time'].min()):.1f}s")

    # Check for gaps
    time_diffs = df['time'].diff().dropna()
    large_gaps = time_diffs[time_diffs > 1.0]
    if len(large_gaps) > 0:
        print(f"Contains {len(large_gaps)} gap(s) > 1s (largest: {large_gaps.max():.1f}s)")

    # Check for NaN values (common in real-world data)
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"Contains {nan_counts.sum()} NaN values")

# Clean DataFrames with NaN values (after iteration to avoid SettingWithCopyWarning)
for filename in list(dfs_physilog.keys()):
    df = dfs_physilog[filename]
    if df.isnull().sum().sum() > 0:
        df_clean = df.dropna()
        dfs_physilog[filename] = df_clean
        print(f"  Cleaned {filename} to {len(df_clean)} samples")

dfs_physilog[list(dfs_physilog.keys())[0]].head()
```

    Loaded 1 Gait-up Physilog file(s):
      - test_file: 876535 samples, 7 columns
        Time range: 0.0s to 10026.1s
        Duration: 10026.1s
    Contains 3 gap(s) > 1s (largest: 1243.6s)
    Contains 30 NaN values
      Cleaned test_file to 876525 samples





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>accelerometer_x</th>
      <th>accelerometer_y</th>
      <th>accelerometer_z</th>
      <th>gyroscope_x</th>
      <th>gyroscope_y</th>
      <th>gyroscope_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>-0.101459</td>
      <td>-0.564616</td>
      <td>0.837138</td>
      <td>0.269809</td>
      <td>-0.044764</td>
      <td>0.067904</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>-0.102069</td>
      <td>-0.565342</td>
      <td>0.835092</td>
      <td>-0.065885</td>
      <td>0.138342</td>
      <td>0.098422</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>-0.101337</td>
      <td>-0.565827</td>
      <td>0.836657</td>
      <td>-0.432096</td>
      <td>0.290930</td>
      <td>0.128939</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>-0.100482</td>
      <td>-0.562073</td>
      <td>0.838463</td>
      <td>-0.767789</td>
      <td>0.382482</td>
      <td>0.128939</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>-0.100849</td>
      <td>-0.563526</td>
      <td>0.837981</td>
      <td>-0.950895</td>
      <td>0.504553</td>
      <td>0.189975</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Example: Processing non-contiguous data with auto-segmentation
# Data already has standard column names and units, but needs segmentation

results_with_segmentation = run_paradigma(
    dfs=dfs_physilog,                     # Pre-loaded data dictionary
    skip_preparation=False,               # Need preparation to add data_segment_nr
    pipelines='gait',
    watch_side="left",
    time_input_unit=TimeUnit.RELATIVE_S,
    # Auto-segmentation parameters
    split_by_gaps=True,                   # Enable automatic segmentation
    max_gap_seconds=1.0,                  # Gaps > 1s create new data segment
    min_segment_seconds=2.0,              # Keep only data segments >= 2s
    save_intermediate=[],                 # No file storage for demo
    verbose=1,
)

gait_results = results_with_segmentation['quantifications']['gait']

print(f"\nTotal arm swings quantified: {len(gait_results)}")
print(f"Number of gait segments: {gait_results['gait_segment_nr'].nunique()}")
if 'data_segment_nr' in gait_results.columns:
    print(f"Number of data segments: {gait_results['data_segment_nr'].nunique()}")
print(f"\nColumns in output: {list(gait_results.columns)}")
gait_results.head()
```

    INFO: Applying ParaDigMa pipelines to provided DataFrame


    INFO: Logging to output\paradigma_run_20260129_0925.log


    INFO: Step 1: Using provided DataFrame(s) as input


    INFO: Steps 2-3: Processing 1 files individually


    INFO: Processing DataFrame 1/1: test_file


    INFO: Starting data preparation pipeline


    INFO: Step 1: Standardizing column names


    INFO: Step 2: Converting sensor units


    INFO: Step 3: Preparing time column


    INFO: Step 4: Correcting orientation for left wrist


    INFO: Step 5: Resampling to 100.0 Hz


    Non-contiguous data detected. Auto-segmenting...


    Created 4 segments: 1713.3s, 1588.3s, 2243.5s, 3220.3s


    INFO: Step 6: Validating prepared data


    WARNING: Time column has irregular sampling


    INFO: Data preparation completed: 876535 rows, 8 columns


    INFO: Step 1: Preprocessing IMU data


    Resampled: 876525 -> 876535 rows at 100.0 Hz


    INFO: Step 2: Extracting gait features


    INFO: Step 3: Detecting gait


    INFO: Step 4: Extracting arm activity features


    INFO: Step 5: Filtering gait


    INFO: Step 6: Quantifying arm swing


    INFO: Gait analysis pipeline completed. Found 1834 windows of gait without other arm activities.


    INFO: Step 4: Combining quantifications from all files


    INFO: Gait: Combined 1834 windows from 1 files


    INFO: Step 5: Aggregating gait results across ALL files


    INFO: Aggregation completed across 4 gait segment categories


    INFO: ParaDigMa analysis completed for all pipelines



    Total arm swings quantified: 1834
    Number of gait segments: 47

    Columns in output: ['gait_segment_nr', 'range_of_motion', 'peak_velocity']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gait_segment_nr</th>
      <th>range_of_motion</th>
      <th>peak_velocity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.394159</td>
      <td>40.731020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>13.593113</td>
      <td>21.522583</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>20.881968</td>
      <td>102.620567</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>16.711927</td>
      <td>48.794681</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>6.565145</td>
      <td>26.096350</td>
    </tr>
  </tbody>
</table>
</div>
