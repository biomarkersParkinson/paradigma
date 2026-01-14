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

### Important required modules


```python
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
- A **single DataFrame**: Will be processed with key `'segment_1'`
- A **list of DataFrames**: Each will get keys like `'segment_1'`, `'segment_2'`, etc.

This means ParaDigMa can run multiple files in sequence. This is useful when you have multiple files
spanning a week, and you want aggregations to be computed across all files.


```python
path_to_ppg_data = Path('../../example_data/verily/ppg')

dfs_ppg = load_data_files(
    data_path=path_to_ppg_data,
    file_patterns='json',
    verbosity=0
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
- No storage: Files are only saved if `store_intermediate` is not empty

**Store Intermediate Results:**

The `store_intermediate` parameter accepts a list of strings:
```python
store_intermediate=['preprocessing', 'quantification', 'aggregation']
```

Valid options are:
- `'preparation'`: Save prepared data
- `'preprocessing'`: Save preprocessed signals
- `'classification'`: Save classification results
- `'quantification'`: Save quantified measures
- `'aggregation'`: Save aggregated results

If `store_intermediate=[]` (empty list), **no files are saved** - results are only returned in memory.

Also, set the correct units of the `time` column. For all options, please check [the API reference](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/constants/index.html#paradigma.constants.TimeUnit).


```python
pipeline = 'pulse_rate'

# Example 1: Using default output directory with storage
results_single_pipeline = run_paradigma(
    dfs=dfs_ppg,
    pipeline_names=pipeline,
    data_prepared=True,
    time_input_unit=TimeUnit.RELATIVE_S,
    store_intermediate=['quantification', 'aggregation'],  # Files saved to ./output
    verbosity=1
)

print(results_single_pipeline['metadata'][pipeline])
print(results_single_pipeline['aggregations'][pipeline])
results_single_pipeline['quantifications'][pipeline].head()
```

    INFO: Applying ParaDigMa pipelines to provided DataFrame


    INFO: Logging to output\paradigma_run_20260114_1954.log


    INFO: Step 1: Using provided DataFrame(s) as input


    INFO: Step 2: Data already prepared, skipping preparation


    INFO: Step 3: Running pipelines ['pulse_rate'] on 2 data files


    INFO: Running pulse_rate pipeline


    INFO: Processing file: PPG_segment0001 (1/2)


    INFO: Step 1: Preprocessing PPG and accelerometer data


    INFO: Step 2: Extracting signal quality features


    INFO: Step 3: Signal quality classification


    INFO: Step 4: Pulse rate estimation


    INFO: Step 5: Quantifying pulse rate


    INFO: Saved pulse rate quantification to output\individual_files\PPG_segment0001\quantification


    INFO: Pulse rate analysis completed: 830 valid pulse rate estimates from 830 total windows


    INFO: Processing file: PPG_segment0002 (2/2)


    INFO: Step 1: Preprocessing PPG and accelerometer data


    INFO: Step 2: Extracting signal quality features


    INFO: Step 3: Signal quality classification


    INFO: Step 4: Pulse rate estimation


    INFO: Step 5: Quantifying pulse rate


    INFO: Saved pulse rate quantification to output\individual_files\PPG_segment0002\quantification


    INFO: Pulse rate analysis completed: 7854 valid pulse rate estimates from 7854 total windows


    INFO: Combined results: 8684 windows from 2 data files


    INFO: Aggregating pulse rate results across all data files


    INFO: Pulse rate aggregation completed with 8684 valid estimates


    INFO: Saved prepared data to output\quantifications_pulse_rate.parquet


    INFO: Results saved to output


    INFO: Pulse_rate pipeline completed


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
    pipeline_names=pipeline,
    data_prepared=True,
    time_input_unit=TimeUnit.RELATIVE_S,
    store_intermediate=[],  # No files saved
    verbosity=1
)

print(f"Results returned without file storage:")
print(f"  Quantifications: {len(results_no_storage['quantifications'][pipeline])} rows")
print(f"  Aggregations: {results_no_storage['aggregations'][pipeline]}")
```

    INFO: Applying ParaDigMa pipelines to provided DataFrame


    INFO: Logging to output\paradigma_run_20260114_1957.log


    INFO: Step 1: Using provided DataFrame(s) as input


    INFO: Step 2: Data already prepared, skipping preparation


    INFO: Step 3: Running pipelines ['pulse_rate'] on 2 data files


    INFO: Running pulse_rate pipeline


    INFO: Processing file: PPG_segment0001 (1/2)


    ERROR: Failed to process data file PPG_segment0001: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'


    INFO: Processing file: PPG_segment0002 (2/2)


    ERROR: Failed to process data file PPG_segment0002: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'


    WARNING: No quantified pulse_rate found in any data file


    INFO: Pulse_rate pipeline completed


    INFO: ParaDigMa analysis completed for all pipelines


    Results returned without file storage:
      Quantifications: 0 rows
      Aggregations: {}


### Example: No File Storage

If you only want to work with results in memory without saving any files, use an empty `store_intermediate` list:

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
    verbosity=0
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
    data_prepared=True,                 # Data is already prepared
    pipeline_names=['gait', 'tremor'],  # Multiple pipelines (list format)
    watch_side='left',                  # Required for gait analysis
    store_intermediate=['quantification'],  # Store quantifications only
    verbosity=1
)
```

    INFO: Applying ParaDigMa pipelines to provided DataFrame


    INFO: Logging to output_multi\paradigma_run_20260114_1957.log


    INFO: Step 1: Using provided DataFrame(s) as input


    INFO: Step 2: Data already prepared, skipping preparation


    INFO: Step 3: Running pipelines ['gait', 'tremor'] on 2 data files


    INFO: Running gait pipeline


    INFO: Processing file: IMU_segment0001 (1/2)


    INFO: Step 1: Preprocessing IMU data


    INFO: Step 2: Extracting gait features


    INFO: Step 3: Detecting gait


    INFO: Step 4: Extracting arm activity features


    INFO: Step 5: Filtering gait


    INFO: Step 6: Quantifying arm swing


    INFO: Gait analysis pipeline completed. Found 1228 windows of gait without other arm activities.


    INFO: Processing file: IMU_segment0002 (2/2)


    INFO: Step 1: Preprocessing IMU data


    INFO: Step 2: Extracting gait features


    INFO: Step 3: Detecting gait


    INFO: Step 4: Extracting arm activity features


    INFO: Step 5: Filtering gait


    INFO: Step 6: Quantifying arm swing


    INFO: Gait analysis pipeline completed. Found 4071 windows of gait without other arm activities.


    INFO: Combined results: 5299 windows from 2 data files


    INFO: Aggregating gait results across all data files


    INFO: Aggregation completed across 4 gait segment length categories


    INFO: Saved prepared data to output_multi\quantifications_gait.parquet


    INFO: Results saved to output_multi


    INFO: Gait pipeline completed


    INFO: Running tremor pipeline


    INFO: Processing file: IMU_segment0001 (1/2)


    INFO: Step 1: Preprocessing gyroscope data


    INFO: Step 2: Extracting tremor features


    INFO: Step 3: Detecting tremor


    INFO: Step 4: Quantifying tremor


    INFO: Tremor analysis completed: 728 tremor windows detected from 8584 total windows


    INFO: Processing file: IMU_segment0002 (2/2)


    INFO: Step 1: Preprocessing gyroscope data


    INFO: Step 2: Extracting tremor features


    INFO: Step 3: Detecting tremor


    INFO: Step 4: Quantifying tremor


    INFO: Tremor analysis completed: 1794 tremor windows detected from 18472 total windows


    INFO: Combined results: 27056 windows from 2 data files


    INFO: Aggregating tremor results across all data files


    INFO: Tremor aggregation completed


    INFO: Saved prepared data to output_multi\quantifications_tremor.parquet


    INFO: Results saved to output_multi


    INFO: Tremor pipeline completed


    INFO: ParaDigMa analysis completed for all pipelines



```python
results_multi_pipeline.keys()
```




    dict_keys(['quantifications', 'aggregations', 'metadata'])




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
       Columns: ['segment_nr', 'range_of_motion', 'peak_velocity', 'file_key']... (4 total)
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


```python
path_to_raw_data = Path('../../example_data/axivity')

device_orientation = ["-x", "-y", "z"]      # Sensor was worn upside-down
pipeline = 'gait'

# Working with raw data - this requires data preparation
# Using custom output directory
results_end_to_end = run_paradigma(
    output_dir=Path('./output_raw'),
    data_path=path_to_raw_data,             # Point to data folder
    data_prepared=False,                    # ParaDigMa will prepare the data
    pipeline_names=pipeline,
    watch_side="left",
    time_input_unit=TimeUnit.RELATIVE_S,    # Specify time unit for raw data
    accelerometer_units='g',
    gyroscope_units='deg/s',
    target_frequency=100.0,
    device_orientation=device_orientation,
    store_intermediate=['aggregation'],     # Only save aggregations
    verbosity=1,
)

print(results_end_to_end['metadata'][pipeline][1])
print(results_end_to_end['aggregations'][pipeline])
results_end_to_end['quantifications'][pipeline].head()
```

    INFO: Applying ParaDigMa pipelines to ..\..\example_data\axivity


    INFO: Logging to output_raw\paradigma_run_20260114_1957.log


    INFO: Step 1: Loading data files


    INFO: Loading Axivity data from ..\..\example_data\axivity\test_data.CWA


    INFO: Loaded Axivity data: 36400 rows, 8 columns


    INFO: Successfully loaded 1 files


    INFO: Loaded 1 data files


    INFO: Step 2: Preparing raw data


    INFO: Starting data preparation pipeline


    INFO: Step 1: Standardizing column names


    INFO: Step 2: Converting sensor units


    INFO: Step 3: Preparing time column


    INFO: Step 4: Correcting orientation for left wrist


    INFO: Step 5: Resampling to 100.0 Hz


    INFO: Step 6: Validating prepared data


    INFO: Data preparation completed: 36400 rows, 8 columns


    INFO: Successfully prepared 1 data files


    INFO: Step 3: Running pipelines ['gait'] on 1 data files


    INFO: Running gait pipeline


    INFO: Processing file: test_data (1/1)


    INFO: Step 1: Preprocessing IMU data


    INFO: Step 2: Extracting gait features


    INFO: Step 3: Detecting gait


    INFO: Step 4: Extracting arm activity features


    INFO: Step 5: Filtering gait


    INFO: Step 6: Quantifying arm swing


    INFO: Gait analysis pipeline completed. Found 27 windows of gait without other arm activities.


    INFO: Combined results: 27 windows from 1 data files


    INFO: Aggregating gait results across all data files


    INFO: Aggregation completed across 4 gait segment length categories


    INFO: Results saved to output_raw


    INFO: Gait pipeline completed


    INFO: ParaDigMa analysis completed for all pipelines


    {'start_time_s': 124.5, 'end_time_s': 127.49, 'duration_unfiltered_segment_s': 124.5, 'duration_filtered_segment_s': 3.0}
    {'0_10': {'duration_s': 0}, '10_20': {'duration_s': 0}, '20_inf': {'duration_s': 18.0, 'median_range_of_motion': np.float64(7.182233339196239), '95p_range_of_motion': np.float64(27.529007915195255), 'median_cov_range_of_motion': np.float64(0.19564530259481105), 'mean_cov_range_of_motion': np.float64(0.2725453668861871), 'median_peak_velocity': np.float64(52.92434205521389), '95p_peak_velocity': np.float64(258.93016146092725), 'median_cov_peak_velocity': np.float64(0.23137490496592453), 'mean_cov_peak_velocity': np.float64(0.2872492141424207)}, '0_inf': {'duration_s': 18.0, 'median_range_of_motion': np.float64(7.182233339196239), '95p_range_of_motion': np.float64(27.529007915195255), 'median_cov_range_of_motion': np.float64(0.19564530259481105), 'mean_cov_range_of_motion': np.float64(0.2725453668861871), 'median_peak_velocity': np.float64(52.92434205521389), '95p_peak_velocity': np.float64(258.93016146092725), 'median_cov_peak_velocity': np.float64(0.23137490496592453), 'mean_cov_peak_velocity': np.float64(0.2872492141424207)}}





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
      <th>segment_nr</th>
      <th>range_of_motion</th>
      <th>peak_velocity</th>
      <th>file_key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>17.182270</td>
      <td>136.030218</td>
      <td>test_data</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22.832489</td>
      <td>181.524445</td>
      <td>test_data</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>23.181810</td>
      <td>283.032602</td>
      <td>test_data</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>29.694767</td>
      <td>253.460914</td>
      <td>test_data</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>29.392093</td>
      <td>221.580715</td>
      <td>test_data</td>
    </tr>
  </tbody>
</table>
</div>
