# Data preparation
ParaDigMa requires the sensor data to be of a specific format. This tutorial provides examples of how to prepare your input data for subsequent analysis. In the end, the input for ParaDigMa is a dataframe consisting of:
* A time column representing the seconds relative to the first row of the dataframe;
* One or multiple of the following sensor column categories:
  * Triaxial accelerometer (x, y, z) in _g_
  * Triaxial gyroscope (x, y, z) in _deg/s_
  * Photoplethysmography (PPG)

The final dataframe should be resampled to 100 Hz, have the correct units for the sensor columns, and the correct format for the time column. Also note that the _gait_ pipeline expects a specific orientation of sensor axes, as explained in [Coordinate system](../guides/coordinate_system).

## Import required modules


```python
import os
from pathlib import Path

from paradigma.constants import TimeUnit
from paradigma.util import (
    convert_units_accelerometer,
    convert_units_gyroscope,
    load_tsdf_dataframe,
    transform_time_array,
)
```

## Load data
This example uses data of the [Personalized Parkinson Project](https://pubmed.ncbi.nlm.nih.gov/31315608/), which is stored in Time Series Data Format (`TSDF`). Inertial Measurements Units (IMU) and photoplethysmography (PPG) data are sampled at a different sampling frequency and therefore stored separately. Note that ParaDigMa works independent of data storage format; it only requires a `pandas.DataFrame` as input.


```python
path_to_raw_data = Path('../../tests/data/data_preparation_tutorial')
path_to_imu_data = path_to_raw_data / 'imu'

df_imu, imu_time, imu_values = load_tsdf_dataframe(
    path_to_data=path_to_imu_data,
    prefix='imu'
)

df_imu.head()
```




<div>
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
      <td>-5.402541</td>
      <td>5.632536</td>
      <td>-2.684842</td>
      <td>-115.670732</td>
      <td>-32.012195</td>
      <td>26.097561</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.040039</td>
      <td>-5.257034</td>
      <td>6.115995</td>
      <td>-2.497091</td>
      <td>-110.609757</td>
      <td>-34.634146</td>
      <td>24.695122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.040039</td>
      <td>-4.947244</td>
      <td>6.392928</td>
      <td>-2.468928</td>
      <td>-103.231708</td>
      <td>-36.768293</td>
      <td>22.926829</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.040039</td>
      <td>-4.792349</td>
      <td>6.735574</td>
      <td>-2.605048</td>
      <td>-96.280488</td>
      <td>-38.719512</td>
      <td>21.158537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.039795</td>
      <td>-4.848675</td>
      <td>7.115770</td>
      <td>-2.731780</td>
      <td>-92.560976</td>
      <td>-41.280488</td>
      <td>20.304878</td>
    </tr>
  </tbody>
</table>
</div>




```python
path_to_ppg_data = os.path.join(path_to_raw_data, 'ppg')

df_ppg, ppg_time, ppg_values = load_tsdf_dataframe(
    path_to_data=path_to_ppg_data,
    prefix='ppg'
)

df_ppg.head()
```




<div>
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
      <td>0.000000</td>
      <td>649511</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.959961</td>
      <td>648214</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.959961</td>
      <td>646786</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.959961</td>
      <td>645334</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.960205</td>
      <td>644317</td>
    </tr>
  </tbody>
</table>
</div>



The timestamps in this dataset correspond to delta milliseconds, and the data is not uniformly distributed as can be observed.

## Prepare dataframe

#### Set column names
You are free to choose column names, although we recommend using the column names set in ParaDigMa for convenience in subsequent data processing steps. These are accessible through the class [`DataColumns`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/constants/index.html#paradigma.constants.DataColumns), which can be imported from [`paradigma.constants`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/constants/index.html). For example, we recommend setting `acc_x_colname` to `DataColumns.ACCELEROMETER_X`. Again, this is not strictly necessary for future steps.


```python
time_colname = 'time'  # DataColumns.TIME

acc_x_colname = 'accelerometer_x'  # DataColumns.ACCELEROMETER_X
acc_y_colname = 'accelerometer_y'  # DataColumns.ACCELEROMETER_Y
acc_z_colname = 'accelerometer_z'  # DataColumns.ACCELEROMETER_Z
gyr_x_colname = 'gyroscope_x'  # DataColumns.GYROSCOPE_X
gyr_y_colname = 'gyroscope_y'  # DataColumns.GYROSCOPE_Y
gyr_z_colname = 'gyroscope_z'  # DataColumns.GYROSCOPE_Z

ppg_colname = 'green'  # DataColumns.PPG
```

#### Change units
ParaDigMa expects acceleration to be measured in g, and rotation in deg/s. Units can be converted conveniently using ParaDigMa functionalities.


```python
# Set to units of the sampled data
accelerometer_units = 'm/s^2'
gyroscope_units = 'deg/s'

# State the column names
accelerometer_columns = [acc_x_colname, acc_y_colname, acc_z_colname]
gyroscope_columns = [gyr_x_colname, gyr_y_colname, gyr_z_colname]

accelerometer_data = df_imu[accelerometer_columns].values
gyroscope_data = df_imu[gyroscope_columns].values

# Convert units to expected format
df_imu[accelerometer_columns] = convert_units_accelerometer(
    data=accelerometer_data,
    units=accelerometer_units
)
df_imu[gyroscope_columns] = convert_units_gyroscope(
    data=gyroscope_data,
    units=gyroscope_units
)

df_imu.head()
```




<div>
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
      <td>-0.550718</td>
      <td>0.574163</td>
      <td>-0.273684</td>
      <td>-115.670732</td>
      <td>-32.012195</td>
      <td>26.097561</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.040039</td>
      <td>-0.535885</td>
      <td>0.623445</td>
      <td>-0.254545</td>
      <td>-110.609757</td>
      <td>-34.634146</td>
      <td>24.695122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.040039</td>
      <td>-0.504306</td>
      <td>0.651675</td>
      <td>-0.251675</td>
      <td>-103.231708</td>
      <td>-36.768293</td>
      <td>22.926829</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.040039</td>
      <td>-0.488517</td>
      <td>0.686603</td>
      <td>-0.265550</td>
      <td>-96.280488</td>
      <td>-38.719512</td>
      <td>21.158537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.039795</td>
      <td>-0.494258</td>
      <td>0.725359</td>
      <td>-0.278469</td>
      <td>-92.560976</td>
      <td>-41.280488</td>
      <td>20.304878</td>
    </tr>
  </tbody>
</table>
</div>



#### Account for watch side
For the Gait & Arm Swing pipeline, it is essential to ensure correct sensor axes orientation. For more information please read [Coordinate System](../guides/coordinate_system) and set the axes of the data accordingly.


```python
# Change the orientation of the sensor according to the documented coordinate system.
# The following changes are specific to the used sensor and its orientation
# relative to predefined coordinate system.
df_imu[acc_y_colname] *= -1
df_imu[acc_z_colname] *= -1
df_imu[gyr_y_colname] *= -1
df_imu[gyr_z_colname] *= -1

df_imu.head()
```




<div>
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
      <td>-0.550718</td>
      <td>-0.574163</td>
      <td>0.273684</td>
      <td>-115.670732</td>
      <td>32.012195</td>
      <td>-26.097561</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.040039</td>
      <td>-0.535885</td>
      <td>-0.623445</td>
      <td>0.254545</td>
      <td>-110.609757</td>
      <td>34.634146</td>
      <td>-24.695122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.040039</td>
      <td>-0.504306</td>
      <td>-0.651675</td>
      <td>0.251675</td>
      <td>-103.231708</td>
      <td>36.768293</td>
      <td>-22.926829</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.040039</td>
      <td>-0.488517</td>
      <td>-0.686603</td>
      <td>0.265550</td>
      <td>-96.280488</td>
      <td>38.719512</td>
      <td>-21.158537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.039795</td>
      <td>-0.494258</td>
      <td>-0.725359</td>
      <td>0.278469</td>
      <td>-92.560976</td>
      <td>41.280488</td>
      <td>-20.304878</td>
    </tr>
  </tbody>
</table>
</div>



#### Change time column
ParaDigMa expects the data to be in seconds relative to the first row, which should be equal to 0. The toolbox has the built-in function [`transform_time_array`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/util/index.html#paradigma.util.transform_time_array) to help users transform their time column to the correct format if the timestamps have been sampled in delta time between timestamps. In the near future, the functionalities for transforming other types (e.g., datetime format) shall be provided.


```python
df_imu[time_colname] = transform_time_array(
    time_array=df_imu[time_colname],
    input_unit_type=TimeUnit.DIFFERENCE_MS,
    output_unit_type=TimeUnit.RELATIVE_S,
)

df_imu.head()
```




<div>
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
      <td>0.00000</td>
      <td>-0.550718</td>
      <td>-0.574163</td>
      <td>0.273684</td>
      <td>-115.670732</td>
      <td>32.012195</td>
      <td>-26.097561</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01004</td>
      <td>-0.535885</td>
      <td>-0.623445</td>
      <td>0.254545</td>
      <td>-110.609757</td>
      <td>34.634146</td>
      <td>-24.695122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02008</td>
      <td>-0.504306</td>
      <td>-0.651675</td>
      <td>0.251675</td>
      <td>-103.231708</td>
      <td>36.768293</td>
      <td>-22.926829</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03012</td>
      <td>-0.488517</td>
      <td>-0.686603</td>
      <td>0.265550</td>
      <td>-96.280488</td>
      <td>38.719512</td>
      <td>-21.158537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04016</td>
      <td>-0.494258</td>
      <td>-0.725359</td>
      <td>0.278469</td>
      <td>-92.560976</td>
      <td>41.280488</td>
      <td>-20.304878</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ppg[time_colname] = transform_time_array(
    time_array=df_ppg[time_colname],
    input_unit_type=TimeUnit.DIFFERENCE_MS,
    output_unit_type=TimeUnit.RELATIVE_S,
)

df_ppg.head()
```




<div>
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
      <td>0.00000</td>
      <td>649511</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00996</td>
      <td>648214</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.01992</td>
      <td>646786</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.02988</td>
      <td>645334</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.03984</td>
      <td>644317</td>
    </tr>
  </tbody>
</table>
</div>



These dataframes are ready to be processed by ParaDigMa.

## Note on Non-Contiguous Data

If your data has gaps or interruptions (e.g., battery died, device removed, multiple recording sessions), you have two options:

### Option 1: Auto-segmentation (Recommended)
Use the `split_by_gaps=True` parameter in `run_paradigma()`:

```python
from paradigma.orchestrator import run_paradigma

results = run_paradigma(
    dfs=df_prepared,
    pipelines='gait',
    watch_side='left',
    split_by_gaps=True,             # Automatically handle gaps
    max_gap_seconds=1.5,            # Split on gaps > 1.5s
    min_segment_seconds=2.0,        # Discard segments < 2s
)
```

This will:
- Automatically detect and split data at gaps larger than `max_gap_seconds`
- Discard segments shorter than `min_segment_seconds`
- Process each contiguous segment independently
- Add a `data_segment_nr` column to track recording chunks

### Option 2: Manual segmentation
Create segments yourself using functions from [segmenting.py](../../src/paradigma/segmenting.py):

```python
from paradigma.segmenting import create_segments, discard_segments

# Create segments based on time gaps
df_prepared['data_segment_nr'] = create_segments(
    time_array=df_prepared['time'].values,
    max_segment_gap_s=1.5
)

# Optionally discard short segments
df_prepared = discard_segments(
    df=df_prepared,
    segment_nr_colname='data_segment_nr',
    min_segment_length_s=2.0,
    fs=100,  # sampling frequency
    format='timestamps'
)

# Then run paradigma normally (split_by_gaps=False)
results = run_paradigma(
    dfs=df_prepared,
    pipelines='gait',
    watch_side='left',
    split_by_gaps=False  # Data already segmented
)
```

For more details and examples, see the [Pipeline Orchestrator Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/pipeline_orchestrator.html#auto-segmentation-for-non-contiguous-data).
