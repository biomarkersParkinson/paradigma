# Data preparation
ParaDigMa requires the sensor data to be of a specific format. This tutorial provides examples of how to prepare your input data for subsequent analysis. In the end, the input for ParaDigMa is a dataframe consisting of:
* A column `time`, representing the seconds relative to the first row of the dataframe;
* One or multiple of the following sensor column categories:
  * Accelerometer: `accelerometer_x`, `accelerometer_y` and `accelerometer_z` in _g_
  * Gyroscope: `gyroscope_x`, `gyroscope_y` and `gyroscope_z` in _deg/s_
  * PPG: `green` 

The final dataframe should be resampled to 100 Hz, have the correct units for the sensor columns, and the correct format for the `time` column. Also note that the _gait_ pipeline expects a specific orientation of sensor axes, as explained in [Coordinate system](../guides/coordinate_system.md).

## Load data
This example uses data of the [Personalized Parkinson Project](https://pubmed.ncbi.nlm.nih.gov/31315608/), which is stored in Time Series Data Format (TSDF). Inertial Measurements Units (IMU) and photoplethysmography (PPG) data are sampled at a different sampling frequency and therefore stored separately. Note that ParaDigMa works independent of data storage format; it only requires a `pandas` dataframe as input.


```python
from pathlib import Path
from paradigma.util import load_tsdf_dataframe

path_to_raw_data = Path('../../tests/data/data_preparation_tutorial')
path_to_imu_data = path_to_raw_data / 'imu'

df_imu, imu_time, imu_values = load_tsdf_dataframe(
    path_to_data=path_to_imu_data, 
    prefix='IMU'
)

df_imu.head()
```




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
      <th>acceleration_x</th>
      <th>acceleration_y</th>
      <th>acceleration_z</th>
      <th>rotation_x</th>
      <th>rotation_y</th>
      <th>rotation_z</th>
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
import os
from paradigma.util import load_tsdf_dataframe

path_to_ppg_data = os.path.join(path_to_raw_data, 'ppg')

df_ppg, ppg_time, ppg_values = load_tsdf_dataframe(
    path_to_data=path_to_ppg_data, 
    prefix='PPG'
)

df_ppg.head()
```




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

#### Change column names
To safeguard robustness of the pipeline, ParaDigMa fixes column names to a predefined standard.


```python
from paradigma.constants import DataColumns

accelerometer_columns = [DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]
gyroscope_columns = [DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z]

# Rename dataframe columns
df_imu = df_imu.rename(columns={
    'time': DataColumns.TIME,
    'acceleration_x': DataColumns.ACCELEROMETER_X,
    'acceleration_y': DataColumns.ACCELEROMETER_Y,
    'acceleration_z': DataColumns.ACCELEROMETER_Z,
    'rotation_x': DataColumns.GYROSCOPE_X,
    'rotation_y': DataColumns.GYROSCOPE_Y,
    'rotation_z': DataColumns.GYROSCOPE_Z,
})

# Set columns to a fixed order
df_imu = df_imu[[DataColumns.TIME] + accelerometer_columns + gyroscope_columns]

df_imu.head()
```




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
from paradigma.constants import DataColumns

ppg_columns = [DataColumns.PPG]

# Rename dataframe columns
df_ppg = df_ppg.rename(columns={
    'time': DataColumns.TIME,
    'ppg': DataColumns.PPG,
})

# Set columns to a fixed order
df_ppg = df_ppg[[DataColumns.TIME] + ppg_columns]

df_ppg.head()
```




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



#### Change units
ParaDigMa expects acceleration to be measured in g, and rotation in deg/s. Units can be converted conveniently using ParaDigMa functionalities.


```python
from paradigma.util import convert_units_accelerometer, convert_units_gyroscope

accelerometer_units = 'm/s^2'
gyroscope_units = 'deg/s'

accelerometer_data = df_imu[accelerometer_columns].values
gyroscope_data = df_imu[gyroscope_columns].values

# Convert units to expected format
df_imu[accelerometer_columns] = convert_units_accelerometer(accelerometer_data, accelerometer_units)
df_imu[gyroscope_columns] = convert_units_gyroscope(gyroscope_data, gyroscope_units)

df_imu.head()
```




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
For the Gait & Arm Swing pipeline, it is essential to ensure correct sensor axes orientation. For more information please read [Coordinate System](../guides/coordinate_system.md) and set the axes of the data accordingly.


```python
# Change the orientation of the sensor according to the documented coordinate system
df_imu[DataColumns.ACCELEROMETER_Y] *= -1
df_imu[DataColumns.ACCELEROMETER_Z] *= -1
df_imu[DataColumns.GYROSCOPE_Y] *= -1
df_imu[DataColumns.GYROSCOPE_Z] *= -1

df_imu.head()
```




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
ParaDigMa expects the data to be in seconds relative to the first row, which should be equal to 0. The toolbox has the built-in function `transform_time_array` to help users transform their time column to the correct format if the timestamps have been sampled in delta time between timestamps. In the near future, the functionalities for transforming other types (e.g., datetime format) shall be provided.


```python
from paradigma.constants import TimeUnit
from paradigma.util import transform_time_array

df_imu[DataColumns.TIME] = transform_time_array(
    time_array=df_imu[DataColumns.TIME], 
    input_unit_type=TimeUnit.DIFFERENCE_MS, 
    output_unit_type=TimeUnit.RELATIVE_S,
)

df_imu.head()
```




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
from paradigma.constants import TimeUnit
from paradigma.util import transform_time_array

df_ppg[DataColumns.TIME] = transform_time_array(
    time_array=df_ppg[DataColumns.TIME], 
    input_unit_type=TimeUnit.DIFFERENCE_MS, 
    output_unit_type=TimeUnit.RELATIVE_S,
)

df_ppg.head()
```




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
