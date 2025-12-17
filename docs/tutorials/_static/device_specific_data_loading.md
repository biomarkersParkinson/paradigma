# Device specific data loading
This tutorial demonstrates how to load sensor data of the following devices into memory:
- Axivity
- Empatica

Note that Paradigma requires further data preparation as outlined in [the data preparation tutorial](../data_preparation.ipynb).

### Axivity
Axivity sensor data (AX3 & AX6) are stored in .CWA format, which requires some preparation to be processable. In this tutorial, we showcase how to transform .CWA files into a workable format in Python using `openmovement`. More information on the `openmovement` package can be found on the [Open Movement GitHub page](https://github.com/openmovementproject/openmovement-python).

For the `openmovement` package, make sure to install the *master* branch, as this branch contains the valid code for preparing .CWA data. This can for example be done using `pip` by running `pip install git+https://github.com/digitalinteraction/openmovement-python.git@master`. Or, when using Poetry, add the following line to the list of dependencies in `pyproject.toml`: `openmovement = { git = "https://github.com/digitalinteraction/openmovement-python.git", branch = "master" }`.


```python
import pandas as pd

from openmovement.load import CwaData
from pathlib import Path
from pprint import pprint

# Load data
path_to_input_data = Path('../../example_data/axivity/')
test_data_filename = 'test_data.CWA'
prepared_data_filename = 'test_data.parquet'

# Note: Set include_gyro to False when using AX3 devices without gyroscope,
# or when gyroscope data is not needed
with CwaData(
    filename=path_to_input_data / test_data_filename,
    include_gyro=True,
    include_temperature=False
    ) as cwa_data:
    print("Data format info:")
    pprint(cwa_data.data_format)

    df = cwa_data.get_samples()  # Load all samples into a DataFrame

# Set time to start at 0 seconds
df['time_dt'] = df['time'].copy()
df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

df.head()
```

    Data format info:
    {'accelAxis': 3,
     'accelRange': 8,
     'accelUnit': 4096,
     'battery': 3.955078125,
     'bytesPerAxis': 2,
     'bytesPerSample': 12,
     'channels': 6,
     'deviceFractional': 56510,
     'estimatedAfterLastSampleTime': 1763370002.7245483,
     'estimatedFirstSampleTime': 1763370002.3245482,
     'events': 1,
     'frequency': 100.0,
     'gyroAxis': 0,
     'gyroRange': 1000,
     'gyroUnit': 32.768,
     'light': 17,
     'numAxesBPS': 98,
     'rateCode': 74,
     'sampleCount': 40,
     'samplesPerSector': 40,
     'sequenceId': 0,
     'sessionId': 0,
     'temperature': 26.171875,
     'timestamp': 1763370002.7245483,
     'timestampOffset': 40,
     'timestampTime': '2025-11-17 09:00:02.724'}





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
      <th>accel_x</th>
      <th>accel_y</th>
      <th>accel_z</th>
      <th>gyro_x</th>
      <th>gyro_y</th>
      <th>gyro_z</th>
      <th>time_dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.837646</td>
      <td>-0.139404</td>
      <td>-0.514160</td>
      <td>-5.981445</td>
      <td>-1.312256</td>
      <td>-10.620117</td>
      <td>2025-11-17 09:00:02.324188160</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.010009</td>
      <td>0.814453</td>
      <td>-0.163086</td>
      <td>-0.491699</td>
      <td>-3.540039</td>
      <td>-2.685547</td>
      <td>-7.141113</td>
      <td>2025-11-17 09:00:02.334197248</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.020018</td>
      <td>0.805908</td>
      <td>-0.148926</td>
      <td>-0.484375</td>
      <td>-1.251221</td>
      <td>-6.530762</td>
      <td>-6.439209</td>
      <td>2025-11-17 09:00:02.344206336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.030027</td>
      <td>0.798096</td>
      <td>-0.148926</td>
      <td>-0.485107</td>
      <td>0.183105</td>
      <td>-11.138916</td>
      <td>-4.852295</td>
      <td>2025-11-17 09:00:02.354215168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.040036</td>
      <td>0.796143</td>
      <td>-0.163574</td>
      <td>-0.480957</td>
      <td>2.777100</td>
      <td>-16.235352</td>
      <td>-3.112793</td>
      <td>2025-11-17 09:00:02.364224256</td>
    </tr>
  </tbody>
</table>
</div>



### Empatica
Empatica sensor data is stored in Aoache Avro (.avro) format. In short, Empatica automatically writes sensor data every 30 minutes to a cloud storage (AWS) with the naming convention [participant_id]_[timestamp].avro. In this tutorial we will show how to read and prepare a single .avro file.

For more detailed documentation on using this data format in Python, consider reading [the official Apache Avro documentation](https://avro.apache.org/docs/). Extensive documentation is available on how to read and write .avro files in Python [here](https://avro.apache.org/docs/++version++/getting-started-python/).


```python
import json

from pathlib import Path

from avro.datafile import DataFileReader
from avro.io import DatumReader

path_to_input_data = Path('../../example_data/empatica/')
empatica_data_filename = 'test_data.avro'

## Read Avro file
# reader = DataFileReader(open(path_to_empatica_data / empatica_data_filename, "rb"), DatumReader())
with open(path_to_input_data / empatica_data_filename, "rb") as f:
    reader = DataFileReader(f, DatumReader())

    schema = json.loads(reader.meta.get("avro.schema").decode("utf-8"))
    empatica_data = next(reader)

accel_data = empatica_data['rawData']['accelerometer']

# The example data does not contain gyroscope data, but if it did, you could access it like this:
# gyro_data = empatica_data['rawData']['gyroscope']

# To convert accelerometer and gyroscope data into the correct format, we need to
# check the Avro schema version. This converts accelerometer into g (9.81 m/s²) units,
# and gyroscope into degrees per second (rad/s). More info on units and conversion
# can be found in the schema object using: print(schema).

avro_version = (
    (empatica_data["schemaVersion"]["major"]),
    (empatica_data["schemaVersion"]["minor"]),
    (empatica_data["schemaVersion"]["patch"]),
)

if avro_version < (6, 5, 0):
    physical_range = accel_data["imuParams"]["physicalMax"] - accel_data["imuParams"]["physicalMin"]
    digital_range = accel_data["imuParams"]["digitalMax"] - accel_data["imuParams"]["digitalMin"]
    accel_x = [val * physical_range / digital_range for val in accel_data["x"]]
    accel_y = [val * physical_range / digital_range for val in accel_data["y"]]
    accel_z = [val * physical_range / digital_range for val in accel_data["z"]]
else:
    conversion_factor = accel_data["imuParams"]["conversionFactor"]
    accel_x = [val * conversion_factor for val in accel_data["x"]]
    accel_y = [val * conversion_factor for val in accel_data["y"]]
    accel_z = [val * conversion_factor for val in accel_data["z"]]

sampling_frequency = accel_data['samplingFrequency']
nrows = len(accel_x)

t_start = accel_data['timestampStart']
t_array = [t_start + i * (1e6 /sampling_frequency) for i in range(nrows)]
t_from_0_array = ([(x - t_array[0]) / 1e6 for x in t_array])

df = pd.DataFrame({
    'time': t_from_0_array,
    'time_dt': pd.to_datetime(t_array, unit='us'),
    'accel_x': accel_x,
    'accel_y': accel_y,
    'accel_z': accel_z,
})

print(f"Data loaded from Avro file with {nrows} rows sampled at {sampling_frequency} Hz.")
print(f"Start time: {pd.to_datetime(t_start, unit='us')}")

df.head()
```

    Data loaded from Avro file with 115904 rows sampled at 63.99989700317383 Hz.
    Start time: 2025-11-26 10:41:54.256034





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
      <th>time_dt</th>
      <th>accel_x</th>
      <th>accel_y</th>
      <th>accel_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>2025-11-26 10:41:54.256034</td>
      <td>-0.999424</td>
      <td>-0.067832</td>
      <td>0.197152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.015625</td>
      <td>2025-11-26 10:41:54.271659</td>
      <td>-1.003328</td>
      <td>-0.063440</td>
      <td>0.187392</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.031250</td>
      <td>2025-11-26 10:41:54.287284</td>
      <td>-0.989664</td>
      <td>-0.072712</td>
      <td>0.184952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.046875</td>
      <td>2025-11-26 10:41:54.302909</td>
      <td>-0.996008</td>
      <td>-0.069784</td>
      <td>0.176656</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.062500</td>
      <td>2025-11-26 10:41:54.318534</td>
      <td>-0.998936</td>
      <td>-0.057584</td>
      <td>0.183000</td>
    </tr>
  </tbody>
</table>
</div>
