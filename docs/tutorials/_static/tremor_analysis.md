# Tremor analysis

This tutorial shows how to run the tremor pipeline to obtain aggregated tremor measures from gyroscope sensor data. Before following along, make sure all data preparation steps have been followed in the data preparation tutorial.

In this tutorial, we use two days of data from a participant of the Personalized Parkinson Project to demonstrate the functionalities. Since `ParaDigMa` expects contiguous time series, the collected data was stored in two segments each with contiguous timestamps. Per segment, we load the data and perform the following steps:
1. Preprocess the time series data
2. Extract tremor features
3. Detect tremor
4. Quantify tremor

We then combine the output of the different segments for the final step:

5. Compute aggregated tremor measures

## Load example data

Here, we start by loading a single contiguous time series (segment), for which we continue running steps 1-3. [Below](#multiple_segments_cell) we show how to run these steps for multiple segments.

We use the interally developed `TSDF` ([documentation](https://biomarkersparkinson.github.io/tsdf/)) to load and store data [[1](https://arxiv.org/abs/2211.11294)]. Depending on the file extension of your time series data, examples of other Python functions for loading the data into memory include:
- _.csv_: `pandas.read_csv()` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html))
- _.json_: `json.load()` ([documentation](https://docs.python.org/3/library/json.html#json.load))


```python
from pathlib import Path
from paradigma.util import load_tsdf_dataframe

# Set the path to where the prepared data is saved and load the data.
# Note: the test data is stored in TSDF, but you can load your data in your own way
path_to_data =  Path('../../example_data')
path_to_prepared_data = path_to_data / 'imu'

segment_nr  = '0001'

df_data, metadata_time, metadata_values = load_tsdf_dataframe(path_to_prepared_data, prefix=f'IMU_segment{segment_nr}')

df_data
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3455326</th>
      <td>34339.561333</td>
      <td>-0.257895</td>
      <td>-0.319139</td>
      <td>-0.761244</td>
      <td>159.329269</td>
      <td>14.634146</td>
      <td>-28.658537</td>
    </tr>
    <tr>
      <th>3455327</th>
      <td>34339.571267</td>
      <td>-0.555502</td>
      <td>-0.153110</td>
      <td>-0.671292</td>
      <td>125.060976</td>
      <td>-213.902440</td>
      <td>-19.329268</td>
    </tr>
    <tr>
      <th>3455328</th>
      <td>34339.581200</td>
      <td>-0.286124</td>
      <td>-0.263636</td>
      <td>-0.981340</td>
      <td>158.658537</td>
      <td>-328.170733</td>
      <td>-3.170732</td>
    </tr>
    <tr>
      <th>3455329</th>
      <td>34339.591133</td>
      <td>-0.232536</td>
      <td>-0.161722</td>
      <td>-0.832536</td>
      <td>288.841465</td>
      <td>-281.707318</td>
      <td>17.073171</td>
    </tr>
    <tr>
      <th>3455330</th>
      <td>34339.601067</td>
      <td>0.180383</td>
      <td>-0.368421</td>
      <td>-1.525837</td>
      <td>376.219514</td>
      <td>-140.853659</td>
      <td>37.256098</td>
    </tr>
  </tbody>
</table>
<p>3455331 rows × 7 columns</p>
</div>



## Step 1: Preprocess data

IMU sensors collect data at a fixed sampling frequency, but the sampling rate is not uniform, causing variation in time differences between timestamps. The [preprocess_imu_data](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/preprocessing.py#:~:text=preprocess_imu_data) function therefore resamples the timestamps to be uniformly distributed, and then interpolates IMU values at these new timestamps using the original timestamps and corresponding IMU values. If the difference between timestamps is larger than a specified tolerance (`config.tolerance`, in seconds), it will return an error that the timestamps are not contiguous.  If you still want to process the data in this case, you can create segments from discontiguous samples using the function [`create_segments`](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/segmenting.py) and analyze these segments consecutively as shown in [here](#multiple_segments_cell). By setting `sensor` to 'gyroscope', only gyroscope data is preprocessed and the accelerometer data is removed from the dataframe. Also a `watch_side` should be provided, although for the tremor analysis it does not matter whether this is the correct side since the tremor features are not influenced by the gyroscope axes orientation.


```python
from paradigma.config import IMUConfig
from paradigma.constants import DataColumns
from paradigma.preprocessing import preprocess_imu_data

# Set column names: replace DataColumn.* with your actual column names.
# It is only necessary to set the columns that are present in your data, and
# only if they differ from the default names defined in DataColumns.
column_mapping = {
    'TIME': DataColumns.TIME,
    'ACCELEROMETER_X': DataColumns.ACCELEROMETER_X,
    'ACCELEROMETER_Y': DataColumns.ACCELEROMETER_Y,
    'ACCELEROMETER_Z': DataColumns.ACCELEROMETER_Z,
    'GYROSCOPE_X': DataColumns.GYROSCOPE_X,
    'GYROSCOPE_Y': DataColumns.GYROSCOPE_Y,
    'GYROSCOPE_Z': DataColumns.GYROSCOPE_Z,
}

config = IMUConfig(column_mapping)
print(f'The data is resampled to {config.resampling_frequency} Hz.')
print(f'The tolerance for checking contiguous timestamps is set to {config.tolerance:.3f} seconds.')

df_preprocessed_data = preprocess_imu_data(df_data, config, sensor='gyroscope', watch_side='left')

df_preprocessed_data
```

    The data is resampled to 100 Hz.
    The tolerance for checking contiguous timestamps is set to 0.030 seconds.





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
      <th>gyroscope_x</th>
      <th>gyroscope_y</th>
      <th>gyroscope_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.402439</td>
      <td>0.243902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>0.432231</td>
      <td>0.665526</td>
      <td>-0.123434</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>1.164277</td>
      <td>-0.069584</td>
      <td>-0.307536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>1.151432</td>
      <td>-0.554928</td>
      <td>-0.554223</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>0.657189</td>
      <td>-0.603207</td>
      <td>-0.731570</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3433956</th>
      <td>34339.56</td>
      <td>130.392434</td>
      <td>29.491627</td>
      <td>-26.868202</td>
    </tr>
    <tr>
      <th>3433957</th>
      <td>34339.57</td>
      <td>135.771133</td>
      <td>-184.515525</td>
      <td>-21.544211</td>
    </tr>
    <tr>
      <th>3433958</th>
      <td>34339.58</td>
      <td>146.364103</td>
      <td>-324.248909</td>
      <td>-5.248641</td>
    </tr>
    <tr>
      <th>3433959</th>
      <td>34339.59</td>
      <td>273.675024</td>
      <td>-293.011330</td>
      <td>14.618256</td>
    </tr>
    <tr>
      <th>3433960</th>
      <td>34339.60</td>
      <td>372.878731</td>
      <td>-158.516265</td>
      <td>35.330770</td>
    </tr>
  </tbody>
</table>
<p>3433961 rows × 4 columns</p>
</div>



## Step 2: Extract tremor features

The function [`extract_tremor_features`](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/pipelines/tremor_pipeline.py#:~:text=extract_tremor_features) extracts windows from the preprocessed gyroscope data using non-overlapping windows of length `config.window_length_s`. Next, from these windows the tremor features are extracted: 12 mel-frequency cepstral coefficients (MFCCs), frequency of the peak in the power spectral density, power below tremor (0.5 - 3 Hz), and power around the tremor peak. The latter is not used for tremor detection, but stored for tremor quantification in Step 4.


```python
from paradigma.config import TremorConfig
from paradigma.pipelines.tremor_pipeline import extract_tremor_features

config = TremorConfig(step='features')
print(f'The window length is {config.window_length_s} seconds')

df_features = extract_tremor_features(df_preprocessed_data, config)

df_features
```

    The window length is 4 seconds





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
      <th>mfcc_1</th>
      <th>mfcc_2</th>
      <th>mfcc_3</th>
      <th>mfcc_4</th>
      <th>mfcc_5</th>
      <th>mfcc_6</th>
      <th>mfcc_7</th>
      <th>mfcc_8</th>
      <th>mfcc_9</th>
      <th>mfcc_10</th>
      <th>mfcc_11</th>
      <th>mfcc_12</th>
      <th>freq_peak</th>
      <th>below_tremor_power</th>
      <th>tremor_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>5.323582</td>
      <td>1.179579</td>
      <td>-0.498552</td>
      <td>-0.149152</td>
      <td>-0.063535</td>
      <td>-0.132090</td>
      <td>-0.112380</td>
      <td>-0.044326</td>
      <td>-0.025917</td>
      <td>0.116045</td>
      <td>0.169869</td>
      <td>0.213884</td>
      <td>3.75</td>
      <td>0.082219</td>
      <td>0.471588</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>5.333162</td>
      <td>1.205712</td>
      <td>-0.607844</td>
      <td>-0.138371</td>
      <td>-0.039518</td>
      <td>-0.137703</td>
      <td>-0.069552</td>
      <td>-0.008029</td>
      <td>-0.087711</td>
      <td>0.089844</td>
      <td>0.152380</td>
      <td>0.195165</td>
      <td>3.75</td>
      <td>0.071260</td>
      <td>0.327252</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>5.180974</td>
      <td>1.039548</td>
      <td>-0.627100</td>
      <td>-0.054816</td>
      <td>-0.016767</td>
      <td>-0.044817</td>
      <td>0.079859</td>
      <td>-0.023155</td>
      <td>0.024729</td>
      <td>0.104989</td>
      <td>0.126502</td>
      <td>0.192319</td>
      <td>7.75</td>
      <td>0.097961</td>
      <td>0.114138</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>5.290298</td>
      <td>1.183957</td>
      <td>-0.627651</td>
      <td>-0.027235</td>
      <td>0.095184</td>
      <td>-0.050455</td>
      <td>-0.024654</td>
      <td>0.029754</td>
      <td>-0.007459</td>
      <td>0.125700</td>
      <td>0.146895</td>
      <td>0.220589</td>
      <td>7.75</td>
      <td>0.193237</td>
      <td>0.180988</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>5.128074</td>
      <td>1.066869</td>
      <td>-0.622282</td>
      <td>0.038557</td>
      <td>-0.034719</td>
      <td>0.045109</td>
      <td>0.076679</td>
      <td>0.057267</td>
      <td>-0.024619</td>
      <td>0.131755</td>
      <td>0.177849</td>
      <td>0.149686</td>
      <td>7.75</td>
      <td>0.156469</td>
      <td>0.090009</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8579</th>
      <td>34316.0</td>
      <td>7.071408</td>
      <td>-0.376556</td>
      <td>0.272322</td>
      <td>0.068750</td>
      <td>0.051588</td>
      <td>0.102012</td>
      <td>0.055017</td>
      <td>0.115942</td>
      <td>0.012746</td>
      <td>0.117970</td>
      <td>0.073279</td>
      <td>0.057367</td>
      <td>13.50</td>
      <td>48.930380</td>
      <td>91.971686</td>
    </tr>
    <tr>
      <th>8580</th>
      <td>34320.0</td>
      <td>1.917642</td>
      <td>0.307927</td>
      <td>0.142330</td>
      <td>0.265357</td>
      <td>0.285635</td>
      <td>0.143886</td>
      <td>0.259636</td>
      <td>0.195724</td>
      <td>0.176947</td>
      <td>0.162205</td>
      <td>0.147897</td>
      <td>0.170488</td>
      <td>11.00</td>
      <td>0.012123</td>
      <td>0.000316</td>
    </tr>
    <tr>
      <th>8581</th>
      <td>34324.0</td>
      <td>2.383806</td>
      <td>0.268580</td>
      <td>0.151254</td>
      <td>0.414430</td>
      <td>0.241540</td>
      <td>0.244071</td>
      <td>0.201109</td>
      <td>0.209611</td>
      <td>0.097146</td>
      <td>0.048798</td>
      <td>0.013239</td>
      <td>0.035379</td>
      <td>2.00</td>
      <td>0.013077</td>
      <td>0.000615</td>
    </tr>
    <tr>
      <th>8582</th>
      <td>34328.0</td>
      <td>1.883626</td>
      <td>0.089983</td>
      <td>0.196880</td>
      <td>0.300523</td>
      <td>0.239185</td>
      <td>0.259342</td>
      <td>0.277586</td>
      <td>0.206517</td>
      <td>0.178499</td>
      <td>0.215561</td>
      <td>0.067234</td>
      <td>0.123958</td>
      <td>13.75</td>
      <td>0.011466</td>
      <td>0.000211</td>
    </tr>
    <tr>
      <th>8583</th>
      <td>34332.0</td>
      <td>2.599103</td>
      <td>0.286252</td>
      <td>-0.014529</td>
      <td>0.475488</td>
      <td>0.229446</td>
      <td>0.188200</td>
      <td>0.173689</td>
      <td>0.033262</td>
      <td>0.138957</td>
      <td>0.106176</td>
      <td>0.036859</td>
      <td>0.082178</td>
      <td>12.50</td>
      <td>0.015068</td>
      <td>0.000891</td>
    </tr>
  </tbody>
</table>
<p>8584 rows × 16 columns</p>
</div>



## Step 3: Detect tremor

The function [`detect_tremor`](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/pipelines/tremor_pipeline.py#:~:text=detect_tremor) uses a pretrained logistic regression classifier to predict the tremor probability (`pred_tremor_proba`) for each window, based on the MFCCs. Using the prespecified threshold, a tremor label of 0 (no tremor) or 1 (tremor) is assigned (`pred_tremor_logreg`). Furthermore, the detected tremor windows are checked for rest tremor in two ways. First, the frequency of the peak should be between 3-7 Hz. Second, we want to exclude windows with significant arm movements. We consider a window to have significant arm movement if `below_tremor_power` exceeds `config.movement_threshold`. The final tremor label is saved in `pred_tremor_checked`. A label for predicted arm at rest (`pred_arm_at_rest`, which is 1 when at rest and 0 when not at rest) was also saved, to control for the amount of arm movement during the observed time period when aggregating the amount of tremor time in Step 4 (if a person is moving their arm, they cannot have rest tremor).


```python
from importlib.resources import files
from paradigma.pipelines.tremor_pipeline import detect_tremor

print(f'A threshold of {config.movement_threshold} deg\u00b2/s\u00b2 \
is used to determine whether the arm is at rest or in stable posture.')

# Load the pre-trained logistic regression classifier
tremor_detection_classifier_package_filename = 'tremor_detection_clf_package.pkl'
full_path_to_classifier_package = files('paradigma') / 'assets' / tremor_detection_classifier_package_filename

# Use the logistic regression classifier to detect tremor and check for rest tremor
df_predictions = detect_tremor(df_features, config, full_path_to_classifier_package)

df_predictions[[config.time_colname, 'pred_tremor_proba', 'pred_tremor_logreg', 'pred_arm_at_rest', 'pred_tremor_checked']]
```

    A threshold of 50 deg²/s² is used to determine whether the arm is at rest or in stable posture.





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
      <th>pred_tremor_proba</th>
      <th>pred_tremor_logreg</th>
      <th>pred_arm_at_rest</th>
      <th>pred_tremor_checked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.038968</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>0.035365</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>0.031255</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>0.021106</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>0.021078</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8579</th>
      <td>34316.0</td>
      <td>0.000296</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8580</th>
      <td>34320.0</td>
      <td>0.000089</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8581</th>
      <td>34324.0</td>
      <td>0.000023</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8582</th>
      <td>34328.0</td>
      <td>0.000053</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8583</th>
      <td>34332.0</td>
      <td>0.000049</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>8584 rows × 5 columns</p>
</div>



#### Store as TSDF
The predicted probabilities (and optionally other features) can be stored and loaded in TSDF as demonstrated below.


```python
import tsdf
from paradigma.util import write_df_data

# Set 'path_to_data' to the directory where you want to save the data
metadata_time_store = tsdf.TSDFMetadata(metadata_time.get_plain_tsdf_dict_copy(), path_to_data)
metadata_values_store = tsdf.TSDFMetadata(metadata_values.get_plain_tsdf_dict_copy(), path_to_data)

<<<<<<< HEAD
# Select the columns to be saved
metadata_time_store.channels = ['time']
=======
# Select the columns to be saved
metadata_time_store.channels = [config.time_colname]
>>>>>>> 37f1c8fea7b90d0e387febafe838f2df6ab9dd47
metadata_values_store.channels = ['tremor_power', 'pred_tremor_proba', 'pred_tremor_logreg', 'pred_arm_at_rest', 'pred_tremor_checked']

# Set the units
metadata_time_store.units = ['Relative seconds']
<<<<<<< HEAD
metadata_values_store.units = ['Unitless', 'Unitless', 'Unitless', 'Unitless', 'Unitless']
=======
metadata_values_store.units = ['Unitless', 'Unitless', 'Unitless', 'Unitless', 'Unitless']
>>>>>>> 37f1c8fea7b90d0e387febafe838f2df6ab9dd47
metadata_time_store.data_type = float
metadata_values_store.data_type = float

# Set the filenames
meta_store_filename = f'segment{segment_nr}_meta.json'
values_store_filename = meta_store_filename.replace('_meta.json', '_values.bin')
time_store_filename = meta_store_filename.replace('_meta.json', '_time.bin')

metadata_values_store.file_name = values_store_filename
metadata_time_store.file_name = time_store_filename

write_df_data(metadata_time_store, metadata_values_store, path_to_data, meta_store_filename, df_predictions)
```


```python
df_predictions, _, _ = load_tsdf_dataframe(path_to_data, prefix=f'segment{segment_nr}')
df_predictions.head()
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
      <th>tremor_power</th>
      <th>pred_tremor_proba</th>
      <th>pred_tremor_logreg</th>
      <th>pred_arm_at_rest</th>
      <th>pred_tremor_checked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.471588</td>
      <td>0.038968</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>0.327252</td>
      <td>0.035365</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>0.114138</td>
      <td>0.031255</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>0.180988</td>
      <td>0.021106</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>0.090009</td>
      <td>0.021078</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Step 4: Quantify tremor

The tremor power of all predicted tremor windows (where `pred_tremor_checked` is 1) is used for tremor quantification. A datetime column is also added, providing necessary information before aggregating over specified hours in Step 5.


```python
import pandas as pd
import datetime
import pytz

df_quantification = df_predictions[[config.time_colname, 'pred_arm_at_rest', 'pred_tremor_checked','tremor_power']].copy()
df_quantification.loc[df_predictions['pred_tremor_checked'] == 0, 'tremor_power'] = None # tremor power of non-tremor windows is set to None

# Create datetime column based on the start time of the segment
start_time = datetime.datetime.strptime(metadata_time.start_iso8601, '%Y-%m-%dT%H:%M:%SZ')
start_time = start_time.replace(tzinfo=pytz.timezone('UTC')).astimezone(pytz.timezone('CET')) # convert to correct timezone if necessary
df_quantification[f'{config.time_colname}_dt'] = start_time + pd.to_timedelta(df_quantification[config.time_colname], unit="s")
df_quantification = df_quantification[[config.time_colname, f'{config.time_colname}_dt', 'pred_arm_at_rest', 'pred_tremor_checked', 'tremor_power']]

df_quantification
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
      <th>time_dt</th>
      <th>pred_arm_at_rest</th>
      <th>pred_tremor_checked</th>
      <th>tremor_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2019-08-20 12:39:16+02:00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.471588</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>2019-08-20 12:39:20+02:00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.327252</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>2019-08-20 12:39:24+02:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>2019-08-20 12:39:28+02:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>2019-08-20 12:39:32+02:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8579</th>
      <td>34316.0</td>
      <td>2019-08-20 22:11:12+02:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8580</th>
      <td>34320.0</td>
      <td>2019-08-20 22:11:16+02:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8581</th>
      <td>34324.0</td>
      <td>2019-08-20 22:11:20+02:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8582</th>
      <td>34328.0</td>
      <td>2019-08-20 22:11:24+02:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8583</th>
      <td>34332.0</td>
      <td>2019-08-20 22:11:28+02:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>8584 rows × 5 columns</p>
</div>



### Run steps 1 - 4 for all segments <a id='multiple_segments_cell'></a>

If your data is also stored in multiple segments, you can modify `segments` in the cell below to a list of the filenames of your respective segmented data.


```python
from pathlib import Path
from importlib.resources import files
import datetime
import pytz
import pandas as pd

from paradigma.util import load_tsdf_dataframe
from paradigma.config import IMUConfig, TremorConfig
from paradigma.preprocessing import preprocess_imu_data
from paradigma.pipelines.tremor_pipeline import extract_tremor_features, detect_tremor

# Set the path to where the prepared data is saved
path_to_data =  Path('../../example_data')
path_to_prepared_data = path_to_data / 'imu'

# Load the pre-trained logistic regression classifier
tremor_detection_classifier_package_filename = 'tremor_detection_clf_package.pkl'
full_path_to_classifier_package = files('paradigma') / 'assets' / tremor_detection_classifier_package_filename

# Create a list of dataframes to store the quantifications of all segments
list_df_quantifications = []

segments  = ['0001','0002'] # list with all  available segments

for segment_nr in segments:

    # Load the data
    df_data, metadata_time, _ = load_tsdf_dataframe(path_to_prepared_data, prefix='IMU_segment'+segment_nr)

    # 1: Preprocess the data
    # Change column names if necessary by creating parameter column_mapping (see previous cells for an example)
    config = IMUConfig()
    df_preprocessed_data = preprocess_imu_data(df_data, config, sensor='gyroscope', watch_side='left')

    # 2: Extract features
    config = TremorConfig(step='features')
    df_features = extract_tremor_features(df_preprocessed_data, config)

    # 3: Detect tremor
    df_predictions = detect_tremor(df_features, config, full_path_to_classifier_package)

    # 4: Quantify tremor
    df_quantification = df_predictions[[config.time_colname, 'pred_arm_at_rest', 'pred_tremor_checked','tremor_power']].copy()
    df_quantification.loc[df_predictions['pred_tremor_checked'] == 0, 'tremor_power'] = None

    # Create datetime column based on the start time of the segment
    start_time = datetime.datetime.strptime(metadata_time.start_iso8601, '%Y-%m-%dT%H:%M:%SZ')
    start_time = start_time.replace(tzinfo=pytz.timezone('UTC')).astimezone(pytz.timezone('CET')) # convert to correct timezone if necessary
    df_quantification[f'{config.time_colname}_dt'] = start_time + pd.to_timedelta(df_quantification[config.time_colname], unit="s")
    df_quantification = df_quantification[[config.time_colname, f'{config.time_colname}_dt', 'pred_arm_at_rest', 'pred_tremor_checked', 'tremor_power']]

    # Add the quantifications of the current segment to the list
    df_quantification['segment_nr'] = segment_nr
    list_df_quantifications.append(df_quantification)

df_quantification = pd.concat(list_df_quantifications, ignore_index=True)
```

## Step 5: Compute aggregated tremor measures

The final step is to compute the amount of tremor time and tremor power with the function [`aggregate_tremor`](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/pipelines/tremor_pipeline.py#:~:text=aggregate_tremor), which aggregates over all windows in the input dataframe. Depending on the size of the input dateframe, you could select the hours and days (both optional) that you want to include in this analysis. In this case we use data collected between 8 am and 10 pm (specified as `select_hours_start` and `select_hours_end`), and days with at least 10 hours of data (`min_hours_per_day`) based on. Based on the selected data, we compute aggregated measures for tremor time and tremor power:
- Tremor time is calculated as the number of detected tremor windows, as percentage of the number of windows while the arm is at rest or in stable posture (when `below_tremor_power` does not exceed `config.movement_threshold`). This way the tremor time is controlled for the amount of time the arm is at rest or in stable posture, when rest tremor and re-emergent tremor could occur.
- For tremor power the following aggregates are derived: the mode, median and 90th percentile of tremor power (specified in `config.aggregates_tremor_power`). The median and modal tremor power reflect the typical tremor severity, whereas the 90th percentile reflects the maximal tremor severity within the observed timeframe. The modal tremor power is computed as the peak in the probability density function of tremor power, which is evaluated at the points specified in `config.evaluation_points_tremor_power` (300 points between 0 and 6 log tremor power). The aggregated tremor measures and metadata are stored in a json file.


```python
import pprint
from paradigma.util import select_hours, select_days
from paradigma.pipelines.tremor_pipeline import aggregate_tremor

select_hours_start = '08:00' # you can specifiy the hours and minutes here
select_hours_end = '22:00'
min_hours_per_day = 10

print(f'Before aggregation we select data collected between {select_hours_start} \
and {select_hours_end}. We also select days with at least {min_hours_per_day} hours of data.')
print(f'The following tremor power aggregates are derived: {config.aggregates_tremor_power}.')

# Select the hours that should be included in the analysis
df_quantification = select_hours(df_quantification, select_hours_start, select_hours_end)

# Remove days with less than the specified minimum amount of hours
df_quantification = select_days(df_quantification, min_hours_per_day)

# Compute the aggregated measures
config = TremorConfig()
d_tremor_aggregates = aggregate_tremor(df = df_quantification, config = config)

pprint.pprint(d_tremor_aggregates)
```

    Before aggregation we select data collected between 08:00 and 22:00. We also select days with at least 10 hours of data.
    The following tremor power aggregates are derived: ['mode_binned', 'median', '90p'].
    {'aggregated_tremor_measures': {'90p_tremor_power': 1.3259483071516063,
                                    'median_tremor_power': 0.5143985314908104,
                                    'modal_tremor_power': 0.3,
                                    'perc_windows_tremor': 19.386769676484793},
     'metadata': {'nr_valid_days': 1,
                  'nr_windows_rest': 8284,
                  'nr_windows_total': 12600}}
