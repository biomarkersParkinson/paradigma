# Tremor analysis

This tutorial shows how to run the tremor pipeline to obtain aggregated tremor measures from gyroscope sensor data. Before following along, make sure all data preparation steps have been followed in the data preparation tutorial.

In this tutorial, we use two days of data from a participant of the Personalized Parkinson Project to demonstrate the functionalities. Since `ParaDigMa` expects contiguous time series, the collected data was stored in two segments each with contiguous timestamps. Per segment, we load the data and perform the following steps:
1. Preprocess the time series data
2. Extract tremor features
3. Detect tremor
4. Quantify tremor

We then combine the output of the different segments for the final step:

5. Compute aggregated tremor measures

## Import required modules


```python
import datetime
import json
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pytz
import tsdf

from paradigma.config import IMUConfig, TremorConfig
from paradigma.constants import DataColumns, DataUnits
from paradigma.pipelines.tremor_pipeline import (
    aggregate_tremor,
    detect_tremor,
    extract_tremor_features,
)
from paradigma.preprocessing import preprocess_imu_data
from paradigma.util import load_tsdf_dataframe, select_days, select_hours, write_df_data
```

## Load example data

Here, we start by loading a single contiguous time series (segment), for which we continue running steps 1-3. [Below](#multiple_segments_cell) we show how to run these steps for multiple segments.

We use the internally developed `TSDF` ([documentation](https://biomarkersparkinson.github.io/tsdf/)) to load and store data [[1](https://arxiv.org/abs/2211.11294)]. Depending on the file extension of your time series data, examples of other Python functions for loading the data into memory include:
- _.csv_: `pandas.read_csv()` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html))
- _.json_: `json.load()` ([documentation](https://docs.python.org/3/library/json.html#json.load))


```python
# Set the path to where the prepared data is saved and load the data.
# Note: the test data is stored in TSDF, but you can load your data in your own way
path_to_data =  Path('../../example_data/verily')
path_to_prepared_data = path_to_data / 'imu'

segment_nr  = '0001'

df_data, metadata_time, metadata_values = load_tsdf_dataframe(
    path_to_prepared_data,
    prefix=f'imu_segment{segment_nr}'
)

df_data
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

IMU sensors collect data at a fixed sampling frequency, but the sampling rate is not uniform, causing variation in time differences between timestamps. The [preprocess_imu_data](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/preprocessing/index.html#paradigma.preprocessing.preprocess_imu_data) function therefore resamples the timestamps to be uniformly distributed, and then interpolates IMU values at these new timestamps using the original timestamps and corresponding IMU values. If the difference between timestamps is larger than a specified tolerance (`config.tolerance`, in seconds), it will return an error that the timestamps are not contiguous.  If you still want to process the data in this case, you can create segments from discontiguous samples using the function [`create_segments`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/segmenting/index.html#paradigma.segmenting.create_segments) and analyze these segments consecutively as shown in [here](#multiple_segments_cell). By setting `sensor` to 'gyroscope', only gyroscope data is preprocessed and the accelerometer data is removed from the dataframe. Also a `watch_side` should be provided, although for the tremor analysis it does not matter whether this is the correct side since the tremor features are not influenced by the gyroscope axes orientation.

Note: The data sampling frequency is automatically detected, and frequency-dependent parameters (such as the tolerance) and features
are subsequently automatically adjusted. For more info, see the [config guide](https://biomarkersparkinson.github.io/paradigma/guides/config.html).


```python
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

imu_config = IMUConfig(column_mapping)

df_preprocessed_data = preprocess_imu_data(
    df_data, imu_config, sensor='gyroscope', watch_side='left'
)

print(f"The data is resampled uniformly to {imu_config.resampling_frequency} Hz.")
print(f"The tolerance for checking contiguous timestamps is "
      f"set to {imu_config.tolerance:.3f} seconds.")

df_preprocessed_data
```

    INFO: Resampled: 3455331 -> 3468300 rows at 101 Hz


    The data is resampled uniformly to 101 Hz.
    The tolerance for checking contiguous timestamps is set to 0.030 seconds.





<div>
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
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.402439</td>
      <td>0.243902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.009901</td>
      <td>0.424212</td>
      <td>0.673259</td>
      <td>-0.121229</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.019802</td>
      <td>1.155682</td>
      <td>-0.056779</td>
      <td>-0.303596</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.029703</td>
      <td>1.161910</td>
      <td>-0.545737</td>
      <td>-0.546139</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.039604</td>
      <td>0.677357</td>
      <td>-0.612758</td>
      <td>-0.731633</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3468295</th>
      <td>34339.554455</td>
      <td>-63.697260</td>
      <td>39.121747</td>
      <td>-14.192411</td>
    </tr>
    <tr>
      <th>3468296</th>
      <td>34339.564356</td>
      <td>181.214291</td>
      <td>-41.963407</td>
      <td>-29.116730</td>
    </tr>
    <tr>
      <th>3468297</th>
      <td>34339.574257</td>
      <td>115.281745</td>
      <td>-269.298210</td>
      <td>-14.416680</td>
    </tr>
    <tr>
      <th>3468298</th>
      <td>34339.584158</td>
      <td>194.076509</td>
      <td>-328.099389</td>
      <td>2.395920</td>
    </tr>
    <tr>
      <th>3468299</th>
      <td>34339.594059</td>
      <td>324.945776</td>
      <td>-247.077559</td>
      <td>23.379297</td>
    </tr>
  </tbody>
</table>
<p>3468300 rows × 4 columns</p>
</div>



## Step 2: Extract tremor features

The function [`extract_tremor_features`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/pipelines/tremor_pipeline/index.html#paradigma.pipelines.tremor_pipeline.extract_tremor_features) extracts windows from the preprocessed gyroscope data using non-overlapping windows of length `config.window_length_s`. Next, from these windows the tremor features are extracted: 12 mel-frequency cepstral coefficients (MFCCs), frequency of the peak in the power spectral density, power below tremor (0.5 - 3 Hz), and power around the tremor peak. The latter is not used for tremor detection, but stored for tremor quantification in Step 4.


```python
tremor_config = TremorConfig()
tremor_config._set_sampling_frequency_detected(imu_config.sampling_frequency)
print(f'The window length is {tremor_config.window_length_s} seconds')

df_features = extract_tremor_features(df_preprocessed_data, tremor_config)

df_features
```

    The window length is 4 seconds





<div>
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
      <td>5.323215</td>
      <td>1.177907</td>
      <td>-0.501607</td>
      <td>-0.149680</td>
      <td>-0.060247</td>
      <td>-0.131664</td>
      <td>-0.113229</td>
      <td>-0.043972</td>
      <td>-0.025087</td>
      <td>0.114878</td>
      <td>0.170381</td>
      <td>0.213038</td>
      <td>3.75</td>
      <td>0.082369</td>
      <td>0.471235</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>5.336619</td>
      <td>1.208349</td>
      <td>-0.607052</td>
      <td>-0.137038</td>
      <td>-0.038054</td>
      <td>-0.135531</td>
      <td>-0.069183</td>
      <td>-0.007556</td>
      <td>-0.085976</td>
      <td>0.089542</td>
      <td>0.153726</td>
      <td>0.195739</td>
      <td>3.75</td>
      <td>0.071459</td>
      <td>0.327415</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>5.180463</td>
      <td>1.042150</td>
      <td>-0.627363</td>
      <td>-0.053882</td>
      <td>-0.016101</td>
      <td>-0.045770</td>
      <td>0.080107</td>
      <td>-0.024386</td>
      <td>0.025677</td>
      <td>0.104405</td>
      <td>0.127752</td>
      <td>0.192906</td>
      <td>7.75</td>
      <td>0.098172</td>
      <td>0.114098</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>5.287420</td>
      <td>1.182862</td>
      <td>-0.627432</td>
      <td>-0.028083</td>
      <td>0.093145</td>
      <td>-0.052906</td>
      <td>-0.025959</td>
      <td>0.030913</td>
      <td>-0.011134</td>
      <td>0.124871</td>
      <td>0.145850</td>
      <td>0.220220</td>
      <td>7.75</td>
      <td>0.193567</td>
      <td>0.181103</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>5.133249</td>
      <td>1.064652</td>
      <td>-0.622849</td>
      <td>0.037858</td>
      <td>-0.035782</td>
      <td>0.043510</td>
      <td>0.075618</td>
      <td>0.058650</td>
      <td>-0.024736</td>
      <td>0.131784</td>
      <td>0.179266</td>
      <td>0.149834</td>
      <td>7.75</td>
      <td>0.156356</td>
      <td>0.089982</td>
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
      <td>6.986879</td>
      <td>-0.354682</td>
      <td>0.273604</td>
      <td>0.057881</td>
      <td>0.054815</td>
      <td>0.118318</td>
      <td>0.064013</td>
      <td>0.112470</td>
      <td>0.014039</td>
      <td>0.117485</td>
      <td>0.074521</td>
      <td>0.060339</td>
      <td>13.50</td>
      <td>48.895782</td>
      <td>91.952490</td>
    </tr>
    <tr>
      <th>8580</th>
      <td>34320.0</td>
      <td>1.917719</td>
      <td>0.307124</td>
      <td>0.145170</td>
      <td>0.265969</td>
      <td>0.283961</td>
      <td>0.144457</td>
      <td>0.258391</td>
      <td>0.195310</td>
      <td>0.176972</td>
      <td>0.162119</td>
      <td>0.147681</td>
      <td>0.169611</td>
      <td>11.00</td>
      <td>0.012124</td>
      <td>0.000316</td>
    </tr>
    <tr>
      <th>8581</th>
      <td>34324.0</td>
      <td>2.380429</td>
      <td>0.268608</td>
      <td>0.152984</td>
      <td>0.414011</td>
      <td>0.240926</td>
      <td>0.245157</td>
      <td>0.200777</td>
      <td>0.210480</td>
      <td>0.097858</td>
      <td>0.047840</td>
      <td>0.014666</td>
      <td>0.036084</td>
      <td>2.00</td>
      <td>0.013080</td>
      <td>0.000614</td>
    </tr>
    <tr>
      <th>8582</th>
      <td>34328.0</td>
      <td>1.884624</td>
      <td>0.093049</td>
      <td>0.196704</td>
      <td>0.299977</td>
      <td>0.240239</td>
      <td>0.259326</td>
      <td>0.276831</td>
      <td>0.205116</td>
      <td>0.178561</td>
      <td>0.213776</td>
      <td>0.067219</td>
      <td>0.123831</td>
      <td>13.75</td>
      <td>0.011465</td>
      <td>0.000211</td>
    </tr>
    <tr>
      <th>8583</th>
      <td>34332.0</td>
      <td>2.597827</td>
      <td>0.286846</td>
      <td>-0.015120</td>
      <td>0.475393</td>
      <td>0.229463</td>
      <td>0.187146</td>
      <td>0.173255</td>
      <td>0.032205</td>
      <td>0.140228</td>
      <td>0.106082</td>
      <td>0.037928</td>
      <td>0.083249</td>
      <td>12.50</td>
      <td>0.015059</td>
      <td>0.000889</td>
    </tr>
  </tbody>
</table>
<p>8584 rows × 16 columns</p>
</div>



## Step 3: Detect tremor

The function [`detect_tremor`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/pipelines/tremor_pipeline/index.html#paradigma.pipelines.tremor_pipeline.detect_tremor) uses a pretrained logistic regression classifier to predict the tremor probability (`pred_tremor_proba`) for each window, based on the MFCCs. Using the prespecified threshold, a tremor label of 0 (no tremor) or 1 (tremor) is assigned (`pred_tremor_logreg`). Furthermore, the detected tremor windows are checked for rest tremor in two ways. First, the frequency of the peak should be between 3-7 Hz. Second, we want to exclude windows with significant arm movements. We consider a window to have significant arm movement if `below_tremor_power` exceeds `config.movement_threshold`. The final tremor label is saved in `pred_tremor_checked`. A label for predicted arm at rest (`pred_arm_at_rest`, which is 1 when at rest and 0 when not at rest) was also saved, to control for the amount of arm movement during the observed time period when aggregating the amount of tremor time in Step 4 (if a person is moving their arm, they cannot have rest tremor).


```python
print(f'A threshold of {tremor_config.movement_threshold} deg\u00b2/s\u00b2 \
is used to determine whether the arm is at rest or in stable posture.')

# Load the pre-trained logistic regression classifier
tremor_detection_classifier_package_filename = 'tremor_detection_clf_package.pkl'
full_path_to_classifier_package = (
    files('paradigma')
    / 'assets'
    / tremor_detection_classifier_package_filename
)

# Use the logistic regression classifier to detect tremor and check for rest tremor
df_predictions = detect_tremor(
    df_features, tremor_config, full_path_to_classifier_package
)

df_predictions[[
    tremor_config.time_colname, DataColumns.PRED_TREMOR_PROBA,
    DataColumns.PRED_TREMOR_LOGREG, DataColumns.PRED_ARM_AT_REST,
    DataColumns.PRED_TREMOR_CHECKED
]]
```

    A threshold of 50 deg²/s² is used to determine whether the arm is at rest or in stable posture.





<div>
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
      <td>0.038855</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>0.035294</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>0.031335</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>0.020958</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>0.021405</td>
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
      <td>0.000292</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8580</th>
      <td>34320.0</td>
      <td>0.000088</td>
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
      <td>0.000050</td>
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
# Set 'path_to_data' to the directory where you want to save the data
metadata_time_store = tsdf.TSDFMetadata(
    metadata_time.get_plain_tsdf_dict_copy(),
    path_to_data
)
metadata_values_store = tsdf.TSDFMetadata(
    metadata_values.get_plain_tsdf_dict_copy(),
    path_to_data
)

# Select the columns to be saved
metadata_time_store.channels = [tremor_config.time_colname]
metadata_values_store.channels = [
    DataColumns.TREMOR_POWER,
    DataColumns.PRED_TREMOR_PROBA,
    DataColumns.PRED_TREMOR_LOGREG,
    DataColumns.PRED_ARM_AT_REST,
    DataColumns.PRED_TREMOR_CHECKED
]

# Set the units
metadata_time_store.units = ['Relative seconds']
metadata_values_store.units = [
    DataUnits.POWER_ROTATION,
    DataUnits.NONE,
    DataUnits.NONE,
    DataUnits.NONE,
    DataUnits.NONE
]
metadata_time_store.data_type = float
metadata_values_store.data_type = float

# Set the filenames
meta_store_filename = f'segment{segment_nr}_meta.json'
values_store_filename = meta_store_filename.replace('_meta.json', '_values.bin')
time_store_filename = meta_store_filename.replace('_meta.json', '_time.bin')

metadata_values_store.file_name = values_store_filename
metadata_time_store.file_name = time_store_filename

write_df_data(metadata_time_store, metadata_values_store,
              path_to_data, meta_store_filename, df_predictions)
```


```python
df_predictions, _, _ = load_tsdf_dataframe(
    path_to_data,
    prefix=f'segment{segment_nr}'
)
df_predictions.head()
```




<div>
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
      <td>0.471235</td>
      <td>0.038855</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>0.327415</td>
      <td>0.035294</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>0.114098</td>
      <td>0.031335</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>0.181103</td>
      <td>0.020958</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>0.089982</td>
      <td>0.021405</td>
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
df_quantification = df_predictions[[
    tremor_config.time_colname, DataColumns.PRED_ARM_AT_REST,
    DataColumns.PRED_TREMOR_CHECKED, DataColumns.TREMOR_POWER
]].copy()
df_quantification.loc[
    df_predictions[DataColumns.PRED_TREMOR_CHECKED] == 0, DataColumns.TREMOR_POWER
] = None # tremor power of non-tremor windows is set to None

# Create datetime column based on the start time of the segment
start_time = datetime.datetime.strptime(
    metadata_time.start_iso8601, '%Y-%m-%dT%H:%M:%SZ'
)
start_time = (
    start_time
    .replace(tzinfo=pytz.timezone('UTC'))
    .astimezone(pytz.timezone('CET')) # convert to correct timezone if necessary
)
df_quantification[f'{tremor_config.time_colname}_dt'] = start_time + \
    pd.to_timedelta(df_quantification[tremor_config.time_colname], unit="s")
df_quantification = df_quantification[[
    tremor_config.time_colname,
    f'{tremor_config.time_colname}_dt',
    DataColumns.PRED_ARM_AT_REST,
    DataColumns.PRED_TREMOR_CHECKED,
    DataColumns.TREMOR_POWER
]]

df_quantification
```




<div>
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
      <td>0.471235</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>2019-08-20 12:39:20+02:00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.327415</td>
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
# Set the path to where the prepared data is saved
path_to_data =  Path('../../example_data/verily')
path_to_prepared_data = path_to_data / 'imu'

# Load the pre-trained logistic regression classifier
tremor_detection_classifier_package_filename = 'tremor_detection_clf_package.pkl'
full_path_to_classifier_package = (
    files('paradigma')
    / 'assets'
    / tremor_detection_classifier_package_filename
)

# Create a list of dataframes to store the quantifications of all segments
list_df_quantifications = []

segments  = ['0001', '0002'] # list with all  available segments

for segment_nr in segments:

    # Load the data
    df_data, metadata_time, _ = load_tsdf_dataframe(
        path_to_prepared_data,
        prefix='imu_segment'+segment_nr
    )

    # 1: Preprocess the data
    # Change column names if necessary by creating parameter
    # column_mapping (see previous cells for an example)
    imu_config = IMUConfig()
    df_preprocessed_data = preprocess_imu_data(
        df_data, imu_config, sensor='gyroscope', watch_side='left'
    )

    # 2: Extract features
    tremor_config = TremorConfig()
    tremor_config._set_sampling_frequency_detected(imu_config.sampling_frequency)

    df_features = extract_tremor_features(df_preprocessed_data, tremor_config)

    # 3: Detect tremor
    df_predictions = detect_tremor(df_features, tremor_config,
                                   full_path_to_classifier_package)

    # 4: Quantify tremor
    df_quantification = df_predictions[[
        tremor_config.time_colname, DataColumns.PRED_ARM_AT_REST,
    DataColumns.PRED_TREMOR_CHECKED, DataColumns.TREMOR_POWER
    ]].copy()
    df_quantification.loc[
        df_predictions[DataColumns.PRED_TREMOR_CHECKED] == 0, DataColumns.TREMOR_POWER
    ] = None

    # Create datetime column based on the start time of the segment
    start_time = datetime.datetime.strptime(
        metadata_time.start_iso8601, '%Y-%m-%dT%H:%M:%SZ'
    )
    start_time = (
        start_time
        .replace(tzinfo=pytz.timezone('UTC'))
        .astimezone(pytz.timezone('CET')) # convert to correct timezone if necessary
    )
    df_quantification[f'{tremor_config.time_colname}_dt'] = start_time + \
        pd.to_timedelta(df_quantification[tremor_config.time_colname], unit="s")
```

    INFO: Resampled: 3455331 -> 3468300 rows at 101 Hz


    INFO: Resampled: 7434685 -> 7462834 rows at 101 Hz


## Step 5: Compute aggregated tremor measures

The final step is to compute the amount of tremor time and tremor power with the function [`aggregate_tremor`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/pipelines/tremor_pipeline/index.html#paradigma.pipelines.tremor_pipeline.aggregate_tremor), which aggregates over all windows in the input dataframe. Depending on the size of the input dateframe, you could select the hours and days (both optional) that you want to include in this analysis. In this case we use data collected between 8 am and 10 pm (specified as `select_hours_start` and `select_hours_end`), and days with at least 10 hours of data (`min_hours_per_day`) based on. Based on the selected data, we compute aggregated measures for tremor time and tremor power:
- Tremor time is calculated as the number of detected tremor windows, as percentage of the number of windows while the arm is at rest or in stable posture (when `below_tremor_power` does not exceed `config.movement_threshold`). This way the tremor time is controlled for the amount of time the arm is at rest or in stable posture, when rest tremor and re-emergent tremor could occur.
- For tremor power the following aggregates are derived: the mode, median and 90th percentile of tremor power (specified in `config.aggregates_tremor_power`). The median and modal tremor power reflect the typical tremor severity, whereas the 90th percentile reflects the maximal tremor severity within the observed timeframe. The modal tremor power is computed as the peak in the probability density function of tremor power, which is evaluated at the points specified in `config.evaluation_points_tremor_power` (300 points between 0 and 6 log tremor power). The aggregated tremor measures and metadata are stored in a json file.


```python
select_hours_start = '08:00' # you can specifiy the hours and minutes here
select_hours_end = '22:00'
min_hours_per_day = 10

print(
    f"Before aggregation we select data collected between {select_hours_start} "
    f"and {select_hours_end}. We also select days with at "
    f"least {min_hours_per_day} hours of data. \nThe following tremor power "
    f"aggregates are derived: {tremor_config.aggregates_tremor_power}."
)

# Select the hours that should be included in the analysis
df_quantification = select_hours(
    df_quantification, select_hours_start, select_hours_end
)

# Remove days with less than the specified minimum amount of hours
df_quantification = select_days(df_quantification, min_hours_per_day)

# Compute the aggregated measures
config = TremorConfig()
d_tremor_aggregates = aggregate_tremor(df = df_quantification, config = config)

print(json.dumps(d_tremor_aggregates, indent=2))
```

    Before aggregation we select data collected between 08:00 and 22:00. We also select days with at least 10 hours of data.
    The following tremor power aggregates are derived: ['mode_binned', 'median', '90p'].
    {
      "metadata": {
        "nr_valid_days": 1,
        "nr_windows_total": 12600,
        "nr_windows_rest": 8282
      },
      "aggregated_tremor_measures": {
        "perc_windows_tremor": 19.415600096595025,
        "median_tremor_power": 0.5144274315149706,
        "modal_tremor_power": 0.3,
        "90p_tremor_power": 1.3229371865246413
      }
    }
