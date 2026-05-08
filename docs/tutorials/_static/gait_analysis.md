# Gait analysis
This tutorial showcases the high-level functions composing the gait pipeline. Before following along, make sure all data preparation steps have been followed in the [Data preparation tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/_static/data_preparation.html).

In this tutorial, we use two days of data from a participant of the Personalized Parkinson Project to demonstrate the functionalities. Since ParaDigMa expects contiguous time series, the collected data was stored in two segments each with contiguous timestamps. Per segment, we load the data and perform the following steps:
1. Data preprocessing
2. Gait feature extraction
3. Gait detection
4. Arm activity feature extraction
5. Filtering gait
6. Arm swing quantification

We then combine the output of the different raw data segments for the final step:

7. Aggregation

To run the complete gait pipeline, a prerequisite is to have both accelerometer and gyroscope data.

## Import required modules


```python
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import tsdf

from paradigma.classification import ClassifierPackage
from paradigma.config import GaitConfig, IMUConfig
from paradigma.constants import DataColumns
from paradigma.pipelines.gait_pipeline import (
    aggregate_arm_swing_params,
    detect_gait,
    extract_arm_activity_features,
    extract_gait_features,
    filter_gait,
    quantify_arm_swing,
)
from paradigma.preprocessing import preprocess_imu_data
from paradigma.util import (
    load_tsdf_dataframe,
    merge_predictions_with_timestamps,
    write_df_data,
)
```

## Load data
Here, we start by loading a single contiguous time series (segment), for which we continue running steps 1-6. [Below](#multiple_segments_cell) we show how to run these steps for multiple raw data segments.

We use the internally developed `TSDF` ([documentation](https://biomarkersparkinson.github.io/tsdf/)) to load and store data [[1](https://arxiv.org/abs/2211.11294)]. Depending on the file extension of your time series data, examples of other Python functions for loading the data into memory include:
- _.csv_: `pandas.read_csv()` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html))
- _.json_: `json.load()` ([documentation](https://docs.python.org/3/library/json.html#json.load))


```python
# Set the path to where the prepared data is saved and load the data.
# Note: the test data is stored in TSDF, but you can load your data in your own way
path_to_data =  Path('../../example_data/verily')
path_to_prepared_data = path_to_data / 'imu'

raw_data_segment_nr  = '0001'

# Load the data from the file
df_imu, metadata_time, metadata_values = load_tsdf_dataframe(
    path_to_data=path_to_prepared_data,
    prefix=f'imu_segment{raw_data_segment_nr}'
)

df_imu
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
The function [`preprocess_imu_data`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/preprocessing/index.html#paradigma.preprocessing.preprocess_imu_data) in the cell below runs all necessary preprocessing steps. It requires the loaded dataframe, a configuration object [`config`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/config/index.html) specifying parameters used for preprocessing, and a selection of sensors. For the sensors, options include `'accelerometer'`, `'gyroscope'`, or `'both'`.  If the difference between timestamps is larger than a specified tolerance (`config.tolerance`, in seconds), it will return an error that the timestamps are not contiguous. If you still want to process the data in this case, you can create segments from discontiguous samples using the function [`create_segments`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/segmenting/index.html#paradigma.segmenting.create_segments) and analyze these segments consecutively as shown in [here](#multiple_segments_cell).

The function [`preprocess_imu_data`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/preprocessing/index.html#paradigma.preprocessing.preprocess_imu_data) processes the data as follows:
1. Resample the data to ensure uniformly distributed sampling rate.
2. Apply filtering to separate the gravity component from the accelerometer.


```python
imu_config = IMUConfig()

df_preprocessed = preprocess_imu_data(
    df=df_imu,
    config=imu_config,
    sensor='both',
    watch_side='left',
)

df_preprocessed.head()
```

    INFO: Resampled: 3455331 -> 3468300 rows at 101 Hz





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
      <th>accelerometer_x_grav</th>
      <th>accelerometer_y_grav</th>
      <th>accelerometer_z_grav</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-0.002357</td>
      <td>-0.001140</td>
      <td>-0.002248</td>
      <td>0.000000</td>
      <td>1.402439</td>
      <td>0.243902</td>
      <td>-0.472284</td>
      <td>-0.378285</td>
      <td>0.772583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.009901</td>
      <td>-0.000438</td>
      <td>-0.000647</td>
      <td>-0.007521</td>
      <td>0.424212</td>
      <td>0.673259</td>
      <td>-0.121229</td>
      <td>-0.472294</td>
      <td>-0.378311</td>
      <td>0.772595</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.019802</td>
      <td>0.000533</td>
      <td>0.002722</td>
      <td>-0.005643</td>
      <td>1.155682</td>
      <td>-0.056779</td>
      <td>-0.303596</td>
      <td>-0.472304</td>
      <td>-0.378336</td>
      <td>0.772606</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.029703</td>
      <td>-0.000393</td>
      <td>0.002789</td>
      <td>-0.002310</td>
      <td>1.161910</td>
      <td>-0.545737</td>
      <td>-0.546139</td>
      <td>-0.472314</td>
      <td>-0.378360</td>
      <td>0.772618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.039604</td>
      <td>-0.002775</td>
      <td>-0.001001</td>
      <td>-0.000398</td>
      <td>0.677357</td>
      <td>-0.612758</td>
      <td>-0.731633</td>
      <td>-0.472323</td>
      <td>-0.378385</td>
      <td>0.772629</td>
    </tr>
  </tbody>
</table>
</div>



The resulting dataframe shown above contains uniformly distributed timestamps with corresponding accelerometer and gyroscope values. Note the for accelerometer values, the following notation is used:
- `accelerometer_x`: the accelerometer signal after filtering out the gravitational component
- `accelerometer_x_grav`: the gravitational component of the accelerometer signal

The accelerometer data is retained and used to compute gravity-related features for the classification tasks, because the gravity is informative of the position of the arm.

## Step 2: Extract gait features
With the data uniformly resampled and the gravitional component separated from the accelerometer signal, features can be extracted from the time series data. This step does not require gyroscope data. To extract the features, the pipeline executes the following steps:
- Use overlapping windows to group timestamps
- Extract temporal features
- Use Fast Fourier Transform the transform the windowed data into the spectral domain
- Extract spectral features
- Combine both temporal and spectral features into a final dataframe

These steps are encapsulated in [`extract_gait_features`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/pipelines/gait_pipeline/index.html#paradigma.pipelines.gait_pipeline.extract_gait_features).


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

# Pass the IMUConfig from preprocessing to inherit the detected sampling frequency
gait_config = GaitConfig(step='gait', column_mapping=column_mapping)
gait_config._set_sampling_frequency_detected(imu_config.sampling_frequency)

df_gait = extract_gait_features(
    df=df_preprocessed,
    config=gait_config
)

print(
    f"A total of {df_gait.shape[1]-1} features have been extracted from "
    f"{df_gait.shape[0]} {gait_config.window_length_s}-second windows with "
    f"{gait_config.window_length_s-gait_config.window_step_length_s} seconds overlap."
)
df_gait.head()
```

    A total of 34 features have been extracted from 34334 6-second windows with 5 seconds overlap.





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>accelerometer_x_grav_mean</th>
      <th>accelerometer_y_grav_mean</th>
      <th>accelerometer_z_grav_mean</th>
      <th>accelerometer_x_grav_std</th>
      <th>accelerometer_y_grav_std</th>
      <th>accelerometer_z_grav_std</th>
      <th>accelerometer_std_norm</th>
      <th>accelerometer_x_power_below_gait</th>
      <th>accelerometer_y_power_below_gait</th>
      <th>...</th>
      <th>accelerometer_mfcc_3</th>
      <th>accelerometer_mfcc_4</th>
      <th>accelerometer_mfcc_5</th>
      <th>accelerometer_mfcc_6</th>
      <th>accelerometer_mfcc_7</th>
      <th>accelerometer_mfcc_8</th>
      <th>accelerometer_mfcc_9</th>
      <th>accelerometer_mfcc_10</th>
      <th>accelerometer_mfcc_11</th>
      <th>accelerometer_mfcc_12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-0.472964</td>
      <td>-0.380610</td>
      <td>0.774297</td>
      <td>0.000275</td>
      <td>0.000748</td>
      <td>0.000542</td>
      <td>0.003370</td>
      <td>0.000003</td>
      <td>1.167951e-06</td>
      <td>...</td>
      <td>-1.102542</td>
      <td>0.524570</td>
      <td>0.215273</td>
      <td>0.429301</td>
      <td>0.900356</td>
      <td>1.136008</td>
      <td>0.672959</td>
      <td>-0.128223</td>
      <td>-0.335890</td>
      <td>-0.060264</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.473002</td>
      <td>-0.380697</td>
      <td>0.774538</td>
      <td>0.000236</td>
      <td>0.000572</td>
      <td>0.000215</td>
      <td>0.003187</td>
      <td>0.000003</td>
      <td>1.203355e-06</td>
      <td>...</td>
      <td>-0.997681</td>
      <td>0.633430</td>
      <td>0.327416</td>
      <td>0.451710</td>
      <td>0.972589</td>
      <td>1.120794</td>
      <td>0.770066</td>
      <td>-0.115923</td>
      <td>-0.395905</td>
      <td>-0.011235</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>-0.473037</td>
      <td>-0.380556</td>
      <td>0.774576</td>
      <td>0.000233</td>
      <td>0.000605</td>
      <td>0.000193</td>
      <td>0.003178</td>
      <td>0.000002</td>
      <td>6.705345e-07</td>
      <td>...</td>
      <td>-1.040508</td>
      <td>0.404687</td>
      <td>0.268458</td>
      <td>0.507542</td>
      <td>0.944616</td>
      <td>1.016366</td>
      <td>0.785605</td>
      <td>-0.071365</td>
      <td>-0.414318</td>
      <td>0.020683</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>-0.472952</td>
      <td>-0.380310</td>
      <td>0.774660</td>
      <td>0.000301</td>
      <td>0.000522</td>
      <td>0.000326</td>
      <td>0.003002</td>
      <td>0.000002</td>
      <td>6.856197e-07</td>
      <td>...</td>
      <td>-1.076265</td>
      <td>0.258896</td>
      <td>0.256840</td>
      <td>0.507037</td>
      <td>0.892664</td>
      <td>0.900503</td>
      <td>0.706305</td>
      <td>-0.080479</td>
      <td>-0.302643</td>
      <td>0.054793</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>-0.472692</td>
      <td>-0.380027</td>
      <td>0.774891</td>
      <td>0.000468</td>
      <td>0.000358</td>
      <td>0.000468</td>
      <td>0.002859</td>
      <td>0.000002</td>
      <td>1.099972e-06</td>
      <td>...</td>
      <td>-1.079838</td>
      <td>0.264684</td>
      <td>0.237084</td>
      <td>0.587880</td>
      <td>0.937095</td>
      <td>0.763024</td>
      <td>0.608238</td>
      <td>-0.160035</td>
      <td>-0.184588</td>
      <td>0.127915</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



Each row in this dataframe corresponds to a single window, with the window length and overlap set in the [`config`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/config/index.html) object. Note that the `time` column has a 1-second interval instead of the 10-millisecond interval before, as it now represents the starting time of the window.

## Step 3: Gait detection
For classification, ParaDigMa uses so-called Classifier Packages which contain a classifier, classification threshold, and a feature scaler as attributes ([documentation](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/classification/index.html#paradigma.classification.ClassifierPackage)). The classifier is a [random forest](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html) trained on a dataset of people with PD performing a wide range of activities in free-living conditions: [The Parkinson@Home Validation Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7584982/). The classification threshold was set to limit the amount of false-positive predictions in the original study, i.e., to limit non-gait to be predicted as gait. The classification threshold can be changed by setting `clf_package.threshold` to a different float value. The feature scaler was similarly fitted on the original dataset, ensuring the features are within expected confined spaces to make reliable predictions.


```python
# Set the path to the classifier package
classifier_package_filename = 'gait_detection_clf_package.pkl'
full_path_to_classifier_package = (
    files('paradigma')
    / 'assets'
    / classifier_package_filename
)

# Load the classifier package
clf_package_detection = ClassifierPackage.load(full_path_to_classifier_package)
gait_threshold = clf_package_detection.threshold

# Detecting gait returns the probability of gait for each window, which is
# concatenated to the original dataframe
df_gait[DataColumns.PRED_GAIT_PROBA] = detect_gait(
    df=df_gait,
    clf_package=clf_package_detection
)

n_windows = df_gait.shape[0]
n_predictions_gait = df_gait.loc[
    df_gait[DataColumns.PRED_GAIT_PROBA] >= gait_threshold
].shape[0]
perc_predictions_gait = round(100 * n_predictions_gait / n_windows, 1)
n_predictions_non_gait = df_gait.loc[
    df_gait[DataColumns.PRED_GAIT_PROBA] < gait_threshold
].shape[0]
perc_predictions_non_gait = round(100 * n_predictions_non_gait / n_windows, 1)

print(
    f"Out of {n_windows} windows, {n_predictions_gait} "
    f"({perc_predictions_gait}%) \n"
    f"were predicted as gait, and {n_predictions_non_gait} "
    f"({perc_predictions_non_gait}%) \n"
    f"as non-gait."
)

# Only the time and the predicted gait probability are shown, but the
# dataframe also contains the extracted features
df_gait[[gait_config.time_colname, DataColumns.PRED_GAIT_PROBA]].head()
```

    Out of 34334 windows, 2755 (8.0%)
    were predicted as gait, and 31579 (92.0%)
    as non-gait.





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>pred_gait_proba</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.000023</td>
    </tr>
  </tbody>
</table>
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
metadata_time_store.channels = [gait_config.time_colname]
metadata_values_store.channels = [DataColumns.PRED_GAIT_PROBA]

# Set the units
metadata_time_store.units = ['Relative seconds']
metadata_values_store.units = ['Unitless']
metadata_time_store.data_type = float
metadata_values_store.data_type = float

# Set the filenames
meta_store_filename = f'segment{raw_data_segment_nr}_meta.json'
values_store_filename = meta_store_filename.replace('_meta.json', '_values.bin')
time_store_filename = meta_store_filename.replace('_meta.json', '_time.bin')

metadata_values_store.file_name = values_store_filename
metadata_time_store.file_name = time_store_filename

write_df_data(metadata_time_store, metadata_values_store, path_to_data,
              meta_store_filename, df_gait)
```


```python
df_gait, _, _ = load_tsdf_dataframe(
    path_to_data,
    prefix=f'segment{raw_data_segment_nr}'
)
df_gait.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>pred_gait_proba</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.000023</td>
    </tr>
  </tbody>
</table>
</div>



Once again, the `time` column indicates the start time of the window. Therefore, it can be observed that probabilities are predicted of overlapping windows, and not of individual timestamps. The function [`merge_timestamps_with_predictions`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/util/index.html#paradigma.util.merge_predictions_with_timestamps) can be used to retrieve predicted probabilities per timestamp by aggregating the predicted probabilities of overlapping windows. This function is included in the next step.

## Step 4: Arm activity feature extraction
The extraction of arm swing features is similar to the extraction of gait features, but we use a different window length and step length (`config.window_length_s`, `config.window_step_length_s`) to distinguish between gait segments with and without other arm activities. Therefore, the following steps are conducted sequentially by [`extract_arm_activity_features`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/pipelines/gait_pipeline/index.html#paradigma.pipelines.gait_pipeline.extract_arm_activity_features):
- Start with the preprocessed data of step 1
- Merge the gait predictions into the preprocessed data
- Discard predicted non-gait activities
- Create windows of the time series data and extract features

But, first, the gait predictions should be merged with the preprocessed time series data, such that individual timestamps have a corresponding probability of gait. The function [`extract_arm_activity_features`](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/pipelines/gait_pipeline/index.html#paradigma.pipelines.gait_pipeline.extract_arm_activity_features) expects a time series dataframe of predicted gait.


```python
# Merge gait predictions into timeseries data
if not any(df_gait[DataColumns.PRED_GAIT_PROBA] >= clf_package_detection.threshold):
    raise ValueError("No gait detected in the input data.")

df = merge_predictions_with_timestamps(
    df_ts=df_preprocessed,
    df_predictions=df_gait,
    pred_proba_colname=DataColumns.PRED_GAIT_PROBA,
    window_length_s=gait_config.window_length_s,
    fs=gait_config.sampling_frequency
)

# Add a column for predicted gait based on a fitted threshold
df[DataColumns.PRED_GAIT] = (
    df[DataColumns.PRED_GAIT_PROBA] >= clf_package_detection.threshold
).astype(int)

# Filter the DataFrame to only include predicted gait (1)
df = df.loc[df[DataColumns.PRED_GAIT]==1].reset_index(drop=True)
```


```python
arm_config = GaitConfig(step='arm_activity')
arm_config._set_sampling_frequency_detected(imu_config.sampling_frequency)

df_arm = extract_arm_activity_features(
    df=df,
    config=arm_config,
)

print(
    f"A total of {df_arm.shape[1] - 1} features have been extracted "
    f"from {df_arm.shape[0]} {arm_config.window_length_s}-second windows "
    f"with {arm_config.window_length_s - arm_config.window_step_length_s} seconds "
    "overlap."
)
df_arm.head()
```

    A total of 61 features have been extracted from 2755 3-second windows with 2.25 seconds overlap.





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>accelerometer_x_grav_mean</th>
      <th>accelerometer_y_grav_mean</th>
      <th>accelerometer_z_grav_mean</th>
      <th>accelerometer_x_grav_std</th>
      <th>accelerometer_y_grav_std</th>
      <th>accelerometer_z_grav_std</th>
      <th>accelerometer_std_norm</th>
      <th>accelerometer_x_power_below_gait</th>
      <th>accelerometer_y_power_below_gait</th>
      <th>...</th>
      <th>gyroscope_mfcc_3</th>
      <th>gyroscope_mfcc_4</th>
      <th>gyroscope_mfcc_5</th>
      <th>gyroscope_mfcc_6</th>
      <th>gyroscope_mfcc_7</th>
      <th>gyroscope_mfcc_8</th>
      <th>gyroscope_mfcc_9</th>
      <th>gyroscope_mfcc_10</th>
      <th>gyroscope_mfcc_11</th>
      <th>gyroscope_mfcc_12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1463.00</td>
      <td>-0.941813</td>
      <td>-0.216146</td>
      <td>-0.129167</td>
      <td>0.031408</td>
      <td>0.089393</td>
      <td>0.060770</td>
      <td>0.166103</td>
      <td>0.000596</td>
      <td>0.007747</td>
      <td>...</td>
      <td>-0.555074</td>
      <td>0.735551</td>
      <td>0.180431</td>
      <td>0.044856</td>
      <td>-0.645180</td>
      <td>-0.255526</td>
      <td>0.122215</td>
      <td>0.297500</td>
      <td>0.326476</td>
      <td>0.348350</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1463.74</td>
      <td>-0.934076</td>
      <td>-0.198583</td>
      <td>-0.092962</td>
      <td>0.045522</td>
      <td>0.066541</td>
      <td>0.038878</td>
      <td>0.360830</td>
      <td>0.001213</td>
      <td>0.002664</td>
      <td>...</td>
      <td>-0.718265</td>
      <td>0.689992</td>
      <td>-0.251657</td>
      <td>-0.279176</td>
      <td>-0.798014</td>
      <td>-0.100938</td>
      <td>0.034502</td>
      <td>0.118056</td>
      <td>0.162835</td>
      <td>0.373055</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1464.49</td>
      <td>-0.883701</td>
      <td>-0.263048</td>
      <td>-0.080902</td>
      <td>0.094025</td>
      <td>0.145017</td>
      <td>0.021149</td>
      <td>0.362067</td>
      <td>0.002425</td>
      <td>0.001264</td>
      <td>...</td>
      <td>-1.126609</td>
      <td>0.759017</td>
      <td>-0.217674</td>
      <td>-0.435664</td>
      <td>-0.443270</td>
      <td>-0.069035</td>
      <td>0.005618</td>
      <td>0.079465</td>
      <td>0.178917</td>
      <td>0.083811</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1465.23</td>
      <td>-0.797733</td>
      <td>-0.400167</td>
      <td>-0.093532</td>
      <td>0.126371</td>
      <td>0.211431</td>
      <td>0.034593</td>
      <td>0.364234</td>
      <td>0.004846</td>
      <td>0.008480</td>
      <td>...</td>
      <td>-1.158651</td>
      <td>1.020173</td>
      <td>-0.163708</td>
      <td>-0.228485</td>
      <td>-0.162279</td>
      <td>-0.015891</td>
      <td>0.126144</td>
      <td>0.099092</td>
      <td>0.233570</td>
      <td>0.016199</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465.97</td>
      <td>-0.695245</td>
      <td>-0.571835</td>
      <td>-0.117326</td>
      <td>0.127879</td>
      <td>0.220583</td>
      <td>0.036150</td>
      <td>0.361459</td>
      <td>0.003841</td>
      <td>0.004359</td>
      <td>...</td>
      <td>-0.777551</td>
      <td>0.784781</td>
      <td>-0.158454</td>
      <td>-0.026845</td>
      <td>-0.001841</td>
      <td>-0.178935</td>
      <td>-0.112552</td>
      <td>0.086241</td>
      <td>0.172922</td>
      <td>-0.038833</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 62 columns</p>
</div>



The features extracted are similar to the features extracted for gait detection, but the gyroscope has been added to extract additional MFCCs of this sensor. The gyroscope (measuring angular velocity) is relevant to distinguish between arm activities. Also note that the `time` column no longer starts at 0, since the first timestamps were predicted as non-gait and therefore discarded.

## Step 5: Filtering gait
This classification task is similar to gait detection, although it uses a different classification object. The trained classifier is a logistic regression, similarly trained on the dataset of the [Parkinson@Home Validation Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7584982/). Filtering gait is the process of detecting and removing gait segments containing other arm activities. This is an important process since individuals entertain a wide array of arm activities during gait: having hands in pockets, holding a dog leash, or carrying a plate to the kitchen. We trained a classifier to detect these other arm activities during gait, enabling accurate estimations of the arm swing.


```python
# Set the path to the classifier package
classifier_package_filename = 'gait_filtering_clf_package.pkl'
full_path_to_classifier_package = (
    files('paradigma')
    / 'assets'
    / classifier_package_filename
)

# Load the classifier package
clf_package_filtering = ClassifierPackage.load(full_path_to_classifier_package)
filt_threshold = clf_package_filtering.threshold

# Detecting no_other_arm_activity returns the probability of
# no_other_arm_activity for each window, which is concatenated to
# the original dataframe
df_arm[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = filter_gait(
    df=df_arm,
    clf_package=clf_package_filtering
)


n_windows = df_arm.shape[0]
n_pred_no_other_arm_act = df_arm.loc[
    df_arm[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= filt_threshold
].shape[0]
perc_no_other_arm_activity = round(
    100 * n_pred_no_other_arm_act / n_windows,
    1
)
n_pred_other_arm_act = df_arm.loc[
    df_arm[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] < filt_threshold
].shape[0]
perc_other_arm_activity = round(
    100 * n_pred_other_arm_act / n_windows,
    1
)

print(
    f"Out of {n_windows} windows, {n_pred_no_other_arm_act} "
    f"({perc_no_other_arm_activity}%) were predicted as no_other_arm_activity, "
    f"and {n_pred_other_arm_act} ({perc_other_arm_activity}%) as other_arm_activity."
)

# Only the time and predicted probabilities are shown,
# but the dataframe also contains the extracted features
df_arm[[arm_config.time_colname, DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]].head()
```

    Out of 2755 windows, 927 (33.6%) were predicted as no_other_arm_activity, and 1828 (66.4%) as other_arm_activity.





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>pred_no_other_arm_activity_proba</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1463.00</td>
      <td>0.200129</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1463.74</td>
      <td>0.109148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1464.49</td>
      <td>0.137190</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1465.23</td>
      <td>0.167751</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465.97</td>
      <td>0.036145</td>
    </tr>
  </tbody>
</table>
</div>



## Step 6: Arm swing quantification
The next step is to extract arm swing estimates from the predicted gait segments. Arm swing estimates can be calculated for both **filtered** gait (i.e., gait without other arm activities) and **unfiltered** gait (i.e., all predicted gait segments including other arm activities).

**Important:** As of version 1.1.0, the high-level gait pipeline functions (such as `run_gait_pipeline()` and `run_paradigma()`) return arm swing results in a **nested** structure with separate entries for filtered and unfiltered gait:
- Quantified arm swing parameters are exposed under keys `'filtered'` and `'unfiltered'`
  - `'filtered'`: DataFrame with arm swings from gait without other arm activities
  - `'unfiltered'`: DataFrame with arm swings from all gait segments
- Corresponding gait segment metadata are also provided separately for the filtered and unfiltered cases

This allows analysis of arm swing with and without filtering for other arm activities when using the pipeline output.

When calling `quantify_arm_swing()` directly (as shown in this tutorial), it still returns a `(DataFrame, dict)` pair, and the `filtered` argument controls whether only gait segments without other arm activities are included in the results.

Specifically, the range of motion (`'range_of_motion'`) and peak angular velocity (`'peak_velocity'`) are extracted.

This step creates gait segments based on consecutively predicted gait windows. A new gait segment is created if the gap between consecutive gait predictions exceeds `config.max_segment_gap_s`. Furthermore, a gait segment is considered valid if it is of at minimum length `config.min_segment_length_s`.

But, first, similar to the step of extracting arm activity features, the predictions of the previous step should be merged with the preprocessed time series data.


```python
# Merge arm activity predictions into timeseries data
if not any(
    df_arm[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= filt_threshold
):
    raise ValueError(
        "No gait without other arm activities detected in the input data."
    )

df = merge_predictions_with_timestamps(
    df_ts=df_preprocessed,
    df_predictions=df_arm,
    pred_proba_colname=DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA,
    window_length_s=arm_config.window_length_s,
    fs=arm_config.sampling_frequency
)

# Add a column for predicted gait based on a fitted threshold
df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY] = (
    df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= filt_threshold
).astype(int)
```


```python
# Set to True to quantify arm swing based on the filtered gait segments, and
# False to quantify arm swing based on all gait segments
filtered = True

if filtered:
    dataset_used = 'filtered'
    print("The arm swing quantification is based on the filtered gait segments.\n")
else:
    dataset_used = 'unfiltered'
    print("The arm swing quantification is based on all gait segments.\n")

quantified_arm_swing, gait_segment_meta = quantify_arm_swing(
    df=df,
    fs=arm_config.sampling_frequency,
    filtered=filtered,
    max_segment_gap_s=arm_config.max_segment_gap_s,
    min_segment_length_s=arm_config.min_segment_length_s,
    start_dt=None,  # if provided, this ensures datetime format of segments is returned
)

print(
    f"Gait segments are created of minimum {arm_config.min_segment_length_s} seconds "
    f"and maximum {arm_config.max_segment_gap_s} seconds gap between segments.\n"
)
print(
    f"A total of {quantified_arm_swing['gait_segment_nr'].nunique()} {dataset_used} "
    f"gait segments have been quantified."
)

print("\nMetadata of the first gait segment:")
first_segment_meta = gait_segment_meta['per_segment'][1]
print(json.dumps(first_segment_meta, indent = 1))

segment_duration_s = first_segment_meta['duration_s']
segment_start_s = first_segment_meta['start_s']
segment_end_s = first_segment_meta['end_s']

print(
    f"\nThe first {dataset_used} gait segment has a duration of "
    f"{segment_duration_s:.2f} seconds, starting at {segment_start_s:.2f}s and "
    f"ending at {segment_end_s:.2f}s."
)

print(
    f"\nIndividual arm swings of the first gait segment of the "
    f" {dataset_used} dataset:"
)
quantified_arm_swing.loc[quantified_arm_swing['gait_segment_nr'] == 1]
```

    The arm swing quantification is based on the filtered gait segments.

    Gait segments are created of minimum 1.5 seconds and maximum 1.5 seconds gap between segments.

    A total of 79 filtered gait segments have been quantified.

    Metadata of the first gait segment:
    {
     "start_s": 2221.71,
     "end_s": 2230.649900990099,
     "duration_s": 8.94059405940594,
     "unfiltered_duration_s": 12.653465346534654,
     "segment_categories": [
      "0_20"
     ]
    }

    The first filtered gait segment has a duration of 8.94 seconds, starting at 2221.71s and ending at 2230.65s.

    Individual arm swings of the first gait segment of the  filtered dataset:





<div>
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
      <td>20.777252</td>
      <td>90.763084</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22.350129</td>
      <td>105.756866</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>24.345197</td>
      <td>103.794565</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>24.725786</td>
      <td>115.074796</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>17.669009</td>
      <td>63.330240</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>12.262003</td>
      <td>59.865183</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>6.787542</td>
      <td>36.696967</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>6.206446</td>
      <td>30.851121</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>7.865740</td>
      <td>42.370003</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>9.697051</td>
      <td>44.037056</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>9.790587</td>
      <td>38.668452</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>9.540712</td>
      <td>34.264631</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>9.189801</td>
      <td>33.028421</td>
    </tr>
  </tbody>
</table>
</div>



### Run steps 1-6 for the all raw data segment(s) <a id='multiple_segments_cell'></a>

If your data is also stored in multiple raw data segments, you can modify `raw_data_segments` in the cell below to a list of the filenames of your respective segmented data.


```python
# Set the path to where the prepared data is saved
path_to_data =  Path('../../example_data/verily')
path_to_prepared_data = path_to_data / 'imu'

# Load the gait detection classifier package
classifier_package_filename = 'gait_detection_clf_package.pkl'
full_path_to_classifier_package = (
    files('paradigma')
    / 'assets'
    / classifier_package_filename
)
clf_package_detection = ClassifierPackage.load(full_path_to_classifier_package)

# Load the gait filtering classifier package
classifier_package_filename = 'gait_filtering_clf_package.pkl'
full_path_to_classifier_package = (
    files('paradigma')
    / 'assets'
    / classifier_package_filename
)
clf_package_filtering = ClassifierPackage.load(full_path_to_classifier_package)

# Set to True to quantify arm swing based on the filtered gait segments, and
# False to quantify arm swing based on all gait segments
filtered = True

# Create a list to store all quantified arm swing segments
list_quantified_arm_swing = []
max_gait_segment_nr = 0

raw_data_segments  = ['0001']  # list of raw data segments to process in this example

for raw_data_segment_nr in raw_data_segments:

    # Load the data
    df_imu, _, _ = load_tsdf_dataframe(
        path_to_prepared_data,
        prefix=f'imu_segment{raw_data_segment_nr}'
    )

    # 1: Preprocess the data
    # Change column names if necessary by creating parameter column_mapping
    # (see previous cells for an example)
    imu_config = IMUConfig()

    df_preprocessed = preprocess_imu_data(
        df=df_imu,
        config=imu_config,
        sensor='both',
        watch_side='left',
    )

    # 2: Extract gait features
    gait_config = GaitConfig(step='gait')
    gait_config._set_sampling_frequency_detected(imu_config.sampling_frequency)

    df_gait = extract_gait_features(
        df=df_preprocessed,
        config=gait_config
    )

    # 3: Detect gait
    df_gait[DataColumns.PRED_GAIT_PROBA] = detect_gait(
        df=df_gait,
        clf_package=clf_package_detection
    )

    # Merge gait predictions into timeseries data
    if not any(
        df_gait[DataColumns.PRED_GAIT_PROBA] >= clf_package_detection.threshold
    ):
        raise ValueError("No gait detected in the input data.")

    df = merge_predictions_with_timestamps(
        df_ts=df_preprocessed,
        df_predictions=df_gait,
        pred_proba_colname=DataColumns.PRED_GAIT_PROBA,
        window_length_s=gait_config.window_length_s,
        fs=gait_config.sampling_frequency
    )

    df[DataColumns.PRED_GAIT] = (
        df[DataColumns.PRED_GAIT_PROBA] >= clf_package_detection.threshold
    ).astype(int)
    df = df.loc[df[DataColumns.PRED_GAIT]==1].reset_index(drop=True)

    # 4: Extract arm activity features
    arm_config = GaitConfig(step='arm_activity')
    arm_config._set_sampling_frequency_detected(imu_config.sampling_frequency)

    df_arm_activity = extract_arm_activity_features(
        df=df,
        config=arm_config,
    )

    # 5: Filter gait
    df_arm_activity[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = filter_gait(
        df=df_arm_activity,
        clf_package=clf_package_filtering
    )

    # Merge arm activity predictions into timeseries data
    if not any(
        df_arm_activity[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= filt_threshold
    ):
        raise ValueError(
            "No gait without other arm activities detected in the input data."
        )

    df = merge_predictions_with_timestamps(
        df_ts=df_preprocessed,
        df_predictions=df_arm_activity,
        pred_proba_colname=DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA,
        window_length_s=arm_config.window_length_s,
        fs=arm_config.sampling_frequency
    )

    df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY] = (
        df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= filt_threshold
    ).astype(int)
    df = df.loc[df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY]==1].reset_index(drop=True)

    # 6: Quantify arm swing
    quantified_arm_swing, gait_segment_meta = quantify_arm_swing(
        df=df,
        fs=arm_config.sampling_frequency,
        filtered=filtered,
        max_segment_gap_s=arm_config.max_segment_gap_s,
        min_segment_length_s=arm_config.min_segment_length_s,
    )

    # Since gait segments start at zero, and we are concatenating multiple segments,
    # we need to update the gait segment numbers to avoid aggregating multiple
    # gait segments with the same number
    if len(list_quantified_arm_swing) == 0:
        max_gait_segment_nr = 0
    else:
        max_gait_segment_nr = quantified_arm_swing['gait_segment_nr'].max()

    quantified_arm_swing['gait_segment_nr'] += max_gait_segment_nr
    gait_segment_meta['per_segment'] = {
        k + max_gait_segment_nr: v for k, v in gait_segment_meta['per_segment'].items()
    }

    # Add the predictions of the current raw data segment to the list
    quantified_arm_swing['raw_data_segment_nr'] = raw_data_segment_nr
    list_quantified_arm_swing.append(quantified_arm_swing)

quantified_arm_swing = pd.concat(list_quantified_arm_swing, ignore_index=True)
```

    INFO: Resampled: 3455331 -> 3468300 rows at 101 Hz


## Step 7: Aggregation
Finally, the arm swing estimates can be aggregated across all gait segments.

Optionally, gait segments can be categorized into bins of specific length. Bins are tuples *(a, b)* including *a* and excluding *b*, i.e., gait segments `≥ a` seconds and `< b` seconds. For example, to analyze gait segments of at least 20 seconds, the tuple `(20, np.inf)` can be used. In case you want to analyze all gait segments combined, use `(0, np.inf)`.


```python
segment_categories = [(0,10), (10,20), (20, np.inf), (0, np.inf)]

arm_swing_aggregations = aggregate_arm_swing_params(
    df_arm_swing_params=quantified_arm_swing,
    segment_meta=gait_segment_meta,
    gait_segment_categories=segment_categories,
    aggregates=['median', '95p']
)

print(json.dumps(arm_swing_aggregations, indent=2))
```

    {
      "0_10": {
        "duration_s": 356.9306930693069,
        "median_range_of_motion": 11.555917616414455,
        "95p_range_of_motion": 41.02419007440517,
        "median_peak_velocity": 59.97911737734337,
        "95p_peak_velocity": 191.59039595290588
      },
      "10_20": {
        "duration_s": 64.66336633663366,
        "median_range_of_motion": 15.7581269688773,
        "95p_range_of_motion": 59.361218461763244,
        "median_peak_velocity": 83.12687400531114,
        "95p_peak_velocity": 238.85273366009574
      },
      "20_inf": {
        "duration_s": 307.54455445544556,
        "median_range_of_motion": 28.12963717852429,
        "95p_range_of_motion": 48.79783015544807,
        "median_peak_velocity": 138.61976910396135,
        "95p_peak_velocity": 258.9169326938605
      },
      "0_inf": {
        "duration_s": 729.1386138613861,
        "median_range_of_motion": 18.139727342193993,
        "95p_range_of_motion": 47.128543297278256,
        "median_peak_velocity": 94.49319571680101,
        "95p_peak_velocity": 247.55749727037238
      }
    }


The output of the aggregation step contains the aggregated arm swing parameters per gait segment category. Additionally, the total time in seconds `time_s` is added to inform based on how much data the aggregations were created.
