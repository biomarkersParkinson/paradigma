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
config = IMUConfig()

df_preprocessed = preprocess_imu_data(
    df=df_imu,
    config=config,
    sensor='both',
    watch_side='left',
)

print(
    f"The dataset of {df_preprocessed.shape[0] / config.sampling_frequency} seconds "
    f"is automatically resampled to {config.resampling_frequency} Hz."
)
print(
    f"The tolerance for checking contiguous timestamps is set "
    f"to {config.tolerance:.3f} seconds."
)
df_preprocessed.head()
```

    INFO: Data at 100.00 Hz meets quality threshold. Skipping unnecessary resampling to 100.00 Hz


    The dataset of 34553.31 seconds is automatically resampled to 100 Hz.
    The tolerance for checking contiguous timestamps is set to 0.030 seconds.





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
      <td>-0.002323</td>
      <td>-0.001227</td>
      <td>-0.002185</td>
      <td>0.000000</td>
      <td>1.402439</td>
      <td>0.243902</td>
      <td>-0.472318</td>
      <td>-0.378199</td>
      <td>0.772520</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.009933</td>
      <td>-0.000400</td>
      <td>-0.000723</td>
      <td>-0.007460</td>
      <td>0.426829</td>
      <td>0.670732</td>
      <td>-0.121951</td>
      <td>-0.472327</td>
      <td>-0.378225</td>
      <td>0.772532</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.019867</td>
      <td>0.000567</td>
      <td>0.002653</td>
      <td>-0.005559</td>
      <td>1.158537</td>
      <td>-0.060976</td>
      <td>-0.304878</td>
      <td>-0.472337</td>
      <td>-0.378251</td>
      <td>0.772544</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.029800</td>
      <td>-0.000380</td>
      <td>0.002679</td>
      <td>-0.002221</td>
      <td>1.158537</td>
      <td>-0.548780</td>
      <td>-0.548780</td>
      <td>-0.472347</td>
      <td>-0.378277</td>
      <td>0.772556</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.039733</td>
      <td>-0.002763</td>
      <td>-0.001123</td>
      <td>-0.000320</td>
      <td>0.670732</td>
      <td>-0.609756</td>
      <td>-0.731707</td>
      <td>-0.472356</td>
      <td>-0.378303</td>
      <td>0.772569</td>
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

config = GaitConfig(step='gait', column_mapping=column_mapping)

df_gait = extract_gait_features(
    df=df_preprocessed,
    config=config
)

print(
    f"A total of {df_gait.shape[1]-1} features have been extracted from "
    f"{df_gait.shape[0]} {config.window_length_s}-second windows with "
    f"{config.window_length_s-config.window_step_length_s} seconds overlap."
)
df_gait.head()
```

    A total of 34 features have been extracted from 34548 6-second windows with 5 seconds overlap.





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
      <td>0.000000</td>
      <td>-0.472968</td>
      <td>-0.380609</td>
      <td>0.774291</td>
      <td>0.000275</td>
      <td>0.000766</td>
      <td>0.000560</td>
      <td>0.003373</td>
      <td>0.000003</td>
      <td>1.151509e-06</td>
      <td>...</td>
      <td>-1.105308</td>
      <td>0.505124</td>
      <td>0.212134</td>
      <td>0.403196</td>
      <td>0.870436</td>
      <td>1.129268</td>
      <td>0.703834</td>
      <td>-0.085600</td>
      <td>-0.322168</td>
      <td>-0.093405</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.993667</td>
      <td>-0.472998</td>
      <td>-0.380703</td>
      <td>0.774541</td>
      <td>0.000245</td>
      <td>0.000574</td>
      <td>0.000226</td>
      <td>0.003204</td>
      <td>0.000003</td>
      <td>1.204678e-06</td>
      <td>...</td>
      <td>-0.994130</td>
      <td>0.621509</td>
      <td>0.322671</td>
      <td>0.414244</td>
      <td>0.942337</td>
      <td>1.122690</td>
      <td>0.811417</td>
      <td>-0.061107</td>
      <td>-0.382304</td>
      <td>-0.048965</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.987667</td>
      <td>-0.473034</td>
      <td>-0.380561</td>
      <td>0.774574</td>
      <td>0.000242</td>
      <td>0.000610</td>
      <td>0.000201</td>
      <td>0.003201</td>
      <td>0.000002</td>
      <td>6.718255e-07</td>
      <td>...</td>
      <td>-1.032682</td>
      <td>0.396731</td>
      <td>0.253735</td>
      <td>0.464214</td>
      <td>0.919187</td>
      <td>1.021736</td>
      <td>0.832672</td>
      <td>-0.011435</td>
      <td>-0.407592</td>
      <td>-0.015557</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.981000</td>
      <td>-0.472960</td>
      <td>-0.380315</td>
      <td>0.774648</td>
      <td>0.000297</td>
      <td>0.000529</td>
      <td>0.000318</td>
      <td>0.003034</td>
      <td>0.000002</td>
      <td>6.617470e-07</td>
      <td>...</td>
      <td>-1.068324</td>
      <td>0.244040</td>
      <td>0.239564</td>
      <td>0.465579</td>
      <td>0.867227</td>
      <td>0.904580</td>
      <td>0.752168</td>
      <td>-0.025484</td>
      <td>-0.301639</td>
      <td>0.026369</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.975667</td>
      <td>-0.472709</td>
      <td>-0.380035</td>
      <td>0.774878</td>
      <td>0.000449</td>
      <td>0.000357</td>
      <td>0.000464</td>
      <td>0.002888</td>
      <td>0.000002</td>
      <td>1.081647e-06</td>
      <td>...</td>
      <td>-1.076344</td>
      <td>0.246587</td>
      <td>0.230658</td>
      <td>0.547408</td>
      <td>0.918235</td>
      <td>0.767667</td>
      <td>0.643226</td>
      <td>-0.118808</td>
      <td>-0.191408</td>
      <td>0.108192</td>
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
df_gait[[config.time_colname, DataColumns.PRED_GAIT_PROBA]].head()
```

    Out of 34548 windows, 2747 (8.0%)
    were predicted as gait, and 31801 (92.0%)
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
      <td>0.000000</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.993667</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.987667</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.981000</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.975667</td>
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
metadata_time_store.channels = [config.time_colname]
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
      <td>0.000000</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.993667</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.987667</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.981000</td>
      <td>0.000023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.975667</td>
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

gait_preprocessing_config = GaitConfig(step='gait')

df = merge_predictions_with_timestamps(
    df_ts=df_preprocessed,
    df_predictions=df_gait,
    pred_proba_colname=DataColumns.PRED_GAIT_PROBA,
    window_length_s=gait_preprocessing_config.window_length_s,
    fs=gait_preprocessing_config.sampling_frequency
)

# Add a column for predicted gait based on a fitted threshold
df[DataColumns.PRED_GAIT] = (
    df[DataColumns.PRED_GAIT_PROBA] >= clf_package_detection.threshold
).astype(int)

# Filter the DataFrame to only include predicted gait (1)
df = df.loc[df[DataColumns.PRED_GAIT]==1].reset_index(drop=True)
```


```python
config = GaitConfig(step='arm_activity')

df_arm = extract_arm_activity_features(
    df=df,
    config=config,
)

print(
    f"A total of {df_arm.shape[1] - 1} features have been extracted "
    f"from {df_arm.shape[0]} {config.window_length_s}-second windows "
    f"with {config.window_length_s - config.window_step_length_s} seconds overlap."
)
df_arm.head()
```

    A total of 61 features have been extracted from 2757 3-second windows with 2.25 seconds overlap.





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
      <td>1463.07</td>
      <td>-0.943119</td>
      <td>-0.211245</td>
      <td>-0.125436</td>
      <td>0.030597</td>
      <td>0.085997</td>
      <td>0.059704</td>
      <td>0.188947</td>
      <td>0.000468</td>
      <td>0.007631</td>
      <td>...</td>
      <td>-0.598599</td>
      <td>0.735297</td>
      <td>0.123057</td>
      <td>0.046734</td>
      <td>-0.658865</td>
      <td>-0.264201</td>
      <td>0.147703</td>
      <td>0.311547</td>
      <td>0.322653</td>
      <td>0.361395</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1463.81</td>
      <td>-0.932621</td>
      <td>-0.198881</td>
      <td>-0.090402</td>
      <td>0.048966</td>
      <td>0.070364</td>
      <td>0.036554</td>
      <td>0.366069</td>
      <td>0.001309</td>
      <td>0.002404</td>
      <td>...</td>
      <td>-0.747559</td>
      <td>0.646198</td>
      <td>-0.258551</td>
      <td>-0.306104</td>
      <td>-0.782468</td>
      <td>-0.123423</td>
      <td>-0.002756</td>
      <td>0.074582</td>
      <td>0.149163</td>
      <td>0.362355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1464.56</td>
      <td>-0.878327</td>
      <td>-0.270713</td>
      <td>-0.080581</td>
      <td>0.098331</td>
      <td>0.152360</td>
      <td>0.021547</td>
      <td>0.364226</td>
      <td>0.002602</td>
      <td>0.001563</td>
      <td>...</td>
      <td>-1.159250</td>
      <td>0.803385</td>
      <td>-0.194169</td>
      <td>-0.412418</td>
      <td>-0.418789</td>
      <td>-0.060248</td>
      <td>0.049491</td>
      <td>0.091165</td>
      <td>0.187773</td>
      <td>0.080063</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1465.30</td>
      <td>-0.788787</td>
      <td>-0.414315</td>
      <td>-0.095250</td>
      <td>0.128551</td>
      <td>0.215131</td>
      <td>0.035795</td>
      <td>0.363348</td>
      <td>0.005378</td>
      <td>0.008291</td>
      <td>...</td>
      <td>-1.144686</td>
      <td>1.019807</td>
      <td>-0.132758</td>
      <td>-0.202768</td>
      <td>-0.134086</td>
      <td>-0.023182</td>
      <td>0.106764</td>
      <td>0.085045</td>
      <td>0.217903</td>
      <td>0.029877</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1466.05</td>
      <td>-0.684683</td>
      <td>-0.588562</td>
      <td>-0.119566</td>
      <td>0.126552</td>
      <td>0.218069</td>
      <td>0.035382</td>
      <td>0.356186</td>
      <td>0.004388</td>
      <td>0.004385</td>
      <td>...</td>
      <td>-0.741050</td>
      <td>0.728900</td>
      <td>-0.139386</td>
      <td>-0.022936</td>
      <td>-0.017016</td>
      <td>-0.189782</td>
      <td>-0.136417</td>
      <td>0.098604</td>
      <td>0.165319</td>
      <td>-0.020198</td>
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
df_arm[[config.time_colname, DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA]].head()
```

    Out of 2757 windows, 933 (33.8%) were predicted as no_other_arm_activity, and 1824 (66.2%) as other_arm_activity.





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
      <td>1463.07</td>
      <td>0.198040</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1463.81</td>
      <td>0.106580</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1464.56</td>
      <td>0.136358</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1465.30</td>
      <td>0.156168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1466.05</td>
      <td>0.030799</td>
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

config = GaitConfig(step='arm_activity')

df = merge_predictions_with_timestamps(
    df_ts=df_preprocessed,
    df_predictions=df_arm,
    pred_proba_colname=DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA,
    window_length_s=config.window_length_s,
    fs=config.sampling_frequency
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
    fs=config.sampling_frequency,
    filtered=filtered,
    max_segment_gap_s=config.max_segment_gap_s,
    min_segment_length_s=config.min_segment_length_s,
)

print(
    f"Gait segments are created of minimum {config.min_segment_length_s} seconds "
    f"and maximum {config.max_segment_gap_s} seconds gap between segments.\n"
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
    f"\nThe first {dataset_used} gait segment has a duration of {segment_duration_s:.2f} seconds, "
    f"starting at {segment_start_s:.2f}s and ending at {segment_end_s:.2f}s."
)

print(
    f"\nIndividual arm swings of the first gait segment of the "
    f" {dataset_used} dataset:"
)
quantified_arm_swing.loc[quantified_arm_swing['gait_segment_nr'] == 1]
```

    The arm swing quantification is based on the filtered gait segments.

    Gait segments are created of minimum 1.5 seconds and maximum 1.5 seconds gap between segments.

    A total of 81 filtered gait segments have been quantified.

    Metadata of the first gait segment:
    {
     "start_s": 1616.12,
     "end_s": 1619.11,
     "duration_s": 3.01
    }

    The first filtered gait segment has a duration of 3.01 seconds, starting at 1616.12s and ending at 1619.11s.

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
      <td>18.682702</td>
      <td>72.068852</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>21.772653</td>
      <td>119.622537</td>
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
    config = IMUConfig()

    df_preprocessed = preprocess_imu_data(
        df=df_imu,
        config=config,
        sensor='both',
        watch_side='left',
    )

    # 2: Extract gait features
    config = GaitConfig(step='gait')

    df_gait = extract_gait_features(
        df=df_preprocessed,
        config=config
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
        window_length_s=config.window_length_s,
        fs=config.sampling_frequency
    )

    df[DataColumns.PRED_GAIT] = (
        df[DataColumns.PRED_GAIT_PROBA] >= clf_package_detection.threshold
    ).astype(int)
    df = df.loc[df[DataColumns.PRED_GAIT]==1].reset_index(drop=True)

    # 4: Extract arm activity features
    config = GaitConfig(step='arm_activity')

    df_arm_activity = extract_arm_activity_features(
        df=df,
        config=config,
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
        window_length_s=config.window_length_s,
        fs=config.sampling_frequency
    )

    df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY] = (
        df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= filt_threshold
    ).astype(int)
    df = df.loc[df[DataColumns.PRED_NO_OTHER_ARM_ACTIVITY]==1].reset_index(drop=True)

    # 6: Quantify arm swing
    quantified_arm_swing, gait_segment_meta = quantify_arm_swing(
        df=df,
        fs=config.sampling_frequency,
        filtered=filtered,
        max_segment_gap_s=config.max_segment_gap_s,
        min_segment_length_s=config.min_segment_length_s,
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

    INFO: Data at 100.00 Hz meets quality threshold. Skipping unnecessary resampling to 100.00 Hz


## Step 7: Aggregation
Finally, the arm swing estimates can be aggregated across all gait segments.

Optionally, gait segments can be categorized into bins of specific length. Bins are tuples *(a, b)* including *a* and excluding *b*, i.e., gait segments `≥ a` seconds and `< b` seconds. For example, to analyze gait segments of at least 20 seconds, the tuple `(20, np.inf)` can be used. In case you want to analyze all gait segments combined, use `(0, np.inf)`.


```python
segment_categories = [(0,10), (10,20), (20, np.inf), (0, np.inf)]

arm_swing_aggregations = aggregate_arm_swing_params(
    df_arm_swing_params=quantified_arm_swing,
    segment_meta=gait_segment_meta,
    segment_cats=segment_categories,
    aggregates=['median', '95p']
)

print(json.dumps(arm_swing_aggregations, indent=2))
```

    {
      "0_10": {
        "duration_s": 343.71,
        "median_range_of_motion": 11.329445979107904,
        "95p_range_of_motion": 38.80715638131665,
        "median_peak_velocity": 58.385719697525076,
        "95p_peak_velocity": 189.35575159312535
      },
      "10_20": {
        "duration_s": 74.12,
        "median_range_of_motion": 16.317311020932507,
        "95p_range_of_motion": 48.28521784250025,
        "median_peak_velocity": 106.07594472083005,
        "95p_peak_velocity": 227.03838591455244
      },
      "20_inf": {
        "duration_s": 310.53000000000003,
        "median_range_of_motion": 25.744038880835415,
        "95p_range_of_motion": 41.978104241282786,
        "median_peak_velocity": 144.09755220382561,
        "95p_peak_velocity": 268.00249051735057
      },
      "0_inf": {
        "duration_s": 728.36,
        "median_range_of_motion": 17.83745601349677,
        "95p_range_of_motion": 41.44733771656344,
        "median_peak_velocity": 95.21419907458392,
        "95p_peak_velocity": 255.84351691132278
      }
    }


The output of the aggregation step contains the aggregated arm swing parameters per gait segment category. Additionally, the total time in seconds `time_s` is added to inform based on how much data the aggregations were created.
