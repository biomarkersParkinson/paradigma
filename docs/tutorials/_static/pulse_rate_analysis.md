# Pulse rate analysis

This tutorial shows how to extract pulse rate estimates using photoplethysmography (PPG) data and accelerometer data. The pipeline consists of a stepwise approach to determine signal quality, assessing both PPG morphology and accounting for periodic artifacts using the accelerometer. The usage of accelerometer is optional but is recommended to specifically account for periodic motion artifacts. Based on the signal quality, we extract high-quality segments and estimate the pulse rate for every 2 s using the smoothed pseudo Wigner-Ville Distribution.

In this tutorial, we use two days of data from a participant of the Personalized Parkinson Project to demonstrate the functionalities. Since `ParaDigMa` expects contiguous time series, the collected data was stored in two segments each with contiguous timestamps. Per segment, we load the data and perform the following steps:
1. Preprocess the time series data
2. Extract signal quality features
3. Signal quality classification
4. Pulse rate estimation

We then combine the output of the different segments for the final step:

5. Pulse rate aggregation

## Load data

This pipeline requires PPG data and preferably accelerometer data (optional). Here, we start by loading a single contiguous time series (segment), for which we continue running steps 1-4. [Below](#multiple_segments_cell) we show how to run these steps for multiple segments. The channel `green` represents the values obtained with PPG using green light.

In this example we use the interally developed `TSDF` ([documentation](https://biomarkersparkinson.github.io/tsdf/)) to load and store data [[1](https://arxiv.org/abs/2211.11294)]. However, we are aware that there are other common data formats. For example, the following functions can be used depending on the file extension of the data:
- _.csv_: `pandas.read_csv()` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html))
- _.json_: `json.load()` ([documentation](https://docs.python.org/3/library/json.html#json.load))



```python
from pathlib import Path
from paradigma.util import load_tsdf_dataframe

# Set the path to where the prepared data is saved and load the data.
# Note: the test data is stored in TSDF, but you can load your data in your own way
path_to_prepared_data =  Path('../../example_data')

ppg_prefix = 'ppg'
imu_prefix = 'imu'

segment_nr = '0001' 

df_ppg, metadata_time_ppg, metadata_values_ppg = load_tsdf_dataframe(
    path_to_data=path_to_prepared_data / ppg_prefix, 
    prefix=f'PPG_segment{segment_nr}'
)

# Only relevant if you have IMU data available
df_imu, metadata_time_imu, metadata_values_imu = load_tsdf_dataframe(
    path_to_data=path_to_prepared_data / imu_prefix, 
    prefix=f'IMU_segment{segment_nr}'
)

# Drop the gyroscope columns from the IMU data to keep only accelerometer data
cols_to_drop = df_imu.filter(regex='^gyroscope_').columns
df_acc = df_imu.drop(cols_to_drop, axis=1)

display(df_ppg, df_acc)
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
      <td>262316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.03340</td>
      <td>262320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.06680</td>
      <td>262446</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.10020</td>
      <td>262770</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.13360</td>
      <td>262623</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1029370</th>
      <td>34339.49720</td>
      <td>1049632</td>
    </tr>
    <tr>
      <th>1029371</th>
      <td>34339.53056</td>
      <td>1049632</td>
    </tr>
    <tr>
      <th>1029372</th>
      <td>34339.56392</td>
      <td>1049632</td>
    </tr>
    <tr>
      <th>1029373</th>
      <td>34339.59728</td>
      <td>1049632</td>
    </tr>
    <tr>
      <th>1029374</th>
      <td>34339.63064</td>
      <td>1020788</td>
    </tr>
  </tbody>
</table>
<p>1029375 rows × 2 columns</p>
</div>



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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-0.474641</td>
      <td>-0.379426</td>
      <td>0.770335</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.009933</td>
      <td>-0.472727</td>
      <td>-0.378947</td>
      <td>0.765072</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.019867</td>
      <td>-0.471770</td>
      <td>-0.375598</td>
      <td>0.766986</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.029800</td>
      <td>-0.472727</td>
      <td>-0.375598</td>
      <td>0.770335</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.039733</td>
      <td>-0.475120</td>
      <td>-0.379426</td>
      <td>0.772249</td>
    </tr>
    <tr>
      <th>...</th>
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
    </tr>
    <tr>
      <th>3455327</th>
      <td>34339.571267</td>
      <td>-0.555502</td>
      <td>-0.153110</td>
      <td>-0.671292</td>
    </tr>
    <tr>
      <th>3455328</th>
      <td>34339.581200</td>
      <td>-0.286124</td>
      <td>-0.263636</td>
      <td>-0.981340</td>
    </tr>
    <tr>
      <th>3455329</th>
      <td>34339.591133</td>
      <td>-0.232536</td>
      <td>-0.161722</td>
      <td>-0.832536</td>
    </tr>
    <tr>
      <th>3455330</th>
      <td>34339.601067</td>
      <td>0.180383</td>
      <td>-0.368421</td>
      <td>-1.525837</td>
    </tr>
  </tbody>
</table>
<p>3455331 rows × 4 columns</p>
</div>


## Step 1: Preprocess data

The first step after loading the data is preprocessing using the [preprocess_ppg_data](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/preprocessing.py#:~:text=preprocess_ppg_data). This begins by isolating segments containing both PPG and IMU data, discarding portions where one modality (e.g., PPG) extends beyond the other, such as when the PPG recording is longer than the accelerometer data. This functionality requires the starting times (`metadata_time_ppg.start_iso8601` and `metadata_time_imu.start_iso8601`) in iso8601 format as inputs. After this step, the preprocess_ppg_data function resamples the PPG and accelerometer data to uniformly distributed timestamps, addressing the fixed but non-uniform sampling rates of the sensors. After this, a bandpass Butterworth filter (4th-order, bandpass frequencies: 0.4--3.5 Hz) is applied to the PPG signal, while a high-pass Butterworth filter (4th-order, cut-off frequency: 0.2 Hz) is applied to the accelerometer data. 

Note: the printed shapes are (rows, columns) with each row corresponding to a single data point and each column representing a data column (e.g.time). The number of rows of the overlapping segments of PPG and accelerometer are not the same due to sampling differences (other sensors and possibly other sampling frequencies).


```python
from paradigma.config import PPGConfig, IMUConfig
from paradigma.preprocessing import preprocess_ppg_data

ppg_config = PPGConfig()
imu_config = IMUConfig()

print(f"Original data shapes:\n- PPG data: {df_ppg.shape}\n- Accelerometer data: {df_imu.shape}")

# Remove optional arguments if you don't have accelerometer data (set to None or remove arguments)
df_ppg_proc, df_acc_proc = preprocess_ppg_data(
    df_ppg=df_ppg, 
    ppg_config=ppg_config,
    start_time_ppg=metadata_time_ppg.start_iso8601, # Optional
    df_acc=df_acc,  # Optional
    imu_config=imu_config, # Optional
    start_time_imu=metadata_time_imu.start_iso8601 # Optional
)

print(f"Overlapping preprocessed data shapes:\n- PPG data: {df_ppg_proc.shape}\n- Accelerometer data: {df_acc_proc.shape}")
display(df_ppg_proc, df_acc_proc)
```

    Original data shapes:
    - PPG data: (1029375, 2)
    - Accelerometer data: (3455331, 7)
    Overlapping preprocessed data shapes:
    - PPG data: (1030188, 2)
    - Accelerometer data: (3433961, 4)
    


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
      <td>-26.315811</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.033333</td>
      <td>91.335299</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.066667</td>
      <td>181.603416</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.100000</td>
      <td>225.760466</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.133333</td>
      <td>219.937282</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1030183</th>
      <td>34339.433333</td>
      <td>224556.234611</td>
    </tr>
    <tr>
      <th>1030184</th>
      <td>34339.466667</td>
      <td>210075.529517</td>
    </tr>
    <tr>
      <th>1030185</th>
      <td>34339.500000</td>
      <td>163811.629247</td>
    </tr>
    <tr>
      <th>1030186</th>
      <td>34339.533333</td>
      <td>94537.897763</td>
    </tr>
    <tr>
      <th>1030187</th>
      <td>34339.566667</td>
      <td>12915.304284</td>
    </tr>
  </tbody>
</table>
<p>1030188 rows × 2 columns</p>
</div>



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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>-0.002324</td>
      <td>-0.001442</td>
      <td>-0.002116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>-0.000390</td>
      <td>-0.000914</td>
      <td>-0.007396</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>0.000567</td>
      <td>0.002474</td>
      <td>-0.005445</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>-0.000425</td>
      <td>0.002414</td>
      <td>-0.002099</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>-0.002807</td>
      <td>-0.001408</td>
      <td>-0.000218</td>
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
      <td>-0.402941</td>
      <td>0.038710</td>
      <td>0.461449</td>
    </tr>
    <tr>
      <th>3433957</th>
      <td>34339.57</td>
      <td>-0.659832</td>
      <td>0.098696</td>
      <td>0.817136</td>
    </tr>
    <tr>
      <th>3433958</th>
      <td>34339.58</td>
      <td>-0.464138</td>
      <td>0.033607</td>
      <td>0.471552</td>
    </tr>
    <tr>
      <th>3433959</th>
      <td>34339.59</td>
      <td>-0.389065</td>
      <td>0.108485</td>
      <td>0.622471</td>
    </tr>
    <tr>
      <th>3433960</th>
      <td>34339.60</td>
      <td>-0.082625</td>
      <td>-0.014490</td>
      <td>0.119875</td>
    </tr>
  </tbody>
</table>
<p>3433961 rows × 4 columns</p>
</div>


## Step 2: Extract signal quality features

The preprocessed data (PPG & accelerometer) is windowed into overlapping windows of length `ppg_config.window_length_s` with a window step of `ppg_config.window_step_length_s`. From the PPG windows 10 time- and frequency domain features are extracted to assess PPG morphology and from the accelerometer windows (optional) one relative power feature is calculated to assess periodic motion artifacts.

The detailed steps are encapsulated in `extract_signal_quality_features` (documentation can be found [here](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/pipelines/pulse_rate_pipeline.py#:~:text=extract_signal_quality_features)).


```python
from paradigma.config import PulseRateConfig
from paradigma.pipelines.pulse_rate_pipeline import extract_signal_quality_features

ppg_config = PulseRateConfig('ppg')
acc_config = PulseRateConfig('imu')

print("The default window length for the signal quality feature extraction is set to", ppg_config.window_length_s, "seconds.")
print("The default step size for the signal quality feature extraction is set to", ppg_config.window_step_length_s, "seconds.")

# Remove optional arguments if you don't have accelerometer data (set to None or remove arguments)
df_features = extract_signal_quality_features(
    df_ppg=df_ppg_proc,
    ppg_config=ppg_config,
    df_acc=df_acc_proc,     # Optional
    acc_config=acc_config,  # Optional
)

df_features

```

    The default window length for the signal quality feature extraction is set to 6 seconds.
    The default step size for the signal quality feature extraction is set to 1 seconds.
    




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
      <th>acc_power_ratio</th>
      <th>var</th>
      <th>mean</th>
      <th>median</th>
      <th>kurtosis</th>
      <th>skewness</th>
      <th>signal_to_noise</th>
      <th>auto_corr</th>
      <th>f_dom</th>
      <th>rel_power</th>
      <th>spectral_entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.026409</td>
      <td>1.145652e+05</td>
      <td>282.401234</td>
      <td>238.829637</td>
      <td>2.170853</td>
      <td>0.107401</td>
      <td>3.320049</td>
      <td>0.544165</td>
      <td>0.585938</td>
      <td>0.138454</td>
      <td>0.516336</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.023402</td>
      <td>1.102401e+05</td>
      <td>271.582177</td>
      <td>236.891936</td>
      <td>2.251393</td>
      <td>-0.029309</td>
      <td>3.041878</td>
      <td>0.491829</td>
      <td>0.585938</td>
      <td>0.160433</td>
      <td>0.511626</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.028592</td>
      <td>1.061479e+05</td>
      <td>262.348604</td>
      <td>225.915756</td>
      <td>2.415221</td>
      <td>0.216631</td>
      <td>2.818552</td>
      <td>0.469092</td>
      <td>0.585938</td>
      <td>0.167007</td>
      <td>0.525025</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.019296</td>
      <td>9.514719e+04</td>
      <td>245.089445</td>
      <td>203.417715</td>
      <td>2.481465</td>
      <td>0.110420</td>
      <td>2.677071</td>
      <td>0.415071</td>
      <td>0.585938</td>
      <td>0.170626</td>
      <td>0.550495</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.020083</td>
      <td>7.393010e+04</td>
      <td>218.379138</td>
      <td>187.583266</td>
      <td>2.405921</td>
      <td>0.084566</td>
      <td>2.796140</td>
      <td>0.338369</td>
      <td>0.585938</td>
      <td>0.121113</td>
      <td>0.595214</td>
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
    </tr>
    <tr>
      <th>34329</th>
      <td>34329.0</td>
      <td>0.110219</td>
      <td>8.176078e+06</td>
      <td>1613.021494</td>
      <td>438.201240</td>
      <td>6.122772</td>
      <td>-1.792336</td>
      <td>1.378694</td>
      <td>0.104389</td>
      <td>0.351562</td>
      <td>0.046616</td>
      <td>0.356027</td>
    </tr>
    <tr>
      <th>34330</th>
      <td>34330.0</td>
      <td>0.178742</td>
      <td>3.512188e+07</td>
      <td>3307.888927</td>
      <td>1069.775894</td>
      <td>8.160698</td>
      <td>1.746472</td>
      <td>1.442643</td>
      <td>0.142226</td>
      <td>0.351562</td>
      <td>0.049424</td>
      <td>0.371163</td>
    </tr>
    <tr>
      <th>34331</th>
      <td>34331.0</td>
      <td>0.153351</td>
      <td>1.181350e+08</td>
      <td>6648.535487</td>
      <td>2743.478312</td>
      <td>5.654373</td>
      <td>0.018587</td>
      <td>1.558314</td>
      <td>0.136803</td>
      <td>0.351562</td>
      <td>0.048211</td>
      <td>0.366386</td>
    </tr>
    <tr>
      <th>34332</th>
      <td>34332.0</td>
      <td>0.154910</td>
      <td>1.252829e+09</td>
      <td>20165.525309</td>
      <td>6452.244225</td>
      <td>6.805051</td>
      <td>-1.222184</td>
      <td>1.310088</td>
      <td>0.666123</td>
      <td>0.351562</td>
      <td>0.037812</td>
      <td>0.359105</td>
    </tr>
    <tr>
      <th>34333</th>
      <td>34333.0</td>
      <td>0.093221</td>
      <td>1.008217e+10</td>
      <td>42271.328020</td>
      <td>12552.656437</td>
      <td>23.756877</td>
      <td>4.167326</td>
      <td>1.212179</td>
      <td>0.044647</td>
      <td>0.585938</td>
      <td>0.113283</td>
      <td>0.632749</td>
    </tr>
  </tbody>
</table>
<p>34334 rows × 12 columns</p>
</div>



## Step 3: Signal quality classification

A trained logistic classifier is used to predict PPG signal quality and returns the `pred_sqa_proba`, which is the posterior probability of a PPG window to look like the typical PPG morphology (higher probability indicates toward the typical PPG morphology). The (optional) relative power feature from the accelerometer is compared to a threshold for periodic artifacts and therefore `pred_sqa_acc_label` is used to return a label indicating predicted periodic motion artifacts (label 0) or no periodic motion artifacts (label 1).

The classification step is implemented in `signal_quality_classification` (documentation can be found [here](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/pipelines/pulse_rate_pipeline.py#:~:text=signal_quality_classification)).

<u>**_Note on scale sensitivity_**</u>  
The PPG sensor used for developing this pipeline records in arbitrary units. Some features are scale sensitive and require rescaling when applying the pipeline to other datasets or PPG sensors.  
In this pipeline, the logistic classifier for PPG morphology was trained on z-scored features, using the means (μ) and standard deviations (σ) from the Personalized Parkinson Project training set. These μ and σ values are stored in the `ppg_quality_classifier_package`.  
When applying the code to another dataset, users are advised to recalculate **_μ_** and **_σ** for each feature on their (training) data and update the classifier package accordingly.


```python
from importlib.resources import files
from paradigma.pipelines.pulse_rate_pipeline import signal_quality_classification

ppg_quality_classifier_package_filename = 'ppg_quality_clf_package.pkl'
full_path_to_classifier_package = files('paradigma') / 'assets' / ppg_quality_classifier_package_filename

config = PulseRateConfig()

df_sqa = signal_quality_classification(
    df=df_features, 
    config=config, 
    full_path_to_classifier_package=full_path_to_classifier_package
)

df_sqa
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
      <th>pred_sqa_proba</th>
      <th>pred_sqa_acc_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.121315e-02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>7.126135e-03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>7.017555e-03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>4.134224e-03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>9.195340e-04</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>34329</th>
      <td>34329.0</td>
      <td>1.782669e-08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34330</th>
      <td>34330.0</td>
      <td>2.078262e-06</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34331</th>
      <td>34331.0</td>
      <td>1.190223e-07</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34332</th>
      <td>34332.0</td>
      <td>1.383614e-08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34333</th>
      <td>34333.0</td>
      <td>7.587516e-07</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>34334 rows × 3 columns</p>
</div>



#### Store as TSDF
The predicted probabilities (and optionally other features) can be stored and loaded in TSDF as demonstrated below. 


```python
import tsdf
from paradigma.util import write_df_data

# Set 'path_to_data' to the directory where you want to save the data
metadata_time_store = tsdf.TSDFMetadata(metadata_time_ppg.get_plain_tsdf_dict_copy(), path_to_prepared_data)
metadata_values_store = tsdf.TSDFMetadata(metadata_values_ppg.get_plain_tsdf_dict_copy(), path_to_prepared_data)

# Select the columns to be saved 
metadata_time_store.channels = ['time']
metadata_values_store.channels = ['pred_sqa_proba', 'pred_sqa_acc_label']

# Set the units
metadata_time_store.units = ['Relative seconds']
metadata_values_store.units = ['Unitless', 'Unitless']
metadata_time_store.data_type = float
metadata_values_store.data_type = float

# Set the filenames
meta_store_filename = f'segment{segment_nr}_meta.json'
values_store_filename = meta_store_filename.replace('_meta.json', '_values.bin')
time_store_filename = meta_store_filename.replace('_meta.json', '_time.bin')

metadata_values_store.file_name = values_store_filename
metadata_time_store.file_name = time_store_filename

write_df_data(metadata_time_store, metadata_values_store, path_to_prepared_data, meta_store_filename, df_sqa)
```


```python
df_sqa, _, _ = load_tsdf_dataframe(path_to_prepared_data, prefix=f'segment{segment_nr}')
df_sqa.head()
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
      <th>pred_sqa_proba</th>
      <th>pred_sqa_acc_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.011213</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.007126</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.007018</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.004134</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.000920</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Step 4: Pulse rate estimation

For pulse rate estimation, we extract segments of `config.tfd_length` using [estimate_pulse_rate](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/pipelines/pulse_rate_pipeline.py#:~:text=estimate_pulse_rate). We calculate the smoothed-pseudo Wigner-Ville Distribution (SPWVD) to obtain the frequency content of the PPG signal over time. We extract for every timestamp in the SPWVD the frequency with the highest power. For every non-overlapping 2 s window we average the corresponding frequencies to obtain a pulse rate per window.

Note: for the test data we set the tfd_length to 10 s instead of the default of 30 s, because the small PPP test data doesn't have 30 s of consecutive high-quality PPG data.


```python
from paradigma.pipelines.pulse_rate_pipeline import estimate_pulse_rate

print("The standard default minimal window length for the pulse rate extraction is set to", config.tfd_length, "seconds.")

df_pr = estimate_pulse_rate(
    df_sqa=df_sqa, 
    df_ppg_preprocessed=df_ppg_proc, 
    config=config
)

df_pr
```

    The standard default minimal window length for the pulse rate extraction is set to 30 seconds.
    




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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47.0</td>
      <td>80.372915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.0</td>
      <td>79.769382</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51.0</td>
      <td>79.136408</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53.0</td>
      <td>78.606477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.0</td>
      <td>77.870461</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>801</th>
      <td>32876.0</td>
      <td>78.220133</td>
    </tr>
    <tr>
      <th>802</th>
      <td>32878.0</td>
      <td>78.047301</td>
    </tr>
    <tr>
      <th>803</th>
      <td>32880.0</td>
      <td>78.047301</td>
    </tr>
    <tr>
      <th>804</th>
      <td>32882.0</td>
      <td>78.238326</td>
    </tr>
    <tr>
      <th>805</th>
      <td>32884.0</td>
      <td>78.556701</td>
    </tr>
  </tbody>
</table>
<p>806 rows × 2 columns</p>
</div>



### Run steps 1 - 4 for multiple segments <a id='multiple_segments_cell'></a>

If your data is also stored in multiple segments, you can modify `segments` in the cell below to a list of the filenames of your respective segmented data.


```python
import pandas as pd
from pathlib import Path
from importlib.resources import files

from paradigma.util import load_tsdf_dataframe
from paradigma.config import PPGConfig, IMUConfig, PulseRateConfig
from paradigma.preprocessing import preprocess_ppg_data
from paradigma.pipelines.pulse_rate_pipeline import extract_signal_quality_features, signal_quality_classification, estimate_pulse_rate

# Set the path to where the prepared data is saved
path_to_prepared_data =  Path('../../example_data')

ppg_prefix = 'ppg'
imu_prefix = 'imu'

# Set the path to the classifier package
ppg_quality_classifier_package_filename = 'ppg_quality_clf_package.pkl'
full_path_to_classifier_package = files('paradigma') / 'assets' / ppg_quality_classifier_package_filename

# Create a list of dataframes to store the estimated pulse rates of all segments
list_df_pr = []

segments = ['0001', '0002'] # list with all available segments

for segment_nr in segments:
    
    # Load the data
    df_ppg, metadata_time_ppg, _ = load_tsdf_dataframe(
        path_to_data=path_to_prepared_data / ppg_prefix, 
        prefix=f'PPG_segment{segment_nr}'
    )
    df_imu, metadata_time_imu, _ = load_tsdf_dataframe(
        path_to_data=path_to_prepared_data / imu_prefix, 
        prefix=f'IMU_segment{segment_nr}'   
    )

    # Drop the gyroscope columns from the IMU data
    cols_to_drop = df_imu.filter(regex='^gyroscope_').columns
    df_acc = df_imu.drop(cols_to_drop, axis=1)

    # 1: Preprocess the data

    ppg_config = PPGConfig()
    imu_config = IMUConfig()

    df_ppg_proc, df_acc_proc = preprocess_ppg_data(
        df_ppg=df_ppg, 
        df_acc=df_acc, 
        ppg_config=ppg_config, 
        imu_config=imu_config, 
        start_time_ppg=metadata_time_ppg.start_iso8601,
        start_time_imu=metadata_time_imu.start_iso8601
    )

    # 2: Extract signal quality features
    ppg_config = PulseRateConfig('ppg')
    acc_config = PulseRateConfig('imu')

    df_features = extract_signal_quality_features(
        df_ppg=df_ppg_proc,
        df_acc=df_acc_proc,
        ppg_config=ppg_config, 
        acc_config=acc_config, 
    )
    
    # 3: Signal quality classification
    config = PulseRateConfig()

    df_sqa = signal_quality_classification(
        df=df_features, 
        config=config, 
        full_path_to_classifier_package=full_path_to_classifier_package
    )

    # 4: Estimate pulse rate
    df_pr = estimate_pulse_rate(
        df_sqa=df_sqa, 
        df_ppg_preprocessed=df_ppg_proc, 
        config=config
    )

    # Add the hr estimations of the current segment to the list
    df_pr['segment_nr'] = segment_nr
    list_df_pr.append(df_pr)

df_pr = pd.concat(list_df_pr, ignore_index=True)
```

## Step 5: Pulse rate aggregation

The final step is to aggregate all 2 s pulse rate estimates using [aggregate_pulse_rate](https://github.com/biomarkersParkinson/paradigma/blob/main/src/paradigma/pipelines/pulse_rate_pipeline.py#:~:text=aggregate_pulse_rate). In the current example, the mode and 99th percentile are calculated. We hypothesize that the mode gives representation of the resting pulse rate while the 99th percentile indicates the maximum pulse rate. In Parkinson's disease, we expect that these two measures could reflect autonomic (dys)functioning. The `nr_pr_est` in the metadata indicates based on how many 2 s windows these aggregates are determined.


```python
import pprint
from paradigma.pipelines.pulse_rate_pipeline import aggregate_pulse_rate

pr_values = df_pr['pulse_rate'].values
df_pr_agg = aggregate_pulse_rate(
    pr_values=pr_values, 
    aggregates = ['mode', '99p']
)

pprint.pprint(df_pr_agg)
```

    {'metadata': {'nr_pr_est': 806},
     'pr_aggregates': {'99p_pulse_rate': 87.65865011636926,
                       'mode_pulse_rate': 81.25613346418058}}
    
