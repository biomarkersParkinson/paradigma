
# TSDF fields in Digital Biomarkers for Parkinson's Disease (`DBPD`) schemas

## Mandatory fields

This is a preliminary list of mandatory fields (to be shaped into schemas) that are used in the `DBPD` project. The list will be updated based on the upcoming discussions.

| Field                      | Type         | Description                                                                       |
|----------------------------|--------------|-----------------------------------------------------------------------------------|
| `window_size_sec`          | `float`               | Size of the window (in seconds) used in the analysis.                             |
| `step_size_sec`            | `float`               | Duration in seconds for each segment in the written data.                         |
| `freq_sampling`            | `int`                 | Sampling frequency (in Hz) of the input data.                                                   |
| `channels`         | [channel_type](tsdf_field_types.md)`[]`         | Description of the content of the data written. `channel_type` is specific to the `Digital biomarkers for PD` extension. |
| `units`         | [unit_type](tsdf_field_types.md)`[]`         | Description of the format of the data written. `unit_type` is specific to the `Digital biomarkers for PD` extension. |



## **Tremor** pipeline specific fields

Non-mandatory fields used in the tremor pipeline.

| Field                      | Type         | Description                                                                  |
|----------------------------|--------------|------------------------------------------------------------------------------|
| `mfcc_num_filters`         | `int`      | Number of filters used for estimating the mel-frequency cepstral coefficients. |
| `mfcc_num_mel_coeff`       | `int`      | Number of coefficients used for estimating the mel-frequency cepstral coefficients. |
| `mfcc_max_freq_filter`     | `float`      | Maximum frequency (in Hz) used for filtering in mel-frequency cepstral coefficients. |
| `mfcc_window_size`         | `float`      | Size of the sub-window in seconds used to estimate the spectrogram used in the evaluation of the mel-frequency cepstral coefficients. |
| `excluded_hours`            | `int[]`                  | `[TODO]` List of the excluded hours from the analysis (vector scaling?)                                                  |
| `sum_features_gyro_scale`   | `float[]`                  | `[TODO]` Scaling factors for the sum of tremor-related features (from gyro)                                          |
| `sum_squared_features_gyro_scale` | `float[]`            | `[TODO]` Scaling factors for the sum of squared tremor-related features (from gyro)                                 |
| `n_features_gyro_scale`     | `int`                    | `[TODO]` Scaling factor for the number of gyro features                                        |

## **PPG** pipeline specific fields

Non-mandatory fields used in the PPG pipeline.

| Field                      | Type                 | Description                                                                  |
|----------------------------|----------------------|------------------------------------------------------------------------------|
| `segment_number`           | `int`                | Order number of the analyzed data segment. |
| `freq_sampling_original`   | `int`                | Sampling frequency (in Hz) of the original data (before adjustments for the analysis).                                              |

## **Gait** pipeline specific fields

Non-mandatory fields used in the gait pipeline. We currently do not have information about the fields used in the gait pipeline, but we will update this section as soon as we have more information.

| Field                      | Type                 | Description                                                                  |
|----------------------------|----------------------|------------------------------------------------------------------------------|
| `side_watch`               | `string`             | `[TODO]` Possible values: ['left', 'right']. |

## Additional generic fields

These fields are non-mandatory, and provide standardised vocabulary for describing the data.


| Field                        | Type         | Description                                          |
|------------------------------|--------------|------------------------------------------------------|
| `week_number`                | `int`        | Denotes the specific study week number used for tracking or comparing data. |
| `columns`        | `int`        | Number of columns in the data matrix.                                        |
| `interpolated`               | `bool`       | Indicates whether interpolation was performed on the data. |
| `high_pass_filter_applied`   | `bool`       | Indicates whether a high-pass filter was applied to remove low-frequency noise. |
| `high_pass_filter_cutoff`    | `float`      | Cutoff frequency (in Hz) for the high-pass filter, in case it was applied. |
| `z_score_normalised`         | `bool`       | Indicates whether z-score normalization was applied to the data. |
| `start_datetime_unix_ms`     | `string`     | UNIX timestamp for the start of the recording (milliseconds). Equivalent to `start_iso8601` in UNIX format.   |
| `end_datetime_unix_ms`       | `string`     | UNIX timestamp for the end of the recording (milliseconds). Equivalent to `end_iso8601` in UNIX format.   | 


