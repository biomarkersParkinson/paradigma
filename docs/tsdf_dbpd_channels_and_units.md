# Channels and units in Digital Biomarkers for Parkinson's Disease (`DBPD`) schemas

Within the `DBPD` project, some of the field types are further specialised to provide a better description of the data. These are described in the following sections.

## Field: `channels`
**Type:** `channel_type[]`         
**Description:** Describes the content of the data written. `channel_type` is specific to the `Digital biomarkers for PD` extension.

---

**General types**

| `channel_type` name       | Recommended `unit` | Description                                                                        
|--------------------------|--------------------|------------------------------------------------------------------------------------|
| `time`                   | `time_relative_ms`               | Time corresponding to each datapoint (also see units below)    |
| `acceleration_x`         | `m/s^2`             | Acceleration along the x-axis.                                                       |
| `acceleration_y`         | `m/s^2`             | Acceleration along the y-axis.                                                       |
| `acceleration_z`         | `m/s^2`             | Acceleration along the z-axis.                                                       |
| `rotation_x`             | `deg/s`            | Angular rotation rate around the x-axis.                                              |
| `rotation_y`             | `deg/s`            | Angular rotation rate around the y-axis.                                              |
| `rotation_z`             | `deg/s`            | Angular rotation rate around the z-axis.                                              |


---

<details markdown="1">

<summary><b>PPG-related types</b></summary>

|   `channel_type` name          | Recommended `unit`       | Description                                                                           |
|----------------------------|-----------------------|---------------------------------------------------------------------------------------|
| `ppg_quality_post_prob`     | `probability`         | `[TODO]` Posterior probability that the corresponding PPG signal is of high quality (0 to 1).   |

</details>

---

<details markdown="1">

<summary><b>Tremor-related types</b></summary>

| `channel_type` name       | Recommended `unit` | Description                                                                         |
|--------------------------|--------------------|-------------------------------------------------------------------------------------|
| `gyro_tremor_prob`       | `probability`      | Probability values (0 to 1) indicating the likelihood of tremor activity for each sample. |
| `gyro_tremor_hat`        | `boolean_num`      | Estimated values representing the presence or absence of tremor activity for each sample. |
| `gyro_arm_actv_prob`     | `probability`      | Probability values (0 to 1) indicating the likelihood of arm activity for each sample.    |
| `gyro_arm_actv_hat`      | `boolean_num`      | Estimated values representing the presence or absence of arm activity for each sample.    |
| `GyMeanDx`               | `unitless`         | Mean gyro derivative in the x axis. |
| `GyMeanDy`               | `unitless`         | Mean gyro derivative in the y axis. |
| `GyMeanDz`               | `unitless`         | Mean gyro derivative in the z axis. |
| `GyLTreDomPowerX`        | `unitless`         | Gyro Low tremor (range [3.5-8 Hz]) dominant power in the x axis. |
| `GyLTreDomPowerY`        | `unitless`         | Gyro Low tremor (range [3.5-8 Hz]) dominant power in the y axis. |
| `GyLTreDomPowerZ`        | `unitless`         | Gyro Low tremor (range [3.5-8 Hz]) dominant power in the z axis. |
| `GyGaitBandPower`        | `unitless`         | Gyro gait bandpower (range [0.4 – 2] Hz) – PSD: sum of the axes. |
| `GyGaitBandpowerRatio`   | `unitless`         | Gyro gait bandpower sum / total bandpower sum up to 15 Hz – PSD: sum of the axes. |
| `GyGaitFreqPeak`         | `unitless`         | Frequency peak of the in the gyro gait range – PSD: sum of the axes. |
| `GyGaitFixedDomPower`    | `unitless`         | `[TODO]` Gyro dominant power in a fixed range (specific frequency range not provided). |
| `GyGaitFixedDomPowerRatio` | `unitless`       | `[TODO]` Ratio of dominant power in the gyro gait range to total power. |
| `GyGaitDomPower`         | `unitless`         | `[TODO]` Dominant power in the gyro gait range. |
| `GyGaitDomPowerRatio`    | `unitless`         | `[TODO]` Ratio of dominant power in the gyro gait range to total power. |
| `GyGaitPeakFreqWidth`    | `unitless`         | `[TODO]` Width of the frequency peak in the gyro gait range. |
| `GyLTreBandPower`        | `unitless`         | `[TODO]` Low tremor bandpower (specific frequency range not provided). |
| `GyLTreBandpower`        | `unitless`         | `[TODO]` Low tremor bandpower (specific frequency range not provided). |
| `GyLTreFreqPeak`         | `unitless`         | `[TODO]` Frequency peak in the low tremor range. |
| `GyLTreFixedDomP`        | `unitless`         | `[TODO]` Low tremor dominant power in a fixed range (specific frequency range not provided). |
| `GyLTreFixedDomP`        | `unitless`         | `[TODO]` Low tremor dominant power in a fixed range (specific frequency range not provided). |
| `GyLTreDomPower`         | `unitless`         | `[TODO]` Low tremor dominant power (specific frequency range not provided). |
| `GyLTreDomPowerR`        | `unitless`         | `[TODO]` Ratio of low tremor dominant power to total power. |
| `GyLTrePeakFreqW`        | `unitless`         | `[TODO]` Width of the frequency peak in the low tremor range. |
| `GyHTreBandPower`        | `unitless`         | `[TODO]` High tremor bandpower (specific frequency range not provided). |
| `GyHTreBandpower`        | `unitless`         | `[TODO]` High tremor bandpower (specific frequency range not provided). |
| `GyHTreFreqPeak`         | `unitless`         | `[TODO]` Frequency peak in the high tremor range. |
| `GyHTreFixedDomP`        | `unitless`         | `[TODO]` High tremor dominant power in a fixed range (specific frequency range not provided). |
| `GyHTreFixedDomP`        | `unitless`         | `[TODO]` High tremor dominant power in a fixed range (specific frequency range not provided). |
| `GyHTreDomPower`         | `unitless`         | `[TODO]` High tremor dominant power (specific frequency range not provided). |
| `GyHTreDomPowerR`        | `unitless`         | `[TODO]` Ratio of high tremor dominant power to total power. |
| `GyHTrePeakFreqW`        | `unitless`         | `[TODO]` Width of the frequency peak in the high tremor range. |
| `GyMFCC1`                | `unitless`         | `[TODO]` Mel-frequency cepstral coefficient 1. |
| `GyMFCC2`                | `unitless`         | `[TODO]` Mel-frequency cepstral coefficient 2. |
| `GyMFCC3`                | `unitless`         | `[TODO]` Mel-frequency cepstral coefficient 3. |
| `GyMFCC4`                | `unitless`         | `[TODO]` Mel-frequency cepstral coefficient 4. |
| `GyMFCC5`                | `unitless`         | `[TODO]` Mel-frequency cepstral coefficient 5. |
| `GyMFCC6`                | `unitless`         | `[TODO]` Mel-frequency cepstral coefficient 6. |
| `GyMFCC7`                | `unitless`         | `[TODO]` Mel-frequency cepstral coefficient 7. |
| `GyMFCC8`                | `unitless`         | `[TODO]` Mel-frequency cepstral coefficient 8. |
| `GyMFCC9`                | `unitless`         | `[TODO]` Mel-frequency cepstral coefficient 9. |


</details>

---

<details markdown="1">

<summary><b>Gait-related types</b></summary>

| `channel_type` name       | Recommended `unit` | Description                                                                         |
|--------------------------|--------------------|-------------------------------------------------------------------------------------|
| `std_accel_norm`       | `m/s^2`            | Standard deviation of the norm of the accelerometer axes in the temporal domain.     |
| `x_accel_grav_mean`    | `m/s^2`            | Mean of the x-axis acceleration gravity component.     |
| `y_accel_grav_mean`    | `m/s^2`            | Mean of the y-axis acceleration gravity component.  |
| `z_accel_grav_mean`    | `m/s^2`            | Mean of the z-axis acceleration gravity component. |
| `x_accel_grav_std`     | `m/s^2`            | Standard deviation of the x-axis acceleration gravity component. |
| `y_accel_grav_std`     | `m/s^2`            | Standard deviation of the y-axis acceleration gravity component. |
| `z_accel_grav_std`     | `m/s^2`            | Standard deviation of the z-axis acceleration gravity component.. |
| `x_accel_power_below_gait`  | `(m/s^2)^2/Hz`            | Total power in the [0, 0.7] Hz range of the x-axis accelerometer. |
| `y_accel_power_below_gait`  | `(m/s^2)^2/Hz`            | Total power in the [0, 0.7] Hz range of the y-axis accelerometer. |
| `z_accel_power_below_gait`  | `(m/s^2)^2/Hz`            | Total power in the [0, 0.7] Hz range of the z-axis accelerometer. |
| `x_accel_power_gait`  | `(m/s^2)^2/Hz`            | Total power in the [0.7, 3.5] Hz range of the x-axis accelerometer. |
| `y_accel_power_gait`  | `(m/s^2)^2/Hz`            | Total power in the [0.7, 3.5] Hz range of the y-axis accelerometer. |
| `z_accel_power_gait`  | `(m/s^2)^2/Hz`            | Total power in the [0.7, 3.5] Hz range of the z-axis accelerometer. |
| `x_accel_power_tremor`  | `(m/s^2)^2/Hz`            | Total power in the [3.5, 8] Hz range of the x-axis accelerometer. |
| `y_accel_power_tremor`  | `(m/s^2)^2/Hz`            | Total power in the [3.5, 8] Hz range of the y-axis accelerometer. |
| `z_accel_power_tremor`  | `(m/s^2)^2/Hz`            | Total power in the [3.5, 8] Hz range of the z-axis accelerometer. |
| `x_accel_power_above_tremor`  | `(m/s^2)^2/Hz`            | Total power in the [8, 50] Hz range of the x-axis accelerometer. |
| `y_accel_power_above_tremor`  | `(m/s^2)^2/Hz`            | Total power in the [8, 50] Hz range of the y-axis accelerometer. |
| `z_accel_power_above_tremor`  | `(m/s^2)^2/Hz`            | Total power in the [8, 50] Hz range of the z-axis accelerometer. |
| `x_accel_dominant_frequency`  | `Hz`            | Dominant frequency of the x-axis accelerometer. |
| `y_accel_dominant_frequency`  | `Hz`            | Dominant frequency of the x-axis accelerometer. |
| `z_accel_dominant_frequency`  | `Hz`            | Dominant frequency of the x-axis accelerometer. |
| `accel_norm_cc_{n}`  | `?`            | Cepstral coefficient n with n $\in$ [1,2,...,16] of the accelerometer. |
| `gd_pred_gait_proba`  | `probability`            | Predicted probability of gait being the predominant activity within the window span. |
| `gyro_norm_cc_{n}` | `?` | Cepstral coefficient n with n $\in$ [1,2,...,16] of the gyroscope. |
| `x_gyro_dominant_frequency`  | `Hz`            | Dominant frequency of the x-axis gyroscope |
| `y_gyro_dominant_frequency`  | `Hz`            | Dominant frequency of the x-axis gyroscope |
| `z_gyro_dominant_frequency`  | `Hz`            | Dominant frequency of the x-axis gyroscope |
| `angle_mean_amplitude` | `deg` | Mean of the sum of consecutive minima and maxima angles (angle amplitude is often referred to as range of motion) | 
| `angle_std_amplitude` | `deg` | Std of the sum of consecutive minima and maxima angles |
| `angle_sum_amplitude` | `deg` | Sum of the sum of consecutive minima and maxima angles |
| `ange_perc_95_amplitude` | `deg` | 95th percentile of the sum of consecutive minima and maxima angles | 
| `forward_peak_ang_vel_mean` | `deg/s` | Angular velocity mean in forward direction of the first principal component |
| `forward_peak_ang_vel_std` | `deg/s` | Angular velocity standard deviation in forward direction of the first principal component |
| `backward_peak_ang_vel_mean` | `deg/s` | Angular velocity mean in backward direction of the first principal component | 
| `backward_peak_ang_vel_std` | `deg/s` | Angular velocity standard deviation in backward direction of the first principal component |
| `angle_perc_power` | `percentage` | Percentage of total power in the arm swing frequency band [0.3 - 3 Hz] |
</details>

---

## Field: `units`

**Type:** `unit_type[]`

**Description:** Describes the format of the data written. `unit_type` is specific to the `Digital biomarkers for PD` extension.

| `unit_type`     | Description                                                                                         |
|-----------------|-----------------------------------------------------------------------------------------------------|
| `time_relative_ms`  | Time in milliseconds, relative to the `start_iso8601`.                                                          |
| `time_absolute_unix_s`  | Absolute time in seconds, relative to unix epoch.                                                                 |
| `time_absolute_unix_ms`  | [TODO] Absolute time in milliseconds, relative to unix epoch.                                                                 |
| `probability`   | Probability values (0 to 1) indicating the likelihood of tremor activity for each sample.           |
| `boolean_num`   | `[TODO]` Integer values (0 or 1) representing the true (1) or false (0) presence of an activity.    |
| `unitless`      | Numerical values without units.                                                                     |
| `m/s^2`         | Acceleration in meters per second squared.                                                          |
| `deg/s`         | Angular velocity in degrees per second.                                                             |
