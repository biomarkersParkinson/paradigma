# Sensor Data Requirements

ParaDigMa is designed for analysis of wrist-worn sensor data collected during passive monitoring in persons with Parkinson's disease. This guide specifies the sensor and data requirements for each pipeline.

## General Requirements

All pipelines require data from a wrist-worn sensor with the following characteristics:

- **Sensor Position**: Either wrist
- **Data Format**: Contiguous, strictly increasing timestamps (see [Data Preparation](https://biomarkersparkinson.github.io/paradigma/tutorials/data_preparation.html))
- **Orientation**: Proper coordinate system alignment (see [Coordinate System Guide](https://biomarkersparkinson.github.io/paradigma/guides/coordinate_system.html))
- **Population**: Persons with Parkinson's disease

## Arm Swing during Gait

### Sensor Specifications

| Specification | Minimum Requirement |
|---------------|-------------------|
| **Accelerometer** | Sampling rate ≥ 100 Hz, Range ≥ ± 4 g |
| **Gyroscope** | Sampling rate ≥ 100 Hz, Range ≥ ± 1000 degrees/sec |

### Data Compliance

For reliable weekly measures:

- **Minimum 3 compliant days** with ≥10 hours of data between 8 am and 10 pm
- **At least 2 minutes** of arm swing activity per week
- **Population**: No walking aid, no severe dyskinesia in the watch-sided arm

### Physical Units

- Accelerometer: **g** (gravitational force)
- Gyroscope: **deg/s** (degrees per second)

## Tremor

### Sensor Specifications

| Specification | Minimum Requirement |
|---------------|-------------------|
| **Gyroscope** | Sampling rate ≥ 100 Hz, Range ≥ ± 1000 degrees/sec |

### Data Compliance

For reliable weekly measures:

- **Minimum 3 compliant days** with ≥10 hours of data between 8 am and 10 pm

### Physical Units

- Gyroscope: **deg/s** (degrees per second)

## Pulse Rate

### Sensor Specifications

| Specification | Minimum Requirement |
|---------------|-------------------|
| **PPG (Photoplethysmography)** | Sampling rate ≥ 30 Hz, Green LED |
| **Accelerometer** | Sampling rate ≥ 100 Hz, Range ≥ ± 4 g |

### Data Compliance

For reliable weekly measures:

- **Minimum 12 hours** of data per day on average
- **Population**: No rhythm disorders (e.g., atrial fibrillation, atrial flutter)

### Physical Units

- Accelerometer: **g** (gravitational force)
- PPG: Blood volume pulse in **arbitrary units** (device-dependent)

> **Note**: The PPG processing pipeline is currently validated with the blood volume pulse (BVP) signal from the Verily Study Watch. To use PPG data from other devices, see the [Pulse Rate Analysis Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/_static/pulse_rate_analysis.html#step-3-signal-quality-classification) for guidance on signal adaptation.

## Validated Devices

ParaDigMa has been empirically validated on:

| Device | Arm Swing | Tremor | Pulse Rate |
|--------|:---------:|:------:|:----------:|
| **Gait-up Physilog 4** | ✓ | ✓ | - |
| **Verily Study Watch** | ✓ | ✓ | ✓ |

See [Supported Devices](https://biomarkersparkinson.github.io/paradigma/guides/supported_devices.html) for more details on device-specific loading and considerations.

## Important Notes

> [!WARNING]
> While ParaDigMa is designed to work on any wrist sensor device meeting the requirements above, performance has only been empirically validated on the Gait-up Physilog 4 and Verily Study Watch.

> [!NOTE]
> The specifications above represent the **minimally validated requirements**. For example, while ParaDigMa works with accelerometer and gyroscope data sampled at 50 Hz, the effect on subsequent processing has not been empirically validated. Higher sampling rates and ranges generally improve accuracy.

## Data Preparation

Raw sensor data must be converted to standardized pandas DataFrames before pipeline processing. See the [Device-specific Data Loading](https://biomarkersparkinson.github.io/paradigma/tutorials/device_specific_data_loading.html) and [Data Preparation](https://biomarkersparkinson.github.io/paradigma/tutorials/data_preparation.html) tutorials for:

- Loading data from various file formats (TSDF, Parquet, CSV, etc.)
- Device-specific loading (Empatica, Axivity)
- Unit conversions and coordinate system alignment
- Resampling and timestamp handling

## Contact

For questions about sensor requirements or device-specific issues:

* paradigma@radboudumc.nl
* [GitHub Issues](https://github.com/biomarkersParkinson/paradigma/issues)
