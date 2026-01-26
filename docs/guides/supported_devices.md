# Supported Devices

This guide documents devices that have been tested with ParaDigMa and provides device-specific guidance for data loading and usage.

ParaDigMa is designed to work with wrist sensor data from any device, as long as the data meets the [Sensor Requirements](sensor_requirements.md). However, validation has only been performed for specific devices listed below.

---

## Scientifically Validated Devices

These devices have undergone rigorous scientific validation with ParaDigMa pipelines.

### Verily Study Watch

A research-grade smartwatch developed by Verily Life Sciences for clinical research.

#### Specifications

- **Sensors**: 3-axis accelerometer, 3-axis gyroscope, photoplethysmography (PPG)
- **Sampling Rates**: 100 Hz (IMU), 30 Hz (PPG)
- **Validated Pipelines**: Gait, Tremor, Pulse Rate

---

### Gait-up Physilog 4

A research-grade IMU worn as a wrist-mounted sensor band.

#### Specifications

- **Sensors**: 3-axis accelerometer, 3-axis gyroscope, 3-axis magnetometer
- **Sampling Rate**: 200 Hz
- **Validated Pipelines**: Gait, Tremor

---

## Empirically Validated Devices

These devices have been tested with ParaDigMa and show promising results, but have not undergone full scientific validation.

### Axivity AX6

A compact wrist-worn IMU designed for long-term monitoring.

#### Specifications

- **Sensors**: 3-axis accelerometer, 3-axis gyroscope
- **Sampling Rate**: Configurable (typically 100 Hz)
- **Data Format**: CWA (native)
- **Validated Pipelines**: Gait, Tremor

See the [Device-Specific Data Loading Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/device_specific_data_loading.html) for more examples.

---

### Empatica EmbracePlus

A wrist-worn research device with accelerometer sensors.

#### Specifications

- **Sensors**: 3-axis accelerometer
- **Sampling Rate**: 64 Hz (accelerometer)
- **Data Format**: AVRO (native)
- **Validated Pipelines**: None (data preparation only)


See the [Device-Specific Data Loading Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/device_specific_data_loading.html) for Empatica-specific loading examples.

#### Considerations

- **No gyroscope**: Tremor analysis not possible
- **No PPG in accelerometer file**: Pulse rate analysis requires separate PPG file
- AVRO format loading is supported but may require additional configuration
- **Not validated**: Gait pipeline results not scientifically validated on this device

---

## Device Comparison Table

| Device | Accel | Gyro | PPG | Sampling Rate (Hz) | Gait | Tremor | Pulse Rate | Validation Status |
|--------|:-----:|:----:|:---:|:------------------:|:----:|:------:|:----------:|-------------------|
| **Verily Study Watch** | ✓ | ✓ | ✓ | 100 (IMU), 30 (PPG) | ✓ | ✓ | ✓ | Scientific |
| **Gait-up Physilog 4** | ✓ | ✓ | - | 200 | ✓ | ✓ | - | Scientific |
| **Axivity AX6** | ✓ | ✓ | - | 100 | ~ | ~ | - | Empirical |
| **Axivity AX3** | ✓ | - | - | 100 | ~ | - | - | Limited |
| **Empatica EmbracePlus** | ✓ | - | - | 64 | - | - | - | Limited |

✓ = Fully supported/validated | ~ = Empirically validated | - = Not possible

---

## File Format Support

ParaDigMa can load data from multiple formats:

| Format | Extension | Device Examples | Notes |
|--------|-----------|-----------------|-------|
| **Pandas DataFrame** | (in-memory) | All | Recommended for pre-processed data |
| **TSDF** | `.meta` + `.bin` | Verily Study Watch | Standard research format |
| **Parquet** | `.parquet` | All | Efficient storage, fast loading |
| **CSV** | `.csv` | All | Universal but slower |
| **Pickle** | `.pkl`, `.pickle` | All | Python-specific |
| **CWA** | `.cwa` | Axivity AX3/AX6 | Native Axivity format |
| **AVRO** | `.avro` | Empatica | Native Empatica format |

See [Data Input Formats](input_formats.md) for detailed loading examples.

---

## Using Unsupported Devices

If your device is not listed, you can still use ParaDigMa:

### Step 1: Export to Standard Format

Convert your device's data to pandas DataFrame or a supported file format (Parquet/CSV recommended).

### Step 2: Verify Sensor Requirements

Ensure your data meets the [Sensor Requirements](sensor_requirements.md):

- Accelerometer: ≥100 Hz, ±4g range (for gait/tremor)
- Gyroscope: ≥100 Hz, ±1000 deg/s range (for tremor)
- PPG: ≥30 Hz (for pulse rate)

### Step 3: Prepare Data Format

Follow the [Data Preparation Tutorial](../tutorials/data_preparation.html) to format your DataFrame:

```python
import pandas as pd

# Example: Format your device data
df = pd.DataFrame({
    'time': your_timestamps,  # Relative seconds or absolute datetime
    'x': your_acc_x,         # Accelerometer X
    'y': your_acc_y,         # Accelerometer Y
    'z': your_acc_z,         # Accelerometer Z
    'gyro_x': your_gyro_x,   # Gyroscope X (if available)
    'gyro_y': your_gyro_y,   # Gyroscope Y (if available)
    'gyro_z': your_gyro_z,   # Gyroscope Z (if available)
})
```

### Step 4: Test with Small Dataset

```python
from paradigma.orchestrator import run_paradigma

# Test on a small sample first
results = run_paradigma(
    dfs=df.iloc[:10000],  # First 10,000 samples
    pipelines=['gait'],
    watch_side='left',
    accelerometer_units='g',
    gyroscope_units='deg/s',
    target_frequency=100.0
)
```

### Step 5: Validate Results

- Compare with known validated devices if possible
- Check for reasonable output values
- Inspect intermediate results with `save_intermediate=['quantification']`

---

## Testing with Example Data

ParaDigMa includes example data in `example_data/`:

```
example_data/
├── axivity/          # Axivity AX6 sample
├── empatica/         # Empatica EmbracePlus sample
├── gait_up_physilog/ # Physilog 4 sample
└── verily/           # Verily Study Watch sample
```

Use these to test your installation:

```python
from paradigma.orchestrator import run_paradigma
from pathlib import Path

# Get example data path
example_dir = Path('path/to/paradigma/example_data/verily/imu')

# Run on example data
results = run_paradigma(
    data_path=example_dir,
    pipelines=['gait', 'tremor'],
    watch_side='left',
    file_pattern='*.parquet'
)
```

---

## Hardware Considerations

### Wrist Placement

All validation was performed with the sensor on the **wrist** (not upper arm or other locations):

- Fitted snugly to minimize motion artifacts
- Worn on either left or right wrist
- Specify wrist side with `watch_side` parameter

### Device Orientation

ParaDigMa expects a standardized coordinate system. See [Coordinate System Guide](coordinate_system.md) for details.

If your device uses a different orientation, use the `device_orientation` parameter:

```python
results = run_paradigma(
    dfs=df,
    pipelines=['gait'],
    watch_side='left',
    device_orientation=['z', '-x', 'y']  # Example custom orientation
)
```

---

## Contributing Device Support

We welcome community contributions! If you have experience with a device not listed here:

1. Test ParaDigMa on your device
2. Document loading procedures and any device-specific considerations
3. Share validation results
4. Open a [GitHub Discussion](https://github.com/biomarkersParkinson/paradigma/discussions) or contact paradigma@radboudumc.nl

Your contribution helps expand ParaDigMa's device ecosystem!

---

## See Also

- [Sensor Requirements](sensor_requirements.md) - Technical specifications for sensors
- [Input Formats Guide](input_formats.md) - How to provide data to ParaDigMa
- [Data Preparation Tutorial](../tutorials/data_preparation.html) - Step-by-step guide
- [Coordinate System Guide](coordinate_system.md) - Orientation requirements
