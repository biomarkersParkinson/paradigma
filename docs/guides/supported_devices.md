# Supported Devices

This guide documents the devices that have been tested with ParaDigMa and provides device-specific loading and usage guidance.

## Validated Devices

### Gait-up Physilog 4

**Validation Status**: Arm Swing & Tremor ✓

A research-grade inertial measurement unit (IMU) worn as a wrist-mounted sensor band.

- **Sensors**: 3-axis accelerometer, 3-axis gyroscope, 3-axis magnetometer
- **Sampling Rate**: 200 Hz

#### Validation Details

- **Arm Swing**: Validated on Parkinson@Home dataset
- **Tremor**: Validated on Parkinson@Home dataset
- **Known Limitations**: No PPG sensor on this device

### Verily Study Watch

**Validation Status**: Arm Swing, Tremor & Pulse Rate ✓

A research-grade smartwatch developed by Verily Life Sciences for clinical research.

- **Sensors**: 3-axis accelerometer, 3-axis gyroscope, photoplethysmography (PPG)
- **Sampling Rate**: 100 Hz (IMU), 30 Hz (PPG)

#### Validation Details

- **Arm Swing**: Validated on PPP dataset
- **Tremor**: Validated on PPP dataset
- **Pulse Rate**: Validated on PPP dataset

## Commonly Used Devices with Community Support

### Empatica E4

**Validation Status**: Community support (not formally validated by ParaDigMa team)

A wrist-worn research device with PPG and accelerometer sensors.

- **Sensors**: 3-axis accelerometer, PPG
- **Data Format**: AVRO format (native)
- **Supported by ParaDigMa**: Partial (data loading supported; not validated)

#### Loading Empatica Data

See the [Device-Specific Data Loading Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/device_specific_data_loading.html) for Empatica E4 data loading examples.

#### Known Considerations

- Empatica E4 lacks gyroscope data; tremor and gait analysis require external gyroscope
- Community examples available but not validated by ParaDigMa team

### Axivity AX3/6

**Validation Status**: Community support (not formally validated by ParaDigMa team)

A compact wrist-worn IMU designed for long-term monitoring.

- **Sensors**: 3-axis accelerometer (AX3), or 3-axis accelerometer and gyroscope (AX6)
- **Data Format**: CWA format (native)
- **Supported by ParaDigMa**: Partial (data loading supported; not validated)

#### Loading Axivity Data

See the [Device-Specific Data Loading Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/device_specific_data_loading.html) for Axivity AX3 data loading examples.

#### Known Considerations

- Axivity AX3 lacks gyroscope and PPG data: tremor and pulse rate analysis not possible
- Axivity AX6 lacks PPG data: pulse rate analysis not possible

## Using Data from Unsupported Devices

If your device is not listed above, you can still use ParaDigMa by:

1. **Export to Standard Format**: Convert your device's proprietary format to pandas DataFrame or a supported file format (Parquet, CSV)
2. **Ensure Data Compliance**: Verify your sensor data meets the [Sensor Requirements](https://biomarkersparkinson.github.io/paradigma/guides/sensor_requirements.html)
3. **Data Preparation**: Follow the [Data Preparation Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/data_preparation.html) to standardize your data
4. **Pipeline Testing**: Test with a small dataset first to validate results

## Data Format Support

ParaDigMa can load data from multiple formats:

| Format | Extension | Status |
|--------|-----------|--------|
| **Pandas DataFrame** | (in-memory) | ✓ Recommended |
| **TSDF** | `.meta` + `.bin` | ✓ Supported |
| **Parquet** | `.parquet` | ✓ Supported |
| **CSV** | `.csv` | ✓ Supported |
| **Pickle** | `.pkl` / `.pickle` | ✓ Supported |
| **AVRO** | `.avro` | ✓ Supported |
| **CWA** | `.cwa` | ✓ Supported |

See [Data Preparation Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/data_preparation.html) for examples of loading each format.

## Testing Example Data

ParaDigMa includes example data from validated devices in the `example_data/` folder:

```
example_data/
├── axivity/
├── empatica/
└── verily/
```

Use these examples to test your ParaDigMa installation and understand expected data formats.

## Hardware Considerations

### Wrist Placement

All validation was performed with the sensor on the wrist (not upper arm or other locations). The sensor should be:

- Fitted snugly to minimize motion artifacts
- Worn on either wrist (left or right)

## Contributing Device Support

If you have experience with a device not listed here or want to contribute validation results, please:

1. Open a [GitHub Discussion](https://github.com/biomarkersParkinson/paradigma/discussions)
2. Share data loading code and validation results
3. Contact: paradigma@radboudumc.nl

We welcome community contributions to expand device support!
