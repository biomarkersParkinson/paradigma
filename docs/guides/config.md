# Configuration Guide

Throughout the ParaDigMa toolbox, configuration objects are used to specify parameters for pipeline processes. All configuration classes are defined in `config.py` and can be imported using `from paradigma.config import X`.

Configuration classes use static column names defined in `constants.py` to ensure robustness and consistency across the codebase.

## Overview

Configuration classes are organized into two categories:

- **Sensor Configurations**: For sensors (IMU, PPG)
- **Domain Configurations**: For analysis pipelines (gait, tremor, pulse rate)

## Sensor Configurations

### IMUConfig

Configuration for inertial measurement unit (IMU) sensors (accelerometer + gyroscope):

```python
from paradigma.config import IMUConfig

imu_config = IMUConfig()
```

#### Adaptive Frequency Handling

The `sampling_frequency` is automatically detected from your data during preprocessing and cannot be manually set. This ensures parameters like filter cutoffs and window sizes always match your actual data. However, you are free to manually change the input data, thereby affecting the automated sampling frequency detection.

```python
from paradigma.preprocessing import preprocess_imu_data

imu_config = IMUConfig()
df_preprocessed = preprocess_imu_data(df_raw, imu_config)

# After preprocessing, sampling_frequency is auto-detected and available:
print(f"Detected sampling frequency: {imu_config.sampling_frequency} Hz")

# To use a different resampling frequency (optional, not recommended):
imu_config.resampling_frequency = 100
```

**Key Behaviors:**
- `sampling_frequency` is **read-only** and auto-detected during preprocessing
- Frequency-dependent parameters (filter cutoffs, tolerance) are automatically calculated
- `resampling_frequency` defaults to `None` (means "use detected sampling_frequency" for uniform sampling)
- Set `resampling_frequency` explicitly only if you need to upsample/downsample to a specific rate

### PPGConfig

Configuration for photoplethysmography (PPG) sensors:

```python
from paradigma.config import PPGConfig

ppg_config = PPGConfig()
```

## Domain Configurations

Domain configurations are defined for each analysis pipeline and correspond to processing steps:

1. **Preprocessing**: Raw signal preparation
2. **Feature Extraction**: Window-based feature computation
3. **Classification**: Segment detection (e.g., gait segments)
4. **Quantification**: Measure extraction from segments
5. **Aggregation**: Time-period aggregation (e.g., weekly)

### Using Domain Configs

Each domain (gait, tremor, pulse rate) has configuration classes for its specific processing steps. See the [API Reference](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/index.html) for complete documentation of available configurations.

Example with gait analysis:

```python
from paradigma.config import IMUConfig, GaitConfig
from paradigma.orchestrator import run_paradigma

imu_config = IMUConfig()  # sampling_frequency will be auto-detected
gait_config = GaitConfig()

results = run_paradigma(
    dfs={'data': df},
    pipelines=['gait'],
    watch_side='left',
    imu_config=imu_config,
    gait_config=gait_config
)
```

## Best Practices

1. **Frequency Detection**: `sampling_frequency` is automatically detected from your data during preprocessing. No manual configuration needed.
2. **Resampling** (Optional): By default, data is uniformly sampled at its detected frequency. Only set `resampling_frequency` if you specifically need to upsample or downsample.
3. **Column Names**: Verify that your DataFrame column names match the configuration
4. **Units**: Confirm that sensor data is in correct physical units (see [Sensor Requirements](https://biomarkersparkinson.github.io/paradigma/guides/sensor_requirements.html))
5. **Documentation**: Document any custom configurations (e.g., custom resampling rates) in your analysis code

## See Also

- [Sensor Requirements](https://biomarkersparkinson.github.io/paradigma/guides/sensor_requirements.html) - Detailed sensor specifications
- [Coordinate System Guide](https://biomarkersparkinson.github.io/paradigma/guides/coordinate_system.html) - IMU axis alignment
- [API Reference](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/config.html) - Complete configuration API
- [Data Preparation Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/data_preparation.html) - Data preparation steps
