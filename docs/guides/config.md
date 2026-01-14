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
imu_config.sampling_frequency = 100
```

**Parameters**
-
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
from paradigma.pipelines import run_paradigma

imu_config = IMUConfig()
imu_config.sampling_frequency = 100
gait_config = GaitConfig()

results = run_paradigma(
    dfs={'data': df},
    pipeline_names=['gait'],
    imu_config=imu_config,
    gait_config=gait_config
)
```

## Best Practices

1. **Validation**: Ensure your sensor `sampling_frequency` matches your actual data
2. **Column Names**: Verify that your DataFrame column names match the configuration
3. **Units**: Confirm that sensor data is in correct physical units (see [Sensor Requirements](https://biomarkersparkinson.github.io/paradigma/guides/sensor_requirements.html))
4. **Documentation**: Document any custom configurations in your analysis code

## See Also

- [Sensor Requirements](https://biomarkersparkinson.github.io/paradigma/guides/sensor_requirements.html) - Detailed sensor specifications
- [Coordinate System Guide](https://biomarkersparkinson.github.io/paradigma/guides/coordinate_system.html) - IMU axis alignment
- [API Reference](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/config.html) - Complete configuration API
- [Data Preparation Tutorial](https://biomarkersparkinson.github.io/paradigma/tutorials/data_preparation.html) - Data preparation steps
