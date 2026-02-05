# Sensor Data Requirements

ParaDigMa is designed for analysis of wrist-worn sensor data collected during passive monitoring in persons with Parkinson's disease. This guide specifies the sensor and data requirements for each pipeline.

## General Requirements

All pipelines require data from a wrist-worn sensor with:

- **Sensor Position**: Either wrist (left or right)
- **Population**: Persons with Parkinson's disease
- **Data Quality**: Strictly increasing timestamps
- **Orientation**: Standardized coordinate system (see [Coordinate System Guide](coordinate_system.md))

## Pipeline-Specific Requirements

### Arm Swing during Gait

#### Sensor Specifications

| Specification | Minimum Requirement |
|---------------|-------------------|
| **Accelerometer** | Sampling rate ≥ 100 Hz<br>Range ≥ ± 4 g |
| **Gyroscope** | Sampling rate ≥ 100 Hz<br>Range ≥ ± 1000 degrees/sec |

#### Physical Units

- **Accelerometer**: `g` (gravitational force)
- **Gyroscope**: `deg/s` (degrees per second)

> ParaDigMa has functionalities for converting [accelerometer](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/util/index.html#paradigma.util.convert_units_accelerometer) and [gyroscope](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/util/index.html#paradigma.util.convert_units_gyroscope) in other units (e.g., `m/s^2`, `rad/s`) to these standard units. This can also be setup when using `run_paradigma`.

#### Data Compliance

For reliable weekly measures:

- **Minimum 3 compliant days** with ≥10 hours of data between 8 am and 10 pm
- **At least 2 minutes** of arm swing activity per week

#### Population Constraints

- No walking aid usage
- No severe dyskinesia in the watch-sided arm

#### Required Parameters

- `watch_side`: Must specify `'left'` or `'right'`

---

### Tremor

#### Sensor Specifications

| Specification | Minimum Requirement |
|---------------|-------------------|
| **Gyroscope** | Sampling rate ≥ 100 Hz<br>Range ≥ ± 1000 degrees/sec |

#### Physical Units

- **Gyroscope**: `deg/s` (degrees per second)

> ParaDigMa has functionalities for converting [gyroscope](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/util/index.html#paradigma.util.convert_units_gyroscope) in other units (e.g., `rad/s`) to these standard units. This can also be setup when using `run_paradigma`.

#### Data Compliance

For reliable weekly measures:

- **Minimum 3 compliant days** with ≥10 hours of data between 8 am and 10 pm

---

### Pulse Rate

#### Sensor Specifications

| Specification | Minimum Requirement |
|---------------|-------------------|
| **PPG (Photoplethysmography)** | Sampling rate ≥ 30 Hz<br>Green LED wavelength |
| **Accelerometer** (optional) | Sampling rate ≥ 100 Hz<br>Range ≥ ± 4 g<br>*(for motion artifact detection)* |

#### Physical Units

- **PPG**: Blood volume pulse in arbitrary units (device-dependent)
- **Accelerometer**: `g` (gravitational force) - if used for motion detection

#### Data Compliance

For reliable weekly measures:

- **Minimum 12 hours** of data per day on average

#### Population Constraints

- No cardiac rhythm disorders (e.g., atrial fibrillation, atrial flutter)

#### Important Notes

> **Device-Specific Adaptation**: The PPG processing pipeline is currently validated with the blood volume pulse (BVP) signal from the Verily Study Watch. To use PPG data from other devices, see the [Pulse Rate Analysis Tutorial](../tutorials/_static/pulse_rate_analysis.html#step-3-signal-quality-classification) for signal adaptation guidance.

> **Cannot Combine with Other Pipelines**: The pulse rate pipeline requires different sensor data (PPG) than gait/tremor (IMU) and must be run separately.

---

## Multi-Pipeline Processing

ParaDigMa supports running compatible pipelines together:

```python
from paradigma.orchestrator import run_paradigma

# Gait and tremor can run together (both use IMU data)
results = run_paradigma(
    dfs=df,
    pipelines=['gait', 'tremor'],
    watch_side='left'
)
```

### Pipeline Compatibility Matrix

| Pipeline Combination | Compatible | Reason |
|---------------------|:----------:|---------|
| Gait + Tremor | ✓ | Both use IMU sensors (accelerometer + gyroscope) |
| Gait + Pulse Rate | ✗ | Different sensor types (IMU vs PPG) |
| Tremor + Pulse Rate | ✗ | Different sensor types (IMU vs PPG) |
| Gait + Tremor + Pulse Rate | ✗ | Cannot mix IMU and PPG pipelines |

---

## Important Notes on Validation

> [!NOTE]
> The specifications above represent **minimally validated requirements**. For example, while ParaDigMa works with accelerometer and gyroscope data sampled at 50 Hz, the effect on processing accuracy has not been empirically validated. **Higher sampling rates and sensor ranges generally improve accuracy.**

---
