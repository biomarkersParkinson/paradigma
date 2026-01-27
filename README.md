<p align="center">
  <img src="https://raw.githubusercontent.com/biomarkersParkinson/paradigma/main/docs/source/_static/img/paradigma-logo-banner.png" alt="ParaDigMa logo"/>
</p>

| Badges | |
|:----:|----|
| **Packages and Releases** | [![Latest release](https://img.shields.io/github/release/biomarkersparkinson/paradigma.svg)](https://github.com/biomarkersparkinson/paradigma/releases/latest) [![PyPI](https://img.shields.io/pypi/v/paradigma.svg)](https://pypi.python.org/pypi/paradigma/)  [![Static Badge](https://img.shields.io/badge/RSD-paradigma-lib)](https://research-software-directory.org/software/paradigma) |
| **DOI** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13838392.svg)](https://doi.org/10.5281/zenodo.13838392) |
| **Build Status** | [![](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![Build and test](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml) [![pages-build-deployment](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment) |
| **License** |  [![GitHub license](https://img.shields.io/github/license/biomarkersParkinson/paradigma)](https://github.com/biomarkersparkinson/paradigma/blob/main/LICENSE) |

## Overview

ParaDigMa (Parkinson's disease Digital Markers) is a Python toolbox for extracting validated digital biomarkers from wrist-worn sensor data in Parkinson's disease. It processes accelerometer, gyroscope, and PPG signals collected during passive monitoring in daily life.

**Key Features:**
- Arm swing during gait analysis
- Tremor analysis
- Pulse Rate analysis
- Scientifically Validated in peer-reviewed publications
- Extensible architecture for custom analyses

## Quick Start

### Installation

**For regular use:**

```bash
pip install paradigma
```

Requires Python 3.11+.

**For development or running tutorials:**

Example data requires git-lfs. See the [installation guide](https://biomarkersparkinson.github.io/paradigma/guides/installation.html) for setup instructions.

### Basic Usage

```python
from paradigma.orchestrator import run_paradigma

# Example 1: Single DataFrame with default output directory
results = run_paradigma(
    dfs=df,
    pipelines=['gait', 'tremor'],
    watch_side='left',  # Required for gait pipeline
    save_intermediate=['quantification', 'aggregation']  # Saves to ./output by default
)

# Example 2: Multiple DataFrames as list (assigned to 'df_1', 'df_2', etc.)
results = run_paradigma(
    dfs=[df1, df2, df3],
    pipelines=['gait', 'tremor'],
    output_dir="./results",  # Custom output directory
    watch_side='left',
    save_intermediate=['quantification', 'aggregation']
)

# Example 3: Dictionary of DataFrames (custom segment/file names)
results = run_paradigma(
    dfs={'morning_session': df1, 'evening_session': df2},
    pipelines=['gait', 'tremor'],
    watch_side='right',
    save_intermediate=[]  # No files saved - results only in memory
)

# Example 4: Load from data directory
results = run_paradigma(
    data_path='./my_data',
    pipelines=['gait', 'tremor'],
    watch_side='left',
    file_pattern='*.parquet',
    save_intermediate=['quantification', 'aggregation']
)

# Access results (nested by pipeline)
gait_measures = results['quantifications']['gait']
tremor_measures = results['quantifications']['tremor']
gait_aggregates = results['aggregations']['gait']
tremor_aggregates = results['aggregations']['tremor']
```

**See our [tutorials](https://biomarkersparkinson.github.io/paradigma/tutorials/index.html) for complete examples.**

## Pipelines

<p align="center">
  <img src="https://raw.githubusercontent.com/biomarkersParkinson/paradigma/main/docs/source/_static/img/pipeline-architecture.png" alt="Pipeline architeecture"/>
</p>

### Validated Processing Pipelines

ParaDigMa contains the following validated processing pipelines:
| Pipeline | Input sensors | Output week-level aggregation | Tutorial |
| ---- | ---- | ------- | ---- |
| **Arm swing during gait** | Accelerometer + Gyroscope | Typical, maximum & variability of arm swing range of motion | [Guide](https://biomarkersparkinson.github.io/paradigma/tutorials/gait_analysis) |
| **Tremor** | Gyroscope | % tremor time, typical & maximum tremor power | [Guide](https://biomarkersparkinson.github.io/paradigma/tutorials/tremor_analysis) |
| **Pulse rate** | PPG (+ Accelerometer) | Resting & maximum pulse rate | [Guide](https://biomarkersparkinson.github.io/paradigma/tutorials/pulse_rate_analysis) |

### Pipeline Architecture

ParaDigMa can best be understood by categorizing the sequential processes:
| Process | Description |
| ---- | ---- |
| **Preprocessing** | Preparing raw sensor signals for further processing |
| **Feature extraction** | Extracting features based on windowed sensor signals |
| **Classification** | Detecting segments of interest using validated classifiers (e.g., gait segments) |
| **Quantification** | Extracting specific measures from the detected segments (e.g., arm swing measures) |
| **Aggregation** | Aggregating the measures over a specific time period (e.g., week-level aggregates)

## Usage
### Documentation

- **[Tutorials](https://biomarkersparkinson.github.io/paradigma/tutorials/index.html)** - Step-by-step usage examples
- **[Installation Guide](https://biomarkersparkinson.github.io/paradigma/guides/installation.html)** - Setup and troubleshooting
- **[Sensor Requirements](https://biomarkersparkinson.github.io/paradigma/guides/sensor_requirements.html)** - Data specifications and compliance
- **[Supported Devices](https://biomarkersparkinson.github.io/paradigma/guides/supported_devices.html)** - Validated hardware
- **[Input Formats Guide](https://biomarkersparkinson.github.io/paradigma/guides/input_formats.html)** - Input format options and data loading
- **[Configuration Guide](https://biomarkersparkinson.github.io/paradigma/guides/config.html)** - Pipeline configuration
- **[Scientific Validation](https://biomarkersparkinson.github.io/paradigma/guides/validation.html)** - Validation studies and publications
- **[API Reference](https://biomarkersparkinson.github.io/paradigma/autoapi/paradigma/index.html)** - Complete API documentation

### Data Formats

ParaDigMa supports the following data formats:

- In-memory (recommend): **Pandas DataFrames** (see examples above)
- Data loading file extensions: **TSDF, Parquet, CSV, Pickle** and **several Device-specific formats** (AVRO (Empatica), CWA (Axivity))

### Troubleshooting

For installation issues, see the [installation guide troubleshooting section](https://biomarkersparkinson.github.io/paradigma/guides/installation.html#troubleshooting).

For other issues, check our [issue tracker](https://github.com/biomarkersParkinson/paradigma/issues) or contact paradigma@radboudumc.nl.

### Sensor Requirements & Supported Devices

ParaDigMa is designed for analysis of wrist-worn sensor data collected during passive monitoring in persons with Parkinson's disease.

All pipelines require data from a wrist-worn sensor with:

- **Sensor Position**: Either wrist (left or right)
- **Population**: Persons with Parkinson's disease
- **Data Quality**: Strictly increasing timestamps
- **Orientation**: Standardized coordinate system (see [Coordinate System Guide](coordinate_system.md))

ParaDigMa is designed to work on prepared sensor data of any device, but it has been emperically validated on:

- **Verily Study Watch** (gait, tremor, pulse rate)
- **Axivity AX6** (gait, tremor)
- **Gait-up Physilog 4** (gait, tremor)
- **Empatica EmbracePlus** (data loading)

See [sensor requirements guide](https://biomarkersparkinson.github.io/paradigma/guides/sensor_requirements.html) for more details on requirements; see [supported devices guide](https://biomarkersparkinson.github.io/paradigma/guides/supported_devices.html) for details on device-specific usage.

## Scientific Validation

ParaDigMa pipelines are validated in peer-reviewed publications:

| Pipeline | Publication |
|----------|-------------|
| **Arm swing during gait** | Post et al. (2025, 2026) |
| **Tremor** | Timmermans et al. (2025a, 2025b) |
| **Pulse rate** | Veldkamp et al. (2025) |

See the [validation guide](https://biomarkersparkinson.github.io/paradigma/guides/validation.html) for full publication details.

## Contributing

We welcome contributions! Please see:

- [Contributing Guidelines](https://biomarkersparkinson.github.io/paradigma/contributing.html)
- [Code of Conduct](https://biomarkersparkinson.github.io/paradigma/conduct.html)

## Citation

If you use ParaDigMa in your research, please cite:

```bibtex
@software{paradigma2024,
  author = {Post, Erik and Veldkamp, Kars and Timmermans, Nienke and
            Soriano, Diogo Coutinho and Kasalica, Vedran and
            Kok, Peter and Evers, Luc},
  title = {ParaDigMa: Parkinson's disease Digital Markers},
  year = {2024},
  doi = {10.5281/zenodo.13838392},
  url = {https://github.com/biomarkersParkinson/paradigma}
}
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

**Core Team**: Erik Post, Kars Veldkamp, Nienke Timmermans, Diogo Coutinho Soriano, Vedran Kasalica, Peter Kok, Luc Evers

**Advisors**: Max Little, Jordan Raykov, Twan van Laarhoven, Hayriye Cagnan, Bas Bloem

**Funding**: Michael J Fox Foundation (grant #020425), Dutch Research Council (grants #ASDI.2020.060, #2023.010)

## Contact

- Email: paradigma@radboudumc.nl
- [Issue Tracker](https://github.com/biomarkersParkinson/paradigma/issues)
