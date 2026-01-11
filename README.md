<p align="center">
  <img src="https://raw.githubusercontent.com/biomarkersParkinson/paradigma/main/docs/source/_static/img/paradigma-logo-banner.png" alt="ParaDigMa logo"/>
</p>

| Quick Links | [![Documentation](https://img.shields.io/badge/Docs-Paradigma-blue)](https://biomarkersparkinson.github.io/paradigma/) [![Tutorials](https://img.shields.io/badge/Tutorials-ParaDigMa-blue)](https://biomarkersparkinson.github.io/paradigma/tutorials/getting_started.html) |
|:----:|----|
| **Packages and Releases** | [![Latest release](https://img.shields.io/github/release/biomarkersparkinson/paradigma.svg)](https://github.com/biomarkersparkinson/paradigma/releases/latest) [![PyPI](https://img.shields.io/pypi/v/paradigma.svg)](https://pypi.python.org/pypi/paradigma/)  [![Static Badge](https://img.shields.io/badge/RSD-paradigma-lib)](https://research-software-directory.org/software/paradigma) |
| **DOI** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13838392.svg)](https://doi.org/10.5281/zenodo.13838392) |
| **Build Status** | [![](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![Build and test](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml) [![pages-build-deployment](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment) |
| **License** |  [![GitHub license](https://img.shields.io/github/license/biomarkersParkinson/paradigma)](https://github.com/biomarkersparkinson/paradigma/blob/main/LICENSE) |

## Overview
The Parkinson's disease Digital Markers (ParaDigMa) toolbox is a Python
software package designed for processing real-life wrist sensor data
to extract digital measures of motor and non-motor signs of Parkinson's disease (PD).

Specifically, the toolbox is designed to process accelerometer, gyroscope and
photoplethysmography (PPG) signals, collected during passive monitoring in daily life.
It contains three data processing pipelines: (1) arm swing during gait, (2) tremor,
and (3) pulse rate. These pipelines are scientifically validated for their
use in persons with PD. Furthermore, the toolbox contains general functionalities for
signal processing and feature extraction, such as filtering, peak detection, and
spectral analysis.

The toolbox is accompanied by a set of [example scripts and notebooks](tutorials/index)
for each processing pipeline that demonstrate how to use the toolbox for extracting
digital measures. In addition, the toolbox is designed to be modular, enabling
researchers to easily extend the toolbox with new algorithms and functionalities.

## Features
The components of ParaDigMa are shown in the diagram below.

<p align="center">
  <img src="https://raw.githubusercontent.com/biomarkersParkinson/paradigma/main/docs/source/_static/img/pipeline-architecture.png" alt="Pipeline architeecture"/>
</p>
The three colored, shaded columns represent the individual pipelines. Processes of the pipelines are represented by blue ellipses, and input/output data by rectangular boxes. The input/output of each step is indicated by yellow horizontal bars denoting the type of data (e.g., 3. Extracted features). Arrows indicate the sequential order of the processes of the pipeline. <br/> <br/>
ParaDigMa can best be understood by categorizing the sequential processes:

| Process | Description |
| ---- | ---- |
| Preprocessing | Preparing raw sensor signals for further processing |
| Feature extraction | Extracting features based on windowed sensor signals |
| Classification | Detecting segments of interest using validated classifiers (e.g., gait segments) |
| Quantification | Extracting specific measures from the detected segments (e.g., arm swing measures) |
| Aggregation | Aggregating the measures over a specific time period (e.g., week-level aggregates) |

<br/>
ParaDigMa contains the following validated processing pipelines (each using the processes described above):

| Pipeline | Input | Output classification | Output quantification | Output week-level aggregation |
| ---- | ---- | ---- | ---- | ---- |
| **Arm swing during gait** | Wrist accelerometer and gyroscope data | Gait probability, gait without other arm activities probability | Arm swing range of motion (RoM) | Typical & maximum arm swing RoM |
| **Tremor** | Wrist gyroscope data | Tremor probability | Tremor power | % tremor time, typical & maximum tremor power |
| **Pulse rate** | Wrist PPG and accelerometer data | PPG signal quality | Pulse rate | Resting & maximum pulse rate |

## Installation

The package is available in PyPI and requires [Python 3.11](https://www.python.org/downloads/) or higher. It can be installed using:

```bash
pip install paradigma
```

> **Note:** ParaDigMa uses Git Large File Storage (git-lfs) for storing some tutorial data.
Ensure git-lfs is installed before cloning the repository:

```bash
# Install git-lfs (example for Linux/macOS)
brew install git-lfs   # macOS
sudo apt install git-lfs  # Ubuntu/Debian

# Initialize git-lfs after cloning
git lfs install
git lfs pull
```

## Usage

### Tutorials & documentation
The tutorials include Jupyter notebooks demonstrating each processing pipeline. Before running the
notebooks, see the [Getting started with the ParaDigMa tutorials](https://biomarkersparkinson.github.io/paradigma/tutorials/getting_started.html)
guide for instructions on configuring the correct Python environment and Jupyter kernel.
The API reference contains detailed documentation of all toolbox modules and functions.
The user guides provide additional information about specific topics (e.g. the required orientation
of the wrist sensor).

### Sensor data requirements
The ParaDigMa toolbox is designed for the analysis of passive monitoring data collected using a wrist sensor in persons with PD.

Specific requirements include:
| Pipeline               | Sensor Configuration                                                                                                       | Context of Use                                                                                             |
|------------------------|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **All**               | - Sensor position: wrist-band on most or least affected side (validated for both, but different sensitivity for measuring disease progression for tremor and arm swing during gait).  <br> - Sensor orientation: orientation as described in [Coordinate System](guides/coordinate_system). <br> - Timeframe: contiguous, strictly increasing timestamps. | - Population: persons with PD. <br> - Data collection protocol: passive monitoring in daily life. |
| **Arm swing during gait** | - Accelerometer: minimum sampling rate of 100 Hz, minimum range of ± 4 g. <br> - Gyroscope: minimum sampling rate of 100 Hz, minimum range of ± 1000 degrees/sec. | - Population: no walking aid, no severe dyskinesia in the watch-sided arm. <br> - Compliance: for weekly measures: at least three compliant days (with ≥10 hours of data between 8 am and 10 pm), and at least 2 minutes of arm swing. |
| **Tremor**            | - Gyroscope: minimum sampling rate of 100 Hz, minimum range of ± 1000 degrees/sec. | - Compliance: for weekly measures: at least three compliant days (with ≥10 hours of data between 8 am and 10 pm). |
| **Pulse rate**        | - PPG*: minimum sampling rate of 30 Hz, green LED. <br> - Accelerometer: minimum sampling rate of 100 Hz, minimum range of ± 4 g. | - Population: no rhythm disorders (e.g. atrial fibrillation, atrial flutter). <br> - Compliance: for weekly measures: minimum average of 12 hours of data per day. |

\* The processing of PPG signals is currently based on the blood volume pulse (arbitrary units) obtained from the Verily Study Watch. [This](https://biomarkersparkinson.github.io/paradigma/tutorials/_static/pulse_rate_analysis.html#step-3-signal-quality-classification) part of the PPG tutorial provides code and documentation on how to use the pipeline with other PPG devices.

> **Warning:** While the toolbox is designed to work on any wrist sensor device which fulfills the requirements,
we have currently verified its performance on data from the Gait-up Physilog 4 (arm swing during gait & tremor) and the Verily Study Watch (all pipelines). Furthermore, the specifications above are the minimally validated requirements. For example, while ParaDigMa works with accelerometer and gyroscope data sampled at 50 Hz, its effect on subsequent processes has not been empirically validated.
<br/>

**Data formats**

ParaDigMa does not require a particular method of data storage and retrieval. However, we use [TSDF](https://biomarkersparkinson.github.io/tsdf/) as format for loading and storing sensor data throughout the tutorials, for which we have added [additional support](autoapi/paradigma/util/index). TSDF enables efficient time series data storage with added metadata. Please see our tutorial [Data preparation](tutorials/data_preparation) for examples of loading TSDF into memory. We are developing support for other data formats, and we currently have tutorials for loading data from Empatica and Axivity in [Device specific data loading](tutorials/device_specific_data_loading).
<br/>

## Scientific validation

The pipelines were developed and validated using data from the Parkinson@Home Validation study [[Evers et al. (2020)]](https://pmc.ncbi.nlm.nih.gov/articles/PMC7584982/) and the Personalized Parkinson Project [[Bloem et al. (2019)]](https://pubmed.ncbi.nlm.nih.gov/31315608/). The following publications contain details and validation of the pipelines:

**Arm swing during gait**
* [Post, E. et al. (2025) - Quantifying arm swing in Parkinson's disease: a method account for arm activities during free-living gait](https://doi.org/10.1186/s12984-025-01578-z)
* [Post, E. et al. (2026) - (Pre-print) Longitudinal progression of digital arm swing measures during free-living gait in early Parkinson's disease](https://doi.org/10.64898/2026.01.06.26343500)

**Tremor**
* [Timmermans, N.A. et al. (2025) - A generalizable and open-source algorithm for real-life monitoring of tremor in Parkinson's disease](https://doi.org/10.1038/s41531-025-01056-2)
* [Timmermans, N.A. et al. (2025) - (Pre-print) Progression of daily-life tremor measures in early Parkinson disease: a longitudinal continuous monitoring study](https://www.medrxiv.org/content/10.64898/2025.12.23.25342892v1)

**Pulse rate**
* [Veldkamp, K.I. et al. (2025) - Heart rate monitoring using wrist photoplethysmography in Parkinson disease: feasibility and relation with autonomic dysfunction](https://doi.org/10.1101/2025.08.15.25333751)

## Contributing

We welcome contributions! Please check out our [contributing guidelines](contributing).
Please note that this project is released with a [Code of Conduct](conduct). By contributing to this project, you agree to abide by its terms.

### Running tests
ParaDigMa uses `pytest` for unit tests. To run all tests:

```bash
# If using Poetry
poetry run pytest

# Or using your Python environment
pytest
```

Tests are located in the `tests/` folder.

## License

It is licensed under the terms of the Apache License 2.0 license. See [License](license) for more details.

## Acknowledgements

The core team of ParaDigMa consists of Erik Post, Kars Veldkamp, Nienke Timmermans, Diogo Coutinho Soriano, Peter Kok, Vedran Kasalica and Luc Evers.
Advisors to the project are Max Little, Jordan Raykov, Twan van Laarhoven, Hayriye Cagnan, and Bas Bloem.
The initial release of ParaDigMa was funded by the Michael J Fox Foundation (grant #020425) and the Dutch Research Council (grant #ASDI.2020.060 & grant #2023.010).
ParaDigMa was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

## Contact
Questions, issues or suggestions about ParaDigMa? Please reach out to paradigma@radboudumc.nl, or open an issue in the GitHub repository.
