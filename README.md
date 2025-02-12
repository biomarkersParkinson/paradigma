<p align="center">
  <img src="https://raw.githubusercontent.com/biomarkersParkinson/paradigma/main/docs/source/_static/img/paradigma-logo-banner.png" alt="ParaDigMa logo"/>
</p>

| Badges | |
|:----:|----|
| **Packages and Releases** | [![Latest release](https://img.shields.io/github/release/biomarkersparkinson/paradigma.svg)](https://github.com/biomarkersparkinson/paradigma/releases/latest) [![PyPI](https://img.shields.io/pypi/v/paradigma.svg)](https://pypi.python.org/pypi/paradigma/)  [![Static Badge](https://img.shields.io/badge/RSD-paradigma-lib)](https://research-software-directory.org/software/paradigma) |
| **DOI** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13838392.svg)](https://doi.org/10.5281/zenodo.13838392) |
| **Build Status** | [![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Build and test](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml) [![pages-build-deployment](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment) |
| **License** |  [![GitHub license](https://img.shields.io/github/license/biomarkersParkinson/paradigma)](https://github.com/biomarkersparkinson/paradigma/blob/main/LICENSE) |
<!-- | **Fairness** |  [![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu) [![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/8083/badge)](https://www.bestpractices.dev/projects/8083) | --> 

## Overview
The Parkinsons Disease Digital Markers (ParaDigMa) toolbox is a Python
software package designed for analyzing real-life wrist sensor data
to extract digital measures of motor and non-motor signs of Parkinson's disease (PD).  

Specifically, the toolbox is designed to process accelerometer, gyroscope and 
photoplethysmography signals, collected during passive monitoring in daily life. 
It contains three data processing pipelines: (1) arm swing during gait, (2) tremor, 
and (3) pulse rate analysis. These pipelines are scientifically validated for their 
use in persons with PD. Furthermore, the toolbox contains general functionalities for 
signal processing and feature extraction, such as filtering, peak detection, and 
spectral analysis. 

The toolbox is accompanied by a set of example scripts and notebooks for 
each processing pipeline that demonstrate how to use the toolbox for extracting 
digital measures. In addition, the toolbox is designed to be modular, enabling
researchers to easily extend the toolbox with new algorithms and functionalities. 

## Features
The components of ParaDigMa are shown in the diagram below.

<p align="center">
  <img src="https://raw.githubusercontent.com/biomarkersParkinson/paradigma/main/docs/source/_static/img/pipeline-architecture.png" alt="Pipeline architeecture"/>
</p>
#LJWE: improve readability, add tremor power in quantification of tremor.

ParaDigMa can best be understood by categorizing the sequential processes:

| Process | Description |
| ---- | ---- |
| Preprocessing | Preparing the raw sensor signals for further processing | 
| Feature extraction | Extracting features based on the windowed sensor signals |
| Classification | Detecting segments of interest using validated classifiers (e.g. gait segments) | 
| Quantification | Extracting specific measures from the detected segments (e.g. arm swing measures) |
| Aggregation | Aggregating the measures over a specific time period (e.g. week-level aggregates) |

ParaDigMa contains the following processing pipelines (which use the processes described above): 

| Pipeline | Input | Output (window-level) | Output (week-level) | 
| ---- | ---- | ---- | ---- |
| Arm swing during gait | Wrist accelerometer and gyroscope data | Gait probability, arm swing probability, arm swing range of motion (RoM) | Typical & maximum arm swing RoM | 
| Tremor | Wrist gyroscope data | Tremor probability, tremor power | % tremor time, typical & maximum tremor power | 
| Pulse rate | Wrist photoplethysmography data | PPG signal quality, pulse rate | Resting pulse rate, maximum pulse rate | 

## Installation

The package is available in PyPi and requires [Python 3.10](https://www.python.org/downloads/) or higher. #LJWE: later 3.11 is mentioned, please align
It can be installed using:

```bash
pip install paradigma
```

## Usage

### Tutorials & documentation
See our tutorials for example scripts on how to use the toolbox to extract digital measures from wrist sensor signals.
The API reference contains detailed documentation of all toolbox modules and functions.
The user guides provide additional information about specific topics (e.g. the required orientation of the wrist sensor).

### Sensor data requirements
The ParaDigMa toolbox is designed for the analysis of passive monitoring data collected using a wrist sensor in persons with PD. 

Specific requirements include: 
#LJWE: see attached word table.

| Pipeline 		| 	
| ----		 	| 
| All	 	 	|
| Arm swing during gait | 
| Tremor 		| 
| Pulse rate 		| 

While the toolbox is designed to work on any wrist sensor device which fulfills the requirements, 
we have currently verified its performance on data from the Gait-up Physilog 4 (arm swing during gait & tremor) and the Verily Study Watch (all pipelines). 
We currently support TSDF (#LJWE: include link) as format for the input sensor data. Please see our tutorial "Data preparation" [#LJWE: insert link] for how to use other data formats.

## Scientific validation

The pipelines were developed and validated using data from the Parkinson@Home Validation study [Evers et al. 2020 #LJWE: INSERT REF] 
and the Personalized Parkinson Project [Bloem et al. 2019 #LJWE: INSERT REF].

Details for the different pipelines can be found in the associated scientific publications:

| Pipeline | Reference |
| ---- | ---- | 
| Arm swing during gait | #LJWE: INSERT REF | 
| Tremor | #LJWE: INSERT REF | 
| Pulse rate | #LJWE: INSERT REF | 

## Contributing

We welcome contributions! Please check out our contributing guidelines [#LJWE: kon deze niet vinden? Graag uploaden bij user guides, en daarin de poetry instructies etc. opnemen, hoeft niet hier in de readme]. 
Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms [#LJWE: kon deze ook niet vinden?]

## License

It is licensed under the terms of the Apache License 2.0 license. See [#LJWE: INSERT LICENSE LINK] for more details.

## Acknowledgements

The core team of ParaDigMa consists of Erik Post, Kars Veldkamp, Nienke Timmermans, Diogo Coutinho Soriano, Peter Kok, Vedran Kasalica and Luc Evers. 
Advisors to the project are Max Little, Jordan Raykov, Twan van Laarhoven, Hayriye Cagnan, and Bas Bloem. 
The initial release of ParaDigMa was funded by the Michael J Fox Foundation (grant #020425) and the Dutch Research Council (grant #ASDI.2020.060 & grant #2023.010).
ParaDigMa was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

## Contact
Questions, issues or suggestions about ParaDigMa? Please reach out to erik.post@radboudumc.nl [#LJWE: let's create a seperate toolbox e-mail address], or open an issue in the GitHub repository.