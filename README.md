![ParaDigMa logo](https://raw.githubusercontent.com/biomarkersParkinson/paradigma/update-readme/docs/source/_static/img/paradigma-logo-banner.png)

| Badges | |
|:----:|----|
| **Packages and Releases** | [![Latest release](https://img.shields.io/github/release/biomarkersparkinson/paradigma.svg)](https://github.com/biomarkersparkinson/paradigma/releases/latest) [![PyPI](https://img.shields.io/pypi/v/paradigma.svg)](https://pypi.python.org/pypi/paradigma/)  [![Static Badge](https://img.shields.io/badge/RSD-paradigma-lib)](https://research-software-directory.org/software/paradigma) |
| **DOI** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13838393.svg)](https://doi.org/10.5281/zenodo.13838393) |
| **Build Status** | [![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Build and test](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml) [![pages-build-deployment](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment) |
| **License** |  [![GitHub license](https://img.shields.io/github/license/biomarkersParkinson/paradigma)](https://github.com/biomarkersparkinson/paradigma/blob/main/LICENSE) |
<!-- | **Fairness** |  [![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu) [![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/8083/badge)](https://www.bestpractices.dev/projects/8083) | --> 

ParaDigMa is a Python package for processing time series data of wrist-worn sensors to quantify signs of Parkinson’s disease (PD). It contains functionalities for processing the following sensor types:
* Inertial Measurement Units (accelerometer, gyroscope)
* Photoplethysmogram (PPG)

And analyses of the following motor and non-motor domains:
* Gait
* Tremor
* Heart Rate

## More about ParaDigMa
The components of ParaDigMa are visually shown in the diagram below.

![Pipeline Architecture](https://raw.githubusercontent.com/biomarkersParkinson/paradigma/update-readme/docs/source/_static/img/pipeline-architecture.png)

#### Processes
ParaDigMa can best be understood by categorizing the sequential processes:

| Process | Description |
| ---- | ---- |
| Preprocessing | Ensuring that the sensor data is ready for further processing | 
| Feature extraction | Creating features based on windowed views of the timestamps |
| Classification | Making predictions using developed and validated classifiers | 
| Quantification | Selecting specific features of interest |
| Aggregation | Aggregating the features at a specified time-level |

#### Domain requirements
ParaDigMa can be used to extract aggregations related to a single or multiple domain(s). Each domain has its specific data requirements. Strict requirements for the domain are marked by X, soft requirements (for some additional functionalities) are marked by O.

| | Gait | Tremor | Heart Rate |
|----------|:-----------:|:-----------:|:-----------:|
| **Accelerometer** | X | | O | 
| **Gyroscope** | X | X | | 
| **PPG** | | | X | 



## Installation

The package is available in PyPi and requires [Python 3.10](https://www.python.org/downloads/) or higher. It can be installed using:

```bash
pip install paradigma
```

## Usage

See our [extended documentation](https://biomarkersparkinson.github.io/paradigma/).

## Development

### Installation

The package requires Python 3.10 or higher. Use [Poetry](https://python-poetry.org/docs/#installation) to set up the environment and install the dependencies:

```bash
poetry install
```

### Testing

```bash
poetry run pytest
```

### Type checking

```bash
poetry run pytype .
```

### Building documentation

```bash
poetry run make html --directory docs/
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`paradigma` was created by Peter Kok, Vedran Kasalica, Erik Post, Kars Veldkamp, Nienke Timmermans, Diogo Coutinho Soriano, Luc Evers. It is licensed under the terms of the Apache License 2.0 license.

## Credits

`paradigma` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
