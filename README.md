# Paradigma: A Toolbox for Identifying Digital Biomarkers in Parkinson's Disease

| Badges | |
|:----:|----|
| **Packages and Releases** | [![Latest release](https://img.shields.io/github/release/biomarkersparkinson/paradigma.svg)](https://github.com/biomarkersparkinson/paradigma/releases/latest) [![PyPI](https://img.shields.io/pypi/v/paradigma.svg)](https://pypi.python.org/pypi/paradigma/)  [![Static Badge](https://img.shields.io/badge/RSD-paradigma-lib)](https://research-software-directory.org/software/paradigma) |
| **DOI** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13838393.svg)](https://doi.org/10.5281/zenodo.13838393) |
| **Build Status** | [![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Build and test](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/build-and-test.yml) [![pages-build-deployment](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/biomarkersParkinson/paradigma/actions/workflows/pages/pages-build-deployment) |
| **License** |  [![GitHub license](https://img.shields.io/github/license/biomarkersParkinson/paradigma)](https://github.com/biomarkersparkinson/paradigma/blob/main/LICENSE) |
<!-- | **Fairness** |  [![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu) [![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/8083/badge)](https://www.bestpractices.dev/projects/8083) | -->


Paradigma is a Python package (see [documentation](https://biomarkersparkinson.github.io/paradigma/)) for processing wearable sensor data to identify digital biomarkers for Parkinson’s disease. It simplifies the analysis of gait and photoplethysmography (PPG) data using a unified Time Series Data Format (TSDF), helping researchers extract key PD biomarkers. Future updates will add tremor detection and more feature extraction tools.


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
