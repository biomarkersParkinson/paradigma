# paradigma

Digital Biomarkers for Parkinson's Disease Toolbox

A package ([documentation](https://biomarkersparkinson.github.io/paradigma/)) to process wearable sensor data for Parkinson's disease.

## Installation

The package is available in PyPi and requires [Python 3.9](https://www.python.org/downloads/) or higher. It can be installed using:

```bash
pip install paradigma
```

## Usage

See our [extended documentation](https://biomarkersparkinson.github.io/paradigma/).


## Development

### Installation
The package requires Python 3.9 or higher. Use [Poetry](https://python-poetry.org/docs/#installation) to set up the environment and install the dependencies:

```bash
poetry install
```

### Testing

```bash
poetry run pytest
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
