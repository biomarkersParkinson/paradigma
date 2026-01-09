# Getting started with the ParaDigMa tutorials

This section explains how to get started with the ParaDigMa tutorials and how to run the example
notebooks successfully.

The tutorials are intended as examples demonstrating how to use the ParaDigMa toolbox for extracting
digital measures from wrist sensor data. They assume basic familiarity with Python and Jupyter
notebooks.

## Installation
First, install ParaDigMa in a Python environment (Python 3.11 or higher):

```bash
pip install paradigma
```

Alternatively, if you are working from the repository and using Poetry:
```bash
poetry install
```

## Running the tutorials
The tutorials include Jupyter notebooks demonstrating each processing pipeline (arm swing during
gait, tremor, and pulse rate).

To run these notebooks, you need to ensure that Jupyter uses the same Python environment in which
ParaDigMa is installed.

If ParaDigMa is installed in a virtual environment (e.g. via Poetry), opening Jupyter directly from
the terminal or an IDE may result in an incorrect kernel being selected.

## Using Jupyter with Poetry (recommended)
If you installed ParaDigMa using Poetry, we recommend registering a Jupyter kernel that points to
the Poetry environment:

```bash
poetry run python -m ipykernel install --user --name paradigma --display-name "Python (paradigma)"
```

After this, start Jupyter (Notebook or Lab) and select the “Python (paradigma)” kernel when opening
a tutorial notebook.

If imports such as `import paradigma` fail, this usually indicates that the wrong kernel is
selected.

## Using Jupyter in an IDE (VSCode / PyCharm)
If you use an IDE such as VSCode or PyCharm:
* Make sure the Python interpreter is set to the environment where ParaDigMa is installed
* Ensure the notebook kernel matches this interpreter

Refer to your IDE’s documentation for selecting Python interpreters and Jupyter kernels.

## Tutorial overview
The tutorials are organized as follows:

* **Device-specific data loading**: Examples of loading data from commonly-used devices.
* **Data preparation**: How to prepare sensor data for use with ParaDigMa.
* **Arm swing during gait**: Example pipeline for detecting gait and quantifying arm swing.
* **Tremor**: Example pipeline for detecting and quantifying tremor.
* **Pulse rate**: Example pipeline for pulse rate estimation.

Each tutorial focuses on demonstrating the relevant pipeline rather than providing a full
end-to-end application.

## Notes on scope
The notebooks are intended as illustrative examples. ParaDigMa is designed to be modular and
callable from Python code, allowing users to build their own analysis workflows without relying
on notebooks.
