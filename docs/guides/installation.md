# Installation Guide

## Prerequisites

- **Python 3.11 or higher** is required
- pip (Python package installer)

## Standard Installation

Install ParaDigMa from PyPI using pip:

```bash
pip install paradigma
```

This installs the latest stable release with all required dependencies.

## Development Installation

For development or to run the tutorial notebooks with example data, you need to:

1. Install git-lfs
2. Clone the repository
3. Install in development mode

### Step 1: Install git-lfs

ParaDigMa uses Git Large File Storage (git-lfs) for example data files. Install it **before cloning**.

**Windows**

Using Chocolatey:

```bash
choco install git-lfs
```

Or download the installer from https://git-lfs.com/

**Linux**

Ubuntu/Debian:

```bash
sudo apt-get install git-lfs
```

Fedora/RHEL:

```bash
sudo dnf install git-lfs
```

Arch Linux:

```bash
sudo pacman -S git-lfs
```

**macOS**

```bash
brew install git-lfs
```

**Enable git-lfs (all platforms)**

```bash
git lfs install
```

### Step 2: Clone Repository

```bash
git clone https://github.com/biomarkersParkinson/paradigma.git
cd paradigma
```

**If you already cloned without git-lfs:**

```bash
git lfs install
git lfs pull
```

### Step 3: Install Dependencies

Using pip:

```bash
pip install -e ".[dev]"
```

Using Poetry:

```bash
poetry install
```

## Verify Installation

After installation, verify that ParaDigMa is installed correctly:

```python
import paradigma
print(paradigma.__version__)
```

## Installation from Source

For development or to use the latest development version, clone the repository and install in editable mode:

```bash
git clone https://github.com/biomarkersParkinson/paradigma.git
cd paradigma
pip install -e ".[dev]"
```

The `[dev]` extra includes development dependencies such as testing tools and linting utilities.

## Dependencies

ParaDigMa requires the following core dependencies:

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning algorithms
- **pyyaml**: Configuration file parsing

Optional dependencies for specific data format support:

- **pyarrow**: For reading/writing Parquet files
- **pytz**: Timezone handling

## Troubleshooting

### JSONDecodeError in Tutorials

**Symptom:** When running tutorials, you see:

```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Cause:** Example data files are git-lfs pointer files instead of actual data.

**Solution:**

1. Install git-lfs (see Step 1 in Development Installation above)
2. Pull the actual data files:

   ```bash
   git lfs install
   git lfs pull
   ```

3. Verify files are real data (not pointers):

   **Linux/macOS:**
   ```bash
   ls -lh example_data/verily/ppg/PPG_segment0001_meta.json
   # Should show ~886 bytes, not ~130 bytes

   head -n 1 example_data/verily/ppg/PPG_segment0001_meta.json
   # Should show JSON like {"file_name": ..., not "version https://git-lfs.github.com/spec/v1"
   ```

   **Windows PowerShell:**
   ```powershell
   Get-Item example_data/verily/ppg/PPG_segment0001_meta.json | Select-Object Length
   # Should show ~886 bytes, not ~130 bytes

   Get-Content example_data/verily/ppg/PPG_segment0001_meta.json -Head 1
   # Should show JSON, not "version https://git-lfs.github.com/spec/v1"
   ```

**Still having issues?**

Check if git-lfs is properly configured:

```bash
git lfs env
git lfs ls-files
```

The second command should list the binary data files being tracked by LFS.

### Python Version Error

If you encounter an error like `Python 3.11 or higher is required`, ensure you have the correct Python version:

```bash
python --version
```

If you have multiple Python versions installed, specify Python 3.11+:

```bash
python3.11 -m pip install paradigma
```

### Import Errors

If you get import errors after installation, try upgrading pip and reinstalling:

```bash
pip install --upgrade pip
pip install --force-reinstall paradigma
```

### Dependency Conflicts

If there are dependency conflicts with your existing environment, consider creating a fresh virtual environment:

```bash
python -m venv paradigma_env
source paradigma_env/bin/activate  # On Windows: paradigma_env\Scripts\activate
pip install paradigma
```

## Virtual Environments (Recommended)

Using a virtual environment is recommended to avoid dependency conflicts:

### Using venv

```bash
# Create virtual environment
python -m venv paradigma_env

# Activate it
source paradigma_env/bin/activate  # On Windows: paradigma_env\Scripts\activate

# Install ParaDigMa
pip install paradigma
```

### Using conda

```bash
# Create conda environment
conda create -n paradigma python=3.11

# Activate it
conda activate paradigma

# Install ParaDigMa
pip install paradigma
```

## Getting Help

If you encounter issues during installation:

1. Search existing [GitHub Issues](https://github.com/biomarkersParkinson/paradigma/issues)
2. Open a new issue with:
   - Python version (`python --version`)
   - Error message (full traceback)
   - Operating system
   - Installation method used

Contact: paradigma@radboudumc.nl
