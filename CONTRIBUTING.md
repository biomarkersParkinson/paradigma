# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version
* Python version
* Any details about your local setup that might be helpful in troubleshooting
* Detailed steps to reproduce the bug

### Fix Bugs

Look through the [GitHub issues](https://github.com/biomarkersParkinson/paradigma/issues) for bugs.
Anything tagged with `bug` or `help wanted` is open to whoever wants to implement it.

### Implement Features

Look through the [GitHub issues](https://github.com/biomarkersParkinson/paradigma/issues) for features.
Anything tagged with `enhancement` or `help wanted` is open to whoever wants to implement it.

### Write Documentation

Documentation contributions are always welcome! You can contribute to:
* Official docs: Located in `docs/`
* Tutorial notebooks: `docs/tutorials/`
* Docstrings: In Python modules
* Articles or blog posts

#### Docstring Style Guide

ParaDigMa follows NumPy/Napoleon docstring conventions with these type annotation guidelines:

**Function Signatures:**
- Use PEP 604 syntax: `str | Path` instead of `Union[str, Path]`
- Use `X | None` instead of `Optional[X]`
- Example:
  ```python
  def load_data(
      path: str | Path,
      config: Config | None = None
  ) -> pd.DataFrame:
  ```

**Parameter Docstrings:**
- Use natural language for types, not Python type syntax
- For union types: `str or Path` not `Union[str, Path]`
- For optional parameters: Add `, optional` suffix
- For lists: `list of str` not `List[str]`
- For dicts: `dict` not `Dict[str, int]`

**Examples:**
```python
def example_function(
    data_path: str | Path,
    columns: List[str] | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """
    Load and process data.

    Parameters
    ----------
    data_path : str or Path
        Path to data directory
    columns : list of str, optional
        Column names to load
    config : Config, optional
        Configuration object

    Returns
    -------
    pd.DataFrame
        Processed data
    """
```

**Return Type Documentation:**
- Use simple descriptions: `dict` or `DataFrame` not `Dict[str, pd.DataFrame]`
- Add details in the description text below

#### Workflow for notebooks and docs:
1. Run and export notebooks:

```bash
poetry run build-docs
```

This will:
* Execute all notebooks in `docs/tutorials/`
* Export them to Markdown in `docs/tutorials/_static/`
* Strip outputs
* Build the HTML documentation

2. Serve documentation locally:

```bash
poetry run serve-docs
```

This will serve the built HTML at `http://localhost:8000`.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome!

## Get Started!

Ready to contribute? Here's how to set up `paradigma` locally:

### Prerequisites

1. **Install git-lfs** (required for example data):

   **Windows:**
   ```bash
   choco install git-lfs
   # Or download from https://git-lfs.com/
   ```

   **Linux:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git-lfs

   # Fedora/RHEL
   sudo dnf install git-lfs

   # Arch Linux
   sudo pacman -S git-lfs
   ```

   **macOS:**
   ```bash
   brew install git-lfs
   ```

2. **Enable git-lfs:**
   ```bash
   git lfs install
   ```

### Clone and Install

1. Clone the repository:

```bash
git clone https://github.com/biomarkersParkinson/paradigma.git
cd paradigma
```

2. Install dependencies via Poetry:

```bash
poetry install
```

3. **Verify example data setup:**

   Check that example data was downloaded correctly (not git-lfs pointers):

   **Linux/macOS:**
   ```bash
   ls -lh example_data/verily/ppg/*.bin
   # Should show file sizes (100KB - several MB), not ~130 bytes

   head -n 1 example_data/verily/ppg/PPG_segment0001_meta.json
   # Should show JSON like {"file_name": ..., not "version https://git-lfs.github.com/spec/v1"
   ```

   **Windows PowerShell:**
   ```powershell
   Get-ChildItem example_data/verily/ppg/*.bin | Select-Object Name, Length
   # Should show file sizes (100KB - several MB), not ~130 bytes

   Get-Content example_data/verily/ppg/PPG_segment0001_meta.json -Head 1
   # Should show JSON, not "version https://git-lfs.github.com/spec/v1"
   ```

   **If files are git-lfs pointers:**
   ```bash
   git lfs pull
   ```

4. Create a new branch for your work:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

5. Make your changes and run the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This ensures code formatting (`black`), import sorting (`isort`), stripping notebook outputs, and other checks.
These pre-commit hooks also run for changed files when committing.

6. If contributing to docs, build and serve them locally to verify:

```bash
poetry run build-docs
poetry run serve-docs
```

`build-docs` accepts the following arguments to speed up development:
- `--notebook <filename>` - Build only a specific notebook (e.g., `--notebook pipeline_orchestrator.ipynb`)
- `--dev` - Skip execution of notebooks
- `--no-nbconvert` - Skip conversion of notebooks to markdown

These options are useful when iterating on documentation changes without rebuilding everything.

7. Commit your changes and open a pull request.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported operating systems and versions of Python.

## Code of Conduct

Please note that the `paradigma` project is released with a
[Code of Conduct](https://biomarkersparkinson.github.io/paradigma/conduct.html).
By contributing to this project you agree to abide by its terms.
