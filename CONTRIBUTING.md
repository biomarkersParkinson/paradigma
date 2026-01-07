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
1. Clone the repository:

```bash
git clone https://github.com/biomarkersParkinson/paradigma.git
cd paradigma
```

2. Install dependencies via Poetry:

```bash
poetry install
```

3. Create a new branch for your work:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

4. Make your changes and run the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This ensures code formatting (`black`), import sorting (`isort`), stripping notebook outputs, and other checks.
These pre-commit hooks also run for changed and staged files when committing.

5. If contributing to docs, build and serve them locally to verify:

```bash
poetry run build-docs
poetry run serve-docs
```

6. Commit your changes and open a pull request.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported operating systems and versions of Python.

## Code of Conduct

Please note that the `paradigma` project is released with a
[Code of Conduct](https://biomarkersparkinson.github.io/paradigma/conduct.html).
By contributing to this project you agree to abide by its terms.

Test
