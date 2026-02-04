# Changelog

<!--next-version-placeholder-->

## v1.1.0 (04/02/2026)

### Features
- **Orchestrator Pipeline**: Added `orchestrator.py` with `run_paradigma()` function for complete end-to-end ParaDigMa analysis pipeline from data loading to aggregated results.
- **Data Preparation**: Added `prepare_data.py` with automatic data preparation capabilities, including column mapping for flexible column names, watch orientation adjustment, and comprehensive validation.
- **Centralized Loading**: Added `load.py` to centralize data loading capabilities across all pipelines.
- **High-Level Pipeline Functions**: Added `run_gait_pipeline()`, `run_tremor_pipeline()`, and `run_pulse_rate_pipeline()` for simplified pipeline execution.
- **Segment Distinction**: Added `GAIT_SEGMENT_NR` and `DATA_SEGMENT_NR` constants to distinguish between temporal gaps and gait bouts. `SEGMENT_NR` kept as deprecated alias for backward compatibility.
- **Test Data**: Added Physilog Gait-Up test data to LFS.

### Improvements
- **Logging System**: Replaced `verbose` parameter with standard Python logging using `logging_level` and `custom_logger` parameters across all pipeline functions for better control and traceability.
- **Data Segment Preservation**: Modified `quantify_arm_swing()` to preserve `data_segment_nr` when present in input data.
- **Bug Fixes**: Fixed NaN propagation in resampling by interpolating NaN values before resampling (scipy cannot handle NaN in non-contiguous data).

### Documentation
- **New Tutorials**: Added `pipeline_orchestrator.ipynb` tutorial demonstrating end-to-end processing with logging control examples.
- **New Guides**: Added comprehensive guides for data_input, installation, sensor_requirements, supported_devices, and validation.
- **Readme**: Shortened README.md for readability and expanded detailed guides.
- **PEP Standards**: Updated all docstrings and function signatures to PEP standards.
- **Build System**: Added single-notebook build support with `--notebook` argument; automated style tag removal in documentation build pipeline.

### Testing
- Added minimal testing for new pipeline functionalities.
- Added tests for segment column naming and backward compatibility.
- Added tests to verify data segment tracking preservation.

### Backward Compatibility
- All existing code using `SEGMENT_NR` continues to work.
- Previously used pipeline code remains functional.

## v1.0.4 (11/11/2025)
- Column names not long have to be set to ParaDigMa standards, but can be flexibly adjusted (see data_preparation.ipynb tutorial for instructions).
- Users can now change the tolerance threshold for determining when consecutive timestamps are contiguous using config.
- The usage of accelerometry is now optional. Accelerometry can be used to detect motion artefacts that can be removed using Paradigma (see pulse_rate_analysis.ipynb for more details).
- We also added instructions for how to scale PPG features using z-scoring.
- Coefficient of variation (CoV) added as aggregation method for arm swing during gait.
- Gait segment duration categories are no longer fixed.

For developers
* We added pre-commit hooks to ensure consistency in formatting and automate cleanup.
* We also created two Python scripts for simplifying building and hosting docs, using poetry build-docs and poetry serve-docs.

## v1.0.3 (08/09/2025)
- Added flexibility to let user specify gait segment duration categories.
- Added the within-segment coefficient of variation to list of arm swing parameters.
- Fixed bug where users were unable to change resampling frequency.
- Added the mode of continuous variables using bins to the list of aggregations.
- Increased tolerance for contiguous segments.
- Changed contact details.

## v1.0.2 (12/06/2025)
- Changed 'heart rate' to 'pulse rate' consistently throughout source code and documentation, in line with scientific publications.

## v1.0.1 (11/06/2025)
- Increased efficiency of data processing pipelines.
- Aligned high-level processes of individual pipelines.

## v1.0.0 (15/04/2025)
- Finalized tutorials per pipeline.
- Added functionality to invert watch side if worn on contralateral side.
- Limited memory requirements.
- Expanded documentation for tutorials and readme.

## v0.3.2 (25/09/2024)
- Fixed citation file (`cff`) formatting issues.
- Improved handling of constants in configuration files.
- Refactored gait configuration files and enhanced constraints.
- Bumped Python version to 3.10 for compatibility.
- Tested `Enum` usage for constraints, but reverted to `str` due to the lack of flexibility.
- Fixed typing issues
- Added project badges and citation file.
- Updated project naming and authors in citation file.
- Enabled and fixed `pytype` for type checking.
- Refined types in code and docstrings.

## v0.2.0 (22/08/2024)
- Reorganized imports and exposed functions at the top level.
- Minor documentation fixes.
- Updated Sphinx auto-API config to exclude imported members.
- Reverted some import reorganizations and applied final corrections.
- Refined documentation and code structure.

## v0.1.0 (15/08/2024)
- First pre-release of `paradigma`!
- Initial implementation of gait and PPG feature extraction.
