# Changelog

<!--next-version-placeholder-->

## v1.0.3 (08/09/2025)
- Added flexibility to let user specify gait segment duration categories.
- Fixed bug where users were unable to change resampling frequency.
- Added the mode of continuous variables to the list of aggregations.
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

