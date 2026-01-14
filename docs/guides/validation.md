# Scientific Validation

ParaDigMa pipelines have been scientifically validated through rigorous testing on real-world datasets in published peer-reviewed studies. This document provides an overview of the validation evidence.

## Validation Overview

ParaDigMa offers three validated processing pipelines for extracting digital biomarkers from wrist sensor data in Parkinson's disease. Details on publications are shown below.

### Arm Swing during Gait

**Post, E. et al. (2025)**
- **Title**: Quantifying arm swing in Parkinson's disease: a method account for arm activities during free-living gait
- **Journal**: Journal of NeuroEngineering and Rehabilitation
- **DOI**: https://doi.org/10.1186/s12984-025-01578-z

**Post, E. et al. (2026)**
- **Title**: Longitudinal progression of digital arm swing measures during free-living gait in early Parkinson's disease
- **Status**: Pre-print
- **DOI**: https://doi.org/10.64898/2026.01.06.26343500

### Tremor

**Timmermans, N.A. et al. (2025)**
- **Title**: A generalizable and open-source algorithm for real-life monitoring of tremor in Parkinson's disease
- **Journal**: npj Parkinson's Disease
- **DOI**: https://doi.org/10.1038/s41531-025-01056-2

**Timmermans, N.A. et al. (2025)**
- **Title**: Progression of daily-life tremor measures in early Parkinson disease: a longitudinal continuous monitoring study
- **Status**: Pre-print
- **DOI**: https://www.medrxiv.org/content/10.64898/2025.12.23.25342892v1

### Pulse Rate

**Veldkamp, K.I. et al. (2025)**
- **Title**: Heart rate monitoring using wrist photoplethysmography in Parkinson disease: feasibility and relation with autonomic dysfunction
- **Status**: Pre-print
- **DOI**: https://doi.org/10.1101/2025.08.15.25333751

## Important Caveats

> [!WARNING]
> While ParaDigMa has been validated in published research, the following limitations should be considered:
>
> 1. **Device-Specific Validation**: Formal validation is limited to Gait-up Physilog 4 and Verily Study Watch
> 2. **Population**: Validation primarily in persons with early-to-moderate PD; extrapolation to advanced stages requires caution
> 3. **Wrist Placement**: Some measures show different sensitivity depending on whether the sensor is on the most or least affected side
> 4. **Data Quality**: All validation studies assume high-quality sensor data with sufficient compliance; poor data quality may affect results
> 5. **Generalization**: While algorithms are designed to be generalizable, use on new devices should be validated by the user

## Best Practices

1. **Data Quality**: Ensure sensor compliance (minimum wearing time, data continuity)
2. **Validation**: For new devices or populations, conduct local validation studies
3. **Documentation**: Record device model, firmware, wearing location, and collection protocol
4. **Reproducibility**: Share data and analysis code to enable reproducibility and validation
