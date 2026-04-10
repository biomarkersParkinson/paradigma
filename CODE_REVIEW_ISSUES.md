# ParaDigMa Code Review - Issues & Vulnerabilities

**Date**: April 9, 2026
**Scope**: Full paradigma package codebase
**Overall Risk Assessment**: MEDIUM

---

## 🔴 HIGH PRIORITY ISSUES

### 1. **File Handle Leaks in Empatica Data Loading** (load.py, line 81-96)
**Severity**: HIGH
**Location**: `src/paradigma/load.py::load_empatica_data()`

**Issue**:
```python
with open(file_path, "rb") as f:
    reader = DataFileReader(f, DatumReader())
    empatica_data = next(reader)
# File closed here, but reader may not be fully consumed
```

The `DataFileReader` is not explicitly closed. If processing large Empatica files or processing many files iteratively, this could lead to file handle exhaustion.

**Risk**: After calling `run_paradigma()` multiple times with Empatica data, the process may fail with "Too many open files" error.

**Recommendation**:
```python
def load_empatica_data(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    logger.info(f"Loading Empatica data from {file_path}")

    with open(file_path, "rb") as f:
        reader = DataFileReader(f, DatumReader())
        empatica_data = next(reader)
        reader.close()  # Explicitly close reader
    # rest of function...
```

---

### 2. **Logger Handler Leaks in Orchestrator** (orchestrator.py, line 333-340)
**Severity**: HIGH
**Location**: `src/paradigma/orchestrator.py::run_paradigma()`

**Issue**:
```python
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(...)
package_logger.addHandler(file_handler)
# Handler is never removed!
```

Calling `run_paradigma()` multiple times in the same Python session will add a new file handler each time without removing the old one. After many runs, this leads to:
- N file handles kept open
- Log output duplicated N times
- File handle exhaustion
- Memory leaks

**Test case that exposes this**:
```python
for i in range(100):
    run_paradigma(...)  # Each call adds a new handler
# After 100 runs, 100+ file handles remain open
```

**Recommendation**:
```python
def run_paradigma(...):
    # At the beginning, remove existing file handlers from paradigma loggers
    for logger_name in ['paradigma', 'paradigma.orchestrator', 'paradigma.pipelines', ...]:
        existing_logger = logging.getLogger(logger_name)
        for handler in existing_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                existing_logger.removeHandler(handler)
                handler.close()

    # Now add the new handler
    file_handler = logging.FileHandler(log_file)
    # ... rest of code
```

---

### 3. **Unsafe Pickle Deserialization** (classification.py, line 112-125)
**Severity**: HIGH (Security)
**Location**: `src/paradigma/classification.py::ClassifierPackage.load()`

**Issue**:
```python
@classmethod
def load(cls, filepath: str | Path):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)  # ⚠️ UNSAFE!
    except Exception as e:
        raise ValueError(f"Failed to load classifier package: {e}") from e
```

`pickle.load()` can execute arbitrary Python code. If a classifier file is compromised or comes from an untrusted source, it could execute malicious code.

**Risk**: Remote code execution if classifier files are obtained from untrusted sources or if a server is compromised and classifier files modified.

**Recommendation** (in priority order):
```python
# Option 1: Use safer alternative (joblib)
import joblib

@classmethod
def load(cls, filepath: str | Path):
    try:
        return joblib.load(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load classifier package: {e}") from e

# Option 2: Add file integrity check (MD5/SHA256 hash verification)
import hashlib
@classmethod
def load(cls, filepath: str | Path, expected_hash: str | None = None):
    if expected_hash:
        actual_hash = hashlib.sha256(open(filepath, 'rb').read()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("Classifier file integrity check failed")
    with open(filepath, "rb") as f:
        return pickle.load(f)

# Option 3: Restrict pickle to safe subset (most conservative)
import pickle
import io

safe_globals = {
    'np': np,
    'pd': pd,
    '__builtins__': {}  # Restrict built-ins
}

@classmethod
def load(cls, filepath: str | Path):
    with open(filepath, "rb") as f:
        return pickle.Unpickler(f).load()  # At minimum, use Unpickler
```

---

## 🟠 MEDIUM PRIORITY ISSUES

### 4. **Pipeline Parameter Validation Too Late** (orchestrator.py, line 86-283)
**Severity**: MEDIUM
**Location**: `src/paradigma/orchestrator.py::run_paradigma()`

**Issue**: Pipeline names are not validated against supported pipelines until after setup. Invalid pipeline names will fail late in execution after directories are created and logging is configured.

**Current flow**:
```
run_paradigma(pipelines=['gait', 'invalid_pipeline'])
  → Setup logging
  → Create output directory
  → Load data
  → ... only now validated
```

**Recommendation**:
```python
def run_paradigma(..., pipelines=None, ...):
    # Add validation first
    if pipelines is None:
        pipelines = ["gait"]
    elif isinstance(pipelines, str):
        pipelines = [pipelines]

    SUPPORTED_PIPELINES = {"gait", "tremor", "pulse_rate"}
    invalid = set(pipelines) - SUPPORTED_PIPELINES
    if invalid:
        raise ValueError(f"Unsupported pipelines: {invalid}. "
                        f"Supported: {SUPPORTED_PIPELINES}")

    # Then continue with setup
```

---

### 5. **Missing Empty DataFrame Handling** (orchestrator.py, line ~600+) + (segmenting.py, line ~210)
**Severity**: MEDIUM
**Location**:
- `src/paradigma/segmenting.py::discard_segments()` line 229-230
- `src/paradigma/orchestrator.py` (result aggregation)

**Issue**:
```python
# segmenting.py
def discard_segments(...):
    df = df[df[segment_nr_colname].map(counts_map) >= min_samples].copy()

    if df.empty:  # ✓ Validates here
        raise ValueError(f"All segments were removed...")

    # But orchestrator doesn't handle this exception:
```

When `discard_segments()` is called from within a pipeline and raises `ValueError`, the orchestrator catches it as a generic error but doesn't provide context about which file/pipeline failed.

**Example failure scenario**:
- File has only short micro-segments (e.g., 0.5s gaps)
- `min_segment_length_s=2.0` filters out all segments
- Exception raised: `"All segments were removed: no segment ≥ X samples"`
- Error logged but continues, results in empty quantifications
- User unclear which file caused the issue

**Recommendation**:
```python
# In orchestrator.py pipeline processing loop:
try:
    quantifications = run_gait_pipeline(...)
except ValueError as e:
    if "All segments were removed" in str(e):
        active_logger.warning(
            f"File {file_name}: No valid segments after filtering. "
            f"Try reducing min_segment_length_s or checking data quality."
        )
        all_results["errors"].append({
            "file": file_name,
            "stage": "segmentation",
            "error": str(e),
            "severity": "warning"
        })
        continue
    else:
        raise  # Re-raise other ValueErrors
```

---

### 6. **Input Validation Gaps in Numeric Parameters** (prepare_data.py, preprocessing.py)
**Severity**: MEDIUM
**Location**: `src/paradigma/prepare_data.py`, `src/paradigma/preprocessing.py`

**Issue**: Functions accept numeric parameters without validating ranges:

```python
def convert_sensor_units(df, accelerometer_units="m/s^2", gyroscope_units="deg/s"):
    # No validation that units are recognized
    # No validation of unit conversion logic for custom units

def resample_data(df, resampling_frequency=100.0, tolerance=None):
    # resampling_frequency not validated > 0
    # tolerance not validated > 0
    # tolerance not validated < 1/sampling_frequency (logical bounds)

def create_segments(time_array, max_segment_gap_s):
    # max_segment_gap_s not validated > 0
    # No check for edge case: max_segment_gap_s larger than entire recording
```

**Recommendation**:
```python
def resample_data(..., resampling_frequency=100.0, tolerance=None, ...):
    if resampling_frequency is None:
        resampling_frequency = 100
        logger.warning("resampling_frequency not provided, set to 100 Hz")

    if not isinstance(resampling_frequency, (int, float)) or resampling_frequency <= 0:
        raise ValueError(f"resampling_frequency must be positive number, got {resampling_frequency}")

    if tolerance is not None and (not isinstance(tolerance, (int, float)) or tolerance < 0):
        raise ValueError(f"tolerance must be non-negative number, got {tolerance}")

    # ... rest of function

def create_segments(time_array, max_segment_gap_s):
    if not isinstance(max_segment_gap_s, (int, float)) or max_segment_gap_s <= 0:
        raise ValueError(f"max_segment_gap_s must be positive, got {max_segment_gap_s}")

    time_range = np.max(time_array) - np.min(time_array)
    if max_segment_gap_s > time_range:
        logger.warning(f"max_segment_gap_s ({max_segment_gap_s}s) > recording duration ({time_range}s). "
                       f"All data will be treated as single segment.")
    # ... rest of function
```

---

### 7. **Missing null/NaN Checks After Data Operations** (processing pipeline)
**Severity**: MEDIUM
**Location**: Multiple files - `feature_extraction.py`, `pipelines/`

**Issue**: After operations like resampling, unit conversion, and filtering, DataFrames may contain NaN values that are not checked. These propagate through the pipeline.

Example:
```python
def resample_data(...):
    # ... interpolation might produce NaN values
    df_resampled = pd.DataFrame(...)  # May have NaN
    return df_resampled  # No validation

# Later in pipeline:
def extract_arm_activity_features(...):
    # Processes df with NaN values
    # Classifier or feature extraction may fail silently or produce unreliable results
```

**Recommendation**:
```python
def resample_data(...):
    # ... resampling code

    # Validate output
    if df_resampled.isnull().any().any():
        n_nans = df_resampled.isnull().sum().sum()
        logger.warning(f"Resampling produced {n_nans} NaN values. "
                      f"Rows: {df_resampled.isnull().any(axis=1).sum()}")

        # Option 1: Forward fill (time-series appropriate)
        df_resampled = df_resampled.fillna(method='ffill').bfill()

        # Option 2: Raise error (stricter)
        # raise ValueError(f"Resampling produced NaN values: {n_nans}")

    return df_resampled
```

---

### 8. **Inconsistent Error Context in Different Pipelines**
**Severity**: MEDIUM
**Location**: `src/paradigma/pipelines/{gait,tremor,pulse_rate}_pipeline.py`

**Issue**: Different pipelines format error dictionaries inconsistently:

**Gait pipeline**:
```python
all_results["errors"].append({
    "file": file_name,           # ✓ Good
    "stage": "loading",          # ✓ Good
    "error": str(e)              # ✓ Good
})
```

**Tremor/Pulse-rate pipelines**:
```python
result_dict["_error"] = f"Classification failed: {str(e)}"  # Different format!
```

This makes uniform error aggregation and reporting difficult.

**Recommendation** (already partially fixed in gait pipeline):
```python
# Use consistent error format across all pipelines
ERROR_TEMPLATE = {
    "pipeline": "tremor",         # Pipeline name
    "file": file_name,            # File being processed
    "stage": "classification",    # Pipeline stage
    "error": str(e),              # Error message
    "timestamp": datetime.now(),  # When error occurred
    "segment_nr": segment_nr,     # If applicable
}

# Return both error dict and _error field for compatibility
result_dict["errors"] = [ERROR_TEMPLATE]
result_dict["_error"] = f"Classification failed: {str(e)}"
```

---

## 🟡 LOW PRIORITY ISSUES

### 9. **Classification.predict_proba Not Scaling Data** (classification.py, line 69-76)
**Severity**: LOW (but can affect predictions)
**Location**: `src/paradigma/classification.py::ClassifierPackage.predict_proba()`

**Issue**:
```python
def predict_proba(self, x) -> float:
    if not self.classifier:
        raise ValueError("Classifier is not loaded.")
    return self.classifier.predict_proba(x)[:, 1]  # x not scaled!
```

If `scaler` is set on the package, data should be scaled before prediction. Currently:
```python
def predict(self, x) -> int:
    ...
    return int(self.predict_proba(x) >= self.threshold)  # Uses unscaled data
```

But `transform_features()` exists but is never called in these methods.

**Recommendation**:
```python
def predict_proba(self, x) -> float:
    if not self.classifier:
        raise ValueError("Classifier is not loaded.")
    x_scaled = self.transform_features(x)  # Scale before prediction
    return self.classifier.predict_proba(x_scaled)[:, 1]

def predict(self, x) -> int:
    if not self.classifier:
        raise ValueError("Classifier is not loaded.")
    x_scaled = self.transform_features(x)
    return int(self.classifier.predict_proba(x_scaled)[:, 1] >= self.threshold)
```

---

### 10. **Deprecated Function Still Used** (segmenting.py)
**Severity**: LOW
**Location**: `src/paradigma/segmenting.py::categorize_segments()` line 324

**Issue**:
```python
@deprecated("This will be removed in v1.1.")
def categorize_segments(df, fs, format="timestamps", window_step_length_s=None):
    ...
```

The function is marked deprecated but:
1. No deprecation warnings issued
2. No clear migration path in docstring
3. Unknown if still used elsewhere

**Recommendation**:
```python
import warnings

def categorize_segments(df, fs, format="timestamps", window_step_length_s=None):
    warnings.warn(
        "categorize_segments() is deprecated and will be removed in v1.1. "
        "Use segment duration directly from df['duration_s'] instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... rest of function
```

---

### 11. **Incomplete Error Handling in Empatica Loader** (load.py, line 97-104)
**Severity**: LOW
**Location**: `src/paradigma/load.py::load_empatica_data()`

**Issue**:
```python
gyro_data = None
if (
    "gyroscope" in empatica_data["rawData"]
    and len(empatica_data["rawData"]["gyroscope"]["x"]) > 0
):
    gyro_data = empatica_data["rawData"]["gyroscope"]
else:
    raise ValueError("Gyroscope data not found in Empatica file.")  # Hard error
```

Raises error if gyro data missing, but user may want to process only accelerometer data.

**Recommendation**:
```python
logger.warning("Gyroscope data not found in Empatica file. "
              "Processing accelerometer data only.")
# Don't raise; let caller handle missing gyroscope
df_data = {...}  # Accelerometer only
return df_data
```

---

## 📋 SUMMARY TABLE

| Issue | Severity | Type | Impact | Fixed? |
|-------|----------|------|--------|--------|
| File handle leaks (Empatica) | HIGH | Resource Leak | Process failure after many runs | ❌ |
| Logger handler leaks | HIGH | Resource Leak | File handle exhaustion | ❌ |
| Unsafe pickle deserialization | HIGH | Security | RCE from malicious classifier | ❌ |
| Missing pipeline validation | MEDIUM | Input Validation | Late failure, poor UX | ❌ |
| Empty DataFrame edge case | MEDIUM | Error Handling | Silent failures or confusing errors | ❌ |
| Missing numeric validation | MEDIUM | Input Validation | Unexpected behavior | ❌ |
| NaN propagation | MEDIUM | Data Quality | Compromised results | ❌ |
| Inconsistent error format | MEDIUM | Code Quality | Difficult error aggregation | ⚠️ (Partial) |
| Unscaled predictions | LOW | Bug | Reduced classification accuracy | ❌ |
| Deprecated function warnings | LOW | Code Quality | Technical debt | ❌ |
| Strict gyro requirement | LOW | Feature Request | Can't process accel-only data | ❌ |

---

## 🎯 RECOMMENDED ACTION PLAN

### Phase 1: Critical Security & Stability (Sprint 1)
1. Fix logger handler leaks (HIGH priority - affects stability)
2. Fix file handle leaks in Empatica loader (HIGH priority)
3. Add safe classifier loading (HIGH security risk)

### Phase 2: Robustness & Data Quality (Sprint 2)
4. Add comprehensive input validation
5. Add NaN detection and handling
6. Handle empty DataFrame edge cases

### Phase 3: Code Quality & Features (Sprint 3)
7. Standardize error formats across pipelines
8. Fix unscaled predictions
9. Add deprecation warnings
10. Make gyroscope optional in sensors

---

## 📝 NOTES

- The 3 error handling improvements from the previous PR (custom_logger parameters) are good foundations for better error context
- Suggest adding a `ErrorAggregator` utility class to standardize error collection across pipelines
- Consider adding pre-flight validation step that checks all inputs before any file operations
