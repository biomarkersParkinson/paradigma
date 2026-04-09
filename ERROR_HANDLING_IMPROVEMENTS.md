# Error Handling and Logging Improvements

## Overview
This document describes improvements to error handling and logging in the ParaDigMa pipeline, specifically addressing issues where subset data failures could silently propagate or lack proper context for debugging.

## Changes Made

### 1. Fixed Silent Failure in Feature Extraction
**File**: `src/paradigma/pipelines/gait_pipeline.py`
**Function**: `extract_arm_activity_features()`
**Issue**: When no windows could be created from gait data, the function returned an empty DataFrame instead of raising an error.

**Before**:
```python
if not windowed_data:
    print("No windows were created from the given data.")
    return pd.DataFrame()
```

**After**:
```python
if not windowed_data:
    error_msg = (
        "No windows were created from the given data. This can occur when all "
        "gait segments are too short after filtering (minimum window size required). "
        f"Input data shape: {df.shape}, Number of segments processed: {len(df_grouped)}"
    )
    active_logger.error(error_msg)
    raise ValueError(error_msg)
```

**Impact**: Errors in feature extraction now properly propagate and are logged with context, preventing silent failures.

---

### 2. Replaced All print() Statements with Logger Calls
**File**: `src/paradigma/pipelines/gait_pipeline.py`
**Function**: `quantify_arm_swing()`
**Issues**:
- Per-segment computing errors (RoM, PAV) used `print()` instead of logging
- Errors were not captured in logfiles
- Missing context about segment type and duration

**Before**:
```python
try:
    rom = compute_range_of_motion(...)
except Exception as e:
    print(f"Error computing range of motion for segment {segment_nr}: {e}")
    rom = np.array([np.nan])
```

**After**:
```python
try:
    rom = compute_range_of_motion(...)
except Exception as e:
    active_logger.warning(
        f"Error computing range of motion for segment {segment_nr}: {e}. "
        f"Setting to NaN. (filtered={filtered}, duration={len(group[DataColumns.TIME]) / fs:.2f}s)"
    )
    rom = np.array([np.nan])
```

**Impact**: All errors now logged to file with full context including segment type and duration.

---

### 3. Added Logging Context Throughout Pipeline
**Modifications**:
- Added `custom_logger` parameter to `extract_arm_activity_features()`
- Added `custom_logger` parameter to `quantify_arm_swing()`
- Updated all calls in `run_gait_pipeline()` to pass logger context
- Logger instance initialized in each function: `active_logger = custom_logger if custom_logger is not None else logger`

**Impact**: Consistent logging throughout the pipeline with proper context passed through call chain.

---

## Error Context Available for Traceability

### At File/Subject Level (Orchestrator)
```
Processing file 1/5: subject_001.CWA
  - File name ✓
  - File index ✓
  - Pipeline name ✓
  - Stage (loading, preparation, pipeline_execution, aggregation) ✓
```

### At Pipeline Execution Level
```
Running gait pipeline on subject_001.CWA
  - File context inherited from orchestrator ✓
  - Pipeline step (preprocessing, classification, quantification) ✓
  - Error type ✓
```

### At Segment Level (New)
```
Error computing range of motion for segment 3: <error details>. Setting to NaN. (filtered=True, duration=2.34s)
  - Segment number ✓
  - Segment type (filtered vs unfiltered) ✓ **NEW**
  - Segment duration ✓ **NEW**
  - Error details ✓
```

### At Feature Extraction Level (New)
```
Error: No windows were created from the given data. Input data shape: (1024, 8), Number of segments processed: 3
  - Data shape ✓ **NEW**
  - Number of segments attempted ✓ **NEW**
  - Root cause explanation ✓ **NEW**
```

---

## Error Handling Flow

### Before Changes
```
Data Loading
    ↓
Preparation
    ↓
Pipeline Execution
    ├─ Preprocessing      [Error logged at orchestrator level]
    ├─ Classification
    │  ├─ Gait Detection
    │  ├─ Arm Activity Extraction    [ERROR ← Silent return of empty DataFrame]
    │  └─ Arm Activity Classification
    └─ Quantification               [May fail with confusing error]
```

### After Changes
```
Data Loading
    ↓
Preparation
    ↓
Pipeline Execution
    ├─ Preprocessing      [Error logged with context]
    ├─ Classification
    │  ├─ Gait Detection
    │  ├─ Arm Activity Extraction    [ERROR → ValueError with context, logged]
    │  └─ Arm Activity Classification
    └─ Quantification                [Per-segment errors logged with context]
        ├─ RoM Computation           [Segment error logged with filters/duration]
        └─ PAV Computation           [Segment error logged with filters/duration]
```

---

## Log Message Examples

### Feature Extraction Failure
```
ERROR: No windows were created from the given data. This can occur when all gait segments are too short after filtering (minimum window size required). Input data shape: (512, 8), Number of segments processed: 8
```

### Segment-Level RoM Computation Error
```
WARNING: Error computing range of motion for segment 5: (error details). Setting to NaN. (filtered=True, duration=3.45s)
```

### Complete Pipeline Failure (with context)
```
ERROR: Gait pipeline failed on subject_001.CWA: Classification failed: No windows were created from the given data...
```

---

## Testing Recommendations

### Test Case 1: Short Segments After Filtering
- Input: Data with brief gait bouts
- Condition: All segments filtered to < 1.5s
- Expected: ValueError with context at extraction stage, logged to file
- Verification: Check log for detailed error message

### Test Case 2: Per-Segment RoM Failure
- Input: Data with invalid angle extrema
- Condition: compute_range_of_motion() throws exception
- Expected: Warning logged, segment gets NaN for RoM, processing continues
- Verification: Check log has segment number, type, duration

### Test Case 3: Multi-File Processing with Mixed Results
- Input: 3 files, middle one fails at feature extraction
- Expected: File 1 succeeds, File 2 fails with logged context, File 3 continues
- Verification: Error dict contains file name, stage, and message

---

## Benefits

1. **Improved Debuggability**: Full context available for each error
2. **No Silent Failures**: Feature extraction now properly signals errors
3. **Consistent Logging**: All errors go to log file, not just console
4. **Hierarchical Context**: Subject → File → Segment → Step
5. **Better User Experience**: Users can trace any error back to specific data subset

---

## Backward Compatibility

- All changes are backward compatible
- Logger parameter is optional with sensible defaults
- Empty fixture validation still works for filtering and discard operations
- Existing error handling at orchestrator level unchanged

---

## Files Modified

1. `src/paradigma/pipelines/gait_pipeline.py`
   - `extract_arm_activity_features()`: +logger param, +error handling
   - `quantify_arm_swing()`: +logger param, +enhanced logging
   - Calls in `run_gait_pipeline()`: Updated to pass logger

---

## Future Improvements

1. Add structured logging (JSON format) for easier automated parsing
2. Add segment-level metadata (start/end time, sample count) to error logs
3. Create error summary report at end of processing
4. Add optional error recovery strategies per error type
5. Track which segments succeeded/failed for partial result handling
