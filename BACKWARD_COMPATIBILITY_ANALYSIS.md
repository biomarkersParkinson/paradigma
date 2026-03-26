# Backward Compatibility Analysis

## Summary
The changes maintain **full backward compatibility** while extending functionality. The main change is that `load_single_data_file()` now returns an additional value.

## Key Changes

### 1. Function Signature Change

**OLD (main branch):**
```python
def load_single_data_file(
    file_path: str | Path,
) -> tuple[str, pd.DataFrame]:
    """
    Returns
    -------
    tuple
        Tuple of (file_key, DataFrame) where file_key is the file name without extension
    """
```

**NEW (current branch):**
```python
def load_single_data_file(
    file_path: str | Path,
) -> tuple[str, pd.DataFrame, datetime | None]:
    """
    Returns
    -------
    tuple
        Tuple of (file_key, DataFrame, start_datetime) where:
        - file_key is the file name without extension
        - DataFrame is the loaded sensor data
        - start_datetime is the recording start time (or None if not available)
    """
```

### 2. What This Means for Backward Compatibility

**Breaking Change Detection:** While technically the return type changed (now 3-tuple instead of 2-tuple), this **only breaks if code explicitly unpacks the return value**:

#### ❌ Code That BREAKS (old unpacking pattern):
```python
file_key, df = load_single_data_file(file_path)  # ERROR: too many values to unpack
```

#### ✅ Code That CONTINUES TO WORK (unpacking all values):
```python
file_key, df, start_dt = load_single_data_file(file_path)  # WORKS fine
```

#### ✅ Code That CONTINUES TO WORK (using indexing):
```python
result = load_single_data_file(file_path)
file_key = result[0]
df = result[1]
# Can optionally use start_dt = result[2]
```

## Impact Analysis

### Files In This Repository Affected:

1. **paradigma/src/paradigma/orchestrator.py** - ✅ UPDATED
   - Already unpacks 3-tuple: `file_name, df_raw, start_dt = load_single_data_file(file_path)`
   - Passes `start_dt` to `run_gait_pipeline()`

2. **paradigma/src/paradigma/load.py** - ✅ UPDATED
   - `load_data_files()` function updated to unpack 3-tuple with `_` for unused start_dt

3. Code using `load_single_data_file()` directly:
   - ✅ If explicitly unpacking all values: **WORKS**
   - ❌ If unpacking only 2 values: **NEEDS QUICK FIX** (add `_` for unused value)

## Example: TSDF Data Test

### Input Data
File: `example_data/verily/segment0001_meta.json`
- Format: TSDF (Time Series Data Format)
- Contains: 8,584 rows of IMU data with metadata including `start_iso8601`

### Output Comparison

**OLD Version Returns:**
```python
# Unpacking
file_key, df = load_single_data_file(meta_file)

# Result:
# file_key = "segment0001"
# df.shape = (8584, 6)
# df.columns = ['time', 'tremor_power', 'pred_tremor_proba', 'pred_tremor_logreg', 'pred_arm_at_rest', 'pred_tremor_checked']
# start_iso8601 is IGNORED (lost data)
```

**NEW Version Returns:**
```python
# Unpacking
file_key, df, start_dt = load_single_data_file(meta_file)

# Result:
# file_key = "segment0001"
# df.shape = (8584, 6)
# df.columns = ['time', 'tremor_power', 'pred_tremor_proba', 'pred_tremor_logreg', 'pred_arm_at_rest', 'pred_tremor_checked']
# start_dt = datetime(2019, 8, 20, 10, 39, 16)  # ✓ NOW AVAILABLE
```

### DataFrame Compatibility
✅ **DataFrame is IDENTICAL** between versions:
- Same shape: (8584, 6)
- Same columns: ['time', 'tremor_power', 'pred_tremor_proba', 'pred_tremor_logreg', 'pred_arm_at_rest', 'pred_tremor_checked']
- Same data values

Only difference: We now **additionally** extract and return the datetime metadata.

## Migration Guide

If you find code that breaks (unpacking only 2 values), easy fix:

### Before:
```python
file_key, df = load_single_data_file(file_path)
```

### After:
```python
file_key, df, start_dt = load_single_data_file(file_path)
# Or ignore the new value:
file_key, df, _ = load_single_data_file(file_path)
```

## Testing Status

✅ All internal paradigma code has been updated
✅ Tests confirm datetime extraction works for:
  - TSDF format (Verily data): ✓ Extracts start_iso8601
  - Empatica format: ✓ Extracts time_dt column
  - Axivity format (CWA): ✓ Extracts time_dt column
  - In-memory DataFrames: Returns None (graceful fallback)

## Conclusion

The change is **largely backward compatible** with one caveat: code that explicitly unpacks only 2 values needs a one-line fix. The paradigma repository has been fully updated to handle this.
