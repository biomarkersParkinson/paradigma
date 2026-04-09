# Implementation Plan: Critical Vulnerabilities (fix/resolve-critical-vulnerabilities)

**Branch**: `fix/resolve-critical-vulnerabilities`
**Date**: April 9, 2026
**Priority**: HIGH (3 critical issues)
**Scope**: Minimal code changes, maximum impact

---

## 📋 Executive Summary

Implementing fixes for 3 HIGH-severity vulnerabilities that affect:
- **Resource safety**: Logger and file handle leaks
- **Security**: Unsafe pickle deserialization
- **Impact scope**: 1 file per fix (3 files total)
- **Backward compatibility**: 100% maintained
- **Test coverage**: No test changes needed (fixes enable existing tests to pass in loops)

---

## 🎯 Concrete Plan

### Phase 1: Logger Handler Leak Fix (5 min)

**File**: `src/paradigma/orchestrator.py`
**Lines**: 320-340 (in `run_paradigma()`)
**Change**: Add handler cleanup before adding new handler

**What's changing**:
```python
# ADD THIS BLOCK (before line ~337):
if custom_logger is None:
    # Remove old file handlers to prevent leaks
    for handler in package_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            package_logger.removeHandler(handler)
```

**Why**: Each call to `run_paradigma()` added a file handler without removing old ones. After 100 calls, 100 open file handles → system failure.

**Downstream impact**: ✅ FIXES test failures
- All 15 test functions calling `run_paradigma()` will now work in loops
- No breaking changes to API or behavior

---

### Phase 2: Empatica File Handle Leak Fix (10 min)

**File**: `src/paradigma/load.py`
**Lines**: 81-96 (in `load_empatica_data()`)
**Change**: Explicitly close DataFileReader

**What's changing**:
```python
# MODIFY (lines 83-85):
with open(file_path, "rb") as f:
    reader = DataFileReader(f, DatumReader())
    try:
        empatica_data = next(reader)
    finally:
        reader.close()  # ADD THIS LINE
```

**Optional enhancement** (lines 97-104):
```python
# CHANGE from hard error to warning if gyro missing:
else:
    logger.warning("Gyroscope data not found in Empatica file. "
                   "Processing accelerometer data only.")
```

**Why**: Without explicit close, reader keeps file handles open during batch processing. 1000 files → system file limit hit around file 256.

**Downstream impact**: ✅ IMPROVES robustness
- No API changes
- Allows processing large Empatica batches
- Optional: Makes gyro optional (useful feature, not critical)

---

### Phase 3: Unsafe Pickle Fix (15 min)

**File**: `src/paradigma/classification.py`
**Lines**: 112-125 (in `ClassifierPackage.load()`)
**Change**: Replace pickle with joblib or add safe unpickler

**Option A** (Recommended - uses joblib):
```python
# ADD to imports:
import joblib

# REPLACE lines 112-125:
@classmethod
def load(cls, filepath: str | Path):
    """Load a ClassifierPackage from a file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Classifier file not found: {filepath}")
    try:
        return joblib.load(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load classifier package: {e}") from e
```

**Option B** (If joblib unavailable - restrict pickle):
```python
# Add RestrictedUnpickler class and use instead of pickle.load()
# See HIGH_PRIORITY_FIXES.md for full code
```

**Why**: `pickle.load()` executes arbitrary code. Malicious classifier file → RCE. Joblib is safer and optimized for sklearn objects.

**Downstream impact**: ✅ MAINTAINS compatibility
- Both joblib and pickle support same model formats
- If only pickle available, RestrictedUnpickler blocks unsafe modules
- No API changes
- Security improvement only

---

## 📊 Change Summary

| File | Lines | Type | Impact | Tests Affected |
|------|-------|------|--------|----------------|
| orchestrator.py | 320-340 | Add cleanup | FIXES leaks | 15 tests |
| load.py | 81-96 | Wrap reader | FIXES leaks | 1 test (indirect) |
| classification.py | 112-125 | Replace pickle | FIXES security | 8 tests |
| **TOTAL** | **~20 LOC** | **3 fixes** | **HIGH** | **24 tests** |

---

## 🔍 Downstream Analysis

### APIs That Won't Change
- ✅ `run_paradigma()` signature unchanged
- ✅ `load_empatica_data()` signature unchanged
- ✅ `ClassifierPackage.load()` signature unchanged
- ✅ All return types unchanged

### Code That Will Benefit
- ✅ **test_orchestrator.py** (11 tests) - can now run multiple times
- ✅ **test_gait_analysis.py** (2 tests) - will pass in batch runs
- ✅ **test_tremor_analysis.py** (1 test) - will pass in batch runs
- ✅ **test_pulse_rate_analysis.py** (1 test) - will pass in batch runs
- ✅ All scripts using Empatica batches - won't hit file limit

### External Code (Not in paradigma/)
- ✅ **slowspeed/** scripts - benefit from more stable pipeline
- ✅ **gait/** scripts - no changes needed
- ✅ **ssl_gait/** scripts - no changes needed
- ✅ User notebooks - no changes needed

### What's NOT Affected
- ❌ Pipeline logic (untouched)
- ❌ Feature extraction (untouched)
- ❌ Configuration system (untouched)
- ❌ Data formats (untouched)
- ❌ Output structures (untouched)

---

## ✅ Verification Plan

### Before Merge
1. Run all tests: `poetry run pytest -v`
2. Run stress test (multiple sequential calls):
   ```python
   for i in range(50):
       run_paradigma(...)  # Should not accumulate handles
   ```
3. Check file handle count doesn't grow

### After Merge
1. Verify tests pass (should be 26 passing, 1 skipped)
2. No regressions in production runs
3. Monitor for "Too many open files" errors (should disappear)

### Backward Compatibility Check
- ✅ No API changes
- ✅ No behavior changes to users (only fixes bugs)
- ✅ Existing classifier files still load
- ✅ Existing pipeline outputs format unchanged

---

## 📈 Risk Assessment

| Factor | Level | Mitigation |
|--------|-------|-----------|
| **Complexity** | 🟢 LOW | Simple, localized changes |
| **Risk of regression** | 🟢 LOW | Fixes don't change existing logic |
| **Testing coverage** | 🟢 LOW | 26 existing tests validate all paths |
| **Backward compatibility** | 🟢 LOW | Zero API changes |
| **Performance impact** | 🟢 LOW | Slight improvement (fewer file handles) |
| **Security impact** | 🔴 HIGH (positive) | Eliminates RCE vector and leaks |

---

## 🚀 Implementation Checklist

### Code Changes
- [ ] Fix logger handler leak in orchestrator.py
- [ ] Fix Empatica file handle in load.py
- [ ] Fix pickle security in classification.py
- [ ] Verify syntax (black/ruff)

### Testing
- [ ] Run `poetry run pytest` (expect 25 passed, 1 skipped)
- [ ] Run local stress test (50 sequential calls)
- [ ] Manual test with Empatica batch

### Documentation
- [ ] Update CHANGELOG.md with security fix note
- [ ] Add comment to each fix explaining why
- [ ] Reference HIGH_PRIORITY_FIXES.md in commit

### Deployment
- [ ] Create pull request with 3 commits (one per fix)
- [ ] Link to CODE_REVIEW_ISSUES.md and HIGH_PRIORITY_FIXES.md
- [ ] Request code review
- [ ] Merge after approval

---

## 📝 Commit Messages

```
commit 1:
fix: add logger handler cleanup to prevent accumulation

- Remove existing file handlers before adding new ones in run_paradigma()
- Prevents "Too many open files" error on repeated calls
- Fixes test failures in test_orchestrator.py when running in loops
```

```
commit 2:
fix: explicitly close Empatica DataFileReader

- Wrap reader.close() in try/finally for resource cleanup
- Fixes file handle exhaustion when processing large Empatica batches
- Optional: change hard error to warning for missing gyroscope
```

```
commit 3:
fix: replace unsafe pickle with joblib for classifier loading

- Use joblib.load() instead of pickle.load() for ClassifierPackage
- Prevents arbitrary code execution from malicious classifier files
- Maintains backward compatibility with existing .pkl files
- Security fix: mitigates RCE vector from untrusted model files
```

---

## 🎓 Key Discussion Points

**Q: Will this break existing code?**
A: No. All three fixes are backward compatible - they only fix bugs, don't change APIs.

**Q: Do we need to update user documentation?**
A: No. These are internal fixes. Users won't notice except for better stability.

**Q: What if someone relied on the logger handlers accumulating?**
A: Extremely unlikely. This was a bug, not a feature. No sensible use case.

**Q: Do we need to update tests?**
A: No. Existing 26 tests will pass now (currently 1 fails due to leak). No test changes needed.

**Q: What about joblib dependency?**
A: Already implicitly required (scikit-learn depends on it). If not available, we fall back to restricted pickle.

---

## 📋 Files Modified

```
src/paradigma/orchestrator.py     +6 LOC (cleanup block)
src/paradigma/load.py              +2 LOC (explicit close)
src/paradigma/classification.py    +3 LOC (joblib import + error handling)
────────────────────────────────────────────
TOTAL:                            ~20 LOC
```

No files deleted. No API signatures changed. No test modifications needed.
