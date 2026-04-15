# High Priority Fixes for ParaDigMa

This document contains code patches for the 3 HIGH severity issues.

## Issue 1: Logger Handler Leaks (orchestrator.py)

**File**: src/paradigma/orchestrator.py
**Current Code** (lines ~320-343):
```python
# Configure package-wide logging level for all paradigma modules
if custom_logger is None:
    package_logger.setLevel(logging_level)

...

# Setup logging to file - add handler to package logger so ALL paradigma modules
# log to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = output_dir / f"paradigma_run_{timestamp}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
package_logger.addHandler(file_handler)
active_logger.info(f"Logging to {log_file}")
```

**Patch needed** (add cleanup before adding new handler):
```python
# Configure package-wide logging level for all paradigma modules
if custom_logger is None:
    package_logger.setLevel(logging_level)

    # CLEANUP: Remove old file handlers from previous runs
    # This prevents file handle leaks when run_paradigma is called multiple times
    for handler in package_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            package_logger.removeHandler(handler)

...

# Setup logging to file - add handler to package logger so ALL paradigma modules
# log to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = output_dir / f"paradigma_run_{timestamp}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
package_logger.addHandler(file_handler)
active_logger.info(f"Logging to {log_file}")
```

**Implementation notes**:
- Add at line ~328 (after `if custom_logger is None` block)
- Use `isinstance(handler, logging.FileHandler)` to check type
- Must iterate over copy of list (`handlers[:]`) to safely remove during iteration
- Call `handler.close()` to properly release file handle

---

## Issue 2: File Handle Leak in Empatica Loader (load.py)

**File**: src/paradigma/load.py
**Current Code** (lines ~81-96):
```python
def load_empatica_data(file_path: str | Path) -> pd.DataFrame:
    """Load Empatica .avro file."""
    file_path = Path(file_path)
    logger.info(f"Loading Empatica data from {file_path}")

    with open(file_path, "rb") as f:
        reader = DataFileReader(f, DatumReader())
        empatica_data = next(reader)
    # ^^^ File closes here but reader may not be fully consumed

    accel_data = empatica_data["rawData"]["accelerometer"]

    # Check for gyroscope data
    gyro_data = None
    if (
        "gyroscope" in empatica_data["rawData"]
        and len(empatica_data["rawData"]["gyroscope"]["x"]) > 0
    ):
        gyro_data = empatica_data["rawData"]["gyroscope"]
    else:
        raise ValueError("Gyroscope data not found in Empatica file.")
    # ...rest of function
```

**Patch needed** (explicitly close reader):
```python
def load_empatica_data(file_path: str | Path) -> pd.DataFrame:
    """Load Empatica .avro file."""
    file_path = Path(file_path)
    logger.info(f"Loading Empatica data from {file_path}")

    with open(file_path, "rb") as f:
        reader = DataFileReader(f, DatumReader())
        try:
            empatica_data = next(reader)
        finally:
            reader.close()  # NEW: Explicitly close reader

    accel_data = empatica_data["rawData"]["accelerometer"]

    # Check for gyroscope data (OPTIONAL: make it optional)
    gyro_data = None
    if (
        "gyroscope" in empatica_data["rawData"]
        and len(empatica_data["rawData"]["gyroscope"]["x"]) > 0
    ):
        gyro_data = empatica_data["rawData"]["gyroscope"]
    else:
        logger.warning("Gyroscope data not found in Empatica file. "
                      "Processing accelerometer data only.")
    # ...rest of function
```

**Alternative: Use context manager if available**:
- Check if DataFileReader supports context manager protocol
- If yes, use: `with DataFileReader(...) as reader:`

**Implementation notes**:
- Add `reader.close()` in finally block to ensure cleanup
- Consider changing hard `raise ValueError` to warning
- Test with large Empatica files to verify fix

---

## Issue 3: Unsafe Pickle Deserialization (classification.py)

**File**: src/paradigma/classification.py
**Current Code** (lines ~112-125):
```python
@classmethod
def load(cls, filepath: str | Path):
    """
    Load a ClassifierPackage from a file.
    ...
    """
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)  # ⚠️ UNSAFE!
    except Exception as e:
        raise ValueError(f"Failed to load classifier package: {e}") from e
```

**Recommended Patch** (use joblib - safer & more efficient):
```python
# At top of file
import joblib  # Add to imports

@classmethod
def load(cls, filepath: str | Path):
    """
    Load a ClassifierPackage from a file.

    Uses joblib instead of pickle for safer deserialization
    of sklearn estimators.

    Parameters
    ----------
    filepath : str or Path
        Path to classifier package file (.pkl or .joblib)

    Returns
    -------
    ClassifierPackage
        Loaded classifier package

    Raises
    ------
    ValueError
        If file cannot be loaded or is corrupted
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Classifier file not found: {filepath}")

    try:
        return joblib.load(filepath)
    except Exception as e:
        raise ValueError(
            f"Failed to load classifier package from {filepath}: {e}"
        ) from e
```

**Alternative Patch if joblib not available** (restrict pickle):
```python
import pickle
import io

@classmethod
def load(cls, filepath: str | Path):
    """
    Load a ClassifierPackage from a file with restricted pickle.

    Uses RestrictedUnpickler to prevent arbitrary code execution.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Classifier file not found: {filepath}")

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Restrict to safe sklearn classes only
            ALLOWED_MODULES = {
                'sklearn', 'numpy', 'pandas', 'scipy',
                'paradigma.classification'
            }
            if not any(module.startswith(m) for m in ALLOWED_MODULES):
                raise pickle.UnpicklingError(
                    f"Attempted to unpickle unsafe module: {module}"
                )
            return super().find_class(module, name)

    try:
        with open(filepath, "rb") as f:
            return RestrictedUnpickler(f).load()
    except Exception as e:
        raise ValueError(
            f"Failed to load classifier package from {filepath}: {e}"
        ) from e
```

**Implementation notes**:
- `joblib.load()` is known to be safer for sklearn objects
- If using restricted unpickler, maintain list of allowed modules
- Add file existence check before attempting load
- Provide clear error messages for debugging

---

## Testing the Fixes

### Test 1: Logger Handler Leak Fix
```python
import tempfile
import os

def test_logger_leak_fix():
    """Test that multiple calls don't leak file handles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        initial_handles = len(os.listdir(f"/proc/{os.getpid()}/fd")) if os.path.exists(f"/proc/{os.getpid()}/fd") else None

        # Call run_paradigma 10 times
        for i in range(10):
            result = run_paradigma(
                dfs=create_test_data(),
                output_dir=tmpdir,
                pipelines=["gait"],
            )

        # Check file handles didn't grow
        if initial_handles:
            final_handles = len(os.listdir(f"/proc/{os.getpid()}/fd"))
            # Should not grow by ~10 handles
            assert final_handles - initial_handles < 5, f"File handles grew: {initial_handles} -> {final_handles}"
```

### Test 2: File Handle Leak Fix (Empatica)
```python
def test_empatica_handle_leak():
    """Test Empatica loader doesn't leak handles."""
    import psutil
    process = psutil.Process()

    for i in range(100):
        df = load_empatica_data("test_data.avro")

    # Check open file count (should stay relatively stable)
    open_files = len(process.open_files())
    assert open_files < 50, f"Too many open files: {open_files}"
```

### Test 3: Pickle Safety Fix
```python
def test_pickle_safety():
    """Test that unsafe pickles are rejected."""
    import pickle

    # Create a malicious pickle
    class MaliciousClass:
        def __reduce__(self):
            import os
            return (os.system, ('echo pwned',))

    bad_pickle = pickle.dumps(MaliciousClass())

    with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
        f.write(bad_pickle)
        f.flush()

        # Should raise or reject
        with pytest.raises((ValueError, pickle.UnpicklingError)):
            ClassifierPackage.load(f.name)
```

---

## Deployment Order

1. **First**: Deploy logger handler leak fix (low risk, high impact)
2. **Second**: Deploy Empatica file handle fix (low risk, needed for stability)
3. **Third**: Deploy pickle safety fix (medium risk, high security impact)

---

## Verification Checklist

- [ ] All three fixes deployed and tests passing
- [ ] Code review by team member
- [ ] Stress test with multiple sequential runs
- [ ] Monitor file handle counts in production
- [ ] Update CHANGELOG.md with security fix note
