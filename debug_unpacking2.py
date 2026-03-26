"""Debug script to understand the unpacking issue - with fresh import."""

import sys

# Remove any cached modules
if "paradigma" in sys.modules:
    del sys.modules["paradigma"]
if "paradigma.load" in sys.modules:
    del sys.modules["paradigma.load"]

# Now import fresh
import importlib
from pathlib import Path

import paradigma.load

importlib.reload(paradigma.load)
from paradigma.load import load_single_data_file

meta_file = Path("example_data/verily/segment0001_meta.json")

print(f"Loading: {meta_file}")

try:
    result = load_single_data_file(meta_file)
    print(f"Result length: {len(result) if isinstance(result, tuple) else 'N/A'}")
    print(
        f"Result types: {[type(r).__name__ for r in result] if isinstance(result, tuple) else result}"
    )

    if len(result) == 3:
        file_key, df, start_dt = result
        print("SUCCESS!")
        print(f"  file_key: {file_key}")
        print(f"start_dt: {start_dt}")
    elif len(result) == 2:
        file_key, df = result
        print("ERROR: Only 2 values returned (no start_dt)")
        print(f"  file_key: {file_key}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
