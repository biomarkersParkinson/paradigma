"""Debug script to understand the unpacking issue."""

from pathlib import Path

from paradigma.load import load_single_data_file

meta_file = Path("example_data/verily/segment0001_meta.json")

print(f"Loading: {meta_file}")
print(f"File exists: {meta_file.exists()}")
print(f"File suffix: {meta_file.suffix}")
print(f"File stem: {meta_file.stem}")

try:
    result = load_single_data_file(meta_file)
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result) if isinstance(result, tuple) else 'N/A'}")
    print(
        f"Result: {[type(r).__name__ for r in result] if isinstance(result, tuple) else result}"
    )

    file_key, df, start_dt = result
    print("✓ Successfully unpacked!")
    print(f"  file_key: {file_key}")
    print(f"  df shape: {df.shape}")
    print(f"  start_dt: {start_dt}")
except Exception as e:
    print(f"✗ Error unpacking: {type(e).__name__}: {e}")
