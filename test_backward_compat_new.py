"""
Backward compatibility check: Compare old vs new behavior.
This script tests the same data with both versions.
"""
import json
import tempfile
from pathlib import Path
from datetime import datetime

def test_load_single_data_file():
    """Test load_single_data_file with both versions."""
    from paradigma.load import load_single_data_file
    
    meta_file = Path("example_data/verily/imu/imu_segment0001_meta.json")
    
    print("\n" + "="*70)
    print("TESTING load_single_data_file()")
    print("="*70)
    
    file_key, df, start_dt = load_single_data_file(meta_file)
    
    print(f"\nFile key: {file_key}")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head(3))
    print(f"\nStart datetime: {start_dt}")
    print(f"Start datetime type: {type(start_dt).__name__}")
    
    return file_key, df, start_dt


def test_run_gait_pipeline():
    """Test run_gait_pipeline with and without datetime."""
    from paradigma.load import load_single_data_file
    from paradigma.pipelines.gait_pipeline import run_gait_pipeline
    
    meta_file = Path("example_data/verily/imu/imu_segment0001_meta.json")
    file_key, df, start_dt = load_single_data_file(meta_file)
    
    print("\n" + "="*70)
    print("TESTING run_gait_pipeline() WITH DATETIME")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results, metadata_dict = run_gait_pipeline(
            df,
            watch_side="left",
            output_dir=tmpdir,
            start_dt=start_dt
        )
    
    if metadata_dict and "Arm_Swing_Parameters" in metadata_dict:
        metadata_with_dt = metadata_dict["Arm_Swing_Parameters"]
        print("\nMetadata WITH datetime:")
        print(json.dumps(metadata_with_dt, indent=2, default=str))
    else:
        print("No arm swing parameters found")
        metadata_with_dt = None
    
    print("\n" + "="*70)
    print("TESTING run_gait_pipeline() WITHOUT DATETIME")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results, metadata_dict = run_gait_pipeline(
            df,
            watch_side="left",
            output_dir=tmpdir,
            start_dt=None
        )
    
    if metadata_dict and "Arm_Swing_Parameters" in metadata_dict:
        metadata_no_dt = metadata_dict["Arm_Swing_Parameters"]
        print("\nMetadata WITHOUT datetime (start_dt=None):")
        print(json.dumps(metadata_no_dt, indent=2, default=str))
    else:
        print("No arm swing parameters found")
        metadata_no_dt = None
    
    # Compare
    if metadata_with_dt and metadata_no_dt:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        
        print("\n1. WITH datetime:")
        if "combined" in metadata_with_dt:
            combined = metadata_with_dt["combined"]
            print(f"   'combined' keys: {list(combined.keys())}")
        
        if "per_segment" in metadata_with_dt and len(metadata_with_dt["per_segment"]) > 0:
            seg1 = metadata_with_dt["per_segment"].get("1", {})
            print(f"   'per_segment[1]' keys: {list(seg1.keys())}")
        
        print("\n2. WITHOUT datetime:")
        if "combined" in metadata_no_dt:
            combined = metadata_no_dt["combined"]
            print(f"   'combined' keys: {list(combined.keys())}")
        
        if "per_segment" in metadata_no_dt and len(metadata_no_dt["per_segment"]) > 0:
            seg1 = metadata_no_dt["per_segment"].get("1", {})
            print(f"   'per_segment[1]' keys: {list(seg1.keys())}")
        
        print("\n3. Key differences:")
        with_dt_keys = set(metadata_with_dt.get("combined", {}).keys())
        no_dt_keys = set(metadata_no_dt.get("combined", {}).keys())
        
        added = with_dt_keys - no_dt_keys
        removed = no_dt_keys - with_dt_keys
        
        if added:
            print(f"   ✓ Added with datetime: {added}")
        if removed:
            print(f"   ✗ Removed with datetime: {removed}")
        if not added and not removed:
            print(f"   No difference in 'combined' keys")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# BACKWARD COMPATIBILITY TEST - NEW VERSION")
    print("#"*70)
    
    file_key, df, start_dt = test_load_single_data_file()
    test_run_gait_pipeline()
    
    print("\n" + "#"*70)
    print("# END OF NEW VERSION TEST")
    print("#"*70)
