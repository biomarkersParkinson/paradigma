"""
Simpler backward compatibility check focusing on the function signatures and return types.
"""
import inspect
from paradigma.load import load_single_data_file


def show_function_signature():
    """Display the function signatures."""
    print("\n" + "="*70)
    print("FUNCTION SIGNATURE ANALYSIS")
    print("="*70)
    
    sig = inspect.signature(load_single_data_file)
    print(f"\nload_single_data_file signature:")
    print(f"  {sig}")
    
    # Get return annotation
    return_annotation = sig.return_annotation
    print(f"\nReturn type annotation: {return_annotation}")
    
    # Check docstring
    doc = load_single_data_file.__doc__
    if doc:
        print(f"\nDocstring (first 500 chars):")
        print(doc[:500])


def test_return_value():
    """Test what the function actually returns."""
    from pathlib import Path
    
    print("\n" + "="*70)
    print("ACTUAL RETURN VALUE TEST")
    print("="*70)
    
    meta_file = Path("example_data/verily/segment0001_meta.json")
    
    result = load_single_data_file(meta_file)
    
    print(f"\nReturn value type: {type(result)}")
    print(f"Return value length: {len(result)}")
    print(f"Return value element types: {[type(x).__name__ for x in result]}")
    
    file_key, df, start_dt = result
    
    print(f"\n1. file_key: {repr(file_key)}")
    print(f"   Type: {type(file_key).__name__}")
    
    print(f"\n2. df:")
    print(f"   Type: {type(df).__name__}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    print(f"\n3. start_dt: {repr(start_dt)}")
    print(f"   Type: {type(start_dt).__name__}")
    print(f"   Value: {start_dt}")


if __name__ == "__main__":
    show_function_signature()
    test_return_value()
    
    print("\n" + "="*70)
    print("SUMMARY: The new implementation returns 3 values including datetime")
    print("="*70)
