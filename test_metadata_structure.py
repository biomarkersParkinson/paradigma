"""Simple test to verify metadata structure with/without datetime."""

import json
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from paradigma.pipelines.gait_pipeline import run_gait_pipeline


def create_dummy_data(duration_seconds=30):
    """Create synthetic IMU data for testing."""
    sampling_rate = 100  # Hz
    n_samples = duration_seconds * sampling_rate
    time = np.linspace(0, duration_seconds, n_samples)

    # Create multi-frequency gait-like signal
    accel_x = 2 * np.sin(
        2 * np.pi * 0.5 * time
    ) + 0.5 * np.sin(  # 0.5 Hz gait frequency
        2 * np.pi * 2 * time
    )  # 2 Hz arm swing
    accel_y = 2 * np.cos(2 * np.pi * 0.5 * time) + 0.5 * np.cos(2 * np.pi * 2 * time)
    accel_z = 10 + 0.1 * np.sin(2 * np.pi * 2 * time)

    gyro_x = 0.2 * np.sin(2 * np.pi * 0.5 * time)
    gyro_y = 0.2 * np.cos(2 * np.pi * 0.5 * time)
    gyro_z = 0.05 * np.sin(2 * np.pi * 2 * time)

    # Add noise
    accel_x += np.random.normal(0, 0.1, len(time))
    accel_y += np.random.normal(0, 0.1, len(time))
    accel_z += np.random.normal(0, 0.1, len(time))
    gyro_x += np.random.normal(0, 0.01, len(time))
    gyro_y += np.random.normal(0, 0.01, len(time))
    gyro_z += np.random.normal(0, 0.01, len(time))

    return pd.DataFrame(
        {
            "time": time,
            "accelerometer_x": accel_x,
            "accelerometer_y": accel_y,
            "accelerometer_z": accel_z,
            "gyroscope_x": gyro_x,
            "gyroscope_y": gyro_y,
            "gyroscope_z": gyro_z,
        }
    )


print("\n" + "=" * 70)
print("TEST A: WITH DATETIME")
print("=" * 70)

df = create_dummy_data()
start_dt = datetime(2019, 8, 20, 10, 39, 16)

with tempfile.TemporaryDirectory() as tmpdir:
    results, metadata_dict = run_gait_pipeline(
        df, watch_side="left", output_dir=tmpdir, start_dt=start_dt
    )

if metadata_dict and "Arm_Swing_Parameters" in metadata_dict:
    metadata = metadata_dict["Arm_Swing_Parameters"]
    print("\nMetadata structure (with datetime):")
    print(json.dumps(metadata, indent=2, default=str))

    # Verify structure
    has_combined = "combined" in metadata
    has_per_segment = "per_segment" in metadata

    if has_combined:
        combined = metadata["combined"]
        has_start_dt = "start_dt" in combined
        has_end_dt = "end_dt" in combined
        has_start_s = "start_s" in combined

        print("\nCOMBINED section contains:")
        print(f"  - duration_s: {'YES' if 'duration_s' in combined else 'NO'}")
        print(f"  - start_dt: {'YES' if has_start_dt else 'NO'}")
        print(f"  - end_dt: {'YES' if has_end_dt else 'NO'}")
        print(f"  - start_s: {'YES (UNEXPECTED!)' if has_start_s else 'NO (correct)'}")

        if has_start_dt and not has_start_s:
            print(
                "\n✓ SUCCESS: DateTime fields present, relative times absent (as expected)"
            )
        elif has_start_dt and has_start_s:
            print("\n✗ WARNING: DateTime fields present but start_s/end_s also present")
        elif not has_start_dt:
            print("\n✗ ERROR: DateTime fields missing!")

    if has_per_segment and len(metadata["per_segment"]) > 0:
        first_segment = metadata["per_segment"].get("1", {})
        print("\nPER_SEGMENT[1] contains:")
        print(f"  - duration_s: {'YES' if 'duration_s' in first_segment else 'NO'}")
        print(f"  - start_dt: {'YES' if 'start_dt' in first_segment else 'NO'}")
        print(f"  - end_dt: {'YES' if 'end_dt' in first_segment else 'NO'}")
        print(
            f"  - start_s: {'YES (UNEXPECTED!)' if 'start_s' in first_segment else 'NO (correct)'}"
        )
else:
    print(
        "No arm swing parameters found - gait may not have been detected in test data"
    )

print("\n" + "=" * 70)
print("TEST B: WITHOUT DATETIME (start_dt=None)")
print("=" * 70)

df = create_dummy_data()

with tempfile.TemporaryDirectory() as tmpdir:
    results, metadata_dict = run_gait_pipeline(
        df, watch_side="left", output_dir=tmpdir, start_dt=None
    )

if metadata_dict and "Arm_Swing_Parameters" in metadata_dict:
    metadata = metadata_dict["Arm_Swing_Parameters"]
    print("\nMetadata structure (without datetime):")
    print(json.dumps(metadata, indent=2, default=str))

    # Verify structure
    if "combined" in metadata:
        combined = metadata["combined"]
        has_start_dt = "start_dt" in combined
        has_start_s = "start_s" in combined

        print("\nCOMBINED section contains:")
        print(f"  - duration_s: {'YES' if 'duration_s' in combined else 'NO'}")
        print(
            f"  - start_dt: {'YES (UNEXPECTED!)' if has_start_dt else 'NO (correct)'}"
        )
        print(
            f"  - end_dt: {'YES (UNEXPECTED!)' if 'end_dt' in combined else 'NO (correct)'}"
        )
        print(f"  - start_s: {'YES' if has_start_s else 'NO'}")

        if not has_start_dt and has_start_s:
            print(
                "\n✓ SUCCESS: DateTime fields absent, relative times present (as expected)"
            )
        elif has_start_dt:
            print(
                "\n✗ ERROR: DateTime fields should not be present when start_dt=None!"
            )
        elif not has_start_s:
            print("\n✗ ERROR: Relative time fields missing!")

    if "per_segment" in metadata and len(metadata["per_segment"]) > 0:
        first_segment = metadata["per_segment"].get("1", {})
        print("\nPER_SEGMENT[1] contains:")
        print(f"  - duration_s: {'YES' if 'duration_s' in first_segment else 'NO'}")
        print(
            f"  - start_dt: {'YES (UNEXPECTED!)' if 'start_dt' in first_segment else 'NO (correct)'}"
        )
        print(
            f"  - end_dt: {'YES (UNEXPECTED!)' if 'end_dt' in first_segment else 'NO (correct)'}"
        )
        print(f"  - start_s: {'YES' if 'start_s' in first_segment else 'NO'}")
else:
    print(
        "No arm swing parameters found - gait may not have been detected in test data"
    )

print("\n" + "=" * 70)
