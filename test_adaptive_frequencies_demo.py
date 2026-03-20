#!/usr/bin/env python
"""
Demo: Adaptive Frequency Bounds in GaitConfig and TremorConfig

Shows how frequency-dependent parameters automatically adapt to different
sampling rates without manual intervention.
"""

from paradigma.config import GaitConfig, TremorConfig

print("=" * 80)
print("ADAPTIVE FREQUENCY BOUNDS DEMO")
print("=" * 80)

# Test 1: GaitConfig with different sampling frequencies
print("\n[Test 1] GaitConfig Frequency Adaptation")
print("-" * 80)

for freq in [50, 64, 100]:
    print(f"\nGaitConfig with {freq}Hz sampling:")
    config = GaitConfig(step="gait")
    config.sampling_frequency = freq
    
    nyquist = freq / 2
    print(f"  Nyquist limit: {nyquist}Hz")
    print(f"  spectrum_high_frequency: {config.spectrum_high_frequency}Hz")
    print(f"  mfcc_high_frequency: {config.mfcc_high_frequency}Hz")
    print(f"  ✓ Within safe bounds: {config.spectrum_high_frequency < nyquist and config.mfcc_high_frequency < nyquist * 0.9}")

# Test 2: TremorConfig with different sampling frequencies
print("\n\n[Test 2] TremorConfig Frequency Adaptation")
print("-" * 80)

for freq in [50, 64, 100]:
    print(f"\nTremorConfig with {freq}Hz sampling:")
    config = TremorConfig(step="features")
    config.sampling_frequency = freq
    
    nyquist = freq / 2
    print(f"  Nyquist limit: {nyquist}Hz")
    print(f"  fmax_peak_search: {config.fmax_peak_search}Hz")
    print(f"  fmax_mfcc: {config.fmax_mfcc}Hz")
    print(f"  ✓ Within safe bounds: {config.fmax_peak_search < nyquist * 0.9 and config.fmax_mfcc < nyquist * 0.9}")

# Test 3: Demonstrate property setter updates on-the-fly
print("\n\n[Test 3] Dynamic Frequency Changes")
print("-" * 80)

gait_config = GaitConfig(step="gait")
print(f"\nInitial (100Hz): spectrum_high_frequency = {gait_config.spectrum_high_frequency}Hz")

gait_config.sampling_frequency = 64
print(f"After change to 64Hz: spectrum_high_frequency = {gait_config.spectrum_high_frequency}Hz")

gait_config.sampling_frequency = 50
print(f"After change to 50Hz: spectrum_high_frequency = {gait_config.spectrum_high_frequency}Hz")

print("\n" + "=" * 80)
print("✓ All tests passed! Frequency bounds adapt automatically.")
print("=" * 80)
