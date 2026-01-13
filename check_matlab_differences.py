#!/usr/bin/env python3
"""
Detailed comparison of missing variables and value mismatches between predictions and original .mat files.
"""

import h5py
import numpy as np
from pathlib import Path

def load_const_values(file_path):
    """Load const values from a .mat file."""
    with h5py.File(file_path, 'r') as f:
        const = {}
        for key in f['const'].keys():
            const[key] = np.array(f['const'][key])
        return const

def load_top_level_vars(file_path):
    """Load top-level variables from a .mat file."""
    vars_dict = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if key == 'const' or key.startswith('#'):
                continue
            if isinstance(f[key], h5py.Dataset):
                vars_dict[key] = np.array(f[key])
    return vars_dict

# File paths
original_file = Path(r"D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat")
predictions_file = Path(r"D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1\out_binarized_1_predictions.mat")

print("=" * 80)
print("Detailed Comparison: Missing Variables and Value Mismatches")
print("=" * 80)

# Load top-level variables
print("\n1. TOP-LEVEL VARIABLES")
print("-" * 80)
orig_vars = load_top_level_vars(original_file)
pred_vars = load_top_level_vars(predictions_file)

orig_keys = set(k for k in orig_vars.keys() if not k.startswith('#'))
pred_keys = set(pred_vars.keys())

missing_in_pred = orig_keys - pred_keys
extra_in_pred = pred_keys - orig_keys

print(f"\nMissing in predictions file ({len(missing_in_pred)}):")
for key in sorted(missing_in_pred):
    print(f"  - {key}: shape {orig_vars[key].shape}, dtype {orig_vars[key].dtype}")

if extra_in_pred:
    print(f"\nExtra in predictions file ({len(extra_in_pred)}):")
    for key in sorted(extra_in_pred):
        print(f"  - {key}")

# Load const values
print("\n2. CONST VALUES")
print("-" * 80)
orig_const = load_const_values(original_file)
pred_const = load_const_values(predictions_file)

orig_const_keys = set(orig_const.keys())
pred_const_keys = set(pred_const.keys())

missing_const = orig_const_keys - pred_const_keys
if missing_const:
    print(f"\nMissing const keys in predictions ({len(missing_const)}):")
    for key in sorted(missing_const):
        print(f"  - {key}: shape {orig_const[key].shape}")

# Compare const values
print(f"\nConst value differences:")
common_const = orig_const_keys & pred_const_keys
for key in sorted(common_const):
    orig_val = orig_const[key]
    pred_val = pred_const[key]
    
    if orig_val.shape != pred_val.shape:
        print(f"  {key}: shape mismatch - orig {orig_val.shape} vs pred {pred_val.shape}")
    elif orig_val.dtype == object or pred_val.dtype == object:
        # String/object comparison
        try:
            orig_str = ''.join([chr(int(x)) for x in orig_val.flatten()])
            pred_str = ''.join([chr(int(x)) for x in pred_val.flatten()])
            if orig_str != pred_str:
                print(f"  {key}: '{orig_str}' vs '{pred_str}'")
        except:
            if not np.array_equal(orig_val, pred_val):
                print(f"  {key}: values differ (object type)")
    elif np.issubdtype(orig_val.dtype, np.floating):
        if not np.allclose(orig_val, pred_val, rtol=1e-5, atol=1e-8, equal_nan=True):
            max_diff = np.nanmax(np.abs(orig_val - pred_val))
            print(f"  {key}: max difference = {max_diff:.6e}, orig={orig_val.flatten()}, pred={pred_val.flatten()}")
    else:
        if not np.array_equal(orig_val, pred_val):
            print(f"  {key}: values differ")

# Check specific missing variables
print("\n3. DETAILED ANALYSIS OF MISSING VARIABLES")
print("-" * 80)

with h5py.File(original_file, 'r') as f:
    for key in sorted(missing_in_pred):
        data = np.array(f[key])
        print(f"\n{key}:")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        if data.size < 20:
            print(f"  Values: {data.flatten()}")
        else:
            print(f"  Sample (first 10): {data.flatten()[:10]}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Missing top-level variables: {len(missing_in_pred)}")
print(f"Const value differences: {len([k for k in common_const if not np.array_equal(orig_const[k], pred_const[k])])}")

