#!/usr/bin/env python3
"""
Compare geometries between two MATLAB .mat files
"""

import numpy as np
import h5py
from pathlib import Path

def load_designs(mat_path):
    """Load designs array from MATLAB file."""
    with h5py.File(mat_path, 'r') as file:
        designs = np.array(file['designs'])
    return designs

def compare_geometries(file1_path, file2_path):
    """Compare geometries (first pane of designs) between two files."""
    print("=" * 80)
    print("Comparing Geometries")
    print("=" * 80)
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print("=" * 80)
    
    # Load designs
    print("\nLoading designs...")
    designs1 = load_designs(file1_path)
    designs2 = load_designs(file2_path)
    
    print(f"\nDesigns shape - File 1: {designs1.shape}")
    print(f"Designs shape - File 2: {designs2.shape}")
    
    # Extract geometries (first pane)
    geometries1 = designs1[:, 0, :, :]  # (n_designs, H, W)
    geometries2 = designs2[:, 0, :, :]  # (n_designs, H, W)
    
    print(f"\nGeometries shape - File 1: {geometries1.shape}")
    print(f"Geometries shape - File 2: {geometries2.shape}")
    
    # Compare shapes
    if geometries1.shape != geometries2.shape:
        print(f"\n✗ Shape mismatch!")
        return
    
    print(f"\n✓ Shapes match")
    
    # Compare values
    print("\n" + "=" * 80)
    print("Numerical Comparison")
    print("=" * 80)
    
    # Check for NaN/inf
    nan1 = np.isnan(geometries1).sum()
    nan2 = np.isnan(geometries2).sum()
    inf1 = np.isinf(geometries1).sum()
    inf2 = np.isinf(geometries2).sum()
    
    print(f"NaN count - File 1: {nan1}, File 2: {nan2}")
    print(f"Inf count - File 1: {inf1}, File 2: {inf2}")
    
    # Compute differences
    abs_diff = np.abs(geometries1 - geometries2)
    rel_diff = abs_diff / (np.abs(geometries1) + 1e-10)
    
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # Check if arrays are close
    is_close = np.allclose(geometries1, geometries2, rtol=1e-5, atol=1e-8)
    
    print(f"\nMatch status: {'✓ MATCH' if is_close else '✗ DIFFERENT'}")
    if not is_close:
        print(f"  Max abs diff: {max_abs_diff:.6e}")
        print(f"  Mean abs diff: {mean_abs_diff:.6e}")
        print(f"  Max rel diff: {max_rel_diff:.6e}")
        print(f"  Mean rel diff: {mean_rel_diff:.6e}")
    
    # Value ranges
    print(f"\nValue ranges - File 1: [{np.min(geometries1):.6e}, {np.max(geometries1):.6e}]")
    print(f"Value ranges - File 2: [{np.min(geometries2):.6e}, {np.max(geometries2):.6e}]")
    
    # Per-design comparison
    print("\n" + "=" * 80)
    print("Per-Design Comparison")
    print("=" * 80)
    n_designs = geometries1.shape[0]
    for d_idx in range(n_designs):
        geo1 = geometries1[d_idx, :, :]
        geo2 = geometries2[d_idx, :, :]
        is_close_design = np.allclose(geo1, geo2, rtol=1e-5, atol=1e-8)
        max_diff_design = np.max(np.abs(geo1 - geo2))
        mean_diff_design = np.mean(np.abs(geo1 - geo2))
        status = '✓' if is_close_design else '✗'
        print(f"  Design {d_idx}: {status} - Max diff: {max_diff_design:.6e}, Mean diff: {mean_diff_design:.6e}")
        print(f"    Range File 1: [{np.min(geo1):.6e}, {np.max(geo1):.6e}]")
        print(f"    Range File 2: [{np.min(geo2):.6e}, {np.max(geo2):.6e}]")
    
    # Check if one is all zeros
    all_zeros1 = np.allclose(geometries1, 0, atol=1e-10)
    all_zeros2 = np.allclose(geometries2, 0, atol=1e-10)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if all_zeros1:
        print("⚠ File 1 geometries are all zeros (or near-zero)")
    if all_zeros2:
        print("⚠ File 2 geometries are all zeros (or near-zero)")
    
    if is_close:
        print("✓ Geometries MATCH")
    else:
        print("✗ Geometries DO NOT MATCH")
        if all_zeros1 or all_zeros2:
            print("  One or both files have zero geometries - this indicates data loss during conversion")
    
    print("=" * 80)

if __name__ == "__main__":
    file1_path = Path(r"D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat")
    file2_path = Path(r"D:\Research\NO-2D-Metamaterials\data\conversion_test_matlab\out_binarized_1.mat")
    
    if not file1_path.exists():
        print(f"ERROR: File 1 not found: {file1_path}")
        exit(1)
    
    if not file2_path.exists():
        print(f"ERROR: File 2 not found: {file2_path}")
        exit(1)
    
    compare_geometries(file1_path, file2_path)

