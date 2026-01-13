#!/usr/bin/env python3
"""
Scan all wavevectors/bands to quantify where stored values deviate from regenerated.

Compares original stored eigenvalues with regenerated ones to identify
systematic discrepancies.
"""

import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

def scan_discrepancies(original_file, regenerated_file):
    """Scan and quantify discrepancies."""
    print("=" * 80)
    print("Scanning Eigenvalue Discrepancies")
    print("=" * 80)
    
    # Load both files
    print(f"\n1. Loading files...")
    with h5py.File(str(original_file), 'r') as f:
        eig_orig = np.array(f['EIGENVALUE_DATA'])
    
    with h5py.File(str(regenerated_file), 'r') as f:
        eig_regen = np.array(f['EIGENVALUE_DATA'])
    
    print(f"   Original shape: {eig_orig.shape}")
    print(f"   Regenerated shape: {eig_regen.shape}")
    
    # Compute statistics
    print(f"\n2. Computing discrepancy statistics...")
    
    # Absolute differences
    abs_diff = np.abs(eig_orig - eig_regen)
    
    # Relative differences (avoid division by zero)
    mask = np.abs(eig_orig) > 1e-10
    rel_diff = np.zeros_like(abs_diff)
    rel_diff[mask] = abs_diff[mask] / np.abs(eig_orig[mask])
    rel_diff[~mask] = np.inf  # Mark zero originals
    
    # Statistics per band
    n_designs, n_bands, n_wavevectors = eig_orig.shape
    
    print(f"\n3. Statistics per band (across all designs and wavevectors):")
    for band_idx in range(n_bands):
        band_abs = abs_diff[:, band_idx, :]
        band_rel = rel_diff[:, band_idx, :]
        band_rel_finite = band_rel[np.isfinite(band_rel)]
        
        print(f"\n   Band {band_idx}:")
        print(f"     Mean absolute diff: {np.mean(band_abs):.6e}")
        print(f"     Max absolute diff: {np.max(band_abs):.6e}")
        if len(band_rel_finite) > 0:
            print(f"     Mean relative diff: {np.mean(band_rel_finite):.6e}")
            print(f"     Max relative diff: {np.max(band_rel_finite):.6e}")
            print(f"     Median relative diff: {np.median(band_rel_finite):.6e}")
        else:
            print(f"     All original values are near zero")
    
    # Find worst cases (only for structures that exist in both)
    n_common_structs = min(eig_orig.shape[0], eig_regen.shape[0])
    abs_diff_common = abs_diff[:n_common_structs, :, :]
    
    print(f"\n4. Worst discrepancies (comparing {n_common_structs} common structures):")
    worst_indices = np.unravel_index(np.argmax(abs_diff_common), abs_diff_common.shape)
    worst_indices_full = (worst_indices[0], worst_indices[1], worst_indices[2])
    print(f"   Max absolute diff: {np.max(abs_diff_common):.6e}")
    print(f"   Location: struct {worst_indices_full[0]}, band {worst_indices_full[1]}, wavevector {worst_indices_full[2]}")
    print(f"   Original: {eig_orig[worst_indices_full]:.6e}")
    print(f"   Regenerated: {eig_regen[worst_indices_full]:.6e}")
    
    # Count large discrepancies (only for common structures)
    large_diff_mask = abs_diff_common > 1.0  # More than 1 Hz difference
    n_large = np.sum(large_diff_mask)
    n_total_common = abs_diff_common.size
    print(f"\n5. Large discrepancies (>1 Hz):")
    print(f"   Count: {n_large} / {n_total_common} ({100*n_large/n_total_common:.2f}%)")
    
    # Per wavevector statistics
    print(f"\n6. Statistics per wavevector (first 10):")
    for wv_idx in range(min(10, n_wavevectors)):
        wv_abs = abs_diff[:, :, wv_idx]
        wv_rel = rel_diff[:, :, wv_idx]
        wv_rel_finite = wv_rel[np.isfinite(wv_rel)]
        
        if len(wv_rel_finite) > 0:
            print(f"   Wavevector {wv_idx}: mean rel diff = {np.mean(wv_rel_finite):.6e}, max = {np.max(wv_rel_finite):.6e}")
        else:
            print(f"   Wavevector {wv_idx}: all original values near zero")
    
    # Save summary plot
    output_dir = Path("data/discrepancy_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n7. Generating plots...")
    
    # Plot 1: Histogram of relative differences
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram of relative differences (log scale)
    rel_finite = rel_diff[np.isfinite(rel_diff)]
    axes[0, 0].hist(np.log10(rel_finite + 1e-10), bins=50)
    axes[0, 0].set_xlabel('log10(relative difference)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Relative Differences')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of absolute differences
    axes[0, 1].hist(np.log10(abs_diff.flatten() + 1e-10), bins=50)
    axes[0, 1].set_xlabel('log10(absolute difference)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Absolute Differences')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean relative difference per band
    mean_rel_per_band = []
    for band_idx in range(n_bands):
        band_rel = rel_diff[:, band_idx, :]
        band_rel_finite = band_rel[np.isfinite(band_rel)]
        if len(band_rel_finite) > 0:
            mean_rel_per_band.append(np.mean(band_rel_finite))
        else:
            mean_rel_per_band.append(np.nan)
    
    axes[1, 0].bar(range(n_bands), mean_rel_per_band)
    axes[1, 0].set_xlabel('Band Index')
    axes[1, 0].set_ylabel('Mean Relative Difference')
    axes[1, 0].set_title('Mean Relative Difference per Band')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter: original vs regenerated
    axes[1, 1].scatter(eig_orig.flatten(), eig_regen.flatten(), alpha=0.1, s=1)
    axes[1, 1].plot([0, eig_orig.max()], [0, eig_orig.max()], 'r--', label='y=x')
    axes[1, 1].set_xlabel('Original Eigenvalue')
    axes[1, 1].set_ylabel('Regenerated Eigenvalue')
    axes[1, 1].set_title('Original vs Regenerated')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "discrepancy_summary.png"
    plt.savefig(plot_path, dpi=150)
    print(f"   Saved plot to {plot_path}")
    plt.close()
    
    return {
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
        'n_large': n_large,
        'worst_indices': worst_indices
    }

if __name__ == "__main__":
    original = Path("data/out_test_10_mat_original/out_binarized_1.mat")
    regenerated = Path("data/out_test_10_mat_regenerated/out_binarized_1.mat")
    
    if not regenerated.exists():
        print(f"Regenerated file not found: {regenerated}")
        print("Please run regenerate_eigenvalue_data.py first")
    else:
        scan_discrepancies(original, regenerated)

