#!/usr/bin/env python3
"""
Plot dispersion curves using regenerated eigenvalue data.

Generates dispersion plots for visual inspection comparing original vs regenerated.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from wavevectors import get_IBZ_contour_wavevectors
from plotting import plot_dispersion

def plot_dispersion_comparison(original_file, regenerated_file, struct_idx=0, output_dir=None):
    """Plot dispersion curves comparing original vs regenerated."""
    if output_dir is None:
        output_dir = Path("data/dispersion_plots_regenerated")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Plotting Dispersion Curves: Original vs Regenerated")
    print("=" * 80)
    
    # Load data
    print(f"\n1. Loading data...")
    with h5py.File(str(original_file), 'r') as f:
        eig_orig = np.array(f['EIGENVALUE_DATA'])
        wavevectors = np.array(f['WAVEVECTOR_DATA'])
        const_dict = {key: np.array(f['const'][key]) for key in f['const']}
    
    with h5py.File(str(regenerated_file), 'r') as f:
        eig_regen = np.array(f['EIGENVALUE_DATA'])
    
    # Get const parameters
    a = float(np.array(const_dict['a']).item())
    
    # Handle wavevector shape
    if wavevectors.ndim == 3:
        wv_struct = wavevectors[struct_idx, :, :].T  # (n_wavevectors, 2)
    else:
        wv_struct = wavevectors  # (n_wavevectors, 2)
    
    print(f"   Original eigenvalues shape: {eig_orig.shape}")
    print(f"   Regenerated eigenvalues shape: {eig_regen.shape}")
    print(f"   Wavevectors shape: {wv_struct.shape}")
    
    # Get IBZ contour
    print(f"\n2. Computing IBZ contour...")
    contour_wv, contour_info = get_IBZ_contour_wavevectors(50, a, 'p4mm')
    print(f"   Contour points: {len(contour_wv)}")
    
    # Interpolate frequencies to contour
    print(f"\n3. Interpolating frequencies to contour...")
    from scipy.interpolate import griddata
    
    n_bands = eig_orig.shape[1]
    frequencies_orig_contour = np.zeros((len(contour_wv), n_bands))
    frequencies_regen_contour = np.zeros((len(contour_wv), n_bands))
    
    for band_idx in range(n_bands):
        # Original
        frequencies_orig_contour[:, band_idx] = griddata(
            wv_struct, 
            eig_orig[struct_idx, band_idx, :],
            contour_wv,
            method='linear',
            fill_value=np.nan
        )
        
        # Regenerated
        frequencies_regen_contour[:, band_idx] = griddata(
            wv_struct,
            eig_regen[struct_idx, band_idx, :],
            contour_wv,
            method='linear',
            fill_value=np.nan
        )
    
    # Plot comparison
    print(f"\n4. Generating plots...")
    
    # Plot 1: Side by side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original
    plot_dispersion(
        wn=contour_info['wavevector_parameter'],
        fr=frequencies_orig_contour,
        N_contour_segments=contour_info['N_segment'] + 1,
        ax=axes[0]
    )
    axes[0].set_title('Original Eigenvalues', fontsize=14, fontweight='bold')
    
    # Regenerated
    plot_dispersion(
        wn=contour_info['wavevector_parameter'],
        fr=frequencies_regen_contour,
        N_contour_segments=contour_info['N_segment'] + 1,
        ax=axes[1]
    )
    axes[1].set_title('Regenerated Eigenvalues (Rayleigh Quotient)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / f"comparison_struct_{struct_idx}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_bands))
    for band_idx in range(n_bands):
        ax.plot(contour_info['wavevector_parameter'], frequencies_orig_contour[:, band_idx],
               '--', color=colors[band_idx], linewidth=2, alpha=0.7, label=f'Original Band {band_idx}')
        ax.plot(contour_info['wavevector_parameter'], frequencies_regen_contour[:, band_idx],
               '-', color=colors[band_idx], linewidth=2, label=f'Regenerated Band {band_idx}')
    
    ax.set_xlabel('Wavevector Parameter', fontsize=12)
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    ax.set_title(f'Dispersion Comparison: Original vs Regenerated (Struct {struct_idx})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f"overlay_struct_{struct_idx}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {plot_path}")
    plt.close()
    
    # Plot 3: Difference
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for band_idx in range(n_bands):
        diff = frequencies_regen_contour[:, band_idx] - frequencies_orig_contour[:, band_idx]
        ax.plot(contour_info['wavevector_parameter'], diff,
               color=colors[band_idx], linewidth=2, label=f'Band {band_idx}')
    
    ax.set_xlabel('Wavevector Parameter', fontsize=12)
    ax.set_ylabel('Frequency Difference [Hz]', fontsize=12)
    ax.set_title(f'Difference: Regenerated - Original (Struct {struct_idx})', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plot_path = output_dir / f"difference_struct_{struct_idx}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {plot_path}")
    plt.close()
    
    print(f"\n   All plots saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot dispersion curves comparing original vs regenerated')
    parser.add_argument('--original', type=str, default='data/out_test_10_mat_original/out_binarized_1.mat',
                       help='Original .mat file')
    parser.add_argument('--regenerated', type=str, default='data/out_test_10_mat_regenerated/out_binarized_1.mat',
                       help='Regenerated .mat file')
    parser.add_argument('--struct', type=int, default=0, help='Structure index to plot')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    plot_dispersion_comparison(args.original, args.regenerated, args.struct, args.output)

