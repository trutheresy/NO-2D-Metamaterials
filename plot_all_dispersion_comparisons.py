#!/usr/bin/env python3
"""
Generate dispersion plots for ALL structures comparing original vs regenerated.

Creates comprehensive comparison plots for visual inspection.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.interpolate import griddata

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from wavevectors import get_IBZ_contour_wavevectors
from plotting import plot_dispersion

def plot_all_structures(original_file, regenerated_file, output_dir=None, max_structs=None):
    """
    Plot dispersion curves for all structures.
    
    Parameters
    ----------
    original_file : str or Path
        Path to original .mat file
    regenerated_file : str or Path
        Path to regenerated .mat file
    output_dir : str or Path, optional
        Output directory for plots
    max_structs : int, optional
        Maximum number of structures to plot (None = all)
    """
    original_file = Path(original_file)
    regenerated_file = Path(regenerated_file)
    
    if output_dir is None:
        output_dir = Path("data/dispersion_plots_all_structures")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating Dispersion Plots for All Structures")
    print("=" * 80)
    print(f"Original:    {original_file}")
    print(f"Regenerated: {regenerated_file}")
    print(f"Output:      {output_dir}")
    
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
    n_designs = eig_orig.shape[0]
    
    if max_structs is not None:
        n_designs = min(n_designs, max_structs)
    
    print(f"   Found {n_designs} structures")
    print(f"   Original eigenvalues shape: {eig_orig.shape}")
    print(f"   Regenerated eigenvalues shape: {eig_regen.shape}")
    
    # Handle wavevector shape
    if wavevectors.ndim == 3:
        wv_all = wavevectors  # (n_designs, 2, n_wavevectors)
    else:
        # Broadcast to all structures
        wv_all = np.tile(wavevectors[np.newaxis, :, :], (n_designs, 1, 1))
    
    # Get IBZ contour (once)
    print(f"\n2. Computing IBZ contour...")
    contour_wv, contour_info = get_IBZ_contour_wavevectors(50, a, 'p4mm')
    print(f"   Contour points: {len(contour_wv)}")
    
    # Process each structure
    print(f"\n3. Processing structures...")
    n_bands = eig_orig.shape[1]
    
    for struct_idx in range(n_designs):
        print(f"\n   Structure {struct_idx + 1}/{n_designs}...")
        
        # Get wavevectors for this structure
        if wavevectors.ndim == 3:
            wv_struct = wavevectors[struct_idx, :, :].T  # (n_wavevectors, 2)
        else:
            wv_struct = wavevectors  # (n_wavevectors, 2)
        
        # Interpolate frequencies to contour
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
        
        # Create comparison plots
        colors = plt.cm.tab10(np.linspace(0, 1, n_bands))
        
        # Plot 1: Side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original
        plot_dispersion(
            wn=contour_info['wavevector_parameter'],
            fr=frequencies_orig_contour,
            N_contour_segments=contour_info['N_segment'] + 1,
            ax=axes[0]
        )
        axes[0].set_title(f'Original (Struct {struct_idx})', fontsize=14, fontweight='bold')
        
        # Regenerated
        plot_dispersion(
            wn=contour_info['wavevector_parameter'],
            fr=frequencies_regen_contour,
            N_contour_segments=contour_info['N_segment'] + 1,
            ax=axes[1]
        )
        axes[1].set_title(f'Regenerated - Rayleigh (Struct {struct_idx})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plot_path = output_dir / f"comparison_struct_{struct_idx}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {plot_path.name}")
        
        # Plot 2: Overlay
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for band_idx in range(n_bands):
            ax.plot(contour_info['wavevector_parameter'], frequencies_orig_contour[:, band_idx],
                   '--', color=colors[band_idx], linewidth=2, alpha=0.7, label=f'Original Band {band_idx}')
            ax.plot(contour_info['wavevector_parameter'], frequencies_regen_contour[:, band_idx],
                   '-', color=colors[band_idx], linewidth=2, label=f'Regenerated Band {band_idx}')
        
        ax.set_xlabel('Wavevector Parameter', fontsize=12)
        ax.set_ylabel('Frequency [Hz]', fontsize=12)
        ax.set_title(f'Dispersion Comparison: Original vs Regenerated (Struct {struct_idx})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / f"overlay_struct_{struct_idx}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {plot_path.name}")
        
        # Plot 3: Difference
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for band_idx in range(n_bands):
            diff = frequencies_regen_contour[:, band_idx] - frequencies_orig_contour[:, band_idx]
            ax.plot(contour_info['wavevector_parameter'], diff,
                   color=colors[band_idx], linewidth=2, label=f'Band {band_idx}')
        
        ax.set_xlabel('Wavevector Parameter', fontsize=12)
        ax.set_ylabel('Frequency Difference [Hz]', fontsize=12)
        ax.set_title(f'Difference: Regenerated - Original (Struct {struct_idx})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plot_path = output_dir / f"difference_struct_{struct_idx}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {plot_path.name}")
    
    print(f"\n" + "=" * 80)
    print("All plots generated successfully!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated plots per structure:")
    print(f"  - comparison_struct_N.png  (side-by-side)")
    print(f"  - overlay_struct_N.png      (overlay)")
    print(f"  - difference_struct_N.png   (difference)")
    print(f"\nTotal: {n_designs} structures × 3 plots = {n_designs * 3} plots")
    
    return output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot dispersion curves for all structures')
    parser.add_argument('--original', type=str, 
                       default='data/out_test_10_mat_original/out_binarized_1.mat',
                       help='Original .mat file')
    parser.add_argument('--regenerated', type=str,
                       default='data/out_test_10_mat_regenerated/out_binarized_1.mat',
                       help='Regenerated .mat file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--max-structs', type=int, default=None,
                       help='Maximum number of structures to plot')
    args = parser.parse_args()
    
    plot_all_structures(args.original, args.regenerated, args.output, args.max_structs)

