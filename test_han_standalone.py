"""
Standalone test script to validate 2D-dispersion-Han Python implementation.

This script generates test designs using the same logic as MATLAB's
ex_dispersion_batch_save.m and runs dispersion calculations to validate
the Python implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime

# Add Python library to path
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))

# Import Python library functions
from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt
from get_design2 import get_design2
from design_parameters import DesignParameters
from design_conversion import convert_design, apply_steel_rubber_paradigm
from wavevectors import get_IBZ_wavevectors
from plotting import plot_design
from utils_han import make_chunks, init_storage


def generate_test_designs(num_designs=5, N_pix=32, rng_seed_offset=0):
    """
    Generate test designs matching MATLAB's ex_dispersion_batch_save.m logic.
    
    Parameters
    ----------
    num_designs : int
        Number of designs to generate
    N_pix : int
        Number of pixels
    rng_seed_offset : int
        Random seed offset
        
    Returns
    -------
    designs : list
        List of design arrays
    design_numbers : list
        List of design numbers (seeds)
    """
    designs = []
    design_numbers = []
    
    # Set up design parameters (matching MATLAB script)
    design_params = DesignParameters(None)
    design_params.design_number = []
    design_params.design_style = 'kernel'
    design_params.design_options = {
        'kernel': 'periodic - not squared',
        'sigma_f': 1.0,
        'sigma_l': 1.0,
        'symmetry_type': 'p4mm',
        'N_value': np.inf  # Continuous designs
    }
    design_params.N_pix = [N_pix, N_pix]
    design_params = design_params.prepare()
    
    for struct_idx in range(1, num_designs + 1):
        # Set design number (seed)
        design_params.design_number = struct_idx + rng_seed_offset
        design_params = design_params.prepare()
        
        # Generate design
        design = get_design2(design_params)
        designs.append(design)
        design_numbers.append(struct_idx + rng_seed_offset)
        
        print(f"Generated design {struct_idx}/{num_designs} (seed: {struct_idx + rng_seed_offset})")
    
    return designs, design_numbers


def run_dispersion_calculation(design, const):
    """
    Run dispersion calculation for a single design.
    
    Parameters
    ----------
    design : ndarray
        Design array (N_pix x N_pix x 3)
    const : dict
        Constants structure
        
    Returns
    -------
    wv : ndarray
        Wavevectors
    fr : ndarray
        Frequencies
    ev : ndarray or None
        Eigenvectors
    """
    # Set up constants for this design
    const_design = const.copy()
    const_design['design'] = design
    
    # Apply steel-rubber paradigm (matching MATLAB)
    const_design['design'] = apply_steel_rubber_paradigm(const_design['design'], const_design)
    
    # Run dispersion calculation
    wv, fr, ev, mesh, K_out, M_out, T_out = dispersion_with_matrix_save_opt(
        const_design, const_design['wavevectors']
    )
    
    return wv, fr, ev


def create_dispersion_plots(frequencies, wavevectors, design, sample_idx, output_dir):
    """
    Create dispersion plots for a sample.
    
    Parameters
    ----------
    frequencies : ndarray
        Frequency data (N_wv x N_eig)
    wavevectors : ndarray
        Wavevector data (N_wv x 2)
    design : ndarray
        Design array
    sample_idx : int
        Sample index
    output_dir : Path
        Output directory for plots
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Design
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(design[:, :, 0], cmap='viridis', origin='lower')
    ax1.set_title(f'Sample {sample_idx + 1}: Elastic Modulus')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(design[:, :, 1], cmap='viridis', origin='lower')
    ax2.set_title(f'Sample {sample_idx + 1}: Density')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(design[:, :, 2], cmap='viridis', origin='lower')
    ax3.set_title(f'Sample {sample_idx + 1}: Poisson Ratio')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    plt.colorbar(im3, ax=ax3)
    
    # Plot 2: Dispersion curves
    ax4 = plt.subplot(2, 3, (4, 6))
    for band in range(frequencies.shape[1]):
        ax4.plot(frequencies[:, band], 'o-', label=f'Band {band+1}', markersize=3, alpha=0.7)
    ax4.set_title(f'Sample {sample_idx + 1}: Dispersion Relations')
    ax4.set_xlabel('Wavevector Index')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'sample_{sample_idx + 1}_dispersion.png', dpi=150)
    plt.close()
    
    print(f"  Saved plot: {output_dir / f'sample_{sample_idx + 1}_dispersion.png'}")


def create_summary_plot(all_frequencies, output_dir):
    """
    Create summary plot comparing all samples.
    
    Parameters
    ----------
    all_frequencies : list
        List of frequency arrays for each sample
    output_dir : Path
        Output directory
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Summary: All Samples Dispersion Relations', fontsize=16)
    
    axes = axes.flatten()
    
    for i, frequencies in enumerate(all_frequencies[:6]):  # Plot up to 6 samples
        ax = axes[i]
        for band in range(frequencies.shape[1]):
            ax.plot(frequencies[:, band], 'o-', label=f'Band {band+1}', markersize=2, alpha=0.7)
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Wavevector Index')
        ax.set_ylabel('Frequency (Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(all_frequencies), 6):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_all_samples.png', dpi=150)
    plt.close()
    
    print(f"  Saved summary plot: {output_dir / 'summary_all_samples.png'}")


def main():
    """Main test function."""
    print("=" * 70)
    print("2D-Dispersion-Han: Standalone Python Validation Test")
    print("=" * 70)
    
    # Test parameters (matching MATLAB ex_dispersion_batch_save.m)
    N_struct = 5
    N_pix = 32
    N_ele = 1
    N_eig = 6
    N_wv = [25, None]  # Will calculate second value
    N_wv[1] = int(np.ceil(N_wv[0] / 2))
    rng_seed_offset = 0
    
    # Material parameters (matching MATLAB)
    const = {
        'N_ele': N_ele,
        'N_pix': N_pix,
        'N_eig': N_eig,
        'a': 1.0,
        'E_min': 2e6,
        'E_max': 200e9,
        'rho_min': 1200,
        'rho_max': 8e3,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 1.0,
        'sigma_eig': 1e-2,
        'design_scale': 'linear',
        'isUseGPU': False,
        'isUseImprovement': True,
        'isUseSecondImprovement': False,
        'isUseParallel': False,
        'isSaveEigenvectors': True,
        'eigenvector_dtype': 'double',
        'isSaveKandM': False,
        'isSaveMesh': False,
    }
    
    # Generate wavevectors
    print(f"\nGenerating wavevectors: N_wv = {N_wv}")
    symmetry_type = 'p4mm'
    const['wavevectors'] = get_IBZ_wavevectors(N_wv, const['a'], symmetry_type, N_tesselations=1)
    print(f"Generated {const['wavevectors'].shape[0]} wavevectors")
    
    # Generate test designs
    print(f"\nGenerating {N_struct} test designs...")
    designs, design_numbers = generate_test_designs(N_struct, N_pix, rng_seed_offset)
    
    # Create output directory
    output_dir = Path('test_output_han')
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Running Dispersion Calculations")
    print(f"{'='*70}")
    
    # Run calculations
    all_frequencies = []
    all_wavevectors = []
    
    for i, design in enumerate(designs):
        print(f"\nProcessing Sample {i+1}/{N_struct} (design number: {design_numbers[i]})")
        print(f"  Design shape: {design.shape}")
        print(f"  Design range: E=[{design[:,:,0].min():.3f}, {design[:,:,0].max():.3f}], "
              f"rho=[{design[:,:,1].min():.3f}, {design[:,:,1].max():.3f}], "
              f"nu=[{design[:,:,2].min():.3f}, {design[:,:,2].max():.3f}]")
        
        try:
            wv, fr, ev = run_dispersion_calculation(design, const)
            print(f"  Calculation complete: frequencies shape {fr.shape}")
            print(f"  Frequency range: [{fr.min():.6f}, {fr.max():.6f}] Hz")
            print(f"  Number of bands: {fr.shape[1]}")
            
            all_frequencies.append(fr)
            all_wavevectors.append(wv)
            
            # Create plots
            create_dispersion_plots(fr, wv, design, i, plots_dir)
            
        except Exception as e:
            print(f"  ERROR in calculation: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary plot
    if all_frequencies:
        print(f"\n{'='*70}")
        print("Creating Summary Plots")
        print(f"{'='*70}")
        create_summary_plot(all_frequencies, plots_dir)
    
    # Print statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully processed: {len(all_frequencies)}/{N_struct} samples")
    
    if all_frequencies:
        # Calculate statistics
        all_freq_array = np.array([f.flatten() for f in all_frequencies])
        print(f"\nFrequency Statistics:")
        print(f"  Mean frequency: {np.mean(all_freq_array):.6f} Hz")
        print(f"  Std frequency: {np.std(all_freq_array):.6f} Hz")
        print(f"  Min frequency: {np.min(all_freq_array):.6f} Hz")
        print(f"  Max frequency: {np.max(all_freq_array):.6f} Hz")
        
        # Check for any issues
        if np.any(np.isnan(all_freq_array)):
            print(f"  WARNING: Found NaN values in frequencies")
        if np.any(np.isinf(all_freq_array)):
            print(f"  WARNING: Found Inf values in frequencies")
        if np.any(all_freq_array < 0):
            print(f"  WARNING: Found negative frequencies")
    
    print(f"\nOutput saved to: {output_dir}/")
    print(f"Plots saved to: {plots_dir}/")
    print("=" * 70)
    
    return all_frequencies, all_wavevectors, designs


if __name__ == '__main__':
    frequencies, wavevectors, designs = main()

