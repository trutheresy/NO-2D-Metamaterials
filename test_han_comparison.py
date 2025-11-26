"""
Test script to compare 2D-dispersion-Han MATLAB library with Python implementation.

This script:
1. Runs 5 samples from MATLAB ex_dispersion_batch_save.m
2. Loads the designs from MATLAB output
3. Runs equivalent Python calculations
4. Compares results with plots
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from datetime import datetime

# Add Python library to path
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))

# Import Python library functions
from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt
from get_design2 import get_design2
from design_parameters import DesignParameters
from design_conversion import convert_design
from wavevectors import get_IBZ_wavevectors
from plotting import plot_dispersion, plot_design
from mat73_loader import load_matlab_v73

# Import apply_steel_rubber_paradigm
from design_conversion import apply_steel_rubber_paradigm


def load_matlab_designs(mat_file_path, num_samples=5):
    """
    Load designs from MATLAB output file.
    
    Parameters
    ----------
    mat_file_path : str or Path
        Path to MATLAB .mat file
    num_samples : int
        Number of designs to load
        
    Returns
    -------
    designs : list
        List of design arrays
    const : dict
        Constants structure from MATLAB
    """
    print(f"Loading MATLAB file: {mat_file_path}")
    data = load_matlab_v73(str(mat_file_path), verbose=False)
    
    # Extract designs
    designs = []
    if 'designs' in data:
        designs_array = data['designs']
        num_available = min(num_samples, designs_array.shape[3])
        for i in range(num_available):
            designs.append(designs_array[:, :, :, i])
    else:
        raise ValueError("'designs' not found in MATLAB file")
    
    # Extract constants
    const = {}
    if 'const' in data:
        mat_const = data['const']
        # Handle different MATLAB struct formats
        if isinstance(mat_const, np.ndarray) and mat_const.dtype.names:
            # Structured array
            for key in mat_const.dtype.names:
                value = mat_const[key].item()
                # Handle arrays
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        const[key] = float(value.item())
                    else:
                        const[key] = value
                else:
                    const[key] = value
        elif isinstance(mat_const, dict):
            # Already a dict (from h5py loading)
            const = mat_const.copy()
        else:
            # Try to access as object
            for key in dir(mat_const):
                if not key.startswith('_'):
                    try:
                        value = getattr(mat_const, key)
                        if isinstance(value, np.ndarray):
                            if value.size == 1:
                                const[key] = float(value.item())
                            else:
                                const[key] = value
                        else:
                            const[key] = value
                    except:
                        pass
    
    # Extract wavevectors
    if 'WAVEVECTOR_DATA' in data:
        const['wavevectors'] = data['WAVEVECTOR_DATA'][:, :, 0]  # Use first structure's wavevectors
    
    return designs, const


def run_python_dispersion(design, const):
    """
    Run Python dispersion calculation for a single design.
    
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
    
    # Ensure required flags are set
    const_design['isSaveEigenvectors'] = True
    const_design['eigenvector_dtype'] = 'double'
    const_design['isSaveKandM'] = False
    const_design['isSaveMesh'] = False
    
    # Run dispersion calculation
    wv, fr, ev, mesh, K_out, M_out, T_out = dispersion_with_matrix_save_opt(
        const_design, const_design['wavevectors']
    )
    
    return wv, fr, ev


def compare_results(mat_fr, py_fr, mat_wv, py_wv, sample_idx):
    """
    Compare MATLAB and Python results and create plots.
    
    Parameters
    ----------
    mat_fr : ndarray
        MATLAB frequencies
    py_fr : ndarray
        Python frequencies
    mat_wv : ndarray
        MATLAB wavevectors
    py_wv : ndarray
        Python wavevectors
    sample_idx : int
        Sample index for labeling
    """
    # Calculate differences
    diff = np.abs(mat_fr - py_fr)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = diff / (np.abs(mat_fr) + 1e-10)
    max_rel_diff = np.max(rel_diff)
    
    print(f"\nSample {sample_idx + 1} Comparison:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Max relative difference: {max_rel_diff:.6e}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Sample {sample_idx + 1} Comparison: MATLAB vs Python', fontsize=14)
    
    # Plot 1: MATLAB frequencies
    ax = axes[0, 0]
    for band in range(mat_fr.shape[1]):
        ax.plot(mat_fr[:, band], label=f'Band {band+1}')
    ax.set_title('MATLAB Frequencies')
    ax.set_xlabel('Wavevector Index')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Python frequencies
    ax = axes[0, 1]
    for band in range(py_fr.shape[1]):
        ax.plot(py_fr[:, band], label=f'Band {band+1}')
    ax.set_title('Python Frequencies')
    ax.set_xlabel('Wavevector Index')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Overlay comparison
    ax = axes[1, 0]
    for band in range(min(mat_fr.shape[1], py_fr.shape[1])):
        ax.plot(mat_fr[:, band], 'o-', label=f'MATLAB Band {band+1}', markersize=3, alpha=0.7)
        ax.plot(py_fr[:, band], 's--', label=f'Python Band {band+1}', markersize=3, alpha=0.7)
    ax.set_title('Overlay Comparison')
    ax.set_xlabel('Wavevector Index')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Difference
    ax = axes[1, 1]
    for band in range(min(mat_fr.shape[1], py_fr.shape[1])):
        ax.plot(diff[:, band], label=f'Band {band+1}')
    ax.set_title(f'Absolute Difference (Max: {max_diff:.2e})')
    ax.set_xlabel('Wavevector Index')
    ax.set_ylabel('|MATLAB - Python|')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('comparison_plots')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'sample_{sample_idx + 1}_comparison.png', dpi=150)
    print(f"  Saved plot to: {output_dir / f'sample_{sample_idx + 1}_comparison.png'}")
    
    plt.close()
    
    return max_diff, mean_diff, max_rel_diff


def main():
    """Main test function."""
    print("=" * 70)
    print("2D-Dispersion-Han: MATLAB vs Python Comparison Test")
    print("=" * 70)
    
    # Check if MATLAB file exists
    # User should run MATLAB script first to generate test data
    mat_file = Path('2D-dispersion-han/OUTPUT') / 'test_output.mat'
    
    # Look for any .mat file in OUTPUT directory
    output_dir = Path('2D-dispersion-han/OUTPUT')
    if not output_dir.exists():
        print(f"\nERROR: MATLAB output directory not found: {output_dir}")
        print("Please run the MATLAB script 'ex_dispersion_batch_save.m' first to generate test data.")
        print("\nAlternatively, you can specify a MATLAB file path as an argument.")
        return
    
    # Find .mat files
    mat_files = list(output_dir.glob('*.mat'))
    if not mat_files:
        print(f"\nERROR: No .mat files found in {output_dir}")
        print("Please run the MATLAB script 'ex_dispersion_batch_save.m' first.")
        return
    
    # Use the most recent file
    mat_file = max(mat_files, key=lambda p: p.stat().st_mtime)
    print(f"\nUsing MATLAB file: {mat_file}")
    
    # Load MATLAB designs
    try:
        designs_mat, const_mat = load_matlab_designs(mat_file, num_samples=5)
        print(f"Loaded {len(designs_mat)} designs from MATLAB file")
    except Exception as e:
        print(f"\nERROR loading MATLAB file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare constants for Python
    const_py = {
        'N_ele': int(const_mat.get('N_ele', 1)),
        'N_pix': int(const_mat.get('N_pix', 32)),
        'N_eig': int(const_mat.get('N_eig', 6)),
        'a': float(const_mat.get('a', 1.0)),
        'E_min': float(const_mat.get('E_min', 2e6)),
        'E_max': float(const_mat.get('E_max', 200e9)),
        'rho_min': float(const_mat.get('rho_min', 1200)),
        'rho_max': float(const_mat.get('rho_max', 8e3)),
        'poisson_min': float(const_mat.get('poisson_min', 0.0)),
        'poisson_max': float(const_mat.get('poisson_max', 0.5)),
        't': float(const_mat.get('t', 1.0)),
        'sigma_eig': float(const_mat.get('sigma_eig', 1e-2)),
        'design_scale': const_mat.get('design_scale', 'linear'),
        'isUseGPU': False,
        'isUseImprovement': True,
        'isUseSecondImprovement': False,
        'isUseParallel': False,
    }
    
    # Load MATLAB results for comparison
    data = load_matlab_v73(str(mat_file), verbose=False)
    mat_frequencies = data.get('EIGENVALUE_DATA', None)
    
    if mat_frequencies is None:
        print("\nWARNING: EIGENVALUE_DATA not found in MATLAB file. Will only run Python calculations.")
        mat_frequencies = None
    else:
        print(f"Loaded MATLAB frequencies: shape {mat_frequencies.shape}")
    
    # Run Python calculations and compare
    results = []
    num_samples = min(5, len(designs_mat))
    
    for i in range(num_samples):
        print(f"\n{'='*70}")
        print(f"Processing Sample {i+1}/{num_samples}")
        print(f"{'='*70}")
        
        design = designs_mat[i]
        print(f"Design shape: {design.shape}")
        print(f"Design range: E=[{design[:,:,0].min():.3f}, {design[:,:,0].max():.3f}], "
              f"rho=[{design[:,:,1].min():.3f}, {design[:,:,1].max():.3f}], "
              f"nu=[{design[:,:,2].min():.3f}, {design[:,:,2].max():.3f}]")
        
        # Run Python dispersion
        try:
            wv_py, fr_py, ev_py = run_python_dispersion(design, const_py)
            print(f"Python calculation complete: frequencies shape {fr_py.shape}")
        except Exception as e:
            print(f"ERROR in Python calculation: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Compare with MATLAB if available
        if mat_frequencies is not None and i < mat_frequencies.shape[2]:
            fr_mat = mat_frequencies[:, :, i]
            
            # Compare results
            max_diff, mean_diff, max_rel_diff = compare_results(
                fr_mat, fr_py, const_py['wavevectors'], wv_py, i
            )
            
            results.append({
                'sample': i + 1,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_rel_diff': max_rel_diff
            })
        else:
            # Just plot Python results
            fig, ax = plt.subplots(figsize=(10, 6))
            for band in range(fr_py.shape[1]):
                ax.plot(fr_py[:, band], label=f'Band {band+1}')
            ax.set_title(f'Sample {i+1}: Python Frequencies')
            ax.set_xlabel('Wavevector Index')
            ax.set_ylabel('Frequency (Hz)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            output_dir = Path('comparison_plots')
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / f'sample_{i+1}_python_only.png', dpi=150)
            plt.close()
            print(f"  Saved Python-only plot")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if results:
        print("\nComparison Results:")
        print(f"{'Sample':<10} {'Max Diff':<15} {'Mean Diff':<15} {'Max Rel Diff':<15}")
        print("-" * 60)
        for r in results:
            print(f"{r['sample']:<10} {r['max_diff']:<15.6e} {r['mean_diff']:<15.6e} {r['max_rel_diff']:<15.6e}")
        
        avg_max_diff = np.mean([r['max_diff'] for r in results])
        avg_max_rel_diff = np.mean([r['max_rel_diff'] for r in results])
        print(f"\nAverage max difference: {avg_max_diff:.6e}")
        print(f"Average max relative difference: {avg_max_rel_diff:.6e}")
    
    print(f"\nPlots saved to: comparison_plots/")
    print("=" * 70)


if __name__ == '__main__':
    main()

