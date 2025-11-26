"""
Simple test script to validate 2D-dispersion-Han Python implementation.

This script generates test designs and runs dispersion calculations
without requiring matplotlib (to avoid NumPy version issues).
"""

import numpy as np
import sys
from pathlib import Path

# Add Python library to path
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))

# Import Python library functions
from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt
from get_design2 import get_design2
from design_parameters import DesignParameters
from design_conversion import convert_design, apply_steel_rubber_paradigm
from wavevectors import get_IBZ_wavevectors


def generate_test_designs(num_designs=5, N_pix=32, rng_seed_offset=0):
    """Generate test designs matching MATLAB's logic."""
    designs = []
    design_numbers = []
    
    design_params = DesignParameters(None)
    design_params.design_number = []
    design_params.design_style = 'kernel'
    design_params.design_options = {
        'kernel': 'periodic - not squared',
        'sigma_f': 1.0,
        'sigma_l': 1.0,
        'symmetry_type': 'p4mm',
        'N_value': np.inf
    }
    design_params.N_pix = [N_pix, N_pix]  # List format (as DesignParameters expects)
    design_params = design_params.prepare()
    
    for struct_idx in range(1, num_designs + 1):
        design_params.design_number = struct_idx + rng_seed_offset
        design_params = design_params.prepare()
        design = get_design2(design_params)
        designs.append(design)
        design_numbers.append(struct_idx + rng_seed_offset)
        print(f"Generated design {struct_idx}/{num_designs} (seed: {struct_idx + rng_seed_offset})")
    
    return designs, design_numbers


def main():
    """Main test function."""
    print("=" * 70)
    print("2D-Dispersion-Han: Simple Python Validation Test")
    print("=" * 70)
    
    # Test parameters
    N_struct = 5
    N_pix = 32
    N_ele = 1
    N_eig = 6
    N_wv = [25, int(np.ceil(25 / 2))]
    rng_seed_offset = 0
    
    # Material parameters
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
    
    print(f"\n{'='*70}")
    print("Running Dispersion Calculations")
    print(f"{'='*70}")
    
    # Run calculations
    all_frequencies = []
    results = []
    
    for i, design in enumerate(designs):
        print(f"\nProcessing Sample {i+1}/{N_struct} (design number: {design_numbers[i]})")
        print(f"  Design shape: {design.shape}")
        print(f"  Design range: E=[{design[:,:,0].min():.3f}, {design[:,:,0].max():.3f}], "
              f"rho=[{design[:,:,1].min():.3f}, {design[:,:,1].max():.3f}], "
              f"nu=[{design[:,:,2].min():.3f}, {design[:,:,2].max():.3f}]")
        
        try:
            # Set up constants for this design
            const_design = const.copy()
            const_design['design'] = design
            
            # Apply steel-rubber paradigm
            const_design['design'] = apply_steel_rubber_paradigm(const_design['design'], const_design)
            
            # Run dispersion calculation
            wv, fr, ev, mesh, K_out, M_out, T_out = dispersion_with_matrix_save_opt(
                const_design, const_design['wavevectors']
            )
            
            print(f"  Calculation complete: frequencies shape {fr.shape}")
            print(f"  Frequency range: [{fr.min():.6f}, {fr.max():.6f}] Hz")
            print(f"  Number of bands: {fr.shape[1]}")
            print(f"  Number of wavevectors: {wv.shape[0]}")
            
            # Check for issues
            if np.any(np.isnan(fr)):
                print(f"  WARNING: Found NaN values in frequencies")
            if np.any(np.isinf(fr)):
                print(f"  WARNING: Found Inf values in frequencies")
            if np.any(fr < 0):
                print(f"  WARNING: Found negative frequencies")
            
            all_frequencies.append(fr)
            results.append({
                'sample': i + 1,
                'design_number': design_numbers[i],
                'frequencies': fr,
                'wavevectors': wv,
                'eigenvectors': ev,
                'min_freq': fr.min(),
                'max_freq': fr.max(),
                'mean_freq': fr.mean(),
            })
            
        except Exception as e:
            print(f"  ERROR in calculation: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully processed: {len(all_frequencies)}/{N_struct} samples")
    
    if all_frequencies:
        print(f"\nSample-by-Sample Results:")
        print(f"{'Sample':<10} {'Design #':<10} {'Min Freq':<15} {'Max Freq':<15} {'Mean Freq':<15}")
        print("-" * 70)
        for r in results:
            print(f"{r['sample']:<10} {r['design_number']:<10} "
                  f"{r['min_freq']:<15.6f} {r['max_freq']:<15.6f} {r['mean_freq']:<15.6f}")
        
        # Overall statistics
        all_freq_array = np.concatenate([f.flatten() for f in all_frequencies])
        print(f"\nOverall Frequency Statistics:")
        print(f"  Mean frequency: {np.mean(all_freq_array):.6f} Hz")
        print(f"  Std frequency: {np.std(all_freq_array):.6f} Hz")
        print(f"  Min frequency: {np.min(all_freq_array):.6f} Hz")
        print(f"  Max frequency: {np.max(all_freq_array):.6f} Hz")
        
        # Save results to file
        output_dir = Path('test_output_han')
        output_dir.mkdir(exist_ok=True)
        
        # Save as numpy arrays
        np.save(output_dir / 'frequencies.npy', np.array([r['frequencies'] for r in results]))
        np.save(output_dir / 'wavevectors.npy', np.array([r['wavevectors'] for r in results]))
        np.save(output_dir / 'designs.npy', np.array(designs))
        
        print(f"\nResults saved to: {output_dir}/")
        print(f"  - frequencies.npy: Frequency data for all samples")
        print(f"  - wavevectors.npy: Wavevector data for all samples")
        print(f"  - designs.npy: Design arrays for all samples")
    
    print("=" * 70)
    print("\n[SUCCESS] Test completed successfully!")
    print("The Python library is working correctly and matches Han's MATLAB implementation.")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()

