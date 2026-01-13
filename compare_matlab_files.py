#!/usr/bin/env python3
"""
Compare original and reconstructed MATLAB .mat files
"""

import numpy as np
import sys
from pathlib import Path
import h5py
import os

# Custom utilities
try:
    import NO_utils
except ImportError as e:
    print(f"Error importing NO_utils: {e}")
    sys.exit(1)

def load_mat_data_safe(data_path):
    """
    Load MATLAB .mat file data, handling optional fields like rng_seed_offset.
    """
    with h5py.File(data_path, 'r') as file:
        EIGENVALUE_DATA = np.array(file['EIGENVALUE_DATA'])
        EIGENVECTOR_DATA_real = np.array(file['EIGENVECTOR_DATA']['real'])
        EIGENVECTOR_DATA_imag = np.array(file['EIGENVECTOR_DATA']['imag'])
        EIGENVECTOR_DATA = EIGENVECTOR_DATA_real + 1j * EIGENVECTOR_DATA_imag
        WAVEVECTOR_DATA = np.array(file['WAVEVECTOR_DATA'])
        
        const = {key: np.array(file['const'][key]) for key in file['const']}
        
        N_struct = np.array(file['N_struct'])
        design_params = np.array(file['design_params'])
        designs = np.array(file['designs'])
        imag_tol = np.array(file['imag_tol'])
        
        # Handle optional rng_seed_offset
        rng_seed_offset = None
        if 'rng_seed_offset' in file:
            rng_seed_offset = np.array(file['rng_seed_offset'])
    
    return {
        'EIGENVALUE_DATA': EIGENVALUE_DATA,
        'EIGENVECTOR_DATA': EIGENVECTOR_DATA,
        'WAVEVECTOR_DATA': WAVEVECTOR_DATA,
        'const': const,
        'N_struct': N_struct,
        'design_params': design_params,
        'designs': designs,
        'imag_tol': imag_tol,
        'rng_seed_offset': rng_seed_offset
    }

def compare_matlab_files(original_path, reconstructed_path):
    """
    Compare two MATLAB .mat files and report differences.
    """
    print("=" * 80)
    print("Comparing MATLAB Files")
    print("=" * 80)
    print(f"Original: {original_path}")
    print(f"Reconstructed: {reconstructed_path}")
    print("=" * 80)
    
    # Load original file
    print("\nLoading original file...")
    data_orig = load_mat_data_safe(original_path)
    
    # Load reconstructed file
    print("Loading reconstructed file...")
    data_recon = load_mat_data_safe(reconstructed_path)
    
    # Extract arrays
    designs_orig = data_orig['designs']
    design_params_orig = data_orig['design_params']
    WAVEVECTOR_DATA_orig = data_orig['WAVEVECTOR_DATA']
    EIGENVALUE_DATA_orig = data_orig['EIGENVALUE_DATA']
    EIGENVECTOR_DATA_orig = data_orig['EIGENVECTOR_DATA']
    const_orig = data_orig['const']
    N_struct_orig = data_orig['N_struct']
    imag_tol_orig = data_orig['imag_tol']
    rng_seed_offset_orig = data_orig['rng_seed_offset']
    
    designs_recon = data_recon['designs']
    design_params_recon = data_recon['design_params']
    WAVEVECTOR_DATA_recon = data_recon['WAVEVECTOR_DATA']
    EIGENVALUE_DATA_recon = data_recon['EIGENVALUE_DATA']
    EIGENVECTOR_DATA_recon = data_recon['EIGENVECTOR_DATA']
    const_recon = data_recon['const']
    N_struct_recon = data_recon['N_struct']
    imag_tol_recon = data_recon['imag_tol']
    rng_seed_offset_recon = data_recon['rng_seed_offset']
    
    # Extract dimensions
    n_designs_orig = designs_orig.shape[0]
    n_panes_orig = designs_orig.shape[1]
    design_res_orig = designs_orig.shape[2]
    
    # Handle WAVEVECTOR_DATA shape: could be (n_designs, 2, n_wv) or (n_designs, n_wv, 2)
    if WAVEVECTOR_DATA_orig.ndim == 3:
        if WAVEVECTOR_DATA_orig.shape[1] == 2:
            n_wavevectors_orig = WAVEVECTOR_DATA_orig.shape[2]
        else:
            n_wavevectors_orig = WAVEVECTOR_DATA_orig.shape[1]
    else:
        n_wavevectors_orig = WAVEVECTOR_DATA_orig.shape[0]
    
    # Handle EIGENVALUE_DATA shape: could be (n_designs, n_bands, n_wv) or (n_designs, n_wv, n_bands)
    if EIGENVALUE_DATA_orig.ndim == 3:
        # Try to infer: if middle dim is 6, it's likely n_bands
        if EIGENVALUE_DATA_orig.shape[1] == 6:
            n_bands_orig = EIGENVALUE_DATA_orig.shape[1]
        elif EIGENVALUE_DATA_orig.shape[2] == 6:
            n_bands_orig = EIGENVALUE_DATA_orig.shape[2]
        else:
            n_bands_orig = EIGENVALUE_DATA_orig.shape[1]  # Default assumption
    else:
        n_bands_orig = EIGENVALUE_DATA_orig.shape[1]
    
    n_designs_recon = designs_recon.shape[0]
    n_panes_recon = designs_recon.shape[1]
    design_res_recon = designs_recon.shape[2]
    
    if WAVEVECTOR_DATA_recon.ndim == 3:
        if WAVEVECTOR_DATA_recon.shape[1] == 2:
            n_wavevectors_recon = WAVEVECTOR_DATA_recon.shape[2]
        else:
            n_wavevectors_recon = WAVEVECTOR_DATA_recon.shape[1]
    else:
        n_wavevectors_recon = WAVEVECTOR_DATA_recon.shape[0]
    
    if EIGENVALUE_DATA_recon.ndim == 3:
        if EIGENVALUE_DATA_recon.shape[1] == 6:
            n_bands_recon = EIGENVALUE_DATA_recon.shape[1]
        elif EIGENVALUE_DATA_recon.shape[2] == 6:
            n_bands_recon = EIGENVALUE_DATA_recon.shape[2]
        else:
            n_bands_recon = EIGENVALUE_DATA_recon.shape[1]
    else:
        n_bands_recon = EIGENVALUE_DATA_recon.shape[1]
    
    print(f"\nDebug: EIGENVECTOR_DATA_orig shape: {EIGENVECTOR_DATA_orig.shape}")
    print(f"Debug: n_designs={n_designs_orig}, n_bands={n_bands_orig}, n_wavevectors={n_wavevectors_orig}, design_res={design_res_orig}")
    
    # Skip splitting EIGENVECTOR_DATA for now - just compare the full arrays
    EIGENVECTOR_DATA_x_orig = None
    EIGENVECTOR_DATA_y_orig = None
    EIGENVECTOR_DATA_x_recon = None
    EIGENVECTOR_DATA_y_recon = None
    
    if EIGENVECTOR_DATA_recon.ndim == 4:
        n_dof = EIGENVECTOR_DATA_recon.shape[3]
        expected_dof = 2 * design_res_recon * design_res_recon
        if n_dof == expected_dof:
            EIGENVECTOR_DATA_x_flat = EIGENVECTOR_DATA_recon[:, :, :, 0::2]
            EIGENVECTOR_DATA_y_flat = EIGENVECTOR_DATA_recon[:, :, :, 1::2]
            EIGENVECTOR_DATA_x_recon = EIGENVECTOR_DATA_x_flat.reshape(n_designs_recon, n_bands_recon, n_wavevectors_recon, design_res_recon, design_res_recon)
            EIGENVECTOR_DATA_y_recon = EIGENVECTOR_DATA_y_flat.reshape(n_designs_recon, n_bands_recon, n_wavevectors_recon, design_res_recon, design_res_recon)
            EIGENVECTOR_DATA_x_recon = EIGENVECTOR_DATA_x_recon.transpose(0, 2, 1, 3, 4)
            EIGENVECTOR_DATA_y_recon = EIGENVECTOR_DATA_y_recon.transpose(0, 2, 1, 3, 4)
        else:
            print(f"  WARNING: EIGENVECTOR_DATA DOF mismatch: {n_dof} vs expected {expected_dof}")
            EIGENVECTOR_DATA_x_recon = None
            EIGENVECTOR_DATA_y_recon = None
    else:
        EIGENVECTOR_DATA_x_recon = None
        EIGENVECTOR_DATA_y_recon = None
    
    # Generate WAVEFORM_DATA (not critical for comparison, skip for now)
    WAVEFORM_DATA_orig = None
    WAVEFORM_DATA_recon = None
    n_dim_orig = 2
    n_dim_recon = 2
    
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    
    # Compare basic dimensions
    print("\n1. Basic Dimensions:")
    print(f"   Designs: {n_designs_orig} vs {n_designs_recon} {'✓' if n_designs_orig == n_designs_recon else '✗'}")
    print(f"   Panes: {n_panes_orig} vs {n_panes_recon} {'✓' if n_panes_orig == n_panes_recon else '✗'}")
    print(f"   Design resolution: {design_res_orig} vs {design_res_recon} {'✓' if design_res_orig == design_res_recon else '✗'}")
    print(f"   Wavevectors: {n_wavevectors_orig} vs {n_wavevectors_recon} {'✓' if n_wavevectors_orig == n_wavevectors_recon else '✗'}")
    print(f"   Bands: {n_bands_orig} vs {n_bands_recon} {'✓' if n_bands_orig == n_bands_recon else '✗'}")
    
    # Compare array shapes
    print("\n2. Array Shapes:")
    arrays_to_compare = [
        ('designs', designs_orig, designs_recon),
        ('design_params', design_params_orig, design_params_recon),
        ('WAVEVECTOR_DATA', WAVEVECTOR_DATA_orig, WAVEVECTOR_DATA_recon),
        ('EIGENVALUE_DATA', EIGENVALUE_DATA_orig, EIGENVALUE_DATA_recon),
        ('EIGENVECTOR_DATA', EIGENVECTOR_DATA_orig, EIGENVECTOR_DATA_recon),
    ]
    
    # Add x/y components if available
    if EIGENVECTOR_DATA_x_orig is not None and EIGENVECTOR_DATA_x_recon is not None:
        arrays_to_compare.extend([
            ('EIGENVECTOR_DATA_x', EIGENVECTOR_DATA_x_orig, EIGENVECTOR_DATA_x_recon),
            ('EIGENVECTOR_DATA_y', EIGENVECTOR_DATA_y_orig, EIGENVECTOR_DATA_y_recon),
        ])
    
    for name, arr_orig, arr_recon in arrays_to_compare:
        shape_match = arr_orig.shape == arr_recon.shape
        print(f"   {name}: {arr_orig.shape} vs {arr_recon.shape} {'✓' if shape_match else '✗'}")
        if not shape_match:
            print(f"      WARNING: Shape mismatch!")
    
    # Compare numerical values
    print("\n3. Numerical Comparisons:")
    print("   (Using relative tolerance: 1e-5, absolute tolerance: 1e-8)")
    
    for name, arr_orig, arr_recon in arrays_to_compare:
        if arr_orig.shape != arr_recon.shape:
            print(f"   {name}: Skipped (shape mismatch)")
            continue
        
        # Handle complex arrays
        if np.iscomplexobj(arr_orig) or np.iscomplexobj(arr_recon):
            arr_orig = arr_orig.astype(np.complex128)
            arr_recon = arr_recon.astype(np.complex128)
        else:
            arr_orig = arr_orig.astype(np.float64)
            arr_recon = arr_recon.astype(np.float64)
        
        # Check for NaN/inf
        nan_orig = np.isnan(arr_orig).sum()
        nan_recon = np.isnan(arr_recon).sum()
        inf_orig = np.isinf(arr_orig).sum()
        inf_recon = np.isinf(arr_recon).sum()
        
        if nan_orig > 0 or nan_recon > 0:
            print(f"   {name}: NaN count - orig: {nan_orig}, recon: {nan_recon}")
        if inf_orig > 0 or inf_recon > 0:
            print(f"   {name}: Inf count - orig: {inf_orig}, recon: {inf_recon}")
        
        # Compare values (excluding NaN/inf)
        mask_valid = np.isfinite(arr_orig) & np.isfinite(arr_recon)
        if mask_valid.sum() == 0:
            print(f"   {name}: No valid values to compare")
            continue
        
        arr_orig_valid = arr_orig[mask_valid]
        arr_recon_valid = arr_recon[mask_valid]
        
        # Compute differences
        abs_diff = np.abs(arr_orig_valid - arr_recon_valid)
        rel_diff = abs_diff / (np.abs(arr_orig_valid) + 1e-10)  # Add small epsilon to avoid division by zero
        
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        # Check if arrays are close
        if np.iscomplexobj(arr_orig_valid):
            is_close = np.allclose(arr_orig_valid, arr_recon_valid, rtol=1e-5, atol=1e-8)
        else:
            is_close = np.allclose(arr_orig_valid, arr_recon_valid, rtol=1e-5, atol=1e-8)
        
        status = '✓' if is_close else '✗'
        print(f"   {name}: {status}")
        if not is_close:
            print(f"      Max abs diff: {max_abs_diff:.6e}")
            print(f"      Mean abs diff: {mean_abs_diff:.6e}")
            print(f"      Max rel diff: {max_rel_diff:.6e}")
            print(f"      Mean rel diff: {mean_rel_diff:.6e}")
            print(f"      Range orig: [{np.min(arr_orig_valid):.6e}, {np.max(arr_orig_valid):.6e}]")
            print(f"      Range recon: [{np.min(arr_recon_valid):.6e}, {np.max(arr_recon_valid):.6e}]")
    
    # Compare const parameters
    print("\n4. Const Parameters:")
    const_keys_orig = set(const_orig.keys()) if isinstance(const_orig, dict) else set()
    const_keys_recon = set(const_recon.keys()) if isinstance(const_recon, dict) else set()
    
    all_keys = const_keys_orig | const_keys_recon
    missing_in_recon = const_keys_orig - const_keys_recon
    extra_in_recon = const_keys_recon - const_keys_orig
    
    if missing_in_recon:
        print(f"   Missing in reconstructed: {missing_in_recon}")
    if extra_in_recon:
        print(f"   Extra in reconstructed: {extra_in_recon}")
    
    # Compare common keys
    common_keys = const_keys_orig & const_keys_recon
    for key in sorted(common_keys):
        val_orig = const_orig[key]
        val_recon = const_recon[key]
        
        if isinstance(val_orig, np.ndarray) and isinstance(val_recon, np.ndarray):
            if val_orig.shape != val_recon.shape:
                print(f"   {key}: Shape mismatch - {val_orig.shape} vs {val_recon.shape}")
            else:
                try:
                    if np.allclose(val_orig, val_recon, rtol=1e-5, atol=1e-8):
                        print(f"   {key}: ✓ Match")
                    else:
                        print(f"   {key}: ✗ Different")
                        print(f"      Orig: {val_orig}")
                        print(f"      Recon: {val_recon}")
                except:
                    print(f"   {key}: ✗ Cannot compare (different types or non-numeric)")
        else:
            if val_orig == val_recon:
                print(f"   {key}: ✓ Match")
            else:
                print(f"   {key}: ✗ Different")
                print(f"      Orig: {val_orig}")
                print(f"      Recon: {val_recon}")
    
    # Compare other metadata
    print("\n5. Other Metadata:")
    print(f"   N_struct: {N_struct_orig} vs {N_struct_recon} {'✓' if np.array_equal(N_struct_orig, N_struct_recon) else '✗'}")
    print(f"   imag_tol: {imag_tol_orig} vs {imag_tol_recon} {'✓' if np.array_equal(imag_tol_orig, imag_tol_recon) else '✗'}")
    if rng_seed_offset_orig is not None or rng_seed_offset_recon is not None:
        print(f"   rng_seed_offset: {rng_seed_offset_orig} vs {rng_seed_offset_recon}")
    
    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare two MATLAB .mat files')
    parser.add_argument('--file1', type=str, help='Path to first .mat file')
    parser.add_argument('--file2', type=str, help='Path to second .mat file')
    args = parser.parse_args()
    
    if args.file1 and args.file2:
        file1_path = Path(args.file1)
        file2_path = Path(args.file2)
    else:
        # Default comparison: ground truth vs conversion test
        file1_path = Path(r"D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat")
        file2_path = Path(r"D:\Research\NO-2D-Metamaterials\data\conversion_test_matlab\out_binarized_1.mat")
    
    if not file1_path.exists():
        print(f"ERROR: First file not found: {file1_path}")
        sys.exit(1)
    
    if not file2_path.exists():
        print(f"ERROR: Second file not found: {file2_path}")
        sys.exit(1)
    
    compare_matlab_files(file1_path, file2_path)
