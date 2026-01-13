"""
Investigate sources of eigenvalue discrepancies:
1. T Matrix (Transformation Matrix) differences
3. Sparse Matrix Operations differences

This script compares T matrices, Kr/Mr matrices, and sparse operations
between original and reconstructed datasets.
"""

import numpy as np
import scipy.sparse as sp
import h5py
from pathlib import Path
import sys

# Add 2d-dispersion-py to path
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))

# Import utilities
try:
    from system_matrices import get_transformation_matrix
    from system_matrices_vec import get_system_matrices_VEC
    import NO_utils
except ImportError as e:
    print(f"Error importing modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def load_mat_data_safe(data_path):
    """Load MATLAB .mat file data."""
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


def compare_matrices(mat1, mat2, name="Matrix", tolerance=1e-10):
    """Compare two matrices (sparse or dense) and report differences."""
    # Convert to dense if sparse
    if sp.issparse(mat1):
        mat1 = mat1.toarray()
    if sp.issparse(mat2):
        mat2 = mat2.toarray()
    
    # Check shapes
    if mat1.shape != mat2.shape:
        print(f"  {name}: Shape mismatch! {mat1.shape} vs {mat2.shape}")
        return False, None, None
    
    # Compute differences
    diff = mat1 - mat2
    abs_diff = np.abs(diff)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Percentage error (relative to original)
    abs_mat1 = np.abs(mat1)
    # Avoid division by zero
    mask = abs_mat1 > 1e-15
    if np.any(mask):
        percent_error = np.zeros_like(abs_diff)
        percent_error[mask] = 100.0 * abs_diff[mask] / abs_mat1[mask]
        max_percent_error = np.max(percent_error)
        mean_percent_error = np.mean(percent_error[mask])
    else:
        max_percent_error = 0.0
        mean_percent_error = 0.0
    
    # Relative difference (alternative metric)
    max_val = max(np.max(np.abs(mat1)), np.max(np.abs(mat2)))
    if max_val > 0:
        max_rel_diff = max_abs_diff / max_val
    else:
        max_rel_diff = 0
    
    # Check if they match
    matches = max_abs_diff < tolerance
    
    print(f"  {name}:")
    print(f"    Shape: {mat1.shape}")
    print(f"    Max absolute difference: {max_abs_diff:.6e}")
    print(f"    Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"    Max relative difference: {max_rel_diff:.6e}")
    print(f"    Max percentage error: {max_percent_error:.6e}%")
    print(f"    Mean percentage error: {mean_percent_error:.6e}%")
    print(f"    Match (tol={tolerance:.1e}): {'OK' if matches else 'FAIL'}")
    
    if not matches and max_abs_diff > 0:
        # Find location of max difference
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"    Max diff location: {idx}")
        print(f"    Original value: {mat1[idx]:.6e}")
        print(f"    Reconstructed value: {mat2[idx]:.6e}")
    
    return matches, max_percent_error, mean_percent_error


def investigate_t_matrices(data_orig, data_recon, struct_idx=0, n_wavevectors_to_test=5):
    """Investigate T matrix differences."""
    print("=" * 80)
    print("INVESTIGATION 1: T MATRIX DIFFERENCES")
    print("=" * 80)
    
    # Get const parameters
    const_orig = data_orig['const']
    const_recon = data_recon['const']
    
    # Get wavevectors
    wavevectors_orig = data_orig['WAVEVECTOR_DATA']
    wavevectors_recon = data_recon['WAVEVECTOR_DATA']
    
    # Check if wavevectors match
    print("\n1.1: Comparing wavevectors")
    if np.array_equal(wavevectors_orig, wavevectors_recon):
        print("  OK: Wavevectors match exactly")
    else:
        diff = np.abs(wavevectors_orig - wavevectors_recon)
        print(f"  FAIL: Wavevectors differ!")
        print(f"    Max difference: {np.max(diff):.6e}")
        print(f"    Mean difference: {np.mean(diff):.6e}")
    
    # Check const parameters
    print("\n1.2: Comparing const parameters")
    const_keys = ['N_pix', 'N_ele', 'a', 'E_min', 'E_max', 'rho_min', 'rho_max', 
                  'nu_min', 'nu_max', 'sigma_eig', 'N_eig']
    for key in const_keys:
        if key in const_orig and key in const_recon:
            val_orig = float(np.array(const_orig[key]).item())
            val_recon = float(np.array(const_recon[key]).item())
            if val_orig == val_recon:
                print(f"  OK: {key}: {val_orig}")
            else:
                print(f"  FAIL: {key}: {val_orig} vs {val_recon} (diff: {abs(val_orig - val_recon):.6e})")
    
    # Test T matrix computation for a few wavevectors
    print(f"\n1.3: Comparing T matrices for {n_wavevectors_to_test} wavevectors")
    
    # Prepare const dict for T matrix computation
    const_for_t = {
        'N_pix': int(np.array(const_orig['N_pix']).item()),
        'N_ele': int(np.array(const_orig['N_ele']).item()),
        'a': float(np.array(const_orig['a']).item())
    }
    
    t_mismatches = 0
    for wv_idx in range(min(n_wavevectors_to_test, wavevectors_orig.shape[0])):
        # Extract wavevector as 1D array
        wv_orig = np.array(wavevectors_orig[wv_idx, :]).flatten()[:2].astype(np.float32)
        wv_recon = np.array(wavevectors_recon[wv_idx, :]).flatten()[:2].astype(np.float32)
        
        print(f"\n  Wavevector {wv_idx}:")
        print(f"    Original: [{wv_orig[0]:.6f}, {wv_orig[1]:.6f}]")
        print(f"    Reconstructed: [{wv_recon[0]:.6f}, {wv_recon[1]:.6f}]")
        
        # Compute T matrices
        try:
            T_orig = get_transformation_matrix(wv_orig, const_for_t)
            T_recon = get_transformation_matrix(wv_recon, const_for_t)
            
            # Compare T matrices
            matches, max_pct_err, mean_pct_err = compare_matrices(T_orig, T_recon, f"T[{wv_idx}]", tolerance=1e-10)
            if not matches:
                t_mismatches += 1
                # Store percentage errors for summary
                if 't_percent_errors' not in locals():
                    t_percent_errors = []
                t_percent_errors.append((max_pct_err, mean_pct_err))
                
                # Check if it's just a storage format difference
                if sp.issparse(T_orig) and sp.issparse(T_recon):
                    print(f"    T[{wv_idx}] storage formats:")
                    print(f"      Original: {type(T_orig).__name__}")
                    print(f"      Reconstructed: {type(T_recon).__name__}")
                    
        except Exception as e:
            print(f"    Error computing T matrices: {e}")
            t_mismatches += 1
    
    print(f"\n1.4: Summary")
    print(f"  T matrix mismatches: {t_mismatches}/{n_wavevectors_to_test}")
    
    # Compute overall statistics for T matrices
    if 't_percent_errors' in locals() and len(t_percent_errors) > 0:
        all_max_pct = [e[0] for e in t_percent_errors]
        all_mean_pct = [e[1] for e in t_percent_errors]
        print(f"  T matrix percentage errors:")
        print(f"    Max percentage error: {max(all_max_pct):.6e}%")
        print(f"    Mean percentage error: {np.mean(all_mean_pct):.6e}%")
        print(f"    Average of max errors: {np.mean(all_max_pct):.6e}%")
    
    return t_mismatches == 0


def investigate_sparse_operations(data_orig, data_recon, struct_idx=0, n_wavevectors_to_test=3, n_bands_to_test=2):
    """Investigate sparse matrix operation differences."""
    print("\n" + "=" * 80)
    print("INVESTIGATION 3: SPARSE MATRIX OPERATIONS")
    print("=" * 80)
    
    # Load K and M matrices (they should match, but let's verify)
    print("\n3.1: Loading K and M matrices")
    
    # For original, we need to compute K and M
    const_orig = data_orig['const']
    const_for_km = {
        'N_pix': int(np.array(const_orig['N_pix']).item()),
        'N_ele': int(np.array(const_orig['N_ele']).item()),
        'a': float(np.array(const_orig['a']).item()),
        'E_min': float(np.array(const_orig['E_min']).item()),
        'E_max': float(np.array(const_orig['E_max']).item()),
        'rho_min': float(np.array(const_orig['rho_min']).item()),
        'rho_max': float(np.array(const_orig['rho_max']).item())
    }
    # Add nu_min and nu_max if they exist
    if 'nu_min' in const_orig:
        const_for_km['nu_min'] = float(np.array(const_orig['nu_min']).item())
    if 'nu_max' in const_orig:
        const_for_km['nu_max'] = float(np.array(const_orig['nu_max']).item())
    
    # Get design for this structure - already in 3-channel format (3, H, W)
    # Need to transpose to (H, W, 3) for get_system_matrices_VEC
    design_orig = data_orig['designs'][struct_idx, :, :, :]  # (3, H, W)
    design_orig_3ch = design_orig.transpose(1, 2, 0)  # (H, W, 3)
    
    # Add design and other required parameters to const_for_km (matching reconstruction code)
    const_for_km['design'] = design_orig_3ch
    const_for_km['t'] = float(np.array(const_orig.get('t', 1.0)).item()) if 't' in const_orig else 1.0
    const_for_km['design_scale'] = 'linear'
    const_for_km['isUseImprovement'] = True
    const_for_km['isUseSecondImprovement'] = False
    
    # Add Poisson ratio parameters (check both nu_* and poisson_* keys)
    if 'nu_min' in const_orig:
        const_for_km['poisson_min'] = float(np.array(const_orig['nu_min']).item())
    elif 'poisson_min' in const_orig:
        const_for_km['poisson_min'] = float(np.array(const_orig['poisson_min']).item())
    else:
        const_for_km['poisson_min'] = 0.3  # Default
    
    if 'nu_max' in const_orig:
        const_for_km['poisson_max'] = float(np.array(const_orig['nu_max']).item())
    elif 'poisson_max' in const_orig:
        const_for_km['poisson_max'] = float(np.array(const_orig['poisson_max']).item())
    else:
        const_for_km['poisson_max'] = 0.3  # Default
    
    # Compute K and M using the same method as reconstruction
    K, M = get_system_matrices_VEC(const_for_km)
    
    print(f"  K: shape={K.shape}, nnz={K.nnz if sp.issparse(K) else K.size}, dtype={K.dtype}")
    print(f"  M: shape={M.shape}, nnz={M.nnz if sp.issparse(M) else M.size}, dtype={M.dtype}")
    
    # Get wavevectors and eigenvectors
    wavevectors = data_orig['WAVEVECTOR_DATA']
    eigenvectors_orig = data_orig['EIGENVECTOR_DATA'][struct_idx, :, :, :]  # (n_eig, n_wv, n_dof)
    eigenvectors_recon = data_recon['EIGENVECTOR_DATA'][struct_idx, :, :, :]
    
    # Prepare const for T matrix
    const_for_t = {
        'N_pix': int(np.array(const_orig['N_pix']).item()),
        'N_ele': int(np.array(const_orig['N_ele']).item()),
        'a': float(np.array(const_orig['a']).item())
    }
    
    print(f"\n3.2: Comparing Kr and Mr matrices for {n_wavevectors_to_test} wavevectors")
    
    # Get wavevectors from both datasets
    wavevectors_orig = data_orig['WAVEVECTOR_DATA']
    wavevectors_recon = data_recon['WAVEVECTOR_DATA']
    
    # Store percentage errors for summary
    kr_percent_errors = []
    mr_percent_errors = []
    
    kr_mr_mismatches = 0
    for wv_idx in range(min(n_wavevectors_to_test, wavevectors_orig.shape[0])):
        # Get wavevectors from both datasets
        wv_orig = np.array(wavevectors_orig[wv_idx, :]).flatten()[:2].astype(np.float32)
        wv_recon = np.array(wavevectors_recon[wv_idx, :]).flatten()[:2].astype(np.float32)
        
        print(f"\n  Wavevector {wv_idx}:")
        print(f"    Original: [{wv_orig[0]:.6f}, {wv_orig[1]:.6f}]")
        print(f"    Reconstructed: [{wv_recon[0]:.6f}, {wv_recon[1]:.6f}]")
        
        # Compute T matrices for both
        T_orig = get_transformation_matrix(wv_orig, const_for_t)
        T_recon = get_transformation_matrix(wv_recon, const_for_t)
        
        # Convert to sparse (matching reconstruction code)
        T_orig_sparse = T_orig if sp.issparse(T_orig) else sp.csr_matrix(T_orig.astype(np.float32))
        T_recon_sparse = T_recon if sp.issparse(T_recon) else sp.csr_matrix(T_recon.astype(np.float32))
        K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
        M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
        
        # Compute Kr and Mr for original wavevector
        Kr_orig = T_orig_sparse.conj().T @ K_sparse @ T_orig_sparse
        Mr_orig = T_orig_sparse.conj().T @ M_sparse @ T_orig_sparse
        
        # Compute Kr and Mr for reconstructed wavevector
        Kr_recon = T_recon_sparse.conj().T @ K_sparse @ T_recon_sparse
        Mr_recon = T_recon_sparse.conj().T @ M_sparse @ T_recon_sparse
        
        print(f"    Kr_orig: shape={Kr_orig.shape}, nnz={Kr_orig.nnz if sp.issparse(Kr_orig) else Kr_orig.size}")
        print(f"    Mr_orig: shape={Mr_orig.shape}, nnz={Mr_orig.nnz if sp.issparse(Mr_orig) else Mr_orig.size}")
        
        # Compare Kr matrices
        print(f"\n    Comparing Kr matrices:")
        kr_matches, kr_max_pct, kr_mean_pct = compare_matrices(Kr_orig, Kr_recon, "Kr", tolerance=1e-10)
        if not kr_matches:
            kr_mr_mismatches += 1
            kr_percent_errors.append((kr_max_pct, kr_mean_pct))
        
        # Compare Mr matrices
        print(f"\n    Comparing Mr matrices:")
        mr_matches, mr_max_pct, mr_mean_pct = compare_matrices(Mr_orig, Mr_recon, "Mr", tolerance=1e-10)
        if not mr_matches:
            mr_percent_errors.append((mr_max_pct, mr_mean_pct))
        
    # Print summary statistics
    print(f"\n3.3: Summary Statistics")
    print(f"  Kr matrix mismatches: {kr_mr_mismatches}/{n_wavevectors_to_test}")
    if len(kr_percent_errors) > 0:
        all_kr_max_pct = [e[0] for e in kr_percent_errors]
        all_kr_mean_pct = [e[1] for e in kr_percent_errors]
        print(f"  Kr matrix percentage errors:")
        print(f"    Max percentage error: {max(all_kr_max_pct):.6e}%")
        print(f"    Mean percentage error: {np.mean(all_kr_mean_pct):.6e}%")
        print(f"    Average of max errors: {np.mean(all_kr_max_pct):.6e}%")
    
    print(f"  Mr matrix mismatches: {len(mr_percent_errors)}/{n_wavevectors_to_test}")
    if len(mr_percent_errors) > 0:
        all_mr_max_pct = [e[0] for e in mr_percent_errors]
        all_mr_mean_pct = [e[1] for e in mr_percent_errors]
        print(f"  Mr matrix percentage errors:")
        print(f"    Max percentage error: {max(all_mr_max_pct):.6e}%")
        print(f"    Mean percentage error: {np.mean(all_mr_mean_pct):.6e}%")
        print(f"    Average of max errors: {np.mean(all_mr_max_pct):.6e}%")
    
    # Test with a few eigenvectors for eigenvalue computation
    print(f"\n3.4: Testing eigenvalue computation with Kr/Mr differences")
    for wv_idx in range(min(2, wavevectors_orig.shape[0])):  # Test just 2 wavevectors
        wv_orig = np.array(wavevectors_orig[wv_idx, :]).flatten()[:2].astype(np.float32)
        wv_recon = np.array(wavevectors_recon[wv_idx, :]).flatten()[:2].astype(np.float32)
        
        T_orig = get_transformation_matrix(wv_orig, const_for_t)
        T_recon = get_transformation_matrix(wv_recon, const_for_t)
        
        T_orig_sparse = T_orig if sp.issparse(T_orig) else sp.csr_matrix(T_orig.astype(np.float32))
        T_recon_sparse = T_recon if sp.issparse(T_recon) else sp.csr_matrix(T_recon.astype(np.float32))
        K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
        M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
        
        Kr_orig = T_orig_sparse.conj().T @ K_sparse @ T_orig_sparse
        Mr_orig = T_orig_sparse.conj().T @ M_sparse @ T_orig_sparse
        Kr_recon = T_recon_sparse.conj().T @ K_sparse @ T_recon_sparse
        Mr_recon = T_recon_sparse.conj().T @ M_sparse @ T_recon_sparse
        
        for band_idx in range(min(n_bands_to_test, eigenvectors_orig.shape[0])):
            print(f"\n    Band {band_idx}:")
            
            # Get eigenvectors
            eigvec_orig = eigenvectors_orig[band_idx, wv_idx, :].astype(np.complex128)
            eigvec_recon = eigenvectors_recon[band_idx, wv_idx, :].astype(np.complex128)
            
            # Check if eigenvectors match
            eigvec_diff = np.abs(eigvec_orig - eigvec_recon)
            if np.max(eigvec_diff) > 1e-10:
                print(f"      WARNING: Eigenvectors differ! Max diff: {np.max(eigvec_diff):.6e}")
            else:
                print(f"      OK: Eigenvectors match")
            
            # Use original eigenvector for both computations
            eigvec = eigvec_orig
            
            # Compute matrix-vector products with original Kr/Mr
            Kr_eigvec_orig = Kr_orig @ eigvec
            Mr_eigvec_orig = Mr_orig @ eigvec
            
            # Compute matrix-vector products with reconstructed Kr/Mr
            Kr_eigvec_recon = Kr_recon @ eigvec
            Mr_eigvec_recon = Mr_recon @ eigvec
            
            # Convert to dense if sparse
            if sp.issparse(Kr_eigvec_orig):
                Kr_eigvec_orig_dense = Kr_eigvec_orig.toarray().flatten()
            else:
                Kr_eigvec_orig_dense = Kr_eigvec_orig.flatten()
            
            if sp.issparse(Mr_eigvec_orig):
                Mr_eigvec_orig_dense = Mr_eigvec_orig.toarray().flatten()
            else:
                Mr_eigvec_orig_dense = Mr_eigvec_orig.flatten()
            
            if sp.issparse(Kr_eigvec_recon):
                Kr_eigvec_recon_dense = Kr_eigvec_recon.toarray().flatten()
            else:
                Kr_eigvec_recon_dense = Kr_eigvec_recon.flatten()
            
            if sp.issparse(Mr_eigvec_recon):
                Mr_eigvec_recon_dense = Mr_eigvec_recon.toarray().flatten()
            else:
                Mr_eigvec_recon_dense = Mr_eigvec_recon.flatten()
            
            # Compute eigenvalues using norm formula
            eigval_orig = np.linalg.norm(Kr_eigvec_orig_dense) / np.linalg.norm(Mr_eigvec_orig_dense)
            eigval_recon = np.linalg.norm(Kr_eigvec_recon_dense) / np.linalg.norm(Mr_eigvec_recon_dense)
            freq_orig = np.sqrt(np.real(eigval_orig)) / (2 * np.pi)
            freq_recon = np.sqrt(np.real(eigval_recon)) / (2 * np.pi)
            
            # Compare with stored eigenvalue
            stored_eigval_orig = data_orig['EIGENVALUE_DATA'][struct_idx, band_idx, wv_idx]
            stored_eigval_recon = data_recon['EIGENVALUE_DATA'][struct_idx, band_idx, wv_idx]
            
            print(f"      Computed frequency (orig wv): {freq_orig:.6e}")
            print(f"      Computed frequency (recon wv): {freq_recon:.6e}")
            print(f"      Stored original: {stored_eigval_orig:.6e}")
            print(f"      Stored reconstructed: {stored_eigval_recon:.6e}")
            print(f"      Frequency difference (orig vs recon wv): {abs(freq_orig - freq_recon):.6e}")
            
            # Compute percentage error in frequency
            if freq_orig > 0:
                freq_pct_err = 100.0 * abs(freq_orig - freq_recon) / freq_orig
                print(f"      Frequency percentage error: {freq_pct_err:.6e}%")
            


def main():
    """Main investigation function."""
    # File paths
    file_orig = Path(r"D:\Research\NO-2D-Metamaterials\data\out_test_10_mat_original\out_binarized_1.mat")
    file_recon = Path(r"D:\Research\NO-2D-Metamaterials\data\out_test_10_mat_reconstructed\out_binarized_1.mat")
    
    print("Loading data files...")
    data_orig = load_mat_data_safe(file_orig)
    data_recon = load_mat_data_safe(file_recon)
    
    print(f"Original: {data_orig['EIGENVALUE_DATA'].shape}")
    print(f"Reconstructed: {data_recon['EIGENVALUE_DATA'].shape}")
    
    # Test with first structure
    struct_idx = 0
    
    # Investigation 1: T Matrix differences
    t_matches = investigate_t_matrices(data_orig, data_recon, struct_idx, n_wavevectors_to_test=5)
    
    # Investigation 3: Sparse matrix operations
    investigate_sparse_operations(data_orig, data_recon, struct_idx, 
                                  n_wavevectors_to_test=3, n_bands_to_test=2)
    
    print("\n" + "=" * 80)
    print("INVESTIGATION SUMMARY")
    print("=" * 80)
    print(f"T matrices match: {'OK' if t_matches else 'FAIL'}")
    print("\nCheck output above for detailed sparse matrix operation differences.")


if __name__ == "__main__":
    main()

