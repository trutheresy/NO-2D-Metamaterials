"""
Compare Python-generated K & M matrices with MATLAB ground truth.

This script loads K & M matrices from both Python and MATLAB outputs,
and compares them strategically to quickly identify any discrepancies.
"""

import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import h5py


def load_python_matrices(data_dir, struct_idx=0):
    """Load K and M matrices from Python .pt files."""
    data_dir = Path(data_dir)
    K_data = torch.load(data_dir / 'K_data.pt', map_location='cpu')
    M_data = torch.load(data_dir / 'M_data.pt', map_location='cpu')
    
    # Convert to sparse if not already
    K = K_data[struct_idx]
    M = M_data[struct_idx]
    
    if not sp.issparse(K):
        K = sp.csr_matrix(K)
    if not sp.issparse(M):
        M = sp.csr_matrix(M)
    
    return K, M


def load_matlab_matrices(mat_file, struct_idx=0):
    """Load K and M matrices from MATLAB .mat file (HDF5 format)."""
    mat_file = Path(mat_file)
    
    with h5py.File(mat_file, 'r') as f:
        # Get references to K_DATA and M_DATA
        K_DATA_ref = f['K_DATA'][struct_idx, 0]
        M_DATA_ref = f['M_DATA'][struct_idx, 0]
        
        # Extract sparse matrix data
        K_data = np.array(f[K_DATA_ref]['data']).flatten()
        K_ir = np.array(f[K_DATA_ref]['ir']).flatten()
        K_jc = np.array(f[K_DATA_ref]['jc']).flatten()
        
        M_data = np.array(f[M_DATA_ref]['data']).flatten()
        M_ir = np.array(f[M_DATA_ref]['ir']).flatten()
        M_jc = np.array(f[M_DATA_ref]['jc']).flatten()
        
        # Get shape (should be (2178, 2178))
        # Try to infer from jc array (last element is nnz)
        n = len(K_jc) - 1
        
        # Create sparse matrices
        K = sp.csr_matrix((K_data, K_ir, K_jc), shape=(n, n))
        M = sp.csr_matrix((M_data, M_ir, M_jc), shape=(n, n))
    
    return K, M


def compare_sparse_matrices(name, K_py, K_ml, rtol=1e-5, atol=1e-8):
    """Compare two sparse matrices strategically."""
    print(f"\n{'='*70}")
    print(f"Comparing {name} matrices")
    print(f"{'='*70}")
    
    # Basic shape check
    print(f"\nShape comparison:")
    print(f"  Python: {K_py.shape}")
    print(f"  MATLAB: {K_ml.shape}")
    
    if K_py.shape != K_ml.shape:
        print(f"  ❌ Shape mismatch!")
        return False
    
    # Sparsity comparison
    print(f"\nSparsity comparison:")
    print(f"  Python nnz: {K_py.nnz}")
    print(f"  MATLAB nnz: {K_ml.nnz}")
    
    if K_py.nnz != K_ml.nnz:
        print(f"  ⚠️  Non-zero count mismatch!")
        print(f"     Difference: {abs(K_py.nnz - K_ml.nnz)}")
    
    # Compare non-zero pattern
    print(f"\nNon-zero pattern comparison:")
    # Check if non-zero counts match
    pattern_match = (K_py.nnz == K_ml.nnz)
    
    if not pattern_match:
        print(f"  ⚠️  Non-zero pattern differs!")
        print(f"     Python nnz: {K_py.nnz}, MATLAB nnz: {K_ml.nnz}")
        
        # Find locations where pattern differs (convert to dense for small matrices)
        if K_py.shape[0] * K_py.shape[1] < 1e6:  # Only for reasonably sized matrices
            py_nnz_mask = (K_py != 0).toarray()
            ml_nnz_mask = (K_ml != 0).toarray()
            diff_mask = py_nnz_mask != ml_nnz_mask
            n_diff = np.sum(diff_mask)
            print(f"     {n_diff} locations differ in non-zero pattern")
            
            if n_diff > 0 and n_diff <= 20:
                diff_indices = np.where(diff_mask)
                print(f"     Sample differences:")
                for i in range(min(10, n_diff)):
                    idx = (diff_indices[0][i], diff_indices[1][i])
                    py_val = K_py[idx] if py_nnz_mask[idx] else 0.0
                    ml_val = K_ml[idx] if ml_nnz_mask[idx] else 0.0
                    print(f"       [{idx[0]}, {idx[1]}]: Py={py_val:.6e}, ML={ml_val:.6e}")
    else:
        # Also check if the actual non-zero locations match
        if K_py.shape[0] * K_py.shape[1] < 1e6:
            py_nnz_mask = (K_py != 0).toarray()
            ml_nnz_mask = (K_ml != 0).toarray()
            locations_match = np.array_equal(py_nnz_mask, ml_nnz_mask)
            if locations_match:
                print(f"  ✅ Non-zero patterns match (both count and locations)")
            else:
                print(f"  ⚠️  Non-zero counts match but locations differ")
        else:
            print(f"  ✅ Non-zero counts match (matrix too large for location check)")
    
    # Compare values using difference matrix
    print(f"\nValue comparison:")
    
    # Compute difference matrix (sparse-aware)
    diff_matrix = K_py - K_ml
    
    # Compute norm of difference
    diff_norm = sp.linalg.norm(diff_matrix)
    K_ml_norm = sp.linalg.norm(K_ml)
    relative_norm_diff = diff_norm / (K_ml_norm + atol)
    
    print(f"  Frobenius norm of difference: {diff_norm:.6e}")
    print(f"  Relative norm difference: {relative_norm_diff:.6e}")
    
    # Compute statistics on non-zero differences
    if diff_matrix.nnz > 0:
        abs_diff_vals = np.abs(diff_matrix.data)
        max_abs_diff = np.max(abs_diff_vals)
        mean_abs_diff = np.mean(abs_diff_vals)
        
        # Sample some locations for detailed comparison
        n_sample = min(1000, diff_matrix.nnz)
        sample_indices = np.random.choice(diff_matrix.nnz, n_sample, replace=False) if diff_matrix.nnz > n_sample else np.arange(diff_matrix.nnz)
        sample_abs_diffs = abs_diff_vals[sample_indices]
        
        # Get corresponding values from original matrices for relative error
        diff_coords = np.array(list(zip(*diff_matrix.nonzero())))
        if len(diff_coords) > 0:
            sample_coords = diff_coords[sample_indices]
            ml_vals_sample = np.array([K_ml[i, j] for i, j in sample_coords])
            rel_diff_sample = sample_abs_diffs / (np.abs(ml_vals_sample) + atol)
            max_rel_diff = np.max(rel_diff_sample)
            mean_rel_diff = np.mean(rel_diff_sample)
        else:
            max_rel_diff = 0.0
            mean_rel_diff = 0.0
        
        print(f"  Max absolute difference: {max_abs_diff:.6e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
        print(f"  Max relative difference (sampled): {max_rel_diff:.6e}")
        print(f"  Mean relative difference (sampled): {mean_rel_diff:.6e}")
        
        # Check if within tolerance
        # Use a combination of norm-based and element-wise checks
        values_match = (diff_norm < atol * K_ml.shape[0]) and (max_abs_diff < atol * 100) and (max_rel_diff < rtol * 100)
        
        if values_match:
            print(f"  ✅ Values match within tolerance (rtol={rtol}, atol={atol})")
        else:
            print(f"  ❌ Values differ beyond tolerance")
            
            # Find worst mismatches
            n_show = min(10, diff_matrix.nnz)
            worst_indices = np.argsort(abs_diff_vals)[-n_show:][::-1]
            print(f"\n  Worst {n_show} mismatches:")
            for idx in worst_indices:
                i, j = diff_coords[idx]
                py_val = float(K_py[i, j])
                ml_val = float(K_ml[i, j])
                abs_diff = abs_diff_vals[idx]
                rel_diff = abs_diff / (abs(ml_val) + atol)
                print(f"    [{i:4d}, {j:4d}]: Py={py_val:12.6e}, ML={ml_val:12.6e}, "
                      f"abs_diff={abs_diff:.6e}, rel_diff={rel_diff:.6e}")
    else:
        print(f"  ✅ Matrices are identical (no differences found)")
        values_match = True
    
    # Check locations where only one matrix is non-zero
    if K_py.shape[0] * K_py.shape[1] < 1e6:
        py_nnz_mask = (K_py != 0).toarray()
        ml_nnz_mask = (K_ml != 0).toarray()
        only_py_mask = py_nnz_mask & ~ml_nnz_mask
        only_ml_mask = ~py_nnz_mask & ml_nnz_mask
        n_only_py = np.sum(only_py_mask)
        n_only_ml = np.sum(only_ml_mask)
    else:
        # For large matrices, check if diff matrix has unexpected non-zeros
        # If patterns match, all differences should be at common locations
        # Estimate from structure
        n_only_py = 0
        n_only_ml = 0
        if diff_matrix.nnz > (K_py.nnz + K_ml.nnz) // 2:
            print(f"  ⚠️  Many differences detected - pattern may differ significantly")
    
    if n_only_py > 0 or n_only_ml > 0:
        print(f"\n  ⚠️  Locations where only one matrix is non-zero:")
        print(f"     Only Python: {n_only_py}")
        print(f"     Only MATLAB: {n_only_ml}")
    
    return values_match and (n_only_py == 0) and (n_only_ml == 0)


def main():
    """Main comparison function."""
    print("="*70)
    print("K & M Matrix Comparison: Python vs MATLAB")
    print("="*70)
    
    # Paths
    python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
    matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
    struct_idx = 0
    
    # Load matrices
    print(f"\nLoading Python matrices from: {python_data_dir}")
    print(f"  Structure index: {struct_idx}")
    try:
        K_py, M_py = load_python_matrices(python_data_dir, struct_idx)
        print(f"  ✅ Loaded K: shape={K_py.shape}, nnz={K_py.nnz}")
        print(f"  ✅ Loaded M: shape={M_py.shape}, nnz={M_py.nnz}")
    except Exception as e:
        print(f"  ❌ Error loading Python matrices: {e}")
        return
    
    print(f"\nLoading MATLAB matrices from: {matlab_mat_file}")
    try:
        K_ml, M_ml = load_matlab_matrices(matlab_mat_file, struct_idx)
        print(f"  ✅ Loaded K: shape={K_ml.shape}, nnz={K_ml.nnz}")
        print(f"  ✅ Loaded M: shape={M_ml.shape}, nnz={M_ml.nnz}")
    except Exception as e:
        print(f"  ❌ Error loading MATLAB matrices: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare K matrices
    K_match = compare_sparse_matrices("K", K_py, K_ml)
    
    # Compare M matrices
    M_match = compare_sparse_matrices("M", M_py, M_ml)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"K matrices match: {'✅ YES' if K_match else '❌ NO'}")
    print(f"M matrices match: {'✅ YES' if M_match else '❌ NO'}")
    
    if K_match and M_match:
        print(f"\n✅ All matrices match! Python implementation is equivalent to MATLAB.")
    else:
        print(f"\n❌ Discrepancies found. Investigate the differences above.")


if __name__ == "__main__":
    main()

