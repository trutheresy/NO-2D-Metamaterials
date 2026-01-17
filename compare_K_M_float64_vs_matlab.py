#!/usr/bin/env python3
"""
Compare K and M matrices computed in Python with float64 precision vs MATLAB.
This will show if using double precision improves the agreement.
"""

import numpy as np
import scipy.sparse as sp
import h5py
from pathlib import Path

def load_matlab_sparse_matrix(filepath, var_name='K'):
    """Load sparse matrix from MATLAB v7.3 HDF5 format."""
    with h5py.File(filepath, 'r') as f:
        if var_name in f:
            K_ref = f[var_name]
            if isinstance(K_ref, h5py.Dataset) and K_ref.shape == (1, 1):
                K_ref = f[K_ref[0, 0]]
            elif isinstance(K_ref, h5py.Dataset):
                K_ref = f[K_ref[()]]
            
            data = np.array(K_ref['data']).flatten()
            ir = np.array(K_ref['ir']).flatten().astype(int)
            jc = np.array(K_ref['jc']).flatten().astype(int)
            n = len(jc) - 1
            K = sp.csr_matrix((data, ir, jc), shape=(n, n))
            return K
        else:
            raise ValueError(f"Variable {var_name} not found in file")

def compare_sparse_matrices(name, mat_py, mat_ml, tolerance=1e-10):
    """Compare two sparse matrices and report differences."""
    if mat_py is None or mat_ml is None:
        return {'error': 'One or both matrices are None'}
    
    # Ensure both are sparse
    if not sp.issparse(mat_py):
        mat_py = sp.csr_matrix(mat_py)
    if not sp.issparse(mat_ml):
        mat_ml = sp.csr_matrix(mat_ml)
    
    # Ensure same shape
    if mat_py.shape != mat_ml.shape:
        return {
            'error': f'Shape mismatch: Python {mat_py.shape} vs MATLAB {mat_ml.shape}'
        }
    
    # Convert to same format for comparison
    mat_py = sp.csr_matrix(mat_py)
    mat_ml = sp.csr_matrix(mat_ml)
    
    # Compute difference
    diff = mat_py - mat_ml
    
    # Convert to dense for norm calculation
    diff_dense = diff.toarray()
    mat_py_dense = mat_py.toarray()
    mat_ml_dense = mat_ml.toarray()
    
    # Statistics
    abs_diff = np.abs(diff_dense)
    
    # Frobenius norm of difference
    frobenius_diff = np.linalg.norm(diff_dense, 'fro')
    frobenius_py = np.linalg.norm(mat_py_dense, 'fro')
    frobenius_ml = np.linalg.norm(mat_ml_dense, 'fro')
    
    # Relative error (normalized by MATLAB norm)
    if frobenius_ml > 0:
        relative_error = frobenius_diff / frobenius_ml
    else:
        relative_error = np.inf if frobenius_diff > 0 else 0.0
    
    # Max absolute difference
    max_abs_diff = np.max(abs_diff)
    
    # Element-wise relative error
    epsilon = 1e-15
    magnitude_ml = np.abs(mat_ml_dense)
    rel_error_elementwise = abs_diff / (magnitude_ml + epsilon)
    max_rel_error = np.max(rel_error_elementwise)
    mean_rel_error = np.mean(rel_error_elementwise[magnitude_ml > epsilon])
    
    # Check sparsity pattern differences
    threshold = 1e-10
    py_nonzero = np.abs(mat_py_dense) > threshold
    ml_nonzero = np.abs(mat_ml_dense) > threshold
    nnz_py_actual = np.sum(py_nonzero)
    nnz_ml_actual = np.sum(ml_nonzero)
    n_py_only = np.sum(py_nonzero & ~ml_nonzero)
    n_ml_only = np.sum(ml_nonzero & ~py_nonzero)
    n_both = np.sum(py_nonzero & ml_nonzero)
    
    stats = {
        'name': name,
        'shape_py': mat_py.shape,
        'shape_ml': mat_ml.shape,
        'nnz_py': mat_py.nnz,
        'nnz_ml': mat_ml.nnz,
        'nnz_py_actual': int(nnz_py_actual),
        'nnz_ml_actual': int(nnz_ml_actual),
        'n_py_only_nonzero': int(n_py_only),
        'n_ml_only_nonzero': int(n_ml_only),
        'n_both_nonzero': int(n_both),
        'frobenius_diff': float(frobenius_diff),
        'frobenius_py': float(frobenius_py),
        'frobenius_ml': float(frobenius_ml),
        'relative_error_frobenius': float(relative_error),
        'max_abs_diff': float(max_abs_diff),
        'max_rel_error': float(max_rel_error),
        'mean_rel_error': float(mean_rel_error) if not np.isnan(mean_rel_error) else 0.0,
        'match': relative_error < tolerance
    }
    
    return stats

def main():
    # Paths
    matlab_mat_file = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/matlab_intermediates/step10_final_matrices.mat")
    python_K_file_float64 = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/python_intermediates_float64/step10_K_float64.npz")
    python_M_file_float64 = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/python_intermediates_float64/step10_M_float64.npz")
    
    # For comparison with previous float32 results
    python_K_file_float32 = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/python_intermediates/step10_K.npz")
    python_M_file_float32 = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/python_intermediates/step10_M.npz")
    
    print("=" * 80)
    print("Comparing K and M Matrices: Python (float64) vs MATLAB")
    print("=" * 80)
    print(f"MATLAB: {matlab_mat_file}")
    print(f"Python (float64): {python_K_file_float64}")
    print()
    
    # Load matrices
    print("Loading matrices...")
    K_ml = load_matlab_sparse_matrix(matlab_mat_file, 'K')
    M_ml = load_matlab_sparse_matrix(matlab_mat_file, 'M')
    K_py_float64 = sp.load_npz(python_K_file_float64)
    M_py_float64 = sp.load_npz(python_M_file_float64)
    
    # Optionally load float32 versions for comparison
    if python_K_file_float32.exists():
        K_py_float32 = sp.load_npz(python_K_file_float32)
        M_py_float32 = sp.load_npz(python_M_file_float32)
        compare_float32 = True
    else:
        compare_float32 = False
    
    print(f"  MATLAB K: shape={K_ml.shape}, nnz={K_ml.nnz}")
    print(f"  MATLAB M: shape={M_ml.shape}, nnz={M_ml.nnz}")
    print(f"  Python (float64) K: shape={K_py_float64.shape}, nnz={K_py_float64.nnz}, dtype={K_py_float64.dtype}")
    print(f"  Python (float64) M: shape={M_py_float64.shape}, nnz={M_py_float64.nnz}, dtype={M_py_float64.dtype}")
    print()
    
    # Compare float64 vs MATLAB
    print("=" * 80)
    print("Comparison: Python (float64) vs MATLAB")
    print("=" * 80)
    print()
    
    K_stats_float64 = compare_sparse_matrices("K (float64)", K_py_float64, K_ml)
    M_stats_float64 = compare_sparse_matrices("M (float64)", M_py_float64, M_ml)
    
    print(f"K Matrix (float64):")
    print(f"  Frobenius relative error: {K_stats_float64['relative_error_frobenius']*100:.6f}%")
    print(f"  Mean relative error: {K_stats_float64['mean_rel_error']*100:.6f}%")
    print(f"  Max relative error: {K_stats_float64['max_rel_error']*100:.6f}%")
    print(f"  Max absolute diff: {K_stats_float64['max_abs_diff']:.6e}")
    print(f"  Sparsity pattern differences:")
    print(f"    Non-zero only in Python: {K_stats_float64['n_py_only_nonzero']}")
    print(f"    Non-zero only in MATLAB: {K_stats_float64['n_ml_only_nonzero']}")
    print(f"    Non-zero in both: {K_stats_float64['n_both_nonzero']}")
    print()
    
    print(f"M Matrix (float64):")
    print(f"  Frobenius relative error: {M_stats_float64['relative_error_frobenius']*100:.6f}%")
    print(f"  Mean relative error: {M_stats_float64['mean_rel_error']*100:.6f}%")
    print(f"  Max relative error: {M_stats_float64['max_rel_error']*100:.6f}%")
    print(f"  Max absolute diff: {M_stats_float64['max_abs_diff']:.6e}")
    print(f"  Sparsity pattern differences:")
    print(f"    Non-zero only in Python: {M_stats_float64['n_py_only_nonzero']}")
    print(f"    Non-zero only in MATLAB: {M_stats_float64['n_ml_only_nonzero']}")
    print(f"    Non-zero in both: {M_stats_float64['n_both_nonzero']}")
    print()
    
    # Compare float64 vs float32 (if available)
    if compare_float32:
        print("=" * 80)
        print("Improvement: float64 vs float32")
        print("=" * 80)
        print()
        
        K_stats_float32 = compare_sparse_matrices("K (float32)", K_py_float32, K_ml)
        M_stats_float32 = compare_sparse_matrices("M (float32)", M_py_float32, M_ml)
        
        print(f"K Matrix:")
        print(f"  float32 Frobenius relative error: {K_stats_float32['relative_error_frobenius']*100:.6f}%")
        print(f"  float64 Frobenius relative error: {K_stats_float64['relative_error_frobenius']*100:.6f}%")
        improvement_K = ((K_stats_float32['relative_error_frobenius'] - K_stats_float64['relative_error_frobenius']) / K_stats_float32['relative_error_frobenius']) * 100
        print(f"  Improvement: {improvement_K:.2f}% reduction in error")
        print(f"  Ratio: {K_stats_float32['relative_error_frobenius'] / K_stats_float64['relative_error_frobenius']:.6f}x better")
        print()
        
        print(f"M Matrix:")
        print(f"  float32 Frobenius relative error: {M_stats_float32['relative_error_frobenius']*100:.6f}%")
        print(f"  float64 Frobenius relative error: {M_stats_float64['relative_error_frobenius']*100:.6f}%")
        improvement_M = ((M_stats_float32['relative_error_frobenius'] - M_stats_float64['relative_error_frobenius']) / M_stats_float32['relative_error_frobenius']) * 100
        print(f"  Improvement: {improvement_M:.2f}% reduction in error")
        print(f"  Ratio: {M_stats_float32['relative_error_frobenius'] / M_stats_float64['relative_error_frobenius']:.6f}x better")
        print()
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Python (float64) vs MATLAB:")
    print(f"  K matrix Frobenius relative error: {K_stats_float64['relative_error_frobenius']*100:.6e}%")
    print(f"  M matrix Frobenius relative error: {M_stats_float64['relative_error_frobenius']*100:.6e}%")
    if compare_float32:
        print(f"\nImprovement from float32 to float64:")
        print(f"  K matrix: {improvement_K:.2f}% error reduction")
        print(f"  M matrix: {improvement_M:.2f}% error reduction")
    print("=" * 80)

if __name__ == "__main__":
    main()

