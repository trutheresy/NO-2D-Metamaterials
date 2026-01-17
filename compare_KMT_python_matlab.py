#!/usr/bin/env python3
"""
Compare K, M, T matrices computed by Python vs MATLAB.

This script:
1. Loads Python K, M, T matrices
2. Loads MATLAB K, M, T matrices from .mat file
3. Compares and reports relative errors
"""

import numpy as np
import scipy.sparse as sp
import torch
import h5py
from pathlib import Path
import sys

def load_sparse_matrix_from_hdf5(h5py_file, dataset_path, struct_idx):
    """
    Load a sparse matrix from MATLAB HDF5 format.
    
    MATLAB stores sparse matrices as structures with 'data', 'ir', 'jc' fields.
    """
    try:
        # Try to access as HDF5 reference
        ref = h5py_file[dataset_path][struct_idx, 0]
        
        # Load data, row indices, column pointers
        data = np.array(h5py_file[ref]['data']).flatten()
        ir = np.array(h5py_file[ref]['ir']).flatten().astype(int)
        jc = np.array(h5py_file[ref]['jc']).flatten().astype(int)
        
        # Reconstruct sparse matrix
        n = len(jc) - 1
        matrix = sp.csr_matrix((data, ir, jc), shape=(n, n))
        
        return matrix
    except Exception as e:
        print(f"      Warning: Could not load {dataset_path}[{struct_idx}]: {e}")
        return None


def compare_sparse_matrices(name, mat_py, mat_ml, tolerance=1e-10):
    """
    Compare two sparse matrices and report differences.
    
    Returns:
    --------
    dict : Comparison statistics
    """
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
    
    # Element-wise relative error (avoid division by zero)
    epsilon = 1e-15
    magnitude_ml = np.abs(mat_ml_dense)
    rel_error_elementwise = abs_diff / (magnitude_ml + epsilon)
    max_rel_error = np.max(rel_error_elementwise)
    mean_rel_error = np.mean(rel_error_elementwise[~(magnitude_ml < epsilon)])
    
    # Find location of max relative error
    max_rel_error_idx = np.unravel_index(np.argmax(rel_error_elementwise), rel_error_elementwise.shape)
    max_rel_error_py_val = mat_py_dense[max_rel_error_idx]
    max_rel_error_ml_val = mat_ml_dense[max_rel_error_idx]
    max_rel_error_abs_diff = abs_diff[max_rel_error_idx]
    
    # Check sparsity pattern differences
    # Define "non-zero" as magnitude > threshold (to account for numerical zeros)
    threshold = 1e-10
    py_nonzero = np.abs(mat_py_dense) > threshold
    ml_nonzero = np.abs(mat_ml_dense) > threshold
    
    nnz_py_actual = np.sum(py_nonzero)
    nnz_ml_actual = np.sum(ml_nonzero)
    
    # Elements that are non-zero in Python but zero in MATLAB
    py_only_nonzero = py_nonzero & ~ml_nonzero
    n_py_only = np.sum(py_only_nonzero)
    
    # Elements that are non-zero in MATLAB but zero in Python
    ml_only_nonzero = ml_nonzero & ~py_nonzero
    n_ml_only = np.sum(ml_only_nonzero)
    
    # Elements that are non-zero in both
    both_nonzero = py_nonzero & ml_nonzero
    n_both = np.sum(both_nonzero)
    
    # For elements that are non-zero in one but not the other, get some sample values
    if n_py_only > 0:
        py_only_indices = np.where(py_only_nonzero)
        sample_py_only_idx = (py_only_indices[0][0], py_only_indices[1][0])
        sample_py_only_val = mat_py_dense[sample_py_only_idx]
    else:
        sample_py_only_idx = None
        sample_py_only_val = None
    
    if n_ml_only > 0:
        ml_only_indices = np.where(ml_only_nonzero)
        sample_ml_only_idx = (ml_only_indices[0][0], ml_only_indices[1][0])
        sample_ml_only_val = mat_ml_dense[sample_ml_only_idx]
    else:
        sample_ml_only_idx = None
        sample_ml_only_val = None
    
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
        'sample_py_only_idx': sample_py_only_idx,
        'sample_py_only_val': float(sample_py_only_val) if sample_py_only_val is not None else None,
        'sample_ml_only_idx': sample_ml_only_idx,
        'sample_ml_only_val': float(sample_ml_only_val) if sample_ml_only_val is not None else None,
        'frobenius_diff': float(frobenius_diff),
        'frobenius_py': float(frobenius_py),
        'frobenius_ml': float(frobenius_ml),
        'relative_error_frobenius': float(relative_error),
        'max_abs_diff': float(max_abs_diff),
        'max_rel_error': float(max_rel_error),
        'mean_rel_error': float(mean_rel_error) if not np.isnan(mean_rel_error) else 0.0,
        'max_rel_error_location': max_rel_error_idx,
        'max_rel_error_py_val': complex(max_rel_error_py_val) if np.iscomplexobj(max_rel_error_py_val) else float(max_rel_error_py_val),
        'max_rel_error_ml_val': complex(max_rel_error_ml_val) if np.iscomplexobj(max_rel_error_ml_val) else float(max_rel_error_ml_val),
        'max_rel_error_abs_diff': float(max_rel_error_abs_diff),
        'match': relative_error < tolerance
    }
    
    return stats


def main():
    # Paths
    python_KMT_dir = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test")
    matlab_mat_file = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/mat/computed_K_M_T_matrices.mat")
    
    print("=" * 80)
    print("Comparing Python vs MATLAB K, M, T Matrices")
    print("=" * 80)
    print(f"Python matrices: {python_KMT_dir}")
    print(f"MATLAB .mat file: {matlab_mat_file}")
    print()
    
    # Load Python matrices
    print("1. Loading Python matrices...")
    K_list_py = torch.load(python_KMT_dir / "K_data_python.pt", map_location='cpu')
    M_list_py = torch.load(python_KMT_dir / "M_data_python.pt", map_location='cpu')
    T_list_py = torch.load(python_KMT_dir / "T_data_python.pt", map_location='cpu')
    
    n_structs = len(K_list_py)
    n_wavevectors = len(T_list_py[0]) if len(T_list_py) > 0 else 0
    
    print(f"   Loaded {n_structs} structures")
    print(f"   Loaded {n_wavevectors} wavevectors per structure")
    
    # Load MATLAB matrices
    print()
    print("2. Loading MATLAB matrices...")
    if not matlab_mat_file.exists():
        print(f"   ERROR: MATLAB file not found: {matlab_mat_file}")
        print("   Please run MATLAB script first to generate K, M, T matrices")
        return 1
    
    with h5py.File(str(matlab_mat_file), 'r') as f:
        # Check available datasets
        print(f"   Available datasets: {list(f.keys())}")
        
        # Try to load K, M, T data
        # MATLAB may store them as K_DATA, M_DATA, T_DATA or with different names
        K_DATA_path = None
        M_DATA_path = None
        T_DATA_path = None
        
        for key in f.keys():
            if 'K' in key.upper() and 'DATA' in key.upper():
                K_DATA_path = f'/{key}'
            elif 'M' in key.upper() and 'DATA' in key.upper():
                M_DATA_path = f'/{key}'
            elif 'T' in key.upper() and 'DATA' in key.upper():
                T_DATA_path = f'/{key}'
        
        if K_DATA_path is None or M_DATA_path is None:
            print(f"   ERROR: Could not find K_DATA or M_DATA in MATLAB file")
            print(f"   Available keys: {list(f.keys())}")
            return 1
        
        print(f"   Found K_DATA at: {K_DATA_path}")
        print(f"   Found M_DATA at: {M_DATA_path}")
        if T_DATA_path:
            print(f"   Found T_DATA at: {T_DATA_path}")
    
    # Compare matrices
    print()
    print("3. Comparing matrices...")
    print()
    
    all_stats = []
    
    for struct_idx in range(n_structs):
        print(f"   Structure {struct_idx + 1}/{n_structs}:")
        
        # Load Python matrices
        K_py = K_list_py[struct_idx]
        M_py = M_list_py[struct_idx]
        
        # Ensure sparse
        if not sp.issparse(K_py):
            K_py = sp.csr_matrix(K_py)
        if not sp.issparse(M_py):
            M_py = sp.csr_matrix(M_py)
        
        # Load MATLAB matrices
        with h5py.File(str(matlab_mat_file), 'r') as f:
            K_ml = load_sparse_matrix_from_hdf5(f, K_DATA_path, struct_idx)
            M_ml = load_sparse_matrix_from_hdf5(f, M_DATA_path, struct_idx)
        
        if K_ml is None or M_ml is None:
            print(f"      ERROR: Could not load MATLAB matrices")
            continue
        
        # Compare K matrices
        K_stats = compare_sparse_matrices(f"K[{struct_idx}]", K_py, K_ml)
        print(f"      K matrix:")
        print(f"        Sparse matrix nnz: Python={K_stats['nnz_py']}, MATLAB={K_stats['nnz_ml']}")
        print(f"        Actual nnz (|val| > 1e-10): Python={K_stats['nnz_py_actual']}, MATLAB={K_stats['nnz_ml_actual']}")
        print(f"        Sparsity pattern differences:")
        print(f"          Non-zero only in Python: {K_stats['n_py_only_nonzero']}")
        if K_stats['n_py_only_nonzero'] > 0:
            print(f"            Sample: location={K_stats['sample_py_only_idx']}, value={K_stats['sample_py_only_val']:.10e}")
        print(f"          Non-zero only in MATLAB: {K_stats['n_ml_only_nonzero']}")
        if K_stats['n_ml_only_nonzero'] > 0:
            print(f"            Sample: location={K_stats['sample_ml_only_idx']}, value={K_stats['sample_ml_only_val']:.10e}")
        print(f"          Non-zero in both: {K_stats['n_both_nonzero']}")
        print(f"        Relative error (Frobenius): {K_stats['relative_error_frobenius']*100:.6f}%")
        print(f"        Max relative error: {K_stats['max_rel_error']*100:.6f}%")
        print(f"          Location: {K_stats['max_rel_error_location']}")
        print(f"          Python value: {K_stats['max_rel_error_py_val']:.10e}")
        print(f"          MATLAB value: {K_stats['max_rel_error_ml_val']:.10e}")
        print(f"          Absolute diff: {K_stats['max_rel_error_abs_diff']:.10e}")
        print(f"        Mean relative error: {K_stats['mean_rel_error']*100:.6f}%")
        print(f"        Max absolute diff: {K_stats['max_abs_diff']:.6e}")
        all_stats.append(K_stats)
        
        # Compare M matrices
        M_stats = compare_sparse_matrices(f"M[{struct_idx}]", M_py, M_ml)
        print(f"      M matrix:")
        print(f"        Sparse matrix nnz: Python={M_stats['nnz_py']}, MATLAB={M_stats['nnz_ml']}")
        print(f"        Actual nnz (|val| > 1e-10): Python={M_stats['nnz_py_actual']}, MATLAB={M_stats['nnz_ml_actual']}")
        print(f"        Sparsity pattern differences:")
        print(f"          Non-zero only in Python: {M_stats['n_py_only_nonzero']}")
        if M_stats['n_py_only_nonzero'] > 0:
            print(f"            Sample: location={M_stats['sample_py_only_idx']}, value={M_stats['sample_py_only_val']:.10e}")
        print(f"          Non-zero only in MATLAB: {M_stats['n_ml_only_nonzero']}")
        if M_stats['n_ml_only_nonzero'] > 0:
            print(f"            Sample: location={M_stats['sample_ml_only_idx']}, value={M_stats['sample_ml_only_val']:.10e}")
        print(f"          Non-zero in both: {M_stats['n_both_nonzero']}")
        print(f"        Relative error (Frobenius): {M_stats['relative_error_frobenius']*100:.6f}%")
        print(f"        Max relative error: {M_stats['max_rel_error']*100:.6f}%")
        print(f"          Location: {M_stats['max_rel_error_location']}")
        print(f"          Python value: {M_stats['max_rel_error_py_val']:.10e}")
        print(f"          MATLAB value: {M_stats['max_rel_error_ml_val']:.10e}")
        print(f"          Absolute diff: {M_stats['max_rel_error_abs_diff']:.10e}")
        print(f"        Mean relative error: {M_stats['mean_rel_error']*100:.6f}%")
        print(f"        Max absolute diff: {M_stats['max_abs_diff']:.6e}")
        all_stats.append(M_stats)
        
        # Compare T matrices (only for first structure, as T is same for all structures)
        if T_DATA_path and struct_idx == 0:
            # T matrices are the same for all structures (only depend on wavevectors)
            print(f"      T matrices (comparing for first structure, but T is same for all):")
            
            # Load MATLAB T matrices (one per wavevector)
            with h5py.File(str(matlab_mat_file), 'r') as f:
                try:
                    # T_DATA is stored as (1, n_wavevectors) object array
                    T_data_ml = f[T_DATA_path]
                    n_wv_ml = T_data_ml.shape[1] if len(T_data_ml.shape) >= 2 else len(T_data_ml)
                    
                    # Compare a few representative wavevectors
                    n_compare = min(5, n_wavevectors, n_wv_ml)  # Compare first 5 wavevectors
                    
                    for wv_idx in range(n_compare):
                        try:
                            # Load MATLAB T matrix (sparse matrix from HDF5)
                            T_ref_ml = T_data_ml[0, wv_idx]
                            
                            # Load sparse matrix components
                            # MATLAB stores sparse matrices in CSC format by default, but HDF5 might save differently
                            # Check the actual format by examining the structure
                            T_sparse_struct = f[T_ref_ml]
                            
                            # Load data, row indices, column pointers
                            T_data_vals = np.array(T_sparse_struct['data']).flatten()
                            
                            # Check if data is complex (structured array with real/imag)
                            if T_data_vals.dtype.names and 'real' in T_data_vals.dtype.names:
                                T_data_vals = T_data_vals['real'] + 1j * T_data_vals['imag']
                            
                            T_ir = np.array(T_sparse_struct['ir']).flatten().astype(int)
                            T_jc = np.array(T_sparse_struct['jc']).flatten().astype(int)
                            
                            # Get shape from Python matrix first
                            if wv_idx < len(T_list_py[0]):
                                T_py_ref = T_list_py[0][wv_idx]
                                if not sp.issparse(T_py_ref):
                                    T_py_ref = sp.csr_matrix(T_py_ref)
                                expected_shape = T_py_ref.shape
                            else:
                                # Fallback: infer from jc and ir
                                # For CSC format (MATLAB default): jc length = n_cols + 1, max(ir) + 1 = n_rows
                                # For CSR format: jc length = n_cols + 1, max(ir) + 1 = n_rows (same!)
                                n_cols = len(T_jc) - 1
                                n_rows = max(T_ir) + 1 if len(T_ir) > 0 else 0
                                expected_shape = (n_rows, n_cols)
                            
                            # MATLAB stores sparse matrices in CSC (column-major) format
                            # Python scipy.sparse CSR uses (row_indices, col_pointers)
                            # But MATLAB CSC uses (col_indices, row_pointers) - swapped!
                            # When saved to HDF5, MATLAB preserves the format
                            # Try CSC format first (MATLAB default)
                            try:
                                T_ml = sp.csc_matrix((T_data_vals, T_ir, T_jc), shape=expected_shape)
                                # Convert to CSR for consistent comparison
                                T_ml = T_ml.tocsr()
                            except:
                                # If that fails, try CSR format
                                try:
                                    T_ml = sp.csr_matrix((T_data_vals, T_ir, T_jc), shape=expected_shape)
                                except Exception as e:
                                    print(f"          Error constructing sparse matrix: {e}")
                                    continue
                            
                            # Get Python T matrix
                            if wv_idx < len(T_list_py[0]):
                                T_py = T_list_py[0][wv_idx]
                                if not sp.issparse(T_py):
                                    T_py = sp.csr_matrix(T_py)
                                
                                # Debug output for first wavevector
                                if wv_idx == 0:
                                    print(f"          Debug T[{wv_idx}]:")
                                    print(f"            Python: shape={T_py.shape}, nnz={T_py.nnz}, dtype={T_py.dtype}")
                                    print(f"            MATLAB: shape={T_ml.shape}, nnz={T_ml.nnz}, dtype={T_ml.dtype}")
                                
                                # Ensure same shape
                                if T_py.shape != T_ml.shape:
                                    print(f"          Warning: Shape mismatch - Python {T_py.shape} vs MATLAB {T_ml.shape}")
                                    # Don't resize - this indicates a real problem
                                    print(f"          Skipping comparison due to shape mismatch")
                                    continue
                                
                                # Check if both are complex
                                if np.iscomplexobj(T_py) and not np.iscomplexobj(T_ml):
                                    print(f"          Warning: Type mismatch - Python complex, MATLAB real")
                                elif not np.iscomplexobj(T_py) and np.iscomplexobj(T_ml):
                                    print(f"          Warning: Type mismatch - Python real, MATLAB complex")
                                
                                T_stats = compare_sparse_matrices(f"T[0][{wv_idx}]", T_py, T_ml)
                                print(f"        T[{wv_idx}]:")
                                print(f"          Relative error (Frobenius): {T_stats['relative_error_frobenius']*100:.6f}%")
                                print(f"          Max relative error: {T_stats['max_rel_error']*100:.6f}%")
                                print(f"          Mean relative error: {T_stats['mean_rel_error']*100:.6f}%")
                                print(f"          Max absolute diff: {T_stats['max_abs_diff']:.6e}")
                                all_stats.append(T_stats)
                        except Exception as e:
                            print(f"          Warning: Could not compare T[{wv_idx}]: {e}")
                except Exception as e:
                    print(f"        Warning: Could not load T_DATA: {e}")
    
    # Summary
    print()
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    K_stats_list = [s for s in all_stats if 'K[' in s.get('name', '') and 'T' not in s.get('name', '')]
    M_stats_list = [s for s in all_stats if 'M[' in s.get('name', '')]
    T_stats_list = [s for s in all_stats if 'T[' in s.get('name', '')]
    
    if K_stats_list:
        print(f"\nK Matrices ({len(K_stats_list)} structures):")
        frob_errors = [s['relative_error_frobenius'] for s in K_stats_list]
        mean_rel_errors = [s['mean_rel_error'] for s in K_stats_list]
        print(f"  Frobenius relative error: {np.mean(frob_errors)*100:.6f}% (mean), {np.min(frob_errors)*100:.6f}% (min), {np.max(frob_errors)*100:.6f}% (max)")
        print(f"  Mean relative error: {np.mean(mean_rel_errors)*100:.6f}% (mean), {np.min(mean_rel_errors)*100:.6f}% (min), {np.max(mean_rel_errors)*100:.6f}% (max)")
    
    if M_stats_list:
        print(f"\nM Matrices ({len(M_stats_list)} structures):")
        frob_errors = [s['relative_error_frobenius'] for s in M_stats_list]
        mean_rel_errors = [s['mean_rel_error'] for s in M_stats_list]
        print(f"  Frobenius relative error: {np.mean(frob_errors)*100:.6f}% (mean), {np.min(frob_errors)*100:.6f}% (min), {np.max(frob_errors)*100:.6f}% (max)")
        print(f"  Mean relative error: {np.mean(mean_rel_errors)*100:.6f}% (mean), {np.min(mean_rel_errors)*100:.6f}% (min), {np.max(mean_rel_errors)*100:.6f}% (max)")
    
    if T_stats_list:
        print(f"\nT Matrices ({len(T_stats_list)} wavevectors):")
        frob_errors = [s['relative_error_frobenius'] for s in T_stats_list]
        mean_rel_errors = [s['mean_rel_error'] for s in T_stats_list]
        print(f"  Frobenius relative error: {np.mean(frob_errors)*100:.6f}% (mean), {np.min(frob_errors)*100:.6f}% (min), {np.max(frob_errors)*100:.6f}% (max)")
        print(f"  Mean relative error: {np.mean(mean_rel_errors)*100:.6f}% (mean), {np.min(mean_rel_errors)*100:.6f}% (min), {np.max(mean_rel_errors)*100:.6f}% (max)")
    
    print()
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

