#!/usr/bin/env python3
"""
Investigate the sparsity pattern differences in K matrices between Python and MATLAB.

This script will:
1. Load K matrices from both Python and MATLAB
2. Identify all locations where sparsity patterns differ
3. Analyze the magnitude and distribution of these differences
4. Check if there's a systematic pattern
"""

import numpy as np
import scipy.sparse as sp
import torch
import h5py
from pathlib import Path

def load_sparse_matrix_from_hdf5(h5py_file, dataset_path, struct_idx):
    """Load a sparse matrix from MATLAB HDF5 format."""
    try:
        ref = h5py_file[dataset_path][struct_idx, 0]
        data = np.array(h5py_file[ref]['data']).flatten()
        ir = np.array(h5py_file[ref]['ir']).flatten().astype(int)
        jc = np.array(h5py_file[ref]['jc']).flatten().astype(int)
        n = len(jc) - 1
        matrix = sp.csr_matrix((data, ir, jc), shape=(n, n))
        return matrix
    except Exception as e:
        print(f"Error loading {dataset_path}[{struct_idx}]: {e}")
        return None

def investigate_sparsity_differences(K_py, K_ml, struct_idx, threshold=1e-10):
    """Investigate sparsity pattern differences in detail."""
    # Convert to dense for analysis
    K_py_dense = K_py.toarray()
    K_ml_dense = K_ml.toarray()
    
    # Identify non-zero patterns
    py_nonzero = np.abs(K_py_dense) > threshold
    ml_nonzero = np.abs(K_ml_dense) > threshold
    
    # Find differences
    py_only = py_nonzero & ~ml_nonzero
    ml_only = ml_nonzero & ~py_nonzero
    both_nonzero = py_nonzero & ml_nonzero
    
    # Get all locations
    py_only_indices = np.where(py_only)
    ml_only_indices = np.where(ml_only)
    
    print(f"\nStructure {struct_idx + 1}:")
    print(f"  Total matrix size: {K_py.shape[0]} × {K_py.shape[1]}")
    print(f"  Non-zero only in Python: {len(py_only_indices[0])}")
    print(f"  Non-zero only in MATLAB: {len(ml_only_indices[0])}")
    print(f"  Non-zero in both: {np.sum(both_nonzero)}")
    
    if len(py_only_indices[0]) > 0:
        print(f"\n  Python-only non-zeros ({len(py_only_indices[0])} elements):")
        py_only_vals = K_py_dense[py_only]
        print(f"    Value range: [{np.min(py_only_vals):.6e}, {np.max(py_only_vals):.6e}]")
        print(f"    Mean magnitude: {np.mean(np.abs(py_only_vals)):.6e}")
        print(f"    Max magnitude: {np.max(np.abs(py_only_vals)):.6e}")
        print(f"    First 5 locations and values:")
        for i in range(min(5, len(py_only_indices[0]))):
            row, col = py_only_indices[0][i], py_only_indices[1][i]
            val = K_py_dense[row, col]
            print(f"      ({row}, {col}): {val:.10e}")
    
    if len(ml_only_indices[0]) > 0:
        print(f"\n  MATLAB-only non-zeros ({len(ml_only_indices[0])} elements):")
        ml_only_vals = K_ml_dense[ml_only]
        print(f"    Value range: [{np.min(ml_only_vals):.6e}, {np.max(ml_only_vals):.6e}]")
        print(f"    Mean magnitude: {np.mean(np.abs(ml_only_vals)):.6e}")
        print(f"    Max magnitude: {np.max(np.abs(ml_only_vals)):.6e}")
        print(f"    First 5 locations and values:")
        for i in range(min(5, len(ml_only_indices[0]))):
            row, col = ml_only_indices[0][i], ml_only_indices[1][i]
            val = K_ml_dense[row, col]
            print(f"      ({row}, {col}): {val:.10e}")
    
    # Check if differences are near diagonal or off-diagonal
    if len(py_only_indices[0]) > 0:
        py_diag_dist = np.abs(py_only_indices[0] - py_only_indices[1])
        print(f"\n  Python-only: Distance from diagonal:")
        print(f"    Mean: {np.mean(py_diag_dist):.2f}, Min: {np.min(py_diag_dist)}, Max: {np.max(py_diag_dist)}")
    
    if len(ml_only_indices[0]) > 0:
        ml_diag_dist = np.abs(ml_only_indices[0] - ml_only_indices[1])
        print(f"\n  MATLAB-only: Distance from diagonal:")
        print(f"    Mean: {np.mean(ml_diag_dist):.2f}, Min: {np.min(ml_diag_dist)}, Max: {np.max(ml_diag_dist)}")
    
    # Check if there's a systematic offset
    if len(py_only_indices[0]) > 0 and len(ml_only_indices[0]) > 0:
        # Try to find if Python locations match MATLAB locations with an offset
        py_only_set = set(zip(py_only_indices[0], py_only_indices[1]))
        ml_only_set = set(zip(ml_only_indices[0], ml_only_indices[1]))
        
        # Check for various offsets
        print(f"\n  Checking for systematic offset patterns:")
        for row_offset in [-2, -1, 0, 1, 2]:
            for col_offset in [-2, -1, 0, 1, 2]:
                if row_offset == 0 and col_offset == 0:
                    continue
                py_offset_set = set((r + row_offset, c + col_offset) for r, c in py_only_set)
                matches = len(py_offset_set & ml_only_set)
                if matches > 0:
                    print(f"    Offset ({row_offset}, {col_offset}): {matches} matches")
        
        # Check if values at nearby locations are similar
        print(f"\n  Checking values at nearby locations:")
        sample_py_idx = (py_only_indices[0][0], py_only_indices[1][0])
        sample_py_val = K_py_dense[sample_py_idx]
        
        # Check nearby locations in MATLAB
        for dr in [-2, -1, 0, 1, 2]:
            for dc in [-2, -1, 0, 1, 2]:
                check_idx = (sample_py_idx[0] + dr, sample_py_idx[1] + dc)
                if 0 <= check_idx[0] < K_ml_dense.shape[0] and 0 <= check_idx[1] < K_ml_dense.shape[1]:
                    ml_val = K_ml_dense[check_idx]
                    if abs(ml_val) > 1e-10:
                        print(f"    MATLAB at ({check_idx[0]}, {check_idx[1]}) = {ml_val:.10e} (Python at {sample_py_idx} = {sample_py_val:.10e})")
        
        # Also check the symmetric location
        sym_py_idx = (sample_py_idx[1], sample_py_idx[0])
        sym_py_val = K_py_dense[sym_py_idx]
        sym_ml_val = K_ml_dense[sym_py_idx]
        print(f"    Symmetric location ({sym_py_idx[0]}, {sym_py_idx[1]}):")
        print(f"      Python: {sym_py_val:.10e}, MATLAB: {sym_ml_val:.10e}")

def main():
    # Paths
    python_KMT_dir = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test")
    matlab_mat_file = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/mat/computed_K_M_T_matrices.mat")
    
    print("=" * 80)
    print("Investigating K Matrix Sparsity Pattern Differences")
    print("=" * 80)
    
    # Load Python matrices
    print("\n1. Loading Python K matrices...")
    K_list_py = torch.load(python_KMT_dir / "K_data_python.pt", map_location='cpu')
    n_structs = len(K_list_py)
    print(f"   Loaded {n_structs} structures")
    
    # Load MATLAB matrices
    print("\n2. Loading MATLAB K matrices...")
    with h5py.File(str(matlab_mat_file), 'r') as f:
        K_DATA_path = '/K_DATA'
        
        # Investigate first few structures in detail
        for struct_idx in range(min(3, n_structs)):
            K_py = K_list_py[struct_idx]
            if not sp.issparse(K_py):
                K_py = sp.csr_matrix(K_py)
            
            K_ml = load_sparse_matrix_from_hdf5(f, K_DATA_path, struct_idx)
            
            if K_ml is None:
                continue
            
            if not sp.issparse(K_ml):
                K_ml = sp.csr_matrix(K_ml)
            
            investigate_sparsity_differences(K_py, K_ml, struct_idx)
    
    print("\n" + "=" * 80)
    print("Investigation complete")
    print("=" * 80)

if __name__ == "__main__":
    main()

