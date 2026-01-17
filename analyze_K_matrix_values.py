#!/usr/bin/env python3
"""
Analyze the distribution of values in the K matrix to understand the scale.
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

def analyze_matrix_values(name, K):
    """Analyze the distribution of non-zero values in a sparse matrix."""
    # Get non-zero values
    values = K.data[np.abs(K.data) > 1e-15]  # Exclude numerical zeros
    
    print(f"\n{name}:")
    print(f"  Shape: {K.shape}")
    print(f"  Total nnz: {K.nnz}")
    print(f"  Non-zero values (>1e-15): {len(values)}")
    
    if len(values) > 0:
        abs_values = np.abs(values)
        
        print(f"  Value statistics (absolute):")
        print(f"    Min: {np.min(abs_values):.6e}")
        print(f"    Max: {np.max(abs_values):.6e}")
        print(f"    Mean: {np.mean(abs_values):.6e}")
        print(f"    Median: {np.median(abs_values):.6e}")
        print(f"    Std: {np.std(abs_values):.6e}")
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n  Percentiles (absolute):")
        for p in percentiles:
            val = np.percentile(abs_values, p)
            print(f"    {p:2d}th: {val:.6e}")
        
        # Count values near 8.0
        near_8 = np.abs(abs_values - 8.0) < 0.1
        print(f"\n  Values near 8.0 (within 0.1): {np.sum(near_8)}")
        if np.sum(near_8) > 0:
            print(f"    Range: [{np.min(abs_values[near_8]):.6e}, {np.max(abs_values[near_8]):.6e}]")
        
        # Count very small values (< 100)
        small_vals = abs_values < 100
        print(f"\n  Very small values (< 100): {np.sum(small_vals)} ({100*np.sum(small_vals)/len(values):.2f}%)")
        if np.sum(small_vals) > 0:
            print(f"    Range: [{np.min(abs_values[small_vals]):.6e}, {np.max(abs_values[small_vals]):.6e}]")
            print(f"    Mean: {np.mean(abs_values[small_vals]):.6e}")
        
        # Count very large values (> 1e10)
        large_vals = abs_values > 1e10
        print(f"\n  Very large values (> 1e10): {np.sum(large_vals)} ({100*np.sum(large_vals)/len(values):.2f}%)")
        if np.sum(large_vals) > 0:
            print(f"    Range: [{np.min(abs_values[large_vals]):.6e}, {np.max(abs_values[large_vals]):.6e}]")
            print(f"    Mean: {np.mean(abs_values[large_vals]):.6e}")

def main():
    # Paths
    matlab_file = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/matlab_intermediates/step10_final_matrices.mat")
    python_file = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/python_intermediates/step10_K.npz")
    
    print("=" * 80)
    print("Analyzing K Matrix Value Distribution")
    print("=" * 80)
    
    # Load matrices
    print("\nLoading matrices...")
    K_ml = load_matlab_sparse_matrix(matlab_file, 'K')
    K_py = sp.load_npz(python_file)
    
    # Analyze both
    analyze_matrix_values("MATLAB K matrix", K_ml)
    analyze_matrix_values("Python K matrix", K_py)
    
    # Compare specific locations where we saw differences
    print("\n" + "=" * 80)
    print("Specific Locations with Differences")
    print("=" * 80)
    
    # Locations where Python has 8.0 but MATLAB has 0.0
    # From earlier investigation: (74, 75), (75, 74), (124, 125), (125, 124), etc.
    test_locations = [(74, 75), (75, 74), (124, 125), (125, 124), (644, 645)]
    
    print("\nSample locations where Python has non-zero but MATLAB has zero:")
    K_ml_dense = K_ml.toarray()
    K_py_dense = K_py.toarray()
    
    for row, col in test_locations:
        ml_val = K_ml_dense[row, col]
        py_val = K_py_dense[row, col]
        if abs(py_val) > 1e-10 and abs(ml_val) < 1e-10:
            print(f"  Location ({row}, {col}):")
            print(f"    MATLAB: {ml_val:.10e}")
            print(f"    Python: {py_val:.10e}")
            
            # Check neighboring values for context
            print(f"    Neighboring values (MATLAB):")
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if 0 <= row + dr < K_ml.shape[0] and 0 <= col + dc < K_ml.shape[1]:
                        nbr_val = K_ml_dense[row + dr, col + dc]
                        if abs(nbr_val) > 1e-10:
                            print(f"      ({row + dr}, {col + dc}): {nbr_val:.6e}")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()

