#!/usr/bin/env python3
"""
Compare the scale and statistics of K matrices between MATLAB and Python.
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

def compute_statistics(name, K):
    """Compute detailed statistics for a sparse matrix."""
    values = K.data[np.abs(K.data) > 1e-15]
    abs_values = np.abs(values)
    
    stats = {
        'name': name,
        'nnz': K.nnz,
        'n_values': len(values),
        'min': np.min(abs_values),
        'max': np.max(abs_values),
        'mean': np.mean(abs_values),
        'median': np.median(abs_values),
        'std': np.std(abs_values),
        'p1': np.percentile(abs_values, 1),
        'p5': np.percentile(abs_values, 5),
        'p10': np.percentile(abs_values, 10),
        'p25': np.percentile(abs_values, 25),
        'p50': np.percentile(abs_values, 50),
        'p75': np.percentile(abs_values, 75),
        'p90': np.percentile(abs_values, 90),
        'p95': np.percentile(abs_values, 95),
        'p99': np.percentile(abs_values, 99),
    }
    
    return stats

def compare_statistics(stats_ml, stats_py):
    """Compare two statistics dictionaries."""
    print(f"\nComparing {stats_ml['name']} vs {stats_py['name']}")
    print("=" * 80)
    
    # Compare each statistic
    metrics = ['nnz', 'n_values', 'min', 'max', 'mean', 'median', 'std',
               'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']
    
    print(f"{'Metric':<15} {'MATLAB':<20} {'Python':<20} {'Difference':<20} {'Relative Error':<15}")
    print("-" * 90)
    
    for metric in metrics:
        ml_val = stats_ml[metric]
        py_val = stats_py[metric]
        diff = abs(ml_val - py_val)
        
        # Relative error
        if ml_val != 0:
            rel_error = (diff / abs(ml_val)) * 100
        else:
            rel_error = float('inf') if diff != 0 else 0.0
        
        # Format based on magnitude
        if abs(ml_val) > 1e6:
            fmt = "{:.6e}"
        elif abs(ml_val) > 1:
            fmt = "{:.6f}"
        else:
            fmt = "{:.10e}"
        
        ml_str = fmt.format(ml_val)
        py_str = fmt.format(py_val)
        diff_str = fmt.format(diff)
        
        if rel_error == float('inf'):
            rel_str = "inf"
        elif rel_error < 0.0001:
            rel_str = f"{rel_error:.2e}%"
        else:
            rel_str = f"{rel_error:.6f}%"
        
        print(f"{metric:<15} {ml_str:<20} {py_str:<20} {diff_str:<20} {rel_str:<15}")
    
    # Summary
    print("\n" + "-" * 90)
    print("Summary:")
    
    # Key statistics comparison
    key_stats = {
        'Mean': ('mean', 'Mean absolute value'),
        'Median': ('median', 'Median absolute value'),
        'Std Dev': ('std', 'Standard deviation'),
        'Min': ('min', 'Minimum absolute value'),
        'Max': ('max', 'Maximum absolute value'),
    }
    
    for label, (key, desc) in key_stats.items():
        ml_val = stats_ml[key]
        py_val = stats_py[key]
        diff = abs(ml_val - py_val)
        if ml_val != 0:
            rel_error = (diff / abs(ml_val)) * 100
            print(f"  {label} ({desc}):")
            print(f"    MATLAB: {ml_val:.10e}")
            print(f"    Python: {py_val:.10e}")
            print(f"    Difference: {diff:.10e} ({rel_error:.6e}%)")

def main():
    # Paths
    matlab_file = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/matlab_intermediates/step10_final_matrices.mat")
    python_file = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/python_intermediates/step10_K.npz")
    
    print("=" * 80)
    print("Comparing K Matrix Statistics: MATLAB vs Python")
    print("=" * 80)
    
    # Load matrices
    print("\nLoading matrices...")
    K_ml = load_matlab_sparse_matrix(matlab_file, 'K')
    K_py = sp.load_npz(python_file)
    
    # Compute statistics
    print("Computing statistics...")
    stats_ml = compute_statistics("MATLAB K", K_ml)
    stats_py = compute_statistics("Python K", K_py)
    
    # Compare
    compare_statistics(stats_ml, stats_py)
    
    print("\n" + "=" * 80)
    print("Conclusion:")
    print("=" * 80)
    
    # Check if differences are significant
    mean_diff = abs(stats_ml['mean'] - stats_py['mean'])
    mean_rel_error = (mean_diff / stats_ml['mean']) * 100
    
    median_diff = abs(stats_ml['median'] - stats_py['median'])
    median_rel_error = (median_diff / stats_ml['median']) * 100 if stats_ml['median'] != 0 else 0
    
    print(f"Mean difference: {mean_diff:.6e} ({mean_rel_error:.6e}%)")
    print(f"Median difference: {median_diff:.6e} ({median_rel_error:.6e}%)")
    
    if mean_rel_error < 1e-10 and median_rel_error < 1e-10:
        print("\n✓ Scale statistics are essentially identical (differences < 1e-10%)")
    elif mean_rel_error < 1e-6 and median_rel_error < 1e-6:
        print("\n✓ Scale statistics are nearly identical (differences < 1e-6%)")
    else:
        print("\n⚠ Scale statistics show some differences")

if __name__ == "__main__":
    main()

