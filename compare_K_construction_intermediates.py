#!/usr/bin/env python3
"""
Compare intermediate steps of K matrix construction between MATLAB and Python.
This script identifies where discrepancies first appear.
"""

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from pathlib import Path

def load_matlab_sparse_matrix(filepath, var_name='K'):
    """Load sparse matrix from MATLAB v7.3 HDF5 format."""
    import h5py
    with h5py.File(filepath, 'r') as f:
        if var_name in f:
            # Sparse matrix stored as structure
            K_ref = f[var_name]
            # Check if it's a reference array
            if hasattr(K_ref, 'shape') and K_ref.shape == (1, 1):
                K_ref = f[K_ref[0, 0]]
            elif isinstance(K_ref, h5py.Dataset):
                # It's a direct reference
                K_ref = f[K_ref[()]]
            
            data = np.array(K_ref['data']).flatten()
            ir = np.array(K_ref['ir']).flatten().astype(int)
            jc = np.array(K_ref['jc']).flatten().astype(int)
            n = len(jc) - 1
            K = sp.csr_matrix((data, ir, jc), shape=(n, n))
            return K
        else:
            raise ValueError(f"Variable {var_name} not found in file")

def compare_arrays(name, arr_ml, arr_py, tolerance=1e-10):
    """Compare two arrays and report differences."""
    arr_ml = np.asarray(arr_ml)
    arr_py = np.asarray(arr_py)
    
    # Normalize shapes (remove singleton dimensions, flatten column vectors)
    arr_ml = arr_ml.squeeze()
    arr_py = arr_py.squeeze()
    
    if arr_ml.shape != arr_py.shape:
        print(f"  {name}: SHAPE MISMATCH - MATLAB {arr_ml.shape} vs Python {arr_py.shape}")
        return False
    
    abs_diff = np.abs(arr_ml - arr_py)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    # Relative error
    epsilon = 1e-15
    magnitude = np.abs(arr_ml)
    rel_error = abs_diff / (magnitude + epsilon)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error[magnitude > epsilon])
    
    match = max_diff < tolerance
    
    print(f"  {name}:")
    print(f"    Shape: {arr_ml.shape}")
    print(f"    Max absolute diff: {max_diff:.6e}")
    print(f"    Mean absolute diff: {mean_diff:.6e}")
    print(f"    Max relative error: {max_rel_error*100:.6f}%")
    if not np.isnan(mean_rel_error):
        print(f"    Mean relative error: {mean_rel_error*100:.6f}%")
    print(f"    Match: {'OK' if match else 'FAIL'}")
    
    if not match:
        # Find location of max difference
        max_idx = np.unravel_index(np.argmax(abs_diff), arr_ml.shape)
        print(f"    Max diff location: {max_idx}")
        print(f"      MATLAB value: {arr_ml[max_idx]:.10e}")
        print(f"      Python value: {arr_py[max_idx]:.10e}")
    
    return match

def main():
    matlab_dir = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/matlab_intermediates")
    python_dir = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/python_intermediates")
    
    print("=" * 80)
    print("Comparing K Matrix Construction Intermediates")
    print("=" * 80)
    print(f"MATLAB: {matlab_dir}")
    print(f"Python: {python_dir}")
    print()
    
    all_match = True
    
    # Step 1: N_ele
    print("Step 1: N_ele_x, N_ele_y")
    ml_data = sio.loadmat(str(matlab_dir / 'step1_N_ele.mat'))
    py_data = np.load(python_dir / 'step1_N_ele.npz')
    match1 = compare_arrays('N_ele_x', ml_data['N_ele_x'], py_data['N_ele_x'])
    match2 = compare_arrays('N_ele_y', ml_data['N_ele_y'], py_data['N_ele_y'])
    all_match = all_match and match1 and match2
    print()
    
    # Step 2: design_expanded
    print("Step 2: design_expanded")
    ml_data = sio.loadmat(str(matlab_dir / 'step2_design_expanded.mat'))
    py_data = np.load(python_dir / 'step2_design_expanded.npy')
    match = compare_arrays('design_expanded', ml_data['design_expanded'], py_data)
    all_match = all_match and match
    print()
    
    # Step 3: Material properties
    print("Step 3: Material properties")
    ml_data = sio.loadmat(str(matlab_dir / 'step3_material_props.mat'))
    py_data = np.load(python_dir / 'step3_material_props.npz')
    match_E = compare_arrays('E', ml_data['E'], py_data['E'])
    match_nu = compare_arrays('nu', ml_data['nu'], py_data['nu'])
    match_rho = compare_arrays('rho', ml_data['rho'], py_data['rho'])
    all_match = all_match and match_E and match_nu and match_rho
    print()
    
    # Step 4: nodenrs
    print("Step 4: nodenrs")
    ml_data = sio.loadmat(str(matlab_dir / 'step4_nodenrs.mat'))
    py_data = np.load(python_dir / 'step4_nodenrs.npy')
    match = compare_arrays('nodenrs', ml_data['nodenrs'], py_data)
    all_match = all_match and match
    print()
    
    # Step 5: edofVec
    print("Step 5: edofVec")
    ml_data = sio.loadmat(str(matlab_dir / 'step5_edofVec.mat'))
    py_data = np.load(python_dir / 'step5_edofVec.npy')
    match = compare_arrays('edofVec', ml_data['edofVec'], py_data)
    all_match = all_match and match
    print()
    
    # Step 6: edofMat
    print("Step 6: edofMat")
    ml_data = sio.loadmat(str(matlab_dir / 'step6_edofMat.mat'))
    py_data = np.load(python_dir / 'step6_edofMat.npy')
    match = compare_arrays('edofMat', ml_data['edofMat'], py_data)
    all_match = all_match and match
    print()
    
    # Step 7: row_idxs, col_idxs
    print("Step 7: row_idxs, col_idxs")
    ml_data = sio.loadmat(str(matlab_dir / 'step7_indices.mat'))
    py_data = np.load(python_dir / 'step7_indices.npz')
    match_row = compare_arrays('row_idxs', ml_data['row_idxs'], py_data['row_idxs'])
    match_col = compare_arrays('col_idxs', ml_data['col_idxs'], py_data['col_idxs'])
    all_match = all_match and match_row and match_col
    print()
    
    # Step 8: Element matrices
    print("Step 8: Element matrices")
    ml_data = sio.loadmat(str(matlab_dir / 'step8_element_matrices.mat'))
    py_data = np.load(python_dir / 'step8_element_matrices.npz')
    
    # MATLAB stores as (64, 1024), Python as (1024, 8, 8)
    # MATLAB: each column is one element's flattened 8x8 matrix (64 elements)
    # Python: (N_ele, 8, 8) - need to reshape to match MATLAB format
    ml_AllLEle = ml_data['AllLEle']  # (64, 1024)
    py_AllLEle = py_data['AllLEle']  # (1024, 8, 8)
    # Reshape Python to match MATLAB: flatten each 8x8 to 64, then transpose
    py_AllLEle_reshaped = py_AllLEle.reshape(1024, 64).T  # (64, 1024)
    match_K = compare_arrays('AllLEle', ml_AllLEle, py_AllLEle_reshaped)
    
    ml_AllLMat = ml_data['AllLMat']  # (64, 1024)
    py_AllLMat = py_data['AllLMat']  # (1024, 8, 8)
    py_AllLMat_reshaped = py_AllLMat.reshape(1024, 64).T  # (64, 1024)
    match_M = compare_arrays('AllLMat', ml_AllLMat, py_AllLMat_reshaped)
    all_match = all_match and match_K and match_M
    print()
    
    # Step 9: Flattened values
    print("Step 9: Flattened values")
    ml_data = sio.loadmat(str(matlab_dir / 'step9_values.mat'))
    py_data = np.load(python_dir / 'step9_values.npz')
    match_K = compare_arrays('value_K', ml_data['value_K'], py_data['value_K'])
    match_M = compare_arrays('value_M', ml_data['value_M'], py_data['value_M'])
    all_match = all_match and match_K and match_M
    print()
    
    # Step 10: Final matrices
    print("Step 10: Final K matrix")
    ml_K = load_matlab_sparse_matrix(matlab_dir / 'step10_final_matrices.mat', 'K')
    py_K = sp.load_npz(python_dir / 'step10_K.npz')
    
    # Compare sparse matrices
    diff = ml_K - py_K
    diff_dense = diff.toarray()
    ml_dense = ml_K.toarray()
    
    abs_diff = np.abs(diff_dense)
    max_diff = np.max(abs_diff)
    frobenius_diff = np.linalg.norm(diff_dense, 'fro')
    frobenius_ml = np.linalg.norm(ml_dense, 'fro')
    rel_error = frobenius_diff / frobenius_ml if frobenius_ml > 0 else 0
    
    print(f"  K matrix:")
    print(f"    MATLAB: shape={ml_K.shape}, nnz={ml_K.nnz}")
    print(f"    Python: shape={py_K.shape}, nnz={py_K.nnz}")
    print(f"    Max absolute diff: {max_diff:.6e}")
    print(f"    Frobenius relative error: {rel_error*100:.6f}%")
    
    # Check sparsity pattern
    threshold = 1e-10
    ml_nonzero = np.abs(ml_dense) > threshold
    py_nonzero = np.abs(py_K.toarray()) > threshold
    n_ml_only = np.sum(ml_nonzero & ~py_nonzero)
    n_py_only = np.sum(py_nonzero & ~ml_nonzero)
    n_both = np.sum(ml_nonzero & py_nonzero)
    
    print(f"    Sparsity pattern:")
    print(f"      Non-zero only in MATLAB: {n_ml_only}")
    print(f"      Non-zero only in Python: {n_py_only}")
    print(f"      Non-zero in both: {n_both}")
    
    print()
    print("=" * 80)
    if all_match:
        print("All intermediate steps match!")
    else:
        print("Discrepancies found in intermediate steps.")
    print("=" * 80)

if __name__ == "__main__":
    main()

