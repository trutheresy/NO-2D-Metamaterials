"""
Compare intermediate variables from MATLAB and Python K matrix computation.
"""
import numpy as np
import scipy.io as sio
from pathlib import Path

print("="*70)
print("COMPARING K MATRIX INTERMEDIATE VARIABLES")
print("="*70)

# Load MATLAB intermediates
try:
    ml_data = sio.loadmat('test_matlab_K_intermediates.mat')
    print("\n✅ Loaded MATLAB intermediates")
except FileNotFoundError:
    print("\n❌ MATLAB intermediates not found. Run test_matlab_K_matrix_detailed.m first.")
    exit(1)

# Load Python intermediates
try:
    py_data = sio.loadmat('test_python_K_intermediates.mat')
    print("✅ Loaded Python intermediates")
except FileNotFoundError:
    print("\n❌ Python intermediates not found. Run test_python_K_matrix_detailed.py first.")
    exit(1)

def compare_array(name, py_arr, ml_arr, rtol=1e-5, atol=1e-8):
    """Compare two arrays and report differences."""
    print(f"\n{name}:")
    print(f"  Python: shape={py_arr.shape}, dtype={py_arr.dtype}")
    print(f"  MATLAB: shape={ml_arr.shape}, dtype={ml_arr.dtype}")
    
    # Handle shape differences (MATLAB might be transposed or reshaped)
    if py_arr.shape != ml_arr.shape:
        # Try to reshape MATLAB array to match Python
        if py_arr.size == ml_arr.size:
            ml_arr = ml_arr.reshape(py_arr.shape)
            print(f"  Reshaped MATLAB to match Python shape")
        else:
            print(f"  ⚠️  Shapes don't match and sizes differ ({py_arr.size} vs {ml_arr.size})")
            return False
    
    # Compare values
    diff = np.abs(py_arr - ml_arr)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    if np.allclose(py_arr, ml_arr, rtol=rtol, atol=atol):
        print(f"  ✅ Values match (max diff: {max_diff:.6e})")
        return True
    else:
        print(f"  ❌ Values differ")
        print(f"    Max difference: {max_diff:.6e}")
        print(f"    Mean difference: {mean_diff:.6e}")
        
        # Find location of max difference
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    Max diff at index {max_idx}")
        print(f"    Python value: {py_arr[max_idx]}")
        print(f"    MATLAB value: {ml_arr[max_idx]}")
        
        # Show relative error
        ml_abs = np.abs(ml_arr)
        rel_error = max_diff / (ml_abs[max_idx] + 1e-15)
        print(f"    Relative error: {rel_error:.6e}")
        
        return False

# Compare material properties
print("\n" + "="*70)
print("MATERIAL PROPERTIES COMPARISON")
print("="*70)

# Extract arrays (handle MATLAB struct format)
E_py = py_data['E']
E_ml = ml_data['E']
compare_array('E (Young\'s modulus)', E_py, E_ml)

nu_py = py_data['nu']
nu_ml = ml_data['nu']
compare_array('nu (Poisson\'s ratio)', nu_py, nu_ml)

# Compare indices
print("\n" + "="*70)
print("INDICES COMPARISON")
print("="*70)

# MATLAB uses 1-based, Python converts to 0-based
# Compare Python's 1-based indices (before conversion) with MATLAB
edofVec_py = py_data['edofVec'].flatten()
edofVec_ml = ml_data['edofVec'].flatten()
compare_array('edofVec', edofVec_py, edofVec_ml)

edofMat_py = py_data['edofMat']
edofMat_ml = ml_data['edofMat']
compare_array('edofMat', edofMat_py, edofMat_ml)

row_idxs_py = py_data['row_idxs'].flatten()
row_idxs_ml = ml_data['row_idxs'].flatten()
compare_array('row_idxs (1-based)', row_idxs_py, row_idxs_ml)

col_idxs_py = py_data['col_idxs'].flatten()
col_idxs_ml = ml_data['col_idxs'].flatten()
compare_array('col_idxs (1-based)', col_idxs_py, col_idxs_ml)

# Compare element matrices
print("\n" + "="*70)
print("ELEMENT STIFFNESS MATRICES COMPARISON")
print("="*70)

k_ele_first_py = py_data['k_ele_first']
k_ele_first_ml = ml_data['k_ele_first']
# MATLAB might have different shape (1, 8, 8) vs (8, 8)
if k_ele_first_py.ndim == 3:
    k_ele_first_py = k_ele_first_py[0]
if k_ele_first_ml.ndim == 3:
    k_ele_first_ml = k_ele_first_ml[0]
compare_array('k_ele_first (first element)', k_ele_first_py, k_ele_first_ml)

AllLEle_py = py_data['AllLEle']
AllLEle_ml = ml_data['AllLEle']
# MATLAB: (N_ele, 8, 8), Python: (N_ele, 8, 8)
# Need to check if they match after accounting for transpose
if AllLEle_py.shape != AllLEle_ml.shape:
    print(f"\nAllLEle shape mismatch:")
    print(f"  Python: {AllLEle_py.shape}")
    print(f"  MATLAB: {AllLEle_ml.shape}")
    # MATLAB might have transposed during get_element_stiffness_VEC
    if AllLEle_py.shape == AllLEle_ml.shape[::-1]:
        AllLEle_ml = AllLEle_ml.T
        print(f"  Transposed MATLAB to match Python")

# Compare flattened values
value_K_py = py_data['value_K'].flatten()
value_K_ml = ml_data['value_K'].flatten()
compare_array('value_K (flattened)', value_K_py, value_K_ml)

# Compare final K matrices
print("\n" + "="*70)
print("FINAL K MATRIX COMPARISON")
print("="*70)

K_dense_py = py_data['K_dense']
K_dense_ml = ml_data['K_dense']
compare_array('K_dense', K_dense_py, K_dense_ml, rtol=1e-4, atol=1e-6)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

