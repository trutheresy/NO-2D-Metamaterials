"""
Compare intermediate values from Python and MATLAB K matrix assembly.
"""
import numpy as np
import scipy.io as sio
from pathlib import Path

print("="*70)
print("COMPARING K MATRIX INTERMEDIATE VALUES")
print("="*70)

# Load Python intermediates
py_data = sio.loadmat('debug_K_intermediates_py.mat')
print("\n✅ Loaded Python intermediates")

# Load MATLAB intermediates
ml_data = sio.loadmat('2D-dispersion-han/debug_K_intermediates_matlab.mat')
print("✅ Loaded MATLAB intermediates")

def compare_array(name, py_arr, ml_arr, rtol=1e-5, atol=1e-8):
    """Compare two arrays and report differences."""
    print(f"\n--- {name} ---")
    
    # Handle shape differences (MATLAB column vectors vs Python 1D arrays)
    if py_arr.ndim == 1 and ml_arr.ndim == 2 and ml_arr.shape[1] == 1:
        ml_arr = ml_arr.flatten()
    elif py_arr.ndim == 2 and py_arr.shape[1] == 1 and ml_arr.ndim == 1:
        py_arr = py_arr.flatten()
    
    print(f"  Python shape: {py_arr.shape}, MATLAB shape: {ml_arr.shape}")
    
    if py_arr.shape != ml_arr.shape:
        print(f"  ❌ Shapes do not match!")
        return False
    
    if np.allclose(py_arr, ml_arr, rtol=rtol, atol=atol):
        print(f"  ✅ Values match (within tolerance)")
        return True
    else:
        diff = np.abs(py_arr - ml_arr)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  ❌ Values differ")
        print(f"    Max abs diff: {max_diff:.6e}")
        print(f"    Mean abs diff: {mean_diff:.6e}")
        
        # Find location of max difference
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    Max diff at index {max_idx}")
        print(f"    Python value: {py_arr[max_idx]}")
        print(f"    MATLAB value: {ml_arr[max_idx]}")
        
        # Sample comparison
        print(f"    First 10 values:")
        print(f"      Python: {py_arr.flatten()[:10]}")
        print(f"      MATLAB: {ml_arr.flatten()[:10]}")
        
        return False

# Compare material properties
compare_array('E', py_data['E'], ml_data['E'])
compare_array('nu', py_data['nu'], ml_data['nu'])
compare_array('t', py_data['t'], ml_data['t'])

# Compare offset_array
compare_array('offset_array', py_data['offset_array'], ml_data['offset_array'])

# Compare edofMat
compare_array('edofMat', py_data['edofMat'], ml_data['edofMat'])

# Compare row_idxs and col_idxs
compare_array('row_idxs', py_data['row_idxs'], ml_data['row_idxs'])
compare_array('col_idxs', py_data['col_idxs'], ml_data['col_idxs'])

# Compare AllLEle (before transpose in MATLAB)
# MATLAB: AllLEle_raw is (8, 8, N_ele), AllLEle is (N_ele, 64) after transpose
# Python: AllLEle is (N_ele, 8, 8)
py_AllLEle = py_data['AllLEle']
ml_AllLEle_raw = ml_data['AllLEle_raw']  # (8, 8, N_ele)

# Reshape MATLAB to (N_ele, 8, 8) for comparison
N_ele = py_AllLEle.shape[0]
ml_AllLEle_reshaped = permute(ml_AllLEle_raw, [3, 1, 2]);  # (N_ele, 8, 8)

print(f"\n--- AllLEle (element matrices) ---")
print(f"  Python shape: {py_AllLEle.shape}")
print(f"  MATLAB raw shape: {ml_AllLEle_raw.shape}")
print(f"  MATLAB reshaped shape: {ml_AllLEle_reshaped.shape}")

if np.allclose(py_AllLEle, ml_AllLEle_reshaped, rtol=1e-5, atol=1e-8):
    print(f"  ✅ Element matrices match")
else:
    print(f"  ❌ Element matrices differ")
    # Compare first element
    print(f"  First element comparison:")
    diff_first = np.abs(py_AllLEle[0] - ml_AllLEle_reshaped[0])
    print(f"    Max diff: {np.max(diff_first):.6e}")
    print(f"    Mean diff: {np.mean(diff_first):.6e}")

# Compare AllLEle after transpose and reshape
# MATLAB: AllLEle is (N_ele, 64) after transpose of AllLEle_raw
# Python: AllLEle_2d is (N_ele, 64) after reshape
compare_array('AllLEle_2d (reshaped)', py_data['AllLEle_2d'], ml_data['AllLEle'])

# Compare AllLEle_transposed
# MATLAB: AllLEle is already (N_ele, 64), so transpose would be (64, N_ele)
# Python: AllLEle_transposed is (64, N_ele)
py_transposed = py_data['AllLEle_transposed']
ml_transposed = ml_data['AllLEle'].T  # Transpose
compare_array('AllLEle_transposed', py_transposed, ml_transposed)

# Compare value_K (flattened)
compare_array('value_K (flattened)', py_data['value_K'], ml_data['value_K'])

print("\n" + "="*70)

