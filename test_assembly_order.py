"""
Test the assembly order of element matrices into global matrix.
Focus on how element matrices are flattened and indexed.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from elements_vec import get_element_stiffness_VEC

# Test with a small grid (2x2 elements)
N_pix = 2
N_ele = 1
N_ele_x = N_pix * N_ele  # 2
N_ele_y = N_pix * N_ele  # 2

print("="*70)
print("TESTING ASSEMBLY ORDER")
print("="*70)

# Create simple test data
E = np.array([100e9] * (N_ele_x * N_ele_y))
nu = np.array([0.3] * (N_ele_x * N_ele_y))
t = 1.0

print(f"\nTest grid: {N_ele_x}x{N_ele_y} elements = {N_ele_x*N_ele_y} total elements")

# Get element matrices
AllLEle = get_element_stiffness_VEC(E, nu, t)
print(f"\nAllLEle shape: {AllLEle.shape}")  # Should be (4, 8, 8)

# MATLAB does: AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)'
# Then: value_K = AllLEle(:)
# So MATLAB transposes the result first, then flattens

# Python currently does:
value_K_py = AllLEle.flatten()

# Test MATLAB way (with transpose first)
AllLEle_transposed = np.transpose(AllLEle, (0, 2, 1))  # Transpose each 8x8 matrix
value_K_ml_way = AllLEle_transposed.flatten()

print(f"\nPython way (direct flatten):")
print(f"  value_K shape: {value_K_py.shape}")
print(f"  First 10 values: {value_K_py[:10]}")

print(f"\nMATLAB way (transpose then flatten):")
print(f"  value_K shape: {value_K_ml_way.shape}")
print(f"  First 10 values: {value_K_ml_way[:10]}")

# Compare
print(f"\nComparison:")
print(f"  Same values? {np.array_equal(value_K_py, value_K_ml_way)}")
if not np.array_equal(value_K_py, value_K_ml_way):
    diff = np.abs(value_K_py - value_K_ml_way)
    print(f"  Max difference: {np.max(diff):.6e}")
    print(f"  Where different: {np.sum(diff > 1e-10)} locations")

# Check how MATLAB indexes into element matrices
# MATLAB: row_idxs = reshape(kron(edofMat,ones(8,1))',64*N_ele_x*N_ele_y,1)
# This creates row indices by repeating each edofMat row 8 times
# Then: value_K[corresponding_idx] = AllLEle flattened

# For element 0, edofMat[0,:] gives 8 DOF indices
# The element matrix is 8x8
# MATLAB's kron(edofMat,ones(8,1)) creates:
#   [edofMat[0,0] edofMat[0,0] ... (8 times)
#    edofMat[0,1] edofMat[0,1] ... (8 times)
#    ...
#    edofMat[0,7] edofMat[0,7] ... (8 times)]
# Then transposed and reshaped

# This means for element i:
# - row_idxs will have 64 entries (8 DOFs × 8 entries)
# - The corresponding value_K entries come from AllLEle[i] flattened

# The question is: how is AllLEle[i] flattened?
# MATLAB: After transpose, it's row-major (MATLAB's default after transpose)
# Python: Without transpose, it's also row-major

# Actually wait - in MATLAB, after transpose of a 3D array, the flattening order changes
# Let me check the actual MATLAB code more carefully

print("\n" + "="*70)

