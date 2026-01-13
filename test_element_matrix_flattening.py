"""
Test different ways of flattening element matrices to match MATLAB behavior.
MATLAB does: AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)' then value_K = AllLEle(:)
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from elements_vec import get_element_stiffness_VEC

# Test with 2 elements
N_ele = 2
E = np.array([100e9, 200e9])
nu = np.array([0.3, 0.4])
t = 1.0

print("="*70)
print("TESTING ELEMENT MATRIX FLATTENING")
print("="*70)

# Python way (current)
AllLEle_py = get_element_stiffness_VEC(E, nu, t)
print(f"\nPython get_element_stiffness_VEC returns:")
print(f"  Shape: {AllLEle_py.shape}")  # Should be (2, 8, 8)
print(f"  Element 0, first 3x3:")
print(AllLEle_py[0, :3, :3])

# Flatten Python way (row-major)
value_K_py = AllLEle_py.flatten()
print(f"\nPython flatten (row-major):")
print(f"  Length: {len(value_K_py)}")
print(f"  First 20 values (should be element 0):")
print(value_K_py[:20])

# Test MATLAB way
# MATLAB: AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)'
# The transpose ' on a (N_ele, 8, 8) array would... actually, transpose doesn't work that way on 3D arrays
# Let me think about what MATLAB actually does

# If get_element_stiffness_VEC returns a 2D array where each row is a flattened element matrix
# Shape would be (N_ele, 64)
# Then transpose makes it (64, N_ele)
# Then AllLEle(:) flattens column-wise: [col0_row0, col0_row1, ..., col1_row0, ...]

# But Python returns (N_ele, 8, 8), not (N_ele, 64)
# So we need to reshape first, then transpose, then flatten

# Test: reshape to (N_ele, 64), transpose to (64, N_ele), flatten
AllLEle_reshaped = AllLEle_py.reshape(N_ele, 64)
AllLEle_transposed = AllLEle_reshaped.T  # (64, N_ele)
value_K_ml_way = AllLEle_transposed.flatten(order='F')  # Column-major flatten (MATLAB default)

print(f"\nMATLAB way (reshape to (N_ele, 64), transpose, flatten column-wise):")
print(f"  Length: {len(value_K_ml_way)}")
print(f"  First 20 values:")
print(value_K_ml_way[:20])

# Compare
print(f"\nComparison:")
print(f"  Same? {np.array_equal(value_K_py, value_K_ml_way)}")
if not np.array_equal(value_K_py, value_K_ml_way):
    diff = np.abs(value_K_py - value_K_ml_way)
    print(f"  Max difference: {np.max(diff):.6e}")
    # Check pattern - are they just reordered?
    if np.allclose(np.sort(value_K_py), np.sort(value_K_ml_way)):
        print(f"  ✅ Values are the same, just reordered (assembly order differs)")
    else:
        print(f"  ⚠️  Values differ, not just reordered")

print("\n" + "="*70)
print("KEY INSIGHT:")
print("If MATLAB transposes before flattening, the element matrices are interleaved")
print("in value_K, whereas Python flattens sequentially. This would cause values to")
print("be placed in different positions in the global matrix!")
print("="*70)

