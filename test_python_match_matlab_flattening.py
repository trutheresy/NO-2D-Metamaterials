"""
Test if Python needs to match MATLAB's transpose-then-flatten behavior.
MATLAB: AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)' then value_K = AllLEle(:)
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from elements_vec import get_element_stiffness_VEC

# Test with 3 elements to see interleaving pattern
N_ele = 3
E = np.array([100e9, 200e9, 300e9])
nu = np.array([0.3, 0.3, 0.3])
t = 1.0

print("="*70)
print("TESTING MATLAB-STYLE FLATTENING")
print("="*70)

# Python way (current)
AllLEle_py = get_element_stiffness_VEC(E, nu, t)
print(f"\nPython get_element_stiffness_VEC returns:")
print(f"  Shape: {AllLEle_py.shape}")  # Should be (3, 8, 8)

# Python current flattening (row-major)
value_K_py_current = AllLEle_py.flatten()
print(f"\nPython current flatten (row-major):")
print(f"  Length: {len(value_K_py_current)}")
print(f"  Indices 0, 64, 128 (first value of each element):")
print(f"    Index 0: {value_K_py_current[0]:.6e}")
print(f"    Index 64: {value_K_py_current[64]:.6e}")
print(f"    Index 128: {value_K_py_current[128]:.6e}")

# MATLAB way: reshape to (N_ele, 64), transpose to (64, N_ele), flatten column-wise
AllLEle_2d = AllLEle_py.reshape(N_ele, 64)  # Each row is one element's flattened matrix
AllLEle_transposed = AllLEle_2d.T  # (64, N_ele) - each column is one element
value_K_matlab_way = AllLEle_transposed.flatten(order='F')  # Column-major flatten

print(f"\nMATLAB way (reshape to (N_ele, 64), transpose, flatten column-wise):")
print(f"  Length: {len(value_K_matlab_way)}")
print(f"  Indices 0, 64, 128 (should be same as Python if no transpose effect):")
print(f"    Index 0: {value_K_matlab_way[0]:.6e}")
print(f"    Index 64: {value_K_matlab_way[64]:.6e}")
print(f"    Index 128: {value_K_matlab_way[128]:.6e}")

# But wait - MATLAB's transpose and column-major flatten interleaves elements
# So index 0 is element 0, index 1 is element 1, index 2 is element 2, index 3 is element 0 again...
# Let me check this pattern
print(f"\n  First 10 values (should show interleaving):")
for i in range(10):
    print(f"    Index {i}: {value_K_matlab_way[i]:.6e}")

# Compare
print(f"\nComparison:")
print(f"  Same? {np.array_equal(value_K_py_current, value_K_matlab_way)}")
if not np.array_equal(value_K_py_current, value_K_matlab_way):
    # Check if they're just reordered
    if np.allclose(np.sort(value_K_py_current), np.sort(value_K_matlab_way)):
        print(f"  ✅ Values are the same, just reordered")
        print(f"  This means assembly order differs - indices must match this reordering!")
    else:
        print(f"  ⚠️  Values differ")

# Key insight: MATLAB's value_K has elements interleaved
# element 0: indices 0, 64, 128, ...
# element 1: indices 1, 65, 129, ...
# element 2: indices 2, 66, 130, ...

# So when MATLAB does sparse(row_idxs, col_idxs, value_K), the indices and values
# must be in the same interleaved order!

print("\n" + "="*70)
print("KEY INSIGHT:")
print("MATLAB interleaves element values: value_K[i] corresponds to")
print("element (i mod N_ele) at position (i // N_ele) in that element's matrix.")
print("Python currently does sequential: all of element 0, then all of element 1, etc.")
print("="*70)

