"""
Test how indices align with values to understand if Python needs to match MATLAB's interleaving.
"""
import numpy as np

# Small test case: 2x2 elements = 4 elements
N_ele_x = 2
N_ele_y = 2
N_ele = N_ele_x * N_ele_y

print("="*70)
print("TESTING INDEX-VALUE ALIGNMENT")
print("="*70)

# Generate node numbers (MATLAB style)
nodenrs = np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1).reshape(1 + N_ele_y, 1 + N_ele_x, order='F')
print(f"\nNode numbers (shape {nodenrs.shape}):")
print(nodenrs)

# Element DOF vector
edofVec = (2 * nodenrs[0:-1, 0:-1] - 1).reshape(N_ele_x * N_ele_y, 1, order='F').flatten()
print(f"\nedofVec (global DOF of first node of each element):")
print(f"  Shape: {edofVec.shape}")
print(f"  Values: {edofVec}")

# Offset array (SIMPLIFIED version)
offset_array = np.concatenate([
    np.array([2, 3]),
    2*(N_ele_x+1) + np.array([2, 3, 0, 1]),
    np.array([0, 1])
])
print(f"\nOffset array: {offset_array}")

# Element DOF matrix
edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(offset_array, (N_ele, 1))
print(f"\nedofMat (shape {edofMat.shape}):")
print(f"  Each row is one element's 8 DOF indices")
for i in range(N_ele):
    print(f"  Element {i}: {edofMat[i]}")

# Generate row indices (MATLAB: reshape(kron(edofMat,ones(8,1))',64*N_ele,1))
row_idxs_mat = np.kron(edofMat, np.ones((8, 1)))  # Shape: (8*N_ele, 8)
print(f"\nrow_idxs_mat after kron (shape {row_idxs_mat.shape}):")
print(f"  First few rows:")
print(row_idxs_mat[:10])

row_idxs = row_idxs_mat.T.reshape(64 * N_ele, 1, order='F').flatten()
print(f"\nrow_idxs after transpose and reshape (length {len(row_idxs)}):")
print(f"  First 20 values: {row_idxs[:20]}")
print(f"  Values at indices 0, 64, 128, 192 (should show pattern):")
for i in [0, 64, 128, 192]:
    if i < len(row_idxs):
        print(f"    Index {i}: {row_idxs[i]:.0f}")

# Generate column indices
col_idxs_mat = np.kron(edofMat, np.ones((1, 8)))  # Shape: (N_ele, 64)
col_idxs = col_idxs_mat.T.reshape(64 * N_ele, 1, order='F').flatten()

# Now check the pattern: for element i, what indices appear in value_K?
print(f"\nIndex pattern analysis:")
print(f"  For element 0's first value (position 0 in flattened element matrix):")
print(f"    row_idx[0] = {row_idxs[0]:.0f}, col_idx[0] = {col_idxs[0]:.0f}")
print(f"  For element 1's first value (if interleaved, at index 1):")
if len(row_idxs) > 1:
    print(f"    row_idx[1] = {row_idxs[1]:.0f}, col_idx[1] = {col_idxs[1]:.0f}")
print(f"  For element 0's second value (if interleaved, at index N_ele):")
if N_ele < len(row_idxs):
    print(f"    row_idx[{N_ele}] = {row_idxs[N_ele]:.0f}, col_idx[{N_ele}] = {col_idxs[N_ele]:.0f}")

# Check if indices show interleaving pattern
print(f"\nChecking interleaving pattern:")
print(f"  If interleaved, row_idxs[0], row_idxs[1], row_idxs[2], row_idxs[3] should be from different elements")
print(f"    {row_idxs[0]:.0f}, {row_idxs[1]:.0f}, {row_idxs[2]:.0f}, {row_idxs[3]:.0f}")
print(f"  If sequential, row_idxs[0..63] should all be from element 0")
print(f"    First 8 (one row of element matrix): {row_idxs[0:8]}")

print("\n" + "="*70)

