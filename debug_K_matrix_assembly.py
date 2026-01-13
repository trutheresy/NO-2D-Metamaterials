"""
Detailed debugging of K matrix assembly to find the root cause of differences.
Compare intermediate values step-by-step.
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys
import h5py
import torch

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
from elements_vec import get_element_stiffness_VEC

# Configuration
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

print("="*70)
print("DETAILED K MATRIX ASSEMBLY DEBUGGING")
print("="*70)

# Load design
geometries = torch.load(python_data_dir / 'geometries_full.pt', map_location='cpu')
if isinstance(geometries, torch.Tensor):
    geometries = geometries.numpy()
design = geometries[struct_idx].astype(np.float32)
N_pix = design.shape[0]
design_3ch = np.stack([design, design, design], axis=-1)

const = {
    'design': design_3ch.astype(np.float32),
    'N_pix': N_pix,
    'N_ele': 1,
    'a': 1.0,
    'E_min': 20e6,
    'E_max': 200e9,
    'rho_min': 1200,
    'rho_max': 8000,
    'poisson_min': 0.0,
    'poisson_max': 0.5,
    't': 1.0,
    'design_scale': 'linear',
    'isUseImprovement': True,
    'isUseSecondImprovement': False,
}

# Manual step-by-step computation to match MATLAB
N_ele_x = N_pix * const['N_ele']
N_ele_y = N_pix * const['N_ele']

# Expand design
design_expanded = np.repeat(
    np.repeat(const['design'], const['N_ele'], axis=0), 
    const['N_ele'], axis=1
)

# Extract material properties
design_ch0 = design_expanded[:, :, 0].astype(np.float64)
design_ch2 = design_expanded[:, :, 2].astype(np.float64)
E = (const['E_min'] + design_ch0 * (const['E_max'] - const['E_min'])).T.astype(np.float32)
nu = (const['poisson_min'] + design_ch2 * (const['poisson_max'] - const['poisson_min'])).T.astype(np.float32)
t = const['t']

# Node numbering
nodenrs = np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1).reshape(1 + N_ele_y, 1 + N_ele_x, order='F')
edofVec = (2 * nodenrs[0:-1, 0:-1] - 1).reshape(N_ele_x * N_ele_y, 1, order='F').flatten()

# Offset array (MATLAB: [2*(N_ele_y+1)+[0 1 2 3] 2 3 0 1])
offset_array = np.concatenate([
    2*(N_ele_y+1) + np.array([0, 1, 2, 3]),
    np.array([2, 3, 0, 1])
])
edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(offset_array, (N_ele_x * N_ele_y, 1))

# Row and column indices
row_idxs_mat = np.kron(edofMat, np.ones((8, 1)))
row_idxs = row_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()
col_idxs_mat = np.kron(edofMat, np.ones((1, 8)))
col_idxs = col_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()

# Element stiffness matrices
AllLEle = get_element_stiffness_VEC(E.flatten(), nu.flatten(), t)
N_ele_total = N_ele_x * N_ele_y

print(f"\nElement matrix info:")
print(f"  AllLEle shape: {AllLEle.shape}")
print(f"  N_ele_total: {N_ele_total}")

# MATLAB: AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)' then value_K = AllLEle(:)
# MATLAB returns (N_ele, 64), transposes to (64, N_ele), then flattens column-wise
AllLEle_2d = AllLEle.reshape(N_ele_total, 64)
AllLEle_transposed = AllLEle_2d.T
value_K_py = AllLEle_transposed.flatten(order='F').astype(np.float32)

print(f"  AllLEle_2d shape: {AllLEle_2d.shape}")
print(f"  AllLEle_transposed shape: {AllLEle_transposed.shape}")
print(f"  value_K_py shape: {value_K_py.shape}")
print(f"  value_K_py range: [{np.min(value_K_py):.6e}, {np.max(value_K_py):.6e}]")

# Convert to 0-based indices
row_idxs_0based = row_idxs - 1
col_idxs_0based = col_idxs - 1

# Compute K matrix
from scipy.sparse import coo_matrix
K_py = coo_matrix((value_K_py, (row_idxs_0based, col_idxs_0based)), 
                  shape=(2178, 2178), dtype=np.float32).tocsr()

# Filter small values
threshold = 1e-10
K_py.data[np.abs(K_py.data) < threshold] = 0
K_py.eliminate_zeros()

# Load MATLAB K matrix
with h5py.File(str(matlab_mat_file), 'r') as f:
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    K_data_ml = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir_ml = np.array(f[K_DATA_ref]['ir']).flatten().astype(int)
    K_jc_ml = np.array(f[K_DATA_ref]['jc']).flatten().astype(int)
    n = len(K_jc_ml) - 1
    K_ml = sp.csr_matrix((K_data_ml, K_ir_ml, K_jc_ml), shape=(n, n))

K_py_dense = K_py.toarray()
K_ml_dense = K_ml.toarray()

# Find specific differences
diff = K_ml_dense - K_py_dense
diff_abs = np.abs(diff)

print(f"\nMatrix comparison:")
print(f"  Shapes: Python {K_py_dense.shape}, MATLAB {K_ml_dense.shape}")
print(f"  nnz: Python {np.count_nonzero(K_py_dense)}, MATLAB {np.count_nonzero(K_ml_dense)}")
print(f"  Max abs diff: {np.max(diff_abs):.6e}")

# Find locations with largest differences
max_diff_locs = np.unravel_index(np.argsort(diff_abs.flatten())[-20:], diff_abs.shape)
print(f"\nTop 10 locations with largest differences:")
for i in range(10):
    idx = -1 - i
    row, col = max_diff_locs[0][idx], max_diff_locs[1][idx]
    ml_val = K_ml_dense[row, col]
    py_val = K_py_dense[row, col]
    diff_val = diff[row, col]
    print(f"  [{row:4d}, {col:4d}]: MATLAB={ml_val:12.6e}, Python={py_val:12.6e}, diff={diff_val:12.6e}, ratio={ml_val/(py_val+1e-15):.6e}")

# Check if differences are in symmetric locations
print(f"\nChecking symmetry of differences:")
print(f"  diff == diff.T: {np.allclose(diff, diff.T, atol=1e-5)}")

# Save intermediate values for MATLAB comparison
import scipy.io as sio
sio.savemat('debug_K_intermediates_py.mat', {
    'row_idxs': row_idxs,
    'col_idxs': col_idxs,
    'value_K': value_K_py,
    'AllLEle': AllLEle,
    'AllLEle_2d': AllLEle_2d,
    'AllLEle_transposed': AllLEle_transposed,
    'edofMat': edofMat,
    'offset_array': offset_array,
    'E': E,
    'nu': nu,
    't': t,
}, oned_as='column')

print(f"\nSaved intermediate values to debug_K_intermediates_py.mat")

