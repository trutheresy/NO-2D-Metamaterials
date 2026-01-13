"""
Detailed Python test script for K matrix computation.
Saves all intermediate variables for comparison with MATLAB.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch
from pathlib import Path
import sys
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC_simplified
from elements_vec import get_element_stiffness_VEC

# Configuration
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
struct_idx = 0

print("="*70)
print("PYTHON DETAILED K MATRIX COMPUTATION")
print("="*70)

# Load design
if (python_data_dir / 'geometries_full.pt').exists():
    geometries = torch.load(python_data_dir / 'geometries_full.pt', map_location='cpu')
    if isinstance(geometries, torch.Tensor):
        geometries = geometries.numpy()
    design = geometries[struct_idx]
else:
    print("Error: Could not find design file")
    sys.exit(1)

# Convert to float32 to avoid overflow
design = design.astype(np.float32)

if design.ndim == 2:
    N_pix = design.shape[0]
    design_3ch = np.stack([design, design, design], axis=-1)
else:
    N_pix = design.shape[0]
    design_3ch = design

print(f"\n1. Design: shape={design.shape}, dtype={design.dtype}")
print(f"   Min: {np.min(design):.6f}, Max: {np.max(design):.6f}")

# Set up const - MATCHING MATLAB VALUES
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
    'isUseSecondImprovement': True,
}

print(f"\n2. const parameters:")
for key in ['N_pix', 'N_ele', 'a', 'E_min', 'E_max', 'rho_min', 'rho_max', 'poisson_min', 'poisson_max', 't']:
    print(f"   {key} = {const[key]}")

# Step 1: Expand design
print(f"\n=== STEP 1: Expand design ===")
N_ele = const['N_ele']
design_expanded = np.repeat(
    np.repeat(const['design'], N_ele, axis=0), 
    N_ele, axis=1
).astype(np.float32)
print(f"Expanded design shape: {design_expanded.shape}, dtype: {design_expanded.dtype}")
print(f"Design range: [{np.min(design_expanded):.6f}, {np.max(design_expanded):.6f}]")

# Step 2: Extract material properties
print(f"\n=== STEP 2: Extract material properties ===")
if const['design_scale'] == 'linear':
    # Use float64 for intermediate calculation to avoid overflow
    design_ch0 = design_expanded[:, :, 0].astype(np.float64)
    design_ch1 = design_expanded[:, :, 1].astype(np.float64)
    design_ch2 = design_expanded[:, :, 2].astype(np.float64)
    
    E = (const['E_min'] + design_ch0 * (const['E_max'] - const['E_min'])).T.astype(np.float32)
    nu = (const['poisson_min'] + design_ch2 * (const['poisson_max'] - const['poisson_min'])).T.astype(np.float32)
    rho = (const['rho_min'] + design_ch1 * (const['rho_max'] - const['rho_min'])).T.astype(np.float32)
    t = const['t']
else:
    raise ValueError("Only linear design_scale supported")

print(f"E: shape={E.shape}, dtype={E.dtype}, min={np.min(E):.6e}, max={np.max(E):.6e}, mean={np.mean(E):.6e}")
print(f"nu: shape={nu.shape}, dtype={nu.dtype}, min={np.min(nu):.6f}, max={np.max(nu):.6f}, mean={np.mean(nu):.6f}")
print(f"rho: shape={rho.shape}, dtype={rho.dtype}, min={np.min(rho):.2f}, max={np.max(rho):.2f}, mean={np.mean(rho):.2f}")
print(f"t: {t:.6f}")

# Step 3: Compute indices
print(f"\n=== STEP 3: Compute indices ===")
N_ele_x = N_pix * N_ele
N_ele_y = N_pix * N_ele
print(f"N_ele_x = {N_ele_x}, N_ele_y = {N_ele_y}")

# Node numbering
nodenrs = np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1).reshape(1 + N_ele_y, 1 + N_ele_x, order='F')

# Element DOF vector
edofVec = (2 * nodenrs[0:-1, 0:-1] - 1).reshape(N_ele_x * N_ele_y, 1, order='F').flatten()

# Element DOF matrix (SIMPLIFIED VERSION)
offset_array = np.concatenate([
    np.array([2, 3]),
    2*(N_ele_x+1) + np.array([2, 3, 0, 1]),
    np.array([0, 1])
])
edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(offset_array, (N_ele_x * N_ele_y, 1))

print(f"edofVec shape: {edofVec.shape}, range: [{np.min(edofVec)}, {np.max(edofVec)}]")
print(f"edofMat shape: {edofMat.shape}, range: [{np.min(edofMat)}, {np.max(edofMat)}]")

# Row and column indices
row_idxs_mat = np.kron(edofMat, np.ones((8, 1)))
row_idxs = row_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()

col_idxs_mat = np.kron(edofMat, np.ones((1, 8)))
col_idxs = col_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()

print(f"row_idxs shape: {row_idxs.shape}, range: [{np.min(row_idxs)}, {np.max(row_idxs)}]")
print(f"col_idxs shape: {col_idxs.shape}, range: [{np.min(col_idxs)}, {np.max(col_idxs)}]")

# Step 4: Get element stiffness matrices
print(f"\n=== STEP 4: Compute element stiffness matrices ===")
AllLEle = get_element_stiffness_VEC(E.flatten(), nu.flatten(), t)
print(f"AllLEle shape: {AllLEle.shape}")
print(f"AllLEle range: [{np.min(AllLEle):.6e}, {np.max(AllLEle):.6e}], mean: {np.mean(AllLEle):.6e}")

# Check first element
E_first = E.flatten()[0]
nu_first = nu.flatten()[0]
t_first = t
k_ele_first = get_element_stiffness_VEC(np.array([E_first]), np.array([nu_first]), t_first)
print(f"\nFirst element (E={E_first:.6e}, nu={nu_first:.6f}, t={t_first:.6f}):")
print(f"  k_ele shape: {k_ele_first.shape}")
print(f"  k_ele range: [{np.min(k_ele_first):.6e}, {np.max(k_ele_first):.6e}], mean: {np.mean(k_ele_first):.6e}")
print(f"  k_ele sample (first 3x3):")
print(k_ele_first[0, :3, :3])

# Step 5: Flatten and assemble
print(f"\n=== STEP 5: Flatten and assemble K matrix ===")
value_K = AllLEle.flatten().astype(np.float32)
print(f"value_K shape: {value_K.shape}")
print(f"value_K range: [{np.min(value_K):.6e}, {np.max(value_K):.6e}], mean: {np.mean(value_K):.6e}")
print(f"value_K nnz: {np.count_nonzero(value_K)}")

# Convert 1-based to 0-based indices
row_idxs_0 = row_idxs - 1
col_idxs_0 = col_idxs - 1

# Calculate dimensions
N_nodes_x = N_ele_x + 1
N_nodes_y = N_ele_y + 1
N_dof = N_nodes_x * N_nodes_y * 2

# Create sparse matrix
K = coo_matrix((value_K, (row_idxs_0, col_idxs_0)), shape=(N_dof, N_dof), dtype=np.float32).tocsr()

# Filter small values
threshold = 1e-10
K.data[np.abs(K.data) < threshold] = 0
K.eliminate_zeros()

print(f"\nAssembled K matrix:")
print(f"  Shape: {K.shape}, nnz: {K.nnz}")
K_dense = K.toarray()
K_nonzero = K_dense[K_dense != 0]
print(f"  Non-zero values: min={np.min(K_nonzero):.6e}, max={np.max(K_nonzero):.6e}, mean={np.mean(K_nonzero):.6e}")
print(f"  All values: min={np.min(K_dense):.6e}, max={np.max(K_dense):.6e}, mean={np.mean(K_dense):.6e}")

# Save intermediate variables
intermediates = {
    'const': const,
    'design_expanded': design_expanded,
    'E': E,
    'nu': nu,
    'rho': rho,
    't': t,
    'nodenrs': nodenrs,
    'edofVec': edofVec,
    'edofMat': edofMat,
    'row_idxs': row_idxs,  # 1-based (before conversion)
    'col_idxs': col_idxs,  # 1-based (before conversion)
    'row_idxs_0': row_idxs_0,  # 0-based (after conversion)
    'col_idxs_0': col_idxs_0,  # 0-based (after conversion)
    'AllLEle': AllLEle,
    'value_K': value_K,
    'K_dense': K_dense,
    'E_first': E_first,
    'nu_first': nu_first,
    't_first': t_first,
    'k_ele_first': k_ele_first,
}

sio.savemat('test_python_K_intermediates.mat', intermediates)
print(f"\n✅ Saved intermediate variables to test_python_K_intermediates.mat")

