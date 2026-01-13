"""
Test script to compare intermediate variables step-by-step with MATLAB.
Based on get_system_matrices_VEC_simplified.
"""
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import sys
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC_simplified
from elements_vec import get_element_stiffness_VEC, get_element_mass_VEC

# Configuration
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
struct_idx = 0

print("="*70)
print("PYTHON STEP-BY-STEP COMPARISON")
print("="*70)

# Load design
if (python_data_dir / 'geometries_full.pt').exists():
    geometries = torch.load(python_data_dir / 'geometries_full.pt', map_location='cpu')
    if isinstance(geometries, torch.Tensor):
        geometries = geometries.numpy()
    design = geometries[struct_idx]
elif (python_data_dir / 'design_params_full.pt').exists():
    design_params = torch.load(python_data_dir / 'design_params_full.pt', map_location='cpu')
    if isinstance(design_params, torch.Tensor):
        design_params = design_params.numpy()
    design = design_params[struct_idx]
else:
    print("Error: Could not find design file")
    sys.exit(1)

if design.ndim == 2:
    N_pix = design.shape[0]
    design_3ch = np.stack([design, design, design], axis=-1)
else:
    N_pix = design.shape[0]
    design_3ch = design

print(f"\n1. Design loaded: shape={design.shape}, N_pix={N_pix}")

# Set up const - MATCHING MATLAB VALUES
const = {
    'design': design_3ch,
    'N_pix': N_pix,
    'N_ele': 1,
    'a': 1.0,
    'E_min': 20e6,
    'E_max': 200e9,
    'rho_min': 1200,  # MATCHING MATLAB (was 400)
    'rho_max': 8000,
    'poisson_min': 0.0,
    'poisson_max': 0.5,
    't': 1.0,  # MATCHING MATLAB (was 0.01)
    'design_scale': 'linear',
    'isUseImprovement': True,
    'isUseSecondImprovement': True,
}

print(f"\n2. const parameters:")
print(f"   N_pix = {const['N_pix']}")
print(f"   N_ele = {const['N_ele']}")
print(f"   a = {const['a']:.6f}")
print(f"   E: [{const['E_min']:.6e}, {const['E_max']:.6e}]")
print(f"   rho: [{const['rho_min']:.2f}, {const['rho_max']:.2f}]")
print(f"   nu: [{const['poisson_min']:.2f}, {const['poisson_max']:.2f}]")
print(f"   t = {const['t']:.6f}")

# Step 1: Expand design
print(f"\n=== STEP 1: Expand design ===")
N_ele = const['N_ele']
design_expanded = np.repeat(
    np.repeat(const['design'], N_ele, axis=0), 
    N_ele, axis=1
)
print(f"Expanded design shape: {design_expanded.shape}")

# Step 2: Extract material properties
print(f"\n=== STEP 2: Extract material properties ===")
if const['design_scale'] == 'linear':
    E = (const['E_min'] + design_expanded[:, :, 0] * (const['E_max'] - const['E_min'])).T
    nu = (const['poisson_min'] + design_expanded[:, :, 2] * (const['poisson_max'] - const['poisson_min'])).T
    rho = (const['rho_min'] + design_expanded[:, :, 1] * (const['rho_max'] - const['rho_min'])).T
    t = const['t']
else:
    raise ValueError("Only linear design_scale supported")

print(f"E: shape={E.shape}, min={np.min(E):.6e}, max={np.max(E):.6e}, mean={np.mean(E):.6e}")
print(f"nu: shape={nu.shape}, min={np.min(nu):.6f}, max={np.max(nu):.6f}, mean={np.mean(nu):.6f}")
print(f"rho: shape={rho.shape}, min={np.min(rho):.2f}, max={np.max(rho):.2f}, mean={np.mean(rho):.2f}")
print(f"t: {t:.6f}")

# Step 3: Compute element size
print(f"\n=== STEP 3: Compute element size ===")
N_ele_x = N_pix * N_ele
N_ele_y = N_pix * N_ele
element_size = const['a'] / (N_ele * N_pix)
element_area = element_size ** 2
print(f"N_ele_x = {N_ele_x}, N_ele_y = {N_ele_y}")
print(f"element_size = {element_size:.6e} m")
print(f"element_area = {element_area:.6e} m²")

# Step 4: Get element matrices for first element
print(f"\n=== STEP 4: Compute element matrices (first element) ===")
E_first = E.flatten()[0]
nu_first = nu.flatten()[0]
rho_first = rho.flatten()[0]
t_first = t

print(f"First element material properties:")
print(f"  E = {E_first:.6e} Pa")
print(f"  nu = {nu_first:.6f}")
print(f"  rho = {rho_first:.2f} kg/m³")
print(f"  t = {t_first:.6f} m")

# Get element stiffness matrix
k_ele = get_element_stiffness_VEC(np.array([E_first]), np.array([nu_first]), t_first)
print(f"\nElement stiffness matrix:")
print(f"  Shape: {k_ele.shape}")
print(f"  Min: {np.min(k_ele):.6e}, Max: {np.max(k_ele):.6e}, Mean: {np.mean(k_ele):.6e}")
print(f"  Sample (first 3x3):")
print(k_ele[0, :3, :3])

# Get element mass matrix
m_ele = get_element_mass_VEC(np.array([rho_first]), t_first, const)
print(f"\nElement mass matrix:")
print(f"  Shape: {m_ele.shape}")
print(f"  Min: {np.min(m_ele):.6e}, Max: {np.max(m_ele):.6e}, Mean: {np.mean(m_ele):.6e}")
print(f"  Sample (first 3x3):")
print(m_ele[0, :3, :3])

# Step 5: Compute system matrices
print(f"\n=== STEP 5: Compute system matrices ===")
K, M = get_system_matrices_VEC_simplified(const)

print(f"K matrix: shape={K.shape}, nnz={K.nnz}")
K_dense = K.toarray()
print(f"  Min: {np.min(K_dense[K_dense != 0]):.6e}, Max: {np.max(K_dense):.6e}, Mean: {np.mean(K_dense[K_dense != 0]):.6e}")

print(f"M matrix: shape={M.shape}, nnz={M.nnz}")
M_dense = M.toarray()
print(f"  Min: {np.min(M_dense[M_dense != 0]):.6e}, Max: {np.max(M_dense):.6e}, Mean: {np.mean(M_dense[M_dense != 0]):.6e}")

# Save intermediate variables for comparison
intermediates = {
    'const': const,
    'E': E,
    'nu': nu,
    'rho': rho,
    't': t,
    'element_size': element_size,
    'element_area': element_area,
    'k_ele': k_ele,
    'm_ele': m_ele,
    'E_first': E_first,
    'nu_first': nu_first,
    'rho_first': rho_first,
    't_first': t_first,
}

# Save K and M as dense arrays for easier comparison
intermediates['K_dense'] = K_dense
intermediates['M_dense'] = M_dense

sio.savemat('test_python_intermediates.mat', intermediates)
print(f"\n✅ Saved intermediate variables to test_python_intermediates.mat")

