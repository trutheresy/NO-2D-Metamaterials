"""
Detailed step-by-step comparison script for Python.
Saves all intermediate variables for comparison with MATLAB.
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
print("PYTHON DETAILED STEP-BY-STEP")
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
if not np.any(np.isnan(k_ele)) and not np.any(np.isinf(k_ele)):
    print(f"  Sample (first 3x3):")
    print(k_ele[0, :3, :3])
else:
    print(f"  ⚠️  Contains NaN or Inf values!")

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
K_nonzero = K_dense[K_dense != 0]
if len(K_nonzero) > 0 and not np.any(np.isnan(K_nonzero)) and not np.any(np.isinf(K_nonzero)):
    print(f"  Min: {np.min(K_nonzero):.6e}, Max: {np.max(K_dense):.6e}, Mean: {np.mean(K_nonzero):.6e}")
else:
    print(f"  ⚠️  Contains NaN or Inf values!")

print(f"M matrix: shape={M.shape}, nnz={M.nnz}")
M_dense = M.toarray()
M_nonzero = M_dense[M_dense != 0]
print(f"  Min: {np.min(M_nonzero):.6e}, Max: {np.max(M_dense):.6e}, Mean: {np.mean(M_nonzero):.6e}")

# Save intermediate variables
intermediates = {
    'const': const,
    'design_expanded': design_expanded,
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
    'K_dense': K_dense,
    'M_dense': M_dense,
}

sio.savemat('test_python_intermediates_detailed.mat', intermediates)
print(f"\n✅ Saved intermediate variables to test_python_intermediates_detailed.mat")

