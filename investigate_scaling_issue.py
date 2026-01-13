"""
Investigate scaling differences between Python and MATLAB K & M matrices.

This script performs incremental tests to identify where the scaling difference comes from.
"""
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import sys

# Add 2d-dispersion-py to path
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC_simplified
from elements_vec import get_element_stiffness_VEC, get_element_mass_VEC

# Configuration
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

print("="*70)
print("INVESTIGATING SCALING DIFFERENCE")
print("="*70)

# Load Python matrices
print("\n1. Loading Python K & M matrices...")
K_data = torch.load(python_data_dir / 'K_data.pt', map_location='cpu')
M_data = torch.load(python_data_dir / 'M_data.pt', map_location='cpu')
K_py = K_data[struct_idx]
M_py = M_data[struct_idx]
if not sp.issparse(K_py):
    K_py = sp.csr_matrix(K_py)
if not sp.issparse(M_py):
    M_py = sp.csr_matrix(M_py)
print(f"   Python K: shape={K_py.shape}, nnz={K_py.nnz}")
print(f"   Python M: shape={M_py.shape}, nnz={M_py.nnz}")

# Load MATLAB matrices
print("\n2. Loading MATLAB K & M matrices...")
try:
    import h5py
    with h5py.File(str(matlab_mat_file), 'r') as f:
        # Get references to K_DATA and M_DATA
        K_DATA_ref = f['K_DATA'][struct_idx, 0]
        M_DATA_ref = f['M_DATA'][struct_idx, 0]
        
        # Extract sparse matrix data
        K_data_matlab = np.array(f[K_DATA_ref]['data']).flatten()
        K_ir = np.array(f[K_DATA_ref]['ir']).flatten().astype(int)
        K_jc = np.array(f[K_DATA_ref]['jc']).flatten().astype(int)
        
        M_data_matlab = np.array(f[M_DATA_ref]['data']).flatten()
        M_ir = np.array(f[M_DATA_ref]['ir']).flatten().astype(int)
        M_jc = np.array(f[M_DATA_ref]['jc']).flatten().astype(int)
        
        # Get shape (should be (2178, 2178))
        n = len(K_jc) - 1
    
    K_ml = sp.csr_matrix((K_data_matlab, K_ir, K_jc), shape=(n, n))
    M_ml = sp.csr_matrix((M_data_matlab, M_ir, M_jc), shape=(n, n))
    print(f"   MATLAB K: shape={K_ml.shape}, nnz={K_ml.nnz}")
    print(f"   MATLAB M: shape={M_ml.shape}, nnz={M_ml.nnz}")
except Exception as e:
    print(f"   Error loading MATLAB data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare value scales
print("\n3. Comparing value scales...")
K_py_dense = K_py.toarray()
K_ml_dense = K_ml.toarray()
M_py_dense = M_py.toarray()
M_ml_dense = M_ml.toarray()

print(f"\n   K matrix values:")
print(f"   Python: min={np.min(K_py_dense[K_py_dense != 0]):.6e}, max={np.max(K_py_dense):.6e}, mean={np.mean(K_py_dense[K_py_dense != 0]):.6e}")
print(f"   MATLAB: min={np.min(K_ml_dense[K_ml_dense != 0]):.6e}, max={np.max(K_ml_dense):.6e}, mean={np.mean(K_ml_dense[K_ml_dense != 0]):.6e}")
ratio_K = np.mean(K_ml_dense[K_ml_dense != 0]) / np.mean(K_py_dense[K_py_dense != 0])
print(f"   Ratio (MATLAB/Python): {ratio_K:.6e}")

print(f"\n   M matrix values:")
print(f"   Python: min={np.min(M_py_dense[M_py_dense != 0]):.6e}, max={np.max(M_py_dense):.6e}, mean={np.mean(M_py_dense[M_py_dense != 0]):.6e}")
print(f"   MATLAB: min={np.min(M_ml_dense[M_ml_dense != 0]):.6e}, max={np.max(M_ml_dense):.6e}, mean={np.mean(M_ml_dense[M_ml_dense != 0]):.6e}")
ratio_M = np.mean(M_ml_dense[M_ml_dense != 0]) / np.mean(M_py_dense[M_py_dense != 0])
print(f"   Ratio (MATLAB/Python): {ratio_M:.6e}")

# Load design to recompute
print("\n4. Loading design data to recompute matrices...")
try:
    designs_dataset = torch.load(python_data_dir / 'designs_dataset.pt', map_location='cpu')
    if hasattr(designs_dataset, 'tensors') and len(designs_dataset.tensors) > 0:
        design = designs_dataset.tensors[0][struct_idx].numpy()
    else:
        design = designs_dataset[struct_idx].numpy()
    
    # Handle single-channel design
    if design.ndim == 2:
        print(f"   Single-channel design: shape={design.shape}")
    else:
        print(f"   Multi-channel design: shape={design.shape}")
    
    # Extract N_pix from design
    if design.ndim == 2:
        N_pix = design.shape[0]
    else:
        N_pix = design.shape[0]
    
    print(f"   N_pix = {N_pix}")
    
except Exception as e:
    print(f"   Error loading design: {e}")
    sys.exit(1)

# Check const parameters
print("\n5. Checking const parameters used for computation...")
# Load from saved const if available, or use defaults
const = {
    'design': design if design.ndim == 3 else np.stack([design, design, design], axis=-1),
    'N_pix': N_pix,
    'N_ele': 1,  # Based on previous investigation
    'a': 1.0,
    'E_min': 20e6,
    'E_max': 200e9,
    'rho_min': 400,
    'rho_max': 8000,
    'poisson_min': 0.0,
    'poisson_max': 0.5,
    't': 0.01,
    'design_scale': 'linear',
    'isUseImprovement': True,
    'isUseSecondImprovement': True,  # Use simplified version
}

print(f"   N_ele = {const['N_ele']}")
print(f"   E: [{const['E_min']:.6e}, {const['E_max']:.6e}]")
print(f"   rho: [{const['rho_min']:.2f}, {const['rho_max']:.2f}]")
print(f"   nu: [{const['poisson_min']:.2f}, {const['poisson_max']:.2f}]")
print(f"   t = {const['t']:.6f}")

# Recompute Python matrices
print("\n6. Recomputing Python K & M matrices...")
try:
    K_py_recomputed, M_py_recomputed = get_system_matrices_VEC_simplified(const)
    print(f"   Recomputed K: shape={K_py_recomputed.shape}, nnz={K_py_recomputed.nnz}")
    print(f"   Recomputed M: shape={M_py_recomputed.shape}, nnz={M_py_recomputed.nnz}")
    
    # Compare with saved
    diff_K_saved = K_py - K_py_recomputed
    diff_M_saved = M_py - M_py_recomputed
    print(f"\n   Comparing saved vs recomputed Python matrices:")
    print(f"   K difference nnz: {diff_K_saved.nnz}, norm: {sp.linalg.norm(diff_K_saved):.6e}")
    print(f"   M difference nnz: {diff_M_saved.nnz}, norm: {sp.linalg.norm(diff_M_saved):.6e}")
    
    # Use recomputed for further analysis
    K_py = K_py_recomputed
    M_py = M_py_recomputed
    
except Exception as e:
    print(f"   Error recomputing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check material property extraction
print("\n7. Checking material property extraction...")
N_ele = const['N_ele']
N_pix_val = const['N_pix']
N_ele_x = N_pix_val * N_ele
N_ele_y = N_pix_val * N_ele

# Replicate design
design_expanded = np.repeat(
    np.repeat(const['design'], N_ele, axis=0), 
    N_ele, axis=1
)

if const['design_scale'] == 'linear':
    E_extracted = (const['E_min'] + design_expanded[:, :, 0] * (const['E_max'] - const['E_min'])).T
    nu_extracted = (const['poisson_min'] + design_expanded[:, :, 2] * (const['poisson_max'] - const['poisson_min'])).T
    rho_extracted = (const['rho_min'] + design_expanded[:, :, 1] * (const['rho_max'] - const['rho_min'])).T
    t_extracted = const['t']
else:
    E_extracted = np.exp(design_expanded[:, :, 0]).T
    nu_extracted = (const['poisson_min'] + design_expanded[:, :, 2] * (const['poisson_max'] - const['poisson_min'])).T
    rho_extracted = np.exp(design_expanded[:, :, 1]).T
    t_extracted = const['t']

print(f"   E: shape={E_extracted.shape}, min={np.min(E_extracted):.6e}, max={np.max(E_extracted):.6e}")
print(f"   nu: shape={nu_extracted.shape}, min={np.min(nu_extracted):.6f}, max={np.max(nu_extracted):.6f}")
print(f"   rho: shape={rho_extracted.shape}, min={np.min(rho_extracted):.2f}, max={np.max(rho_extracted):.2f}")
print(f"   t: {t_extracted:.6f}")

# Check element matrices for first element
print("\n8. Checking element matrices for first element...")
E_first = E_extracted.flatten()[0]
nu_first = nu_extracted.flatten()[0]
rho_first = rho_extracted.flatten()[0]
t_first = t_extracted

print(f"   First element material properties:")
print(f"   E = {E_first:.6e} Pa")
print(f"   nu = {nu_first:.6f}")
print(f"   rho = {rho_first:.2f} kg/m³")
print(f"   t = {t_first:.6f} m")

# Get element matrices
k_ele = get_element_stiffness_VEC(np.array([E_first]), np.array([nu_first]), t_first)
m_ele = get_element_mass_VEC(np.array([rho_first]), t_first, const)

print(f"\n   Element stiffness matrix (first element):")
print(f"   Shape: {k_ele.shape}")
print(f"   Min: {np.min(k_ele):.6e}, Max: {np.max(k_ele):.6e}, Mean: {np.mean(k_ele):.6e}")
print(f"   Sample (first 3x3):")
print(k_ele[0, :3, :3])

print(f"\n   Element mass matrix (first element):")
print(f"   Shape: {m_ele.shape}")
print(f"   Min: {np.min(m_ele):.6e}, Max: {np.max(m_ele):.6e}, Mean: {np.mean(m_ele):.6e}")
print(f"   Sample (first 3x3):")
print(m_ele[0, :3, :3])

# Check if there's a scaling factor in element matrix computation
print("\n9. Checking element matrix computation functions...")
# Read element stiffness and mass functions to look for scaling issues
from get_element_stiffness import get_element_stiffness
from get_element_mass import get_element_mass

# Compare single element computation
k_ele_single = get_element_stiffness(E_first, nu_first, t_first, const)
m_ele_single = get_element_mass(rho_first, t_first, const)

print(f"   Single element stiffness (non-vectorized):")
print(f"   Shape: {k_ele_single.shape}")
print(f"   Min: {np.min(k_ele_single):.6e}, Max: {np.max(k_ele_single):.6e}, Mean: {np.mean(k_ele_single):.6e}")

print(f"   Single element mass (non-vectorized):")
print(f"   Shape: {m_ele_single.shape}")
print(f"   Min: {np.min(m_ele_single):.6e}, Max: {np.max(m_ele_single):.6e}, Mean: {np.mean(m_ele_single):.6e}")

# Compare with vectorized version
k_ele_vec_first = k_ele[0]
m_ele_vec_first = m_ele[0]

diff_k_ele = np.abs(k_ele_single - k_ele_vec_first)
diff_m_ele = np.abs(m_ele_single - m_ele_vec_first)

print(f"\n   Comparison (vectorized vs non-vectorized):")
print(f"   k_ele diff max: {np.max(diff_k_ele):.6e}, mean: {np.mean(diff_k_ele):.6e}")
print(f"   m_ele diff max: {np.max(diff_m_ele):.6e}, mean: {np.mean(diff_m_ele):.6e}")

print("\n" + "="*70)
print("INVESTIGATION COMPLETE")
print("="*70)

