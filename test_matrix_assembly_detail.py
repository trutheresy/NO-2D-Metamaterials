"""
Test script to compare matrix assembly in detail.
Check for duplicate entries and small values that might be treated as zero.
"""

import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import h5py
import sys
sys.path.insert(0, '2d-dispersion-py')

from plot_dispersion_infer_eigenfrequencies import create_const_dict, compute_K_M_matrices, load_pt_dataset

# Load data
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

print("="*70)
print("Detailed Matrix Assembly Comparison")
print("="*70)

# Load design
data = load_pt_dataset(python_data_dir, None, require_eigenvalue_data=False)
design = data['designs'][struct_idx]
print(f"\nDesign shape: {design.shape}")

# Create const and compute matrices
const = create_const_dict(
    design, N_pix=32, N_ele=1,
    a=1.0, E_min=20e6, E_max=200e9,
    rho_min=400, rho_max=8000,
    nu_min=0.0, nu_max=0.5
)

print("\nComputing Python K, M matrices...")
K_py, M_py = compute_K_M_matrices(const)
print(f"Python K: shape={K_py.shape}, nnz={K_py.nnz}")
print(f"Python M: shape={M_py.shape}, nnz={M_py.nnz}")

# Load MATLAB matrices
print("\nLoading MATLAB matrices...")
with h5py.File(matlab_mat_file, 'r') as f:
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    M_DATA_ref = f['M_DATA'][struct_idx, 0]
    
    K_data_ml = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir_ml = np.array(f[K_DATA_ref]['ir']).flatten()
    K_jc_ml = np.array(f[K_DATA_ref]['jc']).flatten()
    
    M_data_ml = np.array(f[M_DATA_ref]['data']).flatten()
    M_ir_ml = np.array(f[M_DATA_ref]['ir']).flatten()
    M_jc_ml = np.array(f[M_DATA_ref]['jc']).flatten()
    
    n = len(K_jc_ml) - 1
    K_ml = sp.csr_matrix((K_data_ml, K_ir_ml, K_jc_ml), shape=(n, n))
    M_ml = sp.csr_matrix((M_data_ml, M_ir_ml, M_jc_ml), shape=(n, n))

print(f"MATLAB K: shape={K_ml.shape}, nnz={K_ml.nnz}")
print(f"MATLAB M: shape={M_ml.shape}, nnz={M_ml.nnz}")

# Check for very small values that might be zero in MATLAB but non-zero in Python
print("\nChecking for small values in Python matrices...")
K_py_coo = K_py.tocoo()
M_py_coo = M_py.tocoo()

# Count values that are very small (near zero)
K_py_small = np.abs(K_py_coo.data) < 1e-10
M_py_small = np.abs(M_py_coo.data) < 1e-10

print(f"Python K values < 1e-10: {np.sum(K_py_small)} out of {len(K_py_coo.data)}")
print(f"Python M values < 1e-10: {np.sum(M_py_small)} out of {len(M_py_coo.data)}")

# Check if MATLAB has zeros that Python doesn't
print("\nComparing non-zero locations...")
# Convert to sets of (row, col) tuples for comparison
K_py_locs = set(zip(K_py_coo.row, K_py_coo.col))
K_ml_coo = K_ml.tocoo()
K_ml_locs = set(zip(K_ml_coo.row, K_ml_coo.col))

only_py = K_py_locs - K_ml_locs
only_ml = K_ml_locs - K_py_locs
common = K_py_locs & K_ml_locs

print(f"K matrix:")
print(f"  Only in Python: {len(only_py)}")
print(f"  Only in MATLAB: {len(only_ml)}")
print(f"  Common: {len(common)}")

if len(only_py) > 0 and len(only_py) <= 20:
    print(f"\n  Sample locations only in Python:")
    for i, (r, c) in enumerate(list(only_py)[:10]):
        val = K_py[r, c]
        print(f"    [{r}, {c}]: {val:.6e}")

# Same for M
M_py_locs = set(zip(M_py_coo.row, M_py_coo.col))
M_ml_coo = M_ml.tocoo()
M_ml_locs = set(zip(M_ml_coo.row, M_ml_coo.col))

only_py_m = M_py_locs - M_ml_locs
only_ml_m = M_ml_locs - M_py_locs
common_m = M_py_locs & M_ml_locs

print(f"\nM matrix:")
print(f"  Only in Python: {len(only_py_m)}")
print(f"  Only in MATLAB: {len(only_ml_m)}")
print(f"  Common: {len(common_m)}")

if len(only_py_m) > 0 and len(only_py_m) <= 20:
    print(f"\n  Sample locations only in Python:")
    for i, (r, c) in enumerate(list(only_py_m)[:10]):
        val = M_py[r, c]
        print(f"    [{r}, {c}]: {val:.6e}")

# Check if the issue is in how the matrices are assembled
# Let's check the raw element matrices
print("\n" + "="*70)
print("Checking element matrix generation...")
print("="*70)

# Import element functions
from elements_vec import get_element_stiffness_VEC, get_element_mass_VEC

# Get material properties as in get_system_matrices_VEC
# Note: design is single channel, need to create 3-channel design first
# For now, just check the saved matrices - we already have the key finding
print("\nKey finding: Python includes many very small values that MATLAB treats as zero.")
print("This suggests MATLAB's sparse() removes near-zero entries or element matrices produce exact zeros.")

# The design needs to be converted to 3-channel format first, but we already have the key insight
# Let's just check if we can filter the matrices
print("\nTesting if filtering small values fixes the issue...")
# Filter very small values from Python matrices
K_py_filtered = K_py.copy()
K_py_filtered.data[np.abs(K_py_filtered.data) < 1e-10] = 0
K_py_filtered.eliminate_zeros()

M_py_filtered = M_py.copy()
M_py_filtered.data[np.abs(M_py_filtered.data) < 1e-10] = 0
M_py_filtered.eliminate_zeros()

print(f"After filtering values < 1e-10:")
print(f"  K: nnz={K_py_filtered.nnz} (MATLAB: {K_ml.nnz})")
print(f"  M: nnz={M_py_filtered.nnz} (MATLAB: {M_ml.nnz})")

# Get element matrices
AllLEle = get_element_stiffness_VEC(E.flatten(), nu.flatten(), t)
AllLMat = get_element_mass_VEC(rho.flatten(), t, const)

print(f"Element stiffness matrices shape: {AllLEle.shape}")
print(f"Element mass matrices shape: {AllLMat.shape}")
print(f"First element stiffness matrix (sample):")
print(AllLEle[0])
print(f"First element mass matrix (sample):")
print(AllLMat[0])

# Check for zeros in element matrices
K_ele_zero = np.abs(AllLEle) < 1e-10
M_ele_zero = np.abs(AllLMat) < 1e-10
print(f"\nElement matrices with values < 1e-10:")
print(f"  K: {np.sum(K_ele_zero)} out of {AllLEle.size}")
print(f"  M: {np.sum(M_ele_zero)} out of {AllLMat.size}")

