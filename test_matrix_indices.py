"""
Test script to check matrix indices and compare with MATLAB.
This will help identify why nnz differs between Python and MATLAB.
"""

import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import h5py
import sys
sys.path.insert(0, '2d-dispersion-py')

from plot_dispersion_infer_eigenfrequencies import create_const_dict, compute_K_M_matrices

# Load data
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

print("="*70)
print("Testing Matrix Index Generation")
print("="*70)

# Load Python matrices
print("\n1. Loading Python matrices...")
K_py_data = torch.load(python_data_dir / 'K_data.pt', map_location='cpu')
M_py_data = torch.load(python_data_dir / 'M_data.pt', map_location='cpu')
K_py = sp.csr_matrix(K_py_data[struct_idx])
M_py = sp.csr_matrix(M_py_data[struct_idx])

print(f"   Python K: shape={K_py.shape}, nnz={K_py.nnz}, max_row={K_py.indices.max() if K_py.nnz > 0 else 0}, max_col={K_py.indptr.max() if K_py.nnz > 0 else 0}")
print(f"   Python M: shape={M_py.shape}, nnz={M_py.nnz}")

# Get actual max indices from data
K_py_coo = K_py.tocoo()
M_py_coo = M_py.tocoo()
print(f"   Python K max row idx: {K_py_coo.row.max()}, max col idx: {K_py_coo.col.max()}")
print(f"   Python M max row idx: {M_py_coo.row.max()}, max col idx: {M_py_coo.col.max()}")

# Load MATLAB matrices
print("\n2. Loading MATLAB matrices...")
with h5py.File(matlab_mat_file, 'r') as f:
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    M_DATA_ref = f['M_DATA'][struct_idx, 0]
    
    K_data = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir = np.array(f[K_DATA_ref]['ir']).flatten()  # Row indices (0-based in HDF5)
    K_jc = np.array(f[K_DATA_ref]['jc']).flatten()  # Column pointers
    
    M_data = np.array(f[M_DATA_ref]['data']).flatten()
    M_ir = np.array(f[M_DATA_ref]['ir']).flatten()
    M_jc = np.array(f[M_DATA_ref]['jc']).flatten()
    
    n = len(K_jc) - 1
    
    K_ml = sp.csr_matrix((K_data, K_ir, K_jc), shape=(n, n))
    M_ml = sp.csr_matrix((M_data, M_ir, M_jc), shape=(n, n))

print(f"   MATLAB K: shape={K_ml.shape}, nnz={K_ml.nnz}")
print(f"   MATLAB M: shape={M_ml.shape}, nnz={M_ml.nnz}")

K_ml_coo = K_ml.tocoo()
M_ml_coo = M_ml.tocoo()
print(f"   MATLAB K max row idx: {K_ml_coo.row.max()}, max col idx: {K_ml_coo.col.max()}")
print(f"   MATLAB M max row idx: {M_ml_coo.row.max()}, max col idx: {M_ml_coo.col.max()}")

# Check if patterns are similar
print("\n3. Comparing sparsity patterns...")
# Sample some entries to see if they're in the same locations
print(f"   Checking first 10 non-zero entries in K:")
print(f"   Python K first 10: rows={K_py_coo.row[:10]}, cols={K_py_coo.col[:10]}")
print(f"   MATLAB K first 10: rows={K_ml_coo.row[:10]}, cols={K_ml_coo.col[:10]}")

# Now check how matrices are computed
print("\n4. Recomputing Python matrices to check index generation...")
# Load the design using the same method as plot_dispersion_infer_eigenfrequencies
from plot_dispersion_infer_eigenfrequencies import load_pt_dataset
data = load_pt_dataset(python_data_dir, None, require_eigenvalue_data=False)
design = data['designs'][struct_idx]

print(f"   Design shape: {design.shape}")

# Create const dict
const = create_const_dict(
    design, 
    N_pix=32, 
    N_ele=1,
    a=1.0,
    E_min=20e6, 
    E_max=200e9,
    rho_min=400, 
    rho_max=8000,
    nu_min=0.0, 
    nu_max=0.5
)

# Recompute matrices
print("   Computing K, M matrices...")
K_test, M_test = compute_K_M_matrices(const)

print(f"   Recomputed K: shape={K_test.shape}, nnz={K_test.nnz}")
print(f"   Recomputed M: shape={M_test.shape}, nnz={M_test.nnz}")

K_test_coo = K_test.tocoo()
M_test_coo = M_test.tocoo()
print(f"   Recomputed K max row idx: {K_test_coo.row.max()}, max col idx: {K_test_coo.col.max()}")
print(f"   Recomputed M max row idx: {M_test_coo.row.max()}, max col idx: {M_test_coo.col.max()}")

# Compare with saved matrices
print("\n5. Comparing saved vs recomputed Python matrices...")
print(f"   K shapes match: {K_py.shape == K_test.shape}")
print(f"   K nnz match: {K_py.nnz == K_test.nnz}")
print(f"   M shapes match: {M_py.shape == M_test.shape}")
print(f"   M nnz match: {M_py.nnz == M_test.nnz}")

