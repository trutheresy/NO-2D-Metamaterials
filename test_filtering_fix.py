"""
Test to verify that the filtering fix works correctly.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys
sys.path.insert(0, '2d-dispersion-py')

from plot_dispersion_infer_eigenfrequencies import load_pt_dataset, create_const_dict, compute_K_M_matrices
import h5py

# Load design
data = load_pt_dataset(Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1'), None, require_eigenvalue_data=False)
design = data['designs'][0]

# Create const
const = create_const_dict(
    design, N_pix=32, N_ele=1,
    a=1.0, E_min=20e6, E_max=200e9,
    rho_min=400, rho_max=8000,
    nu_min=0.0, nu_max=0.5
)

# Force recomputation by temporarily modifying the function to not load saved matrices
# Actually, let's directly call the matrix computation function
from system_matrices_vec import get_system_matrices_VEC

print("Recomputing K, M matrices with the fix...")
K, M = get_system_matrices_VEC(const)

print(f"Python K: shape={K.shape}, nnz={K.nnz}")
print(f"Python M: shape={M.shape}, nnz={M.nnz}")

# Load MATLAB matrices for comparison
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
with h5py.File(matlab_mat_file, 'r') as f:
    K_DATA_ref = f['K_DATA'][0, 0]
    M_DATA_ref = f['M_DATA'][0, 0]
    
    K_data = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir = np.array(f[K_DATA_ref]['ir']).flatten()
    K_jc = np.array(f[K_DATA_ref]['jc']).flatten()
    
    M_data = np.array(f[M_DATA_ref]['data']).flatten()
    M_ir = np.array(f[M_DATA_ref]['ir']).flatten()
    M_jc = np.array(f[M_DATA_ref]['jc']).flatten()
    
    n = len(K_jc) - 1
    K_ml = sp.csr_matrix((K_data, K_ir, K_jc), shape=(n, n))
    M_ml = sp.csr_matrix((M_data, M_ir, M_jc), shape=(n, n))

print(f"\nMATLAB K: shape={K_ml.shape}, nnz={K_ml.nnz}")
print(f"MATLAB M: shape={M_ml.shape}, nnz={M_ml.nnz}")

print(f"\nComparison:")
print(f"  K nnz match: {K.nnz == K_ml.nnz} (Python: {K.nnz}, MATLAB: {K_ml.nnz})")
print(f"  M nnz match: {M.nnz == M_ml.nnz} (Python: {M.nnz}, MATLAB: {M_ml.nnz})")

if K.nnz == K_ml.nnz and M.nnz == M_ml.nnz:
    print("\n✅ SUCCESS: nnz counts now match MATLAB!")
else:
    print("\n❌ FAIL: nnz counts still differ")

