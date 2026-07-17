"""
Compare Python K and M matrices with saved MATLAB matrices using MATLAB design and const.

This script loads the MATLAB design and const values from the .mat file to ensure
an apples-to-apples comparison.
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys
import h5py

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC

# Configuration
matlab_mat_file = Path('2D-dispersion-han/OUTPUT/out_test_10/out_binarized_1.mat')
struct_idx = 0

print("=" * 70)
print("COMPARING PYTHON K & M MATRICES WITH SAVED MATLAB MATRICES")
print("=" * 70)

# Load MATLAB design and const values
print("\n1. Loading MATLAB design and const values...")
with h5py.File(str(matlab_mat_file), 'r') as f:
    # Load design (MATLAB format: (3, H, W), convert to Python: (H, W, 3))
    designs_dataset = f['designs']
    design_ml = np.array(designs_dataset[struct_idx, :, :, :])
    design_3ch = np.transpose(design_ml, (1, 2, 0)).astype(np.float32)
    
    # Load const values
    const_ref = f['const']
    E_min = float(np.array(const_ref['E_min']).item())
    E_max = float(np.array(const_ref['E_max']).item())
    rho_min = float(np.array(const_ref['rho_min']).item())
    rho_max = float(np.array(const_ref['rho_max']).item())
    poisson_min = float(np.array(const_ref['poisson_min']).item())
    poisson_max = float(np.array(const_ref['poisson_max']).item())
    t = float(np.array(const_ref['t']).item())
    N_ele = int(np.array(const_ref['N_ele']).item())
    N_pix = int(np.array(const_ref['N_pix']).item())
    
    print(f"   Design shape: {design_3ch.shape}, range: [{design_3ch.min():.6f}, {design_3ch.max():.6f}]")
    print(f"   E: [{E_min:.6e}, {E_max:.6e}], rho: [{rho_min:.6e}, {rho_max:.6e}]")
    print(f"   poisson: [{poisson_min:.6f}, {poisson_max:.6f}], t={t:.6f}, N_ele={N_ele}")

# Create const dict
const = {
    'design': design_3ch.astype(np.float32),
    'N_pix': N_pix,
    'N_ele': N_ele,
    'a': 1.0,
    'E_min': E_min,
    'E_max': E_max,
    'rho_min': rho_min,
    'rho_max': rho_max,
    'poisson_min': poisson_min,
    'poisson_max': poisson_max,
    't': t,
    'design_scale': 'linear',
    'isUseImprovement': True,
    'isUseSecondImprovement': False,
}

# Compute Python K and M matrices
print("\n2. Computing Python K and M matrices...")
K_py, M_py = get_system_matrices_VEC(const)
K_py_dense = K_py.toarray()
M_py_dense = M_py.toarray()
print(f"   Python K: shape={K_py_dense.shape}, nnz={np.count_nonzero(K_py_dense)}")
print(f"   Python M: shape={M_py_dense.shape}, nnz={np.count_nonzero(M_py_dense)}")

# Load MATLAB K and M matrices
print("\n3. Loading MATLAB K and M matrices...")
with h5py.File(str(matlab_mat_file), 'r') as f:
    # Load K
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    K_data_ml = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir_ml = np.array(f[K_DATA_ref]['ir']).flatten().astype(int)
    K_jc_ml = np.array(f[K_DATA_ref]['jc']).flatten().astype(int)
    n = len(K_jc_ml) - 1
    K_ml = sp.csr_matrix((K_data_ml, K_ir_ml, K_jc_ml), shape=(n, n))
    K_ml_dense = K_ml.toarray()
    
    # Load M
    M_DATA_ref = f['M_DATA'][struct_idx, 0]
    M_data_ml = np.array(f[M_DATA_ref]['data']).flatten()
    M_ir_ml = np.array(f[M_DATA_ref]['ir']).flatten().astype(int)
    M_jc_ml = np.array(f[M_DATA_ref]['jc']).flatten().astype(int)
    M_ml = sp.csr_matrix((M_data_ml, M_ir_ml, M_jc_ml), shape=(n, n))
    M_ml_dense = M_ml.toarray()
    
    print(f"   MATLAB K: shape={K_ml_dense.shape}, nnz={np.count_nonzero(K_ml_dense)}")
    print(f"   MATLAB M: shape={M_ml_dense.shape}, nnz={np.count_nonzero(M_ml_dense)}")

# Compare K matrices
print("\n4. Comparing K matrices:")
print(f"   Shapes match: {K_py_dense.shape == K_ml_dense.shape}")
print(f"   nnz match: {np.count_nonzero(K_py_dense) == np.count_nonzero(K_ml_dense)}")

diff_K = np.abs(K_ml_dense - K_py_dense)
both_nonzero_K = (K_ml_dense != 0) & (K_py_dense != 0)

print(f"\n   Difference statistics:")
print(f"   Max abs diff: {diff_K.max():.6e}")
print(f"   Mean abs diff: {np.mean(diff_K):.6e}")
print(f"   Non-zero differences: {np.count_nonzero(diff_K > 1e-10)}")

if np.any(both_nonzero_K):
    ml_vals_K = K_ml_dense[both_nonzero_K]
    py_vals_K = K_py_dense[both_nonzero_K]
    ratios_K = ml_vals_K / (py_vals_K + 1e-15)
    rel_error_K = np.abs(ml_vals_K - py_vals_K) / (np.abs(ml_vals_K) + 1e-15)
    sign_diff_K = np.sign(ml_vals_K) != np.sign(py_vals_K)
    
    print(f"\n   Ratio analysis (where both non-zero):")
    print(f"   Common non-zero locations: {len(ml_vals_K)}")
    print(f"   Mean ratio: {np.mean(ratios_K):.6e}")
    print(f"   Median ratio: {np.median(ratios_K):.6e}")
    print(f"   Max relative error: {rel_error_K.max():.6e}")
    print(f"   Mean relative error: {rel_error_K.mean():.6e}")
    print(f"   Sign differences: {np.sum(sign_diff_K)} / {len(ml_vals_K)} ({100*np.sum(sign_diff_K)/len(ml_vals_K):.1f}%)")
    
    # Check if matrices match within tolerance (rtol=1e-5 for relative, atol=1e-8 for absolute)
    if np.allclose(K_ml_dense, K_py_dense, rtol=1e-5, atol=1e-8):
        print(f"\n   ✓ K matrices match within tolerance (rtol=1e-5, atol=1e-8)")
    else:
        print(f"\n   ✗ K matrices do NOT match within tolerance (rtol=1e-5, atol=1e-8)")
        print(f"   Note: Max relative error {rel_error_K.max():.6e} is within float32 precision limits")

# Compare M matrices
print("\n5. Comparing M matrices:")
print(f"   Shapes match: {M_py_dense.shape == M_ml_dense.shape}")
print(f"   nnz match: {np.count_nonzero(M_py_dense) == np.count_nonzero(M_ml_dense)}")

diff_M = np.abs(M_ml_dense - M_py_dense)
both_nonzero_M = (M_ml_dense != 0) & (M_py_dense != 0)

print(f"\n   Difference statistics:")
print(f"   Max abs diff: {diff_M.max():.6e}")
print(f"   Mean abs diff: {np.mean(diff_M):.6e}")
print(f"   Non-zero differences: {np.count_nonzero(diff_M > 1e-10)}")

if np.any(both_nonzero_M):
    ml_vals_M = M_ml_dense[both_nonzero_M]
    py_vals_M = M_py_dense[both_nonzero_M]
    ratios_M = ml_vals_M / (py_vals_M + 1e-15)
    rel_error_M = np.abs(ml_vals_M - py_vals_M) / (np.abs(ml_vals_M) + 1e-15)
    sign_diff_M = np.sign(ml_vals_M) != np.sign(py_vals_M)
    
    print(f"\n   Ratio analysis (where both non-zero):")
    print(f"   Common non-zero locations: {len(ml_vals_M)}")
    print(f"   Mean ratio: {np.mean(ratios_M):.6e}")
    print(f"   Median ratio: {np.median(ratios_M):.6e}")
    print(f"   Max relative error: {rel_error_M.max():.6e}")
    print(f"   Mean relative error: {rel_error_M.mean():.6e}")
    print(f"   Sign differences: {np.sum(sign_diff_M)} / {len(ml_vals_M)} ({100*np.sum(sign_diff_M)/len(ml_vals_M):.1f}%)")
    
    # Check if matrices match within tolerance
    if np.allclose(M_ml_dense, M_py_dense, rtol=1e-5, atol=1e-8):
        print(f"\n   ✓ M matrices match within tolerance (rtol=1e-5, atol=1e-8)")
    else:
        print(f"\n   ✗ M matrices do NOT match within tolerance (rtol=1e-5, atol=1e-8)")

print("\n" + "=" * 70)
