"""
Verify that using t=1.0 (matching MATLAB) fixes the scaling issue.
"""
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import sys
import h5py

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC_simplified

# Configuration
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

print("="*70)
print("VERIFYING THICKNESS FIX")
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

if design.ndim == 2:
    N_pix = design.shape[0]
    design_3ch = np.stack([design, design, design], axis=-1)
else:
    N_pix = design.shape[0]
    design_3ch = design

# Clamp design to [0, 1] to avoid overflow
design_3ch = np.clip(design_3ch, 0, 1)

# Load Python matrices (computed with t=0.01)
K_data = torch.load(python_data_dir / 'K_data.pt', map_location='cpu')
M_data = torch.load(python_data_dir / 'M_data.pt', map_location='cpu')
K_py_old = K_data[struct_idx]
M_py_old = M_data[struct_idx]
if not sp.issparse(K_py_old):
    K_py_old = sp.csr_matrix(K_py_old)
if not sp.issparse(M_py_old):
    M_py_old = sp.csr_matrix(M_py_old)

# Load MATLAB matrices
with h5py.File(str(matlab_mat_file), 'r') as f:
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    M_DATA_ref = f['M_DATA'][struct_idx, 0]
    K_data_matlab = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir = np.array(f[K_DATA_ref]['ir']).flatten().astype(int)
    K_jc = np.array(f[K_DATA_ref]['jc']).flatten().astype(int)
    M_data_matlab = np.array(f[M_DATA_ref]['data']).flatten()
    M_ir = np.array(f[M_DATA_ref]['ir']).flatten().astype(int)
    M_jc = np.array(f[M_DATA_ref]['jc']).flatten().astype(int)
    n = len(K_jc) - 1

K_ml = sp.csr_matrix((K_data_matlab, K_ir, K_jc), shape=(n, n))
M_ml = sp.csr_matrix((M_data_matlab, M_ir, M_jc), shape=(n, n))

# Recompute Python matrices with CORRECTED parameters (matching MATLAB)
const_corrected = {
    'design': design_3ch,
    'N_pix': N_pix,
    'N_ele': 1,
    'a': 1.0,
    'E_min': 20e6,
    'E_max': 200e9,
    'rho_min': 1200,  # MATCHING MATLAB
    'rho_max': 8000,
    'poisson_min': 0.0,
    'poisson_max': 0.5,
    't': 1.0,  # MATCHING MATLAB (was 0.01)
    'design_scale': 'linear',
    'isUseImprovement': True,
    'isUseSecondImprovement': True,
}

print(f"\n1. Recomputing Python matrices with corrected parameters:")
print(f"   t = {const_corrected['t']:.6f} (was 0.01)")
print(f"   rho_min = {const_corrected['rho_min']:.2f} (was 400)")

K_py_new, M_py_new = get_system_matrices_VEC_simplified(const_corrected)

print(f"\n2. Comparing matrices:")

# Compare old vs new Python
print(f"\n   Python (old t=0.01) vs Python (new t=1.0):")
diff_K_old_new = K_py_old - K_py_new
diff_M_old_new = M_py_old - M_py_new
print(f"   K difference nnz: {diff_K_old_new.nnz}, norm: {sp.linalg.norm(diff_K_old_new):.6e}")
print(f"   M difference nnz: {diff_M_old_new.nnz}, norm: {sp.linalg.norm(diff_M_old_new):.6e}")

# Compare new Python vs MATLAB
print(f"\n   Python (new t=1.0) vs MATLAB:")
diff_K_new_ml = K_py_new - K_ml
diff_M_new_ml = M_py_new - M_ml
print(f"   K difference nnz: {diff_K_new_ml.nnz}, norm: {sp.linalg.norm(diff_K_new_ml):.6e}")
print(f"   M difference nnz: {diff_M_new_ml.nnz}, norm: {sp.linalg.norm(diff_M_new_ml):.6e}")

# Compare value scales
K_py_new_dense = K_py_new.toarray()
K_ml_dense = K_ml.toarray()
M_py_new_dense = M_py_new.toarray()
M_ml_dense = M_ml.toarray()

print(f"\n3. Value scale comparison (new Python vs MATLAB):")
print(f"   K matrix:")
print(f"     Python: min={np.min(K_py_new_dense[K_py_new_dense != 0]):.6e}, max={np.max(K_py_new_dense):.6e}, mean={np.mean(K_py_new_dense[K_py_new_dense != 0]):.6e}")
print(f"     MATLAB: min={np.min(K_ml_dense[K_ml_dense != 0]):.6e}, max={np.max(K_ml_dense):.6e}, mean={np.mean(K_ml_dense[K_ml_dense != 0]):.6e}")
ratio_K = np.mean(K_ml_dense[K_ml_dense != 0]) / np.mean(K_py_new_dense[K_py_new_dense != 0])
print(f"     Ratio (MATLAB/Python): {ratio_K:.6e}")

print(f"\n   M matrix:")
print(f"     Python: min={np.min(M_py_new_dense[M_py_new_dense != 0]):.6e}, max={np.max(M_py_new_dense):.6e}, mean={np.mean(M_py_new_dense[M_py_new_dense != 0]):.6e}")
print(f"     MATLAB: min={np.min(M_ml_dense[M_ml_dense != 0]):.6e}, max={np.max(M_ml_dense):.6e}, mean={np.mean(M_ml_dense[M_ml_dense != 0]):.6e}")
ratio_M = np.mean(M_ml_dense[M_ml_dense != 0]) / np.mean(M_py_new_dense[M_py_new_dense != 0])
print(f"     Ratio (MATLAB/Python): {ratio_M:.6e}")

if np.isclose(ratio_K, 1.0, rtol=0.1) and np.isclose(ratio_M, 1.0, rtol=0.1):
    print(f"\n✅ Scaling issue FIXED! Matrices now have similar scales.")
else:
    print(f"\n⚠️  Scaling still differs. Need further investigation.")

print("\n" + "="*70)

