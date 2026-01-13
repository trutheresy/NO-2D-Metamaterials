"""
Analyze the diagonal differences in K matrix.
All largest differences are on diagonal with same ratio (0.89).
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys
import h5py
import torch

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC

# Configuration
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

# Load design and compute Python K
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

K_py, M_py = get_system_matrices_VEC(const)

# Load MATLAB K
with h5py.File(str(matlab_mat_file), 'r') as f:
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    K_data_ml = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir_ml = np.array(f[K_DATA_ref]['ir']).flatten().astype(int)
    K_jc_ml = np.array(f[K_DATA_ref]['jc']).flatten().astype(int)
    n = len(K_jc_ml) - 1
    K_ml = sp.csr_matrix((K_data_ml, K_ir_ml, K_jc_ml), shape=(n, n))

K_py_dense = K_py.toarray()
K_ml_dense = K_ml.toarray()

# Extract diagonal
diag_py = np.diag(K_py_dense)
diag_ml = np.diag(K_ml_dense)

# Find diagonal differences
diag_diff = diag_ml - diag_py
diag_nonzero_mask = (diag_py != 0) & (diag_ml != 0)

print("Diagonal analysis:")
print(f"  Total diagonal entries: {len(diag_py)}")
print(f"  Non-zero in both: {np.sum(diag_nonzero_mask)}")
print(f"  Max diagonal diff: {np.max(np.abs(diag_diff)):.6e}")
print(f"  Mean diagonal diff: {np.mean(np.abs(diag_diff[diag_nonzero_mask])):.6e}")

if np.any(diag_nonzero_mask):
    diag_py_nonzero = diag_py[diag_nonzero_mask]
    diag_ml_nonzero = diag_ml[diag_nonzero_mask]
    ratios = diag_ml_nonzero / diag_py_nonzero
    
    print(f"\nDiagonal ratios (MATLAB/Python):")
    print(f"  Mean ratio: {np.mean(ratios):.6e}")
    print(f"  Median ratio: {np.median(ratios):.6e}")
    print(f"  Std ratio: {np.std(ratios):.6e}")
    print(f"  Min ratio: {np.min(ratios):.6e}")
    print(f"  Max ratio: {np.max(ratios):.6e}")
    
    # Find locations with largest differences
    diag_diff_abs = np.abs(diag_diff)
    largest_diff_indices = np.argsort(diag_diff_abs)[-10:]
    
    print(f"\nTop 10 diagonal differences:")
    for idx in reversed(largest_diff_indices):
        print(f"  [{idx:4d}, {idx:4d}]: MATLAB={diag_ml[idx]:12.6e}, Python={diag_py[idx]:12.6e}, diff={diag_diff[idx]:12.6e}, ratio={diag_ml[idx]/(diag_py[idx]+1e-15):.6e}")

