"""
Investigate the specific differences in K matrix values.
Look for patterns in the differences.
"""
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from pathlib import Path
import sys
import h5py

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
import torch

# Configuration
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

# Load Python design and compute K matrix
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
K_py_dense = K_py.toarray()

# Load MATLAB saved K matrix
with h5py.File(str(matlab_mat_file), 'r') as f:
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    K_data_ml = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir_ml = np.array(f[K_DATA_ref]['ir']).flatten().astype(int)
    K_jc_ml = np.array(f[K_DATA_ref]['jc']).flatten().astype(int)
    n = len(K_jc_ml) - 1
    K_ml = sp.csr_matrix((K_data_ml, K_ir_ml, K_jc_ml), shape=(n, n))
K_ml_dense = K_ml.toarray()

# Find locations with sign differences
both_nonzero = (K_ml_dense != 0) & (K_py_dense != 0)
ml_vals = K_ml_dense[both_nonzero]
py_vals = K_py_dense[both_nonzero]
ratios = ml_vals / (py_vals + 1e-15)
sign_diff = np.sign(ml_vals) != np.sign(py_vals)

# Analyze ratios
unique_ratios, counts = np.unique(np.round(ratios[sign_diff], 6), return_counts=True)
print("Most common ratios for sign differences:")
sorted_indices = np.argsort(counts)[::-1]
for i in range(min(10, len(sorted_indices))):
    idx = sorted_indices[i]
    ratio_val = unique_ratios[idx]
    count = counts[idx]
    print(f"  Ratio {ratio_val:.6e}: {count} occurrences ({100*count/np.sum(sign_diff):.1f}%)")

# Check if differences are symmetric (transpose pattern)
diff = K_ml_dense - K_py_dense
diff_T = K_ml_dense.T - K_py_dense.T
print(f"\nChecking if differences are transpose-related:")
print(f"  diff == -diff.T: {np.allclose(diff, -diff_T, atol=1e-5)}")
print(f"  diff == diff.T: {np.allclose(diff, diff_T, atol=1e-5)}")

# Check if K matrix is symmetric (should be)
print(f"\nChecking matrix symmetry:")
print(f"  MATLAB K is symmetric: {np.allclose(K_ml_dense, K_ml_dense.T, atol=1e-5)}")
print(f"  Python K is symmetric: {np.allclose(K_py_dense, K_py_dense.T, atol=1e-5)}")

