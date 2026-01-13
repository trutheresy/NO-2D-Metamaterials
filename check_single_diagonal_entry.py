"""
Check how a single diagonal entry is assembled in the global K matrix.
Compare Python and MATLAB for one specific diagonal location.
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

# Pick a diagonal entry with large difference (from previous analysis)
diag_idx = 910  # One of the entries with large difference

py_val = K_py_dense[diag_idx, diag_idx]
ml_val = K_ml_dense[diag_idx, diag_idx]

print(f"Diagonal entry [{diag_idx}, {diag_idx}]:")
print(f"  Python: {py_val:.6e}")
print(f"  MATLAB: {ml_val:.6e}")
print(f"  Ratio (MATLAB/Python): {ml_val/py_val:.6f}")
print(f"  Difference: {ml_val - py_val:.6e}")

# Check which elements contribute to this diagonal entry
# For a diagonal entry at DOF i, we need to find which elements have DOF i
# and which local DOF in that element corresponds to the diagonal
print(f"\nAnalysis:")
print(f"  Expected ratio if Python is 4x too large: 0.25")
print(f"  Actual ratio: {ml_val/py_val:.6f}")
print(f"  This suggests Python's diagonal values are {py_val/ml_val:.2f}x larger than MATLAB")

