"""
Create a small test case (2x2 elements) and compare Python vs MATLAB expected behavior.
Use actual values from saved matrices to understand the pattern.
"""
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import sys
import h5py

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC_simplified

# Use actual saved data but check specific entries
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

print("="*70)
print("COMPARING SPECIFIC MATRIX ENTRIES")
print("="*70)

# Load design
if (python_data_dir / 'geometries_full.pt').exists():
    geometries = torch.load(python_data_dir / 'geometries_full.pt', map_location='cpu')
    if isinstance(geometries, torch.Tensor):
        geometries = geometries.numpy()
    design = geometries[struct_idx].astype(np.float32)
else:
    print("Error: Could not find design file")
    sys.exit(1)

if design.ndim == 2:
    N_pix = design.shape[0]
    design_3ch = np.stack([design, design, design], axis=-1)
else:
    N_pix = design.shape[0]
    design_3ch = design

# Set up const
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

# Compute Python K matrix
K_py, _ = get_system_matrices_VEC_simplified(const)
K_py_dense = K_py.toarray()

# Load MATLAB K matrix
with h5py.File(str(matlab_mat_file), 'r') as f:
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    K_data_matlab = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir = np.array(f[K_DATA_ref]['ir']).flatten().astype(int)
    K_jc = np.array(f[K_DATA_ref]['jc']).flatten().astype(int)
    n = len(K_jc) - 1

K_ml = sp.csr_matrix((K_data_matlab, K_ir, K_jc), shape=(n, n))
K_ml_dense = K_ml.toarray()

# Find locations where both have non-zero values
both_nonzero = (K_py_dense != 0) & (K_ml_dense != 0)
py_vals = K_py_dense[both_nonzero]
ml_vals = K_ml_dense[both_nonzero]
locations = np.where(both_nonzero)

# Sample some entries to see the pattern
n_sample = min(20, len(py_vals))
sample_indices = np.random.choice(len(py_vals), n_sample, replace=False)

print(f"\nSampling {n_sample} entries where both matrices are non-zero:")
print(f"{'Row':>6} {'Col':>6} {'Python Value':>15} {'MATLAB Value':>15} {'Ratio (ML/PY)':>15} {'Diff':>15}")
print("-" * 80)

for idx in sample_indices:
    row, col = locations[0][idx], locations[1][idx]
    py_val = py_vals[idx]
    ml_val = ml_vals[idx]
    ratio = ml_val / py_val if abs(py_val) > 1e-15 else np.nan
    diff = ml_val - py_val
    print(f"{row:6d} {col:6d} {py_val:15.6e} {ml_val:15.6e} {ratio:15.6e} {diff:15.6e}")

# Check diagonal entries
print(f"\nDiagonal entries (first 10):")
print(f"{'Index':>6} {'Python':>15} {'MATLAB':>15} {'Ratio':>15} {'Diff':>15}")
print("-" * 70)
for i in range(min(10, K_py.shape[0])):
    py_val = K_py_dense[i, i]
    ml_val = K_ml_dense[i, i]
    if abs(py_val) > 1e-10 or abs(ml_val) > 1e-10:
        ratio = ml_val / py_val if abs(py_val) > 1e-15 else np.nan
        diff = ml_val - py_val
        print(f"{i:6d} {py_val:15.6e} {ml_val:15.6e} {ratio:15.6e} {diff:15.6e}")

# Check if there's a consistent pattern in the ratios
ratios = ml_vals / (py_vals + 1e-15)
ratios_abs = np.abs(ratios)
print(f"\nRatio statistics (MATLAB/Python):")
print(f"  Mean ratio: {np.mean(ratios):.6e}")
print(f"  Median ratio: {np.median(ratios):.6e}")
print(f"  Std ratio: {np.std(ratios):.6e}")
print(f"  Min ratio: {np.min(ratios):.6e}")
print(f"  Max ratio: {np.max(ratios):.6e}")

# Check for sign differences
sign_different = np.sign(py_vals) != np.sign(ml_vals)
print(f"\nSign differences:")
print(f"  Count with different signs: {np.sum(sign_different)} / {len(py_vals)}")
if np.any(sign_different):
    print(f"  This suggests assembly order or matrix indexing issue")

print("\n" + "="*70)

