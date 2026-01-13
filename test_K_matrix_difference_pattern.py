"""
Analyze the pattern of differences between Python and MATLAB K matrices.
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
print("ANALYZING K MATRIX DIFFERENCE PATTERN")
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

# Set up const - MATCHING MATLAB
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

# Load MATLAB K matrix
with h5py.File(str(matlab_mat_file), 'r') as f:
    K_DATA_ref = f['K_DATA'][struct_idx, 0]
    K_data_matlab = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir = np.array(f[K_DATA_ref]['ir']).flatten().astype(int)
    K_jc = np.array(f[K_DATA_ref]['jc']).flatten().astype(int)
    n = len(K_jc) - 1

K_ml = sp.csr_matrix((K_data_matlab, K_ir, K_jc), shape=(n, n))

# Convert to dense for comparison
K_py_dense = K_py.toarray()
K_ml_dense = K_ml.toarray()

# Compute difference
diff_K = K_py_dense - K_ml_dense

print(f"\nK matrix shapes: Python {K_py.shape}, MATLAB {K_ml.shape}")
print(f"K matrix nnz: Python {K_py.nnz}, MATLAB {K_ml.nnz}")

# Analyze difference
diff_K_nonzero = diff_K[diff_K != 0]
if len(diff_K_nonzero) > 0:
    print(f"\nDifference statistics (where non-zero):")
    print(f"  Count of non-zero differences: {len(diff_K_nonzero)}")
    print(f"  Max absolute difference: {np.max(np.abs(diff_K_nonzero)):.6e}")
    print(f"  Mean absolute difference: {np.mean(np.abs(diff_K_nonzero)):.6e}")
    print(f"  Min difference: {np.min(diff_K_nonzero):.6e}")
    print(f"  Max difference: {np.max(diff_K_nonzero):.6e}")
    print(f"  Mean difference: {np.mean(diff_K_nonzero):.6e}")
    
    # Check if differences are systematic (constant ratio)
    # Compare where both are non-zero
    both_nonzero = (K_py_dense != 0) & (K_ml_dense != 0)
    if np.any(both_nonzero):
        py_vals = K_py_dense[both_nonzero]
        ml_vals = K_ml_dense[both_nonzero]
        ratios = ml_vals / (py_vals + 1e-15)
        
        print(f"\nRatio analysis (where both non-zero, {np.sum(both_nonzero)} locations):")
        print(f"  Mean ratio (MATLAB/Python): {np.mean(ratios):.6e}")
        print(f"  Median ratio: {np.median(ratios):.6e}")
        print(f"  Std of ratios: {np.std(ratios):.6e}")
        print(f"  Min ratio: {np.min(ratios):.6e}")
        print(f"  Max ratio: {np.max(ratios):.6e}")
        
        if np.std(ratios) / np.abs(np.mean(ratios)) < 0.01:  # Less than 1% variation
            print(f"  ✅ Ratio is approximately constant - suggests a scaling factor")
        else:
            print(f"  ⚠️  Ratio varies significantly")
    
    # Check if there's a pattern based on matrix location
    # Sample some differences to see pattern
    diff_abs = np.abs(diff_K)
    max_diff_locations = np.unravel_index(np.argsort(diff_abs.flatten())[-20:], diff_abs.shape)
    
    print(f"\nTop 20 locations with largest absolute differences:")
    for i in range(min(20, len(max_diff_locations[0]))):
        row, col = max_diff_locations[0][i], max_diff_locations[1][i]
        py_val = K_py_dense[row, col]
        ml_val = K_ml_dense[row, col]
        diff_val = diff_K[row, col]
        print(f"  [{row:4d}, {col:4d}]: Python={py_val:12.6e}, MATLAB={ml_val:12.6e}, diff={diff_val:12.6e}")
    
    # Check if differences are in specific regions (diagonal, off-diagonal, etc.)
    diagonal_mask = np.eye(K_py.shape[0], dtype=bool)
    off_diagonal_mask = ~diagonal_mask
    
    diff_diag = diff_K[diagonal_mask]
    diff_offdiag = diff_K[off_diagonal_mask]
    
    print(f"\nDifference by region:")
    print(f"  Diagonal: max={np.max(np.abs(diff_diag)):.6e}, mean={np.mean(np.abs(diff_diag)):.6e}")
    print(f"  Off-diagonal: max={np.max(np.abs(diff_offdiag)):.6e}, mean={np.mean(np.abs(diff_offdiag)):.6e}")
    
else:
    print("\n✅ No differences found - matrices match exactly!")

print("\n" + "="*70)

