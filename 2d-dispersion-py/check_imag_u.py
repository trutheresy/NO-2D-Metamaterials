"""Check why imag(u) is not constant."""
import scipy.io as sio
import numpy as np
import sys
sys.path.insert(0, '.')

from tests.test_plotting import create_test_const
from dispersion import dispersion
from get_transformation_matrix import get_transformation_matrix

# Load saved data
py_data = sio.loadmat('test_plots/plot_eigenvector_components_data.mat')
mat_data = sio.loadmat('test_plots/plot_eigenvector_components_data_matlab.mat')

print("=" * 80)
print("INVESTIGATING imag(u) NON-CONSTANT ISSUE")
print("=" * 80)

# Check imag(u) values
u_py = py_data['u']
u_mat = mat_data['u']
u_reshaped_py = py_data['u_reshaped']
u_reshaped_mat = mat_data['u_reshaped']

print("\n1. CHECKING imag(u) VALUES:")
print(f"  Python u shape: {u_py.shape}")
print(f"  MATLAB u shape: {u_mat.shape}")

imag_u_py = np.imag(u_py)
imag_u_mat = np.imag(u_mat)

print(f"\n  Python imag(u) stats:")
print(f"    Min: {np.min(imag_u_py):.6e}")
print(f"    Max: {np.max(imag_u_py):.6e}")
print(f"    Mean: {np.mean(imag_u_py):.6e}")
print(f"    Std: {np.std(imag_u_py):.6e}")
print(f"    First 10 values: {imag_u_py[:10].flatten()}")

print(f"\n  MATLAB imag(u) stats:")
print(f"    Min: {np.min(imag_u_mat):.6e}")
print(f"    Max: {np.max(imag_u_mat):.6e}")
print(f"    Mean: {np.mean(imag_u_mat):.6e}")
print(f"    Std: {np.std(imag_u_mat):.6e}")
print(f"    First 10 values: {imag_u_mat[:10].flatten()}")

# Check imag(u_reshaped)
imag_u_reshaped_py = np.imag(u_reshaped_py)
imag_u_reshaped_mat = np.imag(u_reshaped_mat)

print(f"\n  Python imag(u_reshaped) stats:")
print(f"    Min: {np.min(imag_u_reshaped_py):.6e}")
print(f"    Max: {np.max(imag_u_reshaped_py):.6e}")
print(f"    Mean: {np.mean(imag_u_reshaped_py):.6e}")
print(f"    Std: {np.std(imag_u_reshaped_py):.6e}")
print(f"    First 5x5 values:")
print(imag_u_reshaped_py[:5, :5])

print(f"\n  MATLAB imag(u_reshaped) stats:")
print(f"    Min: {np.min(imag_u_reshaped_mat):.6e}")
print(f"    Max: {np.max(imag_u_reshaped_mat):.6e}")
print(f"    Mean: {np.mean(imag_u_reshaped_mat):.6e}")
print(f"    Std: {np.std(imag_u_reshaped_mat):.6e}")
print(f"    First 5x5 values:")
print(imag_u_reshaped_mat[:5, :5])

# Check if it's just numerical noise
print(f"\n2. IS THIS NUMERICAL NOISE?")
print(f"  Max |imag(u)| / max |real(u)| ratio (Python): {np.max(np.abs(imag_u_py)) / np.max(np.abs(np.real(u_py))):.6e}")
print(f"  Max |imag(u)| / max |real(u)| ratio (MATLAB): {np.max(np.abs(imag_u_mat)) / np.max(np.abs(np.real(u_mat))):.6e}")

# Check the full eigenvector
print(f"\n3. CHECKING FULL EIGENVECTOR:")
eigenvector_full_py = py_data['eigenvector_full']
eigenvector_full_mat = mat_data['u_full']

imag_eig_full_py = np.imag(eigenvector_full_py)
imag_eig_full_mat = np.imag(eigenvector_full_mat)

print(f"  Python imag(eigenvector_full) stats:")
print(f"    Min: {np.min(imag_eig_full_py):.6e}")
print(f"    Max: {np.max(imag_eig_full_py):.6e}")
print(f"    Mean: {np.mean(imag_eig_full_py):.6e}")
print(f"    Std: {np.std(imag_eig_full_py):.6e}")

print(f"\n  MATLAB imag(u_full) stats:")
print(f"    Min: {np.min(imag_eig_full_mat):.6e}")
print(f"    Max: {np.max(imag_eig_full_mat):.6e}")
print(f"    Mean: {np.mean(imag_eig_full_mat):.6e}")
print(f"    Std: {np.std(imag_eig_full_mat):.6e}")

# Recompute to see if we get the same
print(f"\n4. RECOMPUTING TO CHECK:")
const = create_test_const()
wv_gamma = np.array([[0.0, 0.0]])
wv, fr, ev, mesh = dispersion(const, wv_gamma)

k_idx = 0
eig_idx = 0
eigenvector_reduced = ev[:, k_idx, eig_idx]

wv_vec = wv_gamma[k_idx, :]
T = get_transformation_matrix(wv_vec, const)
eigenvector_full_recomp = T @ eigenvector_reduced

u_recomp = eigenvector_full_recomp[0::2]
imag_u_recomp = np.imag(u_recomp)

print(f"  Recomputed imag(u) stats:")
print(f"    Min: {np.min(imag_u_recomp):.6e}")
print(f"    Max: {np.max(imag_u_recomp):.6e}")
print(f"    Mean: {np.mean(imag_u_recomp):.6e}")
print(f"    Std: {np.std(imag_u_recomp):.6e}")

print("\n" + "=" * 80)

