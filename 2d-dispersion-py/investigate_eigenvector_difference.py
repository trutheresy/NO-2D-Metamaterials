"""Investigate why eigenvector values differ between Python and MATLAB."""
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
print("INVESTIGATING EIGENVECTOR DATA DIFFERENCES")
print("=" * 80)

# Compare reduced space eigenvectors
print("\n1. REDUCED SPACE EIGENVECTOR COMPARISON:")
ev_red_py = py_data['eigenvector_reduced']
ev_red_mat = mat_data['u_reduced']
print(f"  Python shape: {ev_red_py.shape}")
print(f"  MATLAB shape: {ev_red_mat.shape}")
if ev_red_py.shape == ev_red_mat.shape:
    diff_red = np.abs(ev_red_py.flatten() - ev_red_mat.flatten())
    print(f"  Max difference: {np.max(diff_red):.6e}")
    print(f"  Mean difference: {np.mean(diff_red):.6e}")
    print(f"  Relative error: {np.max(diff_red) / np.max(np.abs(ev_red_mat)):.6e}")
    print(f"  First 10 values Python: {ev_red_py[:10]}")
    print(f"  First 10 values MATLAB: {ev_red_mat[:10].flatten()}")

# Compare transformation matrices
print("\n2. TRANSFORMATION MATRIX COMPARISON:")
const = create_test_const()
wv_gamma = np.array([0.0, 0.0])
T_py = get_transformation_matrix(wv_gamma, const)
print(f"  Python T shape: {T_py.shape}")
print(f"  Python T type: {type(T_py)}")
print(f"  Python T first 5x5:")
if hasattr(T_py, 'toarray'):
    print(T_py.toarray()[:5, :5])
else:
    print(T_py[:5, :5])

# Recompute full space eigenvector in Python
print("\n3. RECOMPUTING FULL SPACE EIGENVECTOR:")
u_full_py_recomp = T_py @ ev_red_py
print(f"  Recomputed u_full shape: {u_full_py_recomp.shape}")
print(f"  Saved u_full shape: {py_data['eigenvector_full'].shape}")
print(f"  Max difference (recomputed vs saved): {np.max(np.abs(u_full_py_recomp.flatten() - py_data['eigenvector_full'].flatten())):.6e}")

# Compare full space eigenvectors (before any normalization)
print("\n4. FULL SPACE EIGENVECTOR COMPARISON (before normalization):")
u_full_py = py_data['eigenvector_full']
u_full_mat = mat_data['u_full']
print(f"  Python shape: {u_full_py.shape}")
print(f"  MATLAB shape: {u_full_mat.shape}")
if u_full_py.shape == u_full_mat.shape:
    diff_full = np.abs(u_full_py.flatten() - u_full_mat.flatten())
    print(f"  Max difference: {np.max(diff_full):.6e}")
    print(f"  Mean difference: {np.mean(diff_full):.6e}")
    print(f"  Relative error: {np.max(diff_full) / np.max(np.abs(u_full_mat)):.6e}")
    print(f"  Python max(abs): {np.max(np.abs(u_full_py)):.6e}")
    print(f"  MATLAB max(abs): {np.max(np.abs(u_full_mat)):.6e}")

# Check normalization
print("\n5. NORMALIZATION CHECK:")
# MATLAB: u = u/max(abs(u))*(1/10)*const.a
# Python should do the same
scale_factor = (1/10) * const['a']
u_py_normalized = u_full_py / np.max(np.abs(u_full_py)) * scale_factor
u_mat_normalized = u_full_mat / np.max(np.abs(u_full_mat)) * scale_factor
print(f"  Scale factor: {scale_factor}")
print(f"  Python max(abs) after normalization: {np.max(np.abs(u_py_normalized)):.6e}")
print(f"  MATLAB max(abs) after normalization: {np.max(np.abs(u_mat_normalized)):.6e}")

# Compare u and v components
print("\n6. U AND V COMPONENTS COMPARISON:")
u_py = py_data['u']
u_mat = mat_data['u']
print(f"  Python u shape: {u_py.shape}")
print(f"  MATLAB u shape: {u_mat.shape}")
if u_py.shape == u_mat.shape:
    diff_u = np.abs(u_py.flatten() - u_mat.flatten())
    print(f"  u max difference: {np.max(diff_u):.6e}")
    print(f"  u mean difference: {np.mean(diff_u):.6e}")
    print(f"  Python u first 10: {u_py[:10]}")
    print(f"  MATLAB u first 10: {u_mat[:10].flatten()}")

# Check if the issue is in the dispersion calculation itself
print("\n7. CHECKING DISPERSION CALCULATION:")
const_test = create_test_const()
wv_test = np.array([[0.0, 0.0]])
wv, fr, ev, mesh = dispersion(const_test, wv_test)
print(f"  Python frequency [0,0]: {fr[0, 0]:.6f} Hz")
print(f"  MATLAB frequency [0,0]: {mat_data['fr'][0, 0]:.6f} Hz")
print(f"  Frequency difference: {np.abs(fr[0, 0] - mat_data['fr'][0, 0]):.6e}")

# Check eigenvector normalization in dispersion
print("\n8. EIGENVECTOR NORMALIZATION IN DISPERSION:")
ev_test = ev[:, 0, 0]  # First wavevector, first mode
ev_norm_py = np.linalg.norm(ev_test)
print(f"  Python eigenvector norm: {ev_norm_py:.6e}")
ev_norm_mat = np.linalg.norm(ev_red_mat.flatten())
print(f"  MATLAB eigenvector norm: {ev_norm_mat:.6e}")
print(f"  Norm difference: {np.abs(ev_norm_py - ev_norm_mat):.6e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Check the differences above to identify where the discrepancy originates.")

