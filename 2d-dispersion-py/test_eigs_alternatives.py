"""Test different eigs parameter combinations to match MATLAB."""
import numpy as np
import sys
sys.path.insert(0, '.')

from tests.test_plotting import create_test_const
from system_matrices import get_system_matrices
from system_matrices import get_transformation_matrix
from scipy.sparse.linalg import eigs

const = create_test_const()
wv_gamma = np.array([0.0, 0.0])

K, M = get_system_matrices(const, use_vectorized=False)
T = get_transformation_matrix(wv_gamma, const)
Kr = T.conj().T @ K @ T
Mr = T.conj().T @ M @ T

print("Testing different eigs parameter combinations:")
print("=" * 80)

# Method 1: SM (current - wrong)
print("\n1. which='SM' (current):")
try:
    eig_vals, eig_vecs = eigs(Kr, M=Mr, k=5, which='SM')
    idxs = np.argsort(eig_vals)
    eig_vals_sorted = eig_vals[idxs]
    print(f"   First 3 eigenvalues: {eig_vals_sorted[:3]}")
    print(f"   First 3 frequencies: {[np.sqrt(np.maximum(0, np.real(ev))) / (2*np.pi) for ev in eig_vals_sorted[:3]]}")
    print(f"   Are they real? {np.allclose(eig_vals_sorted[:3].imag, 0, atol=1e-10)}")
except Exception as e:
    print(f"   Error: {e}")

# Method 2: sigma=0 (shift at zero - should find eigenvalues near zero)
print("\n2. sigma=0 (shift at zero):")
try:
    eig_vals, eig_vecs = eigs(Kr, M=Mr, k=5, sigma=0)
    idxs = np.argsort(eig_vals)
    eig_vals_sorted = eig_vals[idxs]
    print(f"   First 3 eigenvalues: {eig_vals_sorted[:3]}")
    print(f"   First 3 frequencies: {[np.sqrt(np.maximum(0, np.real(ev))) / (2*np.pi) for ev in eig_vals_sorted[:3]]}")
    print(f"   Are they real? {np.allclose(eig_vals_sorted[:3].imag, 0, atol=1e-10)}")
except Exception as e:
    print(f"   Error: {e}")

# Method 3: sigma=0 with which='LM' (largest magnitude near zero)
print("\n3. sigma=0, which='LM' (largest magnitude near zero):")
try:
    eig_vals, eig_vecs = eigs(Kr, M=Mr, k=5, sigma=0, which='LM')
    idxs = np.argsort(eig_vals)
    eig_vals_sorted = eig_vals[idxs]
    print(f"   First 3 eigenvalues: {eig_vals_sorted[:3]}")
    print(f"   First 3 frequencies: {[np.sqrt(np.maximum(0, np.real(ev))) / (2*np.pi) for ev in eig_vals_sorted[:3]]}")
    print(f"   Are they real? {np.allclose(eig_vals_sorted[:3].imag, 0, atol=1e-10)}")
except Exception as e:
    print(f"   Error: {e}")

# Method 4: Check what MATLAB actually does
# MATLAB: eigs(Kr, Mr, N_eig, 'SM')
# In MATLAB, 'SM' for generalized eigenvalue problem finds smallest magnitude eigenvalues
# But scipy's eigs might behave differently
print("\n4. Full eig for reference:")
Kr_dense = Kr.toarray()
Mr_dense = Mr.toarray()
eig_vals_full, _ = np.linalg.eig(np.linalg.solve(Mr_dense, Kr_dense))
idxs = np.argsort(eig_vals_full)
eig_vals_full_sorted = eig_vals_full[idxs]
print(f"   First 3 eigenvalues: {eig_vals_full_sorted[:3]}")
print(f"   First 3 frequencies: {[np.sqrt(np.maximum(0, np.real(ev))) / (2*np.pi) for ev in eig_vals_full_sorted[:3]]}")

print("\n" + "=" * 80)
print("MATLAB result: first frequency ~0.000190 Hz")
print("=" * 80)

