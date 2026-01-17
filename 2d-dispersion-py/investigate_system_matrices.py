"""Investigate system matrices and eigenvalue solving."""
import numpy as np
import sys
sys.path.insert(0, '.')

from tests.test_plotting import create_test_const
from system_matrices import get_system_matrices
from system_matrices import get_transformation_matrix
from scipy.sparse.linalg import eigs

const = create_test_const()
wv_gamma = np.array([0.0, 0.0])

print("=" * 80)
print("INVESTIGATING SYSTEM MATRICES AND EIGENVALUE SOLVING")
print("=" * 80)

# Get system matrices
print("\n1. SYSTEM MATRICES:")
K, M = get_system_matrices(const, use_vectorized=False)
print(f"  K shape: {K.shape}, type: {type(K)}")
print(f"  M shape: {M.shape}, type: {type(M)}")
print(f"  K density: {K.nnz / (K.shape[0] * K.shape[1]):.6f}")
print(f"  M density: {M.nnz / (M.shape[0] * M.shape[1]):.6f}")

# Get transformation matrix at k=0
print("\n2. TRANSFORMATION MATRIX AT k=0:")
T = get_transformation_matrix(wv_gamma, const)
print(f"  T shape: {T.shape}")
print(f"  T type: {type(T)}")

# Transform to reduced space
print("\n3. REDUCED SPACE MATRICES:")
Kr = T.conj().T @ K @ T
Mr = T.conj().T @ M @ T
print(f"  Kr shape: {Kr.shape}, type: {type(Kr)}")
print(f"  Mr shape: {Mr.shape}, type: {type(Mr)}")

# Check if matrices are Hermitian
Kr_herm_diff = np.max(np.abs(Kr - Kr.conj().T))
Mr_herm_diff = np.max(np.abs(Mr - Mr.conj().T))
print(f"  Kr Hermitian check (max diff): {Kr_herm_diff:.6e}")
print(f"  Mr Hermitian check (max diff): {Mr_herm_diff:.6e}")

# Solve eigenvalue problem
print("\n4. EIGENVALUE PROBLEM:")
sigma_eig = const.get('sigma_eig', 'SM')
N_eig = const['N_eig']
print(f"  sigma_eig: {sigma_eig}")
print(f"  N_eig: {N_eig}")

eigs_kwargs = {'k': N_eig}
if isinstance(sigma_eig, str):
    eigs_kwargs['which'] = sigma_eig
else:
    eigs_kwargs['sigma'] = sigma_eig

print(f"  eigs_kwargs: {eigs_kwargs}")

try:
    eig_vals, eig_vecs = eigs(Kr, M=Mr, **eigs_kwargs)
    print(f"  ✅ Eigenvalue solve successful")
    print(f"  Eigenvalue shape: {eig_vals.shape}")
    print(f"  Eigenvector shape: {eig_vecs.shape}")
    
    # Sort eigenvalues
    idxs = np.argsort(eig_vals)
    eig_vals_sorted = eig_vals[idxs]
    eig_vecs_sorted = eig_vecs[:, idxs]
    
    print(f"\n  First 5 eigenvalues (sorted):")
    for i in range(min(5, len(eig_vals_sorted))):
        print(f"    [{i}]: {eig_vals_sorted[i]:.6e} -> freq: {np.sqrt(np.maximum(0, np.real(eig_vals_sorted[i]))) / (2*np.pi):.6f} Hz")
    
    # Check first eigenvalue (should be near zero for rigid body mode)
    first_eigval = eig_vals_sorted[0]
    first_freq = np.sqrt(np.maximum(0, np.real(first_eigval))) / (2*np.pi)
    print(f"\n  First eigenvalue: {first_eigval:.6e}")
    print(f"  First frequency: {first_freq:.6f} Hz")
    print(f"  Is first eigenvalue near zero? {np.abs(first_eigval) < 1e-6}")
    
    # Normalize eigenvectors (matching MATLAB)
    norms = np.linalg.norm(eig_vecs_sorted, axis=0)
    eig_vecs_normalized = eig_vecs_sorted / norms
    phase_align = np.exp(-1j * np.angle(eig_vecs_sorted[0, :]))
    ev_normalized = (eig_vecs_normalized * phase_align)
    
    print(f"\n  First eigenvector (normalized) first 10 values:")
    print(f"    {ev_normalized[:10, 0]}")
    print(f"  First eigenvector norm: {np.linalg.norm(ev_normalized[:, 0]):.6e}")
    
except Exception as e:
    print(f"  ❌ Eigenvalue solve failed: {e}")

# Compare with saved data
print("\n5. COMPARISON WITH SAVED DATA:")
import scipy.io as sio
py_data = sio.loadmat('test_plots/plot_eigenvector_components_data.mat')
mat_data = sio.loadmat('test_plots/plot_eigenvector_components_data_matlab.mat')

print(f"  Python saved frequency: {py_data['fr'][0, 0]:.6f} Hz")
print(f"  MATLAB saved frequency: {mat_data['fr'][0, 0]:.6f} Hz")

print("\n" + "=" * 80)

