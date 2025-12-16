"""Check matrix properties and eigenvalue solver behavior."""
import numpy as np
import sys
sys.path.insert(0, '.')

from tests.test_plotting import create_test_const
from system_matrices import get_system_matrices
from get_transformation_matrix import get_transformation_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eig

const = create_test_const()
wv_gamma = np.array([0.0, 0.0])

print("=" * 80)
print("CHECKING MATRIX PROPERTIES")
print("=" * 80)

# Get matrices
K, M = get_system_matrices(const, use_vectorized=False)
T = get_transformation_matrix(wv_gamma, const)
Kr = T.conj().T @ K @ T
Mr = T.conj().T @ M @ T

print("\n1. MATRIX PROPERTIES:")
print(f"  Kr shape: {Kr.shape}")
print(f"  Mr shape: {Mr.shape}")

# Check Hermitian
Kr_herm = Kr - Kr.conj().T
Mr_herm = Mr - Mr.conj().T
print(f"\n  Kr Hermitian check:")
print(f"    Max |Kr - Kr*|: {np.max(np.abs(Kr_herm)):.6e}")
print(f"    Mean |Kr - Kr*|: {np.mean(np.abs(Kr_herm)):.6e}")

print(f"\n  Mr Hermitian check:")
print(f"    Max |Mr - Mr*|: {np.max(np.abs(Mr_herm)):.6e}")
print(f"    Mean |Mr - Mr*|: {np.mean(np.abs(Mr_herm)):.6e}")

# Check if matrices are real (they should be at k=0)
Kr_imag_max = np.max(np.abs(Kr.imag)) if hasattr(Kr, 'imag') else np.max(np.abs((Kr - Kr.real).data)) if hasattr(Kr, 'data') else 0
Mr_imag_max = np.max(np.abs(Mr.imag)) if hasattr(Mr, 'imag') else np.max(np.abs((Mr - Mr.real).data)) if hasattr(Mr, 'data') else 0
print(f"\n  Kr is real? {Kr_imag_max < 1e-10}")
print(f"  Mr is real? {Mr_imag_max < 1e-10}")
print(f"    Max |imag(Kr)|: {Kr_imag_max:.6e}")
print(f"    Max |imag(Mr)|: {Mr_imag_max:.6e}")

# Check positive definiteness of M
print(f"\n  Mr eigenvalues (first 5):")
Mr_eigvals = np.linalg.eigvalsh(Mr.toarray() if hasattr(Mr, 'toarray') else Mr)
Mr_eigvals_sorted = np.sort(Mr_eigvals)
print(f"    {Mr_eigvals_sorted[:5]}")
print(f"    All positive? {np.all(Mr_eigvals_sorted > 0)}")
print(f"    Min eigenvalue: {np.min(Mr_eigvals_sorted):.6e}")

# Try different eigenvalue solvers
print("\n2. EIGENVALUE SOLVER COMPARISON:")

# Method 1: eigs with 'SM' (smallest magnitude)
print("\n  Method 1: eigs with 'SM' (smallest magnitude)")
try:
    eig_vals_sm, eig_vecs_sm = eigs(Kr, M=Mr, k=5, which='SM')
    idxs = np.argsort(eig_vals_sm)
    eig_vals_sm_sorted = eig_vals_sm[idxs]
    print(f"    First 3 eigenvalues: {eig_vals_sm_sorted[:3]}")
    print(f"    First 3 frequencies: {[np.sqrt(np.maximum(0, np.real(ev))) / (2*np.pi) for ev in eig_vals_sm_sorted[:3]]}")
except Exception as e:
    print(f"    Error: {e}")

# Method 2: eigs with 'SA' (smallest algebraic)
print("\n  Method 2: eigs with 'SA' (smallest algebraic)")
try:
    eig_vals_sa, eig_vecs_sa = eigs(Kr, M=Mr, k=5, which='SA')
    idxs = np.argsort(eig_vals_sa)
    eig_vals_sa_sorted = eig_vals_sa[idxs]
    print(f"    First 3 eigenvalues: {eig_vals_sa_sorted[:3]}")
    print(f"    First 3 frequencies: {[np.sqrt(np.maximum(0, np.real(ev))) / (2*np.pi) for ev in eig_vals_sa_sorted[:3]]}")
except Exception as e:
    print(f"    Error: {e}")

# Method 3: Full eigenvalue decomposition (for small matrices)
print("\n  Method 3: Full eig (dense)")
if Kr.shape[0] <= 1000:
    try:
        Kr_dense = Kr.toarray() if hasattr(Kr, 'toarray') else Kr
        Mr_dense = Mr.toarray() if hasattr(Mr, 'toarray') else Mr
        eig_vals_full, eig_vecs_full = eig(Kr_dense, Mr_dense)
        idxs = np.argsort(eig_vals_full)
        eig_vals_full_sorted = eig_vals_full[idxs]
        print(f"    First 3 eigenvalues: {eig_vals_full_sorted[:3]}")
        print(f"    First 3 frequencies: {[np.sqrt(np.maximum(0, np.real(ev))) / (2*np.pi) for ev in eig_vals_full_sorted[:3]]}")
        print(f"    First eigenvalue is real? {np.abs(eig_vals_full_sorted[0].imag) < 1e-10}")
    except Exception as e:
        print(f"    Error: {e}")
else:
    print(f"    Skipped (matrix too large: {Kr.shape[0]}x{Kr.shape[0]})")

# Check if there's a zero eigenvalue
print("\n3. CHECKING FOR ZERO EIGENVALUE (rigid body mode):")
if Kr.shape[0] <= 1000:
    Kr_dense = Kr.toarray() if hasattr(Kr, 'toarray') else Kr
    Mr_dense = Mr.toarray() if hasattr(Mr, 'toarray') else Mr
    eig_vals_full, _ = eig(Kr_dense, Mr_dense)
    eig_vals_full_sorted = np.sort(eig_vals_full)
    near_zero = np.abs(eig_vals_full_sorted) < 1e-6
    print(f"    Eigenvalues near zero (< 1e-6): {np.sum(near_zero)}")
    if np.any(near_zero):
        print(f"    Near-zero eigenvalues: {eig_vals_full_sorted[near_zero][:5]}")
    else:
        print(f"    Smallest eigenvalue: {eig_vals_full_sorted[0]:.6e}")
        print(f"    Smallest frequency: {np.sqrt(np.maximum(0, np.real(eig_vals_full_sorted[0]))) / (2*np.pi):.6f} Hz")

print("\n" + "=" * 80)

