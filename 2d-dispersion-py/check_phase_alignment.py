"""Check if phase alignment introduces the constant imag(v) component."""
import sys
sys.path.insert(0, '.')

from tests.test_plotting import create_test_const
from system_matrices import get_system_matrices
from system_matrices import get_transformation_matrix
import numpy as np
from scipy.linalg import eig

const = create_test_const()
wv_gamma = np.array([0.0, 0.0])

# Build matrices and solve
K, M = get_system_matrices(const, use_vectorized=False)
T = get_transformation_matrix(wv_gamma, const)
Kr = T.conj().T @ K @ T
Mr = T.conj().T @ M @ T

# Use full eig
Kr_dense = Kr.toarray()
Mr_dense = Mr.toarray()
eig_vals, eig_vecs = eig(np.linalg.solve(Mr_dense, Kr_dense))
idxs = np.argsort(np.real(eig_vals))[:const['N_eig']]
eig_vals = eig_vals[idxs]
eig_vecs = eig_vecs[:, idxs]

print("=" * 80)
print("CHECKING PHASE ALIGNMENT STEP")
print("=" * 80)

print(f"\n1. EIGENVECTORS BEFORE NORMALIZATION (first mode):")
eig_vec_0 = eig_vecs[:, 0]
print(f"  Type: {eig_vec_0.dtype}")
print(f"  Is real? {np.allclose(eig_vec_0.imag, 0, atol=1e-10)}")
print(f"  Max |imag|: {np.max(np.abs(eig_vec_0.imag)):.10e}")
u_before = eig_vec_0[0::2]
v_before = eig_vec_0[1::2]
print(f"  imag(u) stats: min={np.min(u_before.imag):.10e}, max={np.max(u_before.imag):.10e}, std={np.std(u_before.imag):.10e}")
print(f"  imag(v) stats: min={np.min(v_before.imag):.10e}, max={np.max(v_before.imag):.10e}, std={np.std(v_before.imag):.10e}")

print(f"\n2. AFTER NORMALIZATION:")
norms = np.linalg.norm(eig_vecs, axis=0)
eig_vecs_normalized = eig_vecs / norms
eig_vec_norm_0 = eig_vecs_normalized[:, 0]
u_after_norm = eig_vec_norm_0[0::2]
v_after_norm = eig_vec_norm_0[1::2]
print(f"  imag(u) stats: min={np.min(u_after_norm.imag):.10e}, max={np.max(u_after_norm.imag):.10e}, std={np.std(u_after_norm.imag):.10e}")
print(f"  imag(v) stats: min={np.min(v_after_norm.imag):.10e}, max={np.max(v_after_norm.imag):.10e}, std={np.std(v_after_norm.imag):.10e}")

print(f"\n3. AFTER PHASE ALIGNMENT:")
phase_align = np.exp(-1j * np.angle(eig_vecs[0, :]))
eig_vecs_aligned = eig_vecs_normalized * phase_align
eig_vec_align_0 = eig_vecs_aligned[:, 0]
u_after_align = eig_vec_align_0[0::2]
v_after_align = eig_vec_align_0[1::2]
print(f"  phase_align[0] = {phase_align[0]}")
print(f"  imag(u) stats: min={np.min(u_after_align.imag):.10e}, max={np.max(u_after_align.imag):.10e}, std={np.std(u_after_align.imag):.10e}")
print(f"  imag(v) stats: min={np.min(v_after_align.imag):.10e}, max={np.max(v_after_align.imag):.10e}, std={np.std(v_after_align.imag):.10e}")
print(f"  imag(v) is constant? {np.allclose(v_after_align.imag, np.mean(v_after_align.imag), atol=1e-10)}")

print(f"\n4. CHECKING FIRST COMPONENT:")
print(f"  eig_vecs[0, 0] (first component, first mode) = {eig_vecs[0, 0]}")
print(f"  angle(eig_vecs[0, 0]) = {np.angle(eig_vecs[0, 0]):.10e}")
print(f"  phase_align[0] = {phase_align[0]}")
print(f"  After alignment, first component = {eig_vecs_aligned[0, 0]}")

print("\n" + "=" * 80)

