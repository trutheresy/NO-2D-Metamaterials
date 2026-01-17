"""Check eigenvector component details."""
import sys
sys.path.insert(0, '.')

from tests.test_plotting import create_test_const
from dispersion import dispersion
from system_matrices import get_transformation_matrix
import numpy as np

const = create_test_const()
wv_gamma = np.array([[0.0, 0.0]])

# Get eigenvector from dispersion
wv, fr, ev, mesh = dispersion(const, wv_gamma)
ev_red = ev[:, 0, 0]  # First wavevector, first mode

print("=" * 80)
print("EIGENVECTOR COMPONENT ANALYSIS")
print("=" * 80)

print(f"\n1. REDUCED SPACE EIGENVECTOR (before transformation):")
print(f"  Type: {ev_red.dtype}")
print(f"  Shape: {ev_red.shape}")
print(f"  Is purely real? {np.allclose(ev_red.imag, 0, atol=1e-10)}")
print(f"  Max |imag|: {np.max(np.abs(ev_red.imag)):.10e}")

# Extract u and v components from reduced space
u_red = ev_red[0::2]
v_red = ev_red[1::2]

print(f"\n  u_red (first 10): {u_red[:10]}")
print(f"  v_red (first 10): {v_red[:10]}")

print(f"\n  u_red imag stats:")
print(f"    Min: {np.min(u_red.imag):.10e}")
print(f"    Max: {np.max(u_red.imag):.10e}")
print(f"    Mean: {np.mean(u_red.imag):.10e}")
print(f"    Std: {np.std(u_red.imag):.10e}")

print(f"\n  v_red imag stats:")
print(f"    Min: {np.min(v_red.imag):.10e}")
print(f"    Max: {np.max(v_red.imag):.10e}")
print(f"    Mean: {np.mean(v_red.imag):.10e}")
print(f"    Std: {np.std(v_red.imag):.10e}")

# Transform to full space
T = get_transformation_matrix(wv_gamma[0, :], const)
ev_full = T @ ev_red

print(f"\n2. FULL SPACE EIGENVECTOR (after transformation):")
print(f"  Type: {ev_full.dtype}")
print(f"  Shape: {ev_full.shape}")
print(f"  Is purely real? {np.allclose(ev_full.imag, 0, atol=1e-10)}")
print(f"  Max |imag|: {np.max(np.abs(ev_full.imag)):.10e}")

u_full = ev_full[0::2]
v_full = ev_full[1::2]

print(f"\n  u_full imag stats:")
print(f"    Min: {np.min(u_full.imag):.10e}")
print(f"    Max: {np.max(u_full.imag):.10e}")
print(f"    Mean: {np.mean(u_full.imag):.10e}")
print(f"    Std: {np.std(u_full.imag):.10e}")

print(f"\n  v_full imag stats:")
print(f"    Min: {np.min(v_full.imag):.10e}")
print(f"    Max: {np.max(v_full.imag):.10e}")
print(f"    Mean: {np.mean(v_full.imag):.10e}")
print(f"    Std: {np.std(v_full.imag):.10e}")

# Check if the constant imag(v) comes from the reduced space or transformation
print(f"\n3. CHECKING CONSTANCY:")
print(f"  imag(v_red) is constant? {np.allclose(v_red.imag, np.mean(v_red.imag), atol=1e-10)}")
print(f"  imag(v_full) is constant? {np.allclose(v_full.imag, np.mean(v_full.imag), atol=1e-10)}")

# Check the transformation matrix
print(f"\n4. TRANSFORMATION MATRIX AT k=0:")
print(f"  T type: {type(T)}")
print(f"  T shape: {T.shape}")
print(f"  T is real? {np.allclose(T.imag, 0, atol=1e-10)}")
if hasattr(T, 'toarray'):
    T_dense = T.toarray()
    print(f"  Max |imag(T)|: {np.max(np.abs(T_dense.imag)):.10e}")
else:
    print(f"  Max |imag(T)|: {np.max(np.abs(T.imag)):.10e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("If imag(v) is constant at k=0, it suggests the eigenvector has a small")
print("imaginary component. This could be due to:")
print("1. Numerical precision in eigenvalue solver")
print("2. Phase alignment step introducing small imaginary part")
print("3. Normalization step not preserving realness")

