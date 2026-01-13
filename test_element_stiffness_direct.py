"""
Direct comparison of element stiffness matrix computation.
Test with a single element to isolate differences.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from elements_vec import get_element_stiffness_VEC

# Test with single element
E = np.array([200e9])
nu = np.array([0.3])
t = 1.0

print("="*70)
print("ELEMENT STIFFNESS MATRIX DIRECT COMPARISON")
print("="*70)

print(f"\nInput parameters:")
print(f"  E = {E[0]:.6e} Pa")
print(f"  nu = {nu[0]:.6f}")
print(f"  t = {t:.6f}")

# Compute using Python
k_ele_py = get_element_stiffness_VEC(E, nu, t)
print(f"\nPython result:")
print(f"  Shape: {k_ele_py.shape}")
print(f"  Min: {np.min(k_ele_py):.6e}, Max: {np.max(k_ele_py):.6e}, Mean: {np.mean(k_ele_py):.6e}")
print(f"  Full matrix (element 0):")
print(k_ele_py[0])

# Compute manually using MATLAB formula
coeff = (1/48) * E[0] * t / (1 - nu[0]**2)
nu_val = nu[0]

# MATLAB matrix template
k_template = np.array([
    [24-8*nu_val,  6*nu_val+6,   -12-4*nu_val, 18*nu_val-6,  -12+4*nu_val, -6*nu_val-6,   8*nu_val,    -18*nu_val+6],
    [6*nu_val+6,   24-8*nu_val,  -18*nu_val+6, 8*nu_val,     -6*nu_val-6,  -12+4*nu_val,  18*nu_val-6, -12-4*nu_val],
    [-12-4*nu_val, -18*nu_val+6, 24-8*nu_val,  -6*nu_val-6,  8*nu_val,     18*nu_val-6,  -12+4*nu_val, 6*nu_val+6],
    [18*nu_val-6,  8*nu_val,     -6*nu_val-6,  24-8*nu_val, -18*nu_val+6, -12-4*nu_val,  6*nu_val+6,  -12+4*nu_val],
    [-12+4*nu_val, -6*nu_val-6,  8*nu_val,     -18*nu_val+6, 24-8*nu_val,  6*nu_val+6,   -12-4*nu_val, 18*nu_val-6],
    [-6*nu_val-6,  -12+4*nu_val, 18*nu_val-6,  -12-4*nu_val, 6*nu_val+6,   24-8*nu_val,  -18*nu_val+6, 8*nu_val],
    [8*nu_val,     18*nu_val-6,  -12+4*nu_val, 6*nu_val+6,   -12-4*nu_val, -18*nu_val+6, 24-8*nu_val,  -6*nu_val-6],
    [-18*nu_val+6, -12-4*nu_val, 6*nu_val+6,   -12+4*nu_val, 18*nu_val-6,  8*nu_val,     -6*nu_val-6,  24-8*nu_val]
], dtype=np.float32)

k_ele_manual = coeff * k_template

print(f"\nManual computation (MATLAB formula):")
print(f"  Coeff = {coeff:.6e}")
print(f"  Min: {np.min(k_ele_manual):.6e}, Max: {np.max(k_ele_manual):.6e}, Mean: {np.mean(k_ele_manual):.6e}")
print(f"  Full matrix:")
print(k_ele_manual)

# Compare
diff = np.abs(k_ele_py[0] - k_ele_manual)
print(f"\nDifference (Python - Manual):")
print(f"  Max diff: {np.max(diff):.6e}")
print(f"  Mean diff: {np.mean(diff):.6e}")
if np.allclose(k_ele_py[0], k_ele_manual, rtol=1e-5):
    print(f"  ✅ Matrices match!")
else:
    print(f"  ❌ Matrices differ")
    print(f"  Difference matrix:")
    print(diff)

print("\n" + "="*70)

