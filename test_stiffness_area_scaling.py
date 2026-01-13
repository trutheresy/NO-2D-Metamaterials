"""
Test if adding element area scaling to stiffness matrix fixes the issue.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))

# Test parameters
a = 1.0
N_ele = 1
N_pix = 32

element_size = a / (N_ele * N_pix)
element_area = element_size ** 2

print("="*70)
print("TESTING STIFFNESS AREA SCALING")
print("="*70)
print(f"\nParameters:")
print(f"  a = {a}")
print(f"  N_ele = {N_ele}")
print(f"  N_pix = {N_pix}")
print(f"  element_size = {element_size:.6e} m")
print(f"  element_area = {element_area:.6e} m²")
print(f"  1/element_area = {1.0/element_area:.6e}")

# From the scaling check, median ratio was ~77.7
median_ratio = 77.7
print(f"\nObserved median ratio (MATLAB/Python): {median_ratio:.2f}")

# Check if this relates to element size
print(f"\nPossible relationships:")
print(f"  1/element_size = {1.0/element_size:.2f}")
print(f"  element_size = {element_size:.6f}")
print(f"  element_size * N_pix = {element_size * N_pix:.2f} (should equal a = {a})")
print(f"  N_pix = {N_pix}")

# The Q4 stiffness matrix formula (1/48)*E*t/(1-nu^2)*[...] is for a unit element
# For a square element of side h, the stiffness should be scaled by h^2 (area)
# So if Python is NOT scaling, and MATLAB IS, the ratio should be 1/element_area

expected_ratio_if_python_missing_area = 1.0 / element_area
print(f"\nIf Python is missing element_area scaling:")
print(f"  Expected ratio = 1/element_area = {expected_ratio_if_python_missing_area:.2f}")
print(f"  Observed ratio = {median_ratio:.2f}")
print(f"  Match: {np.isclose(median_ratio, expected_ratio_if_python_missing_area, rtol=0.5)}")

# Check other possibilities
print(f"\nOther possibilities:")
print(f"  N_pix^2 = {N_pix**2} (no match)")
print(f"  element_size * 100 = {element_size * 100:.2f} (no match)")
print(f"  (N_pix/4)^2 = {(N_pix/4)**2:.2f} (no match)")

