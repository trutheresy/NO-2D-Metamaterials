"""
Test element matrix scaling to identify if element size affects stiffness matrix.

For Q4 plane stress elements, the stiffness matrix should be scaled by element area
if the element is not unit size.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from get_element_stiffness import get_element_stiffness
from get_element_mass import get_element_mass

# Test parameters
E = 100e9  # Pa
nu = 0.3
rho = 4000  # kg/m³
t = 0.01  # m
a = 1.0  # m (lattice parameter)
N_ele = 1
N_pix = 32

# Element size
element_size = a / (N_ele * N_pix)
element_area = element_size ** 2

print("="*70)
print("ELEMENT MATRIX SCALING TEST")
print("="*70)

print(f"\nParameters:")
print(f"  a (lattice parameter) = {a} m")
print(f"  N_ele = {N_ele}")
print(f"  N_pix = {N_pix}")
print(f"  Element size = {element_size:.6f} m")
print(f"  Element area = {element_area:.6f} m²")

const = {
    'N_pix': N_pix,
    'N_ele': N_ele,
    'a': a,
}

# Get element matrices
k_ele = get_element_stiffness(E, nu, t, const)
m_ele = get_element_mass(rho, t, const)

print(f"\nElement matrices (current implementation):")
print(f"  Stiffness matrix: min={np.min(k_ele):.6e}, max={np.max(k_ele):.6e}, mean={np.mean(k_ele):.6e}")
print(f"  Mass matrix: min={np.min(m_ele):.6e}, max={np.max(m_ele):.6e}, mean={np.mean(m_ele):.6e}")

# Check if stiffness should be scaled by element area
# For Q4 plane stress with full integration, the standard formula assumes unit element
# If element is not unit size, we may need to scale by area

# Let's check what happens if we scale by area
k_ele_scaled = k_ele * element_area
print(f"\nIf stiffness is scaled by element area ({element_area:.6e}):")
print(f"  Scaled stiffness matrix: min={np.min(k_ele_scaled):.6e}, max={np.max(k_ele_scaled):.6e}, mean={np.mean(k_ele_scaled):.6e}")

# Check MATLAB formula
# MATLAB: k_ele = (1/48)*E*t/(1-nu^2)*[...]
# This formula is for a unit square element
# For a non-unit element, we need to check if MATLAB scales it

# Check mass matrix formula
# MATLAB: m = rho*t*(const.a/(const.N_ele*const.N_pix(1)))^2
#        m_ele = (1/36)*m*[...]
# So m_ele already includes element area scaling!

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print("Mass matrix includes element area: m = rho*t*element_area")
print("Stiffness matrix formula (1/48)*E*t/(1-nu^2)*[...] is for UNIT element")
print("If element is not unit size, stiffness should be scaled by element area")

