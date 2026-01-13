"""
Check if element stiffness matrix diagonal values match between Python and MATLAB.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from elements_vec import get_element_stiffness_VEC
import torch

# Use same values as in the actual computation
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
struct_idx = 0

# Load design
geometries = torch.load(python_data_dir / 'geometries_full.pt', map_location='cpu')
if isinstance(geometries, torch.Tensor):
    geometries = geometries.numpy()
design = geometries[struct_idx].astype(np.float32)
N_pix = design.shape[0]
design_3ch = np.stack([design, design, design], axis=-1)

# Expand design
design_expanded = np.repeat(
    np.repeat(design_3ch, 1, axis=0), 
    1, axis=1
)

# Extract material properties
E_min, E_max = 20e6, 200e9
poisson_min, poisson_max = 0.0, 0.5
t = 1.0

design_ch0 = design_expanded[:, :, 0].astype(np.float64)
design_ch2 = design_expanded[:, :, 2].astype(np.float64)
E = (E_min + design_ch0 * (E_max - E_min)).T.astype(np.float32)
nu = (poisson_min + design_ch2 * (poisson_max - poisson_min)).T.astype(np.float32)

# Get element matrices
AllLEle = get_element_stiffness_VEC(E.flatten(), nu.flatten(), t)

# Extract diagonal of each element matrix
N_ele = AllLEle.shape[0]
diagonals = np.array([np.diag(AllLEle[i]) for i in range(N_ele)])

print(f"Element stiffness matrix diagonal analysis:")
print(f"  Number of elements: {N_ele}")
print(f"  Diagonal shape per element: {np.diag(AllLEle[0]).shape}")
print(f"  All diagonals shape: {diagonals.shape}")
print(f"  Diagonal values range: [{np.min(diagonals):.6e}, {np.max(diagonals):.6e}]")
print(f"  Diagonal values mean: {np.mean(diagonals):.6e}")

# Check first few elements
print(f"\nFirst 5 elements diagonal values:")
for i in range(min(5, N_ele)):
    diag_vals = np.diag(AllLEle[i])
    print(f"  Element {i}: E={E.flatten()[i]:.6e}, nu={nu.flatten()[i]:.6f}")
    print(f"    Diagonal: min={np.min(diag_vals):.6e}, max={np.max(diag_vals):.6e}, mean={np.mean(diag_vals):.6e}")

# Check if diagonal values match expected formula
# For Q4 element, diagonal entries should be (1/48)*E*t/(1-nu^2)*24*(1-nu/3)
# Actually, let me check the actual formula from the element stiffness matrix
print(f"\nChecking diagonal formula consistency:")
E_test = E.flatten()[0]
nu_test = nu.flatten()[0]
k_ele_test = AllLEle[0]
diag_test = np.diag(k_ele_test)

# Formula from get_element_stiffness: k_ele = (1/48)*E*t/(1-nu^2)*[matrix]
# The diagonal entries in the matrix template are 24-8*nu
coeff = (1/48) * E_test * t / (1 - nu_test**2)
expected_diag = coeff * (24 - 8*nu_test)

print(f"  First element: E={E_test:.6e}, nu={nu_test:.6f}")
print(f"  Coefficient: {coeff:.6e}")
print(f"  Expected diagonal (assuming 24-8*nu): {expected_diag:.6e}")
print(f"  Actual diagonal values: {diag_test}")
print(f"  Do they match? {np.allclose(diag_test, expected_diag, rtol=1e-4)}")

