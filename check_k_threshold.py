"""
Check what threshold would be needed for K matrix.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys
sys.path.insert(0, '2d-dispersion-py')

from plot_dispersion_infer_eigenfrequencies import load_pt_dataset, create_const_dict
from system_matrices_vec import get_element_stiffness_VEC, get_element_mass_VEC

# Load design
data = load_pt_dataset(Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1'), None, require_eigenvalue_data=False)
design = data['designs'][0]

# Create const
const = create_const_dict(
    design, N_pix=32, N_ele=1,
    a=1.0, E_min=20e6, E_max=200e9,
    rho_min=400, rho_max=8000,
    nu_min=0.0, nu_max=0.5
)

# Get to the point where we have the values
N_pix = 32
N_ele = 1
N_ele_x = N_pix * N_ele
N_ele_y = N_pix * N_ele

design_expanded = np.repeat(
    np.repeat(const['design'], N_ele, axis=0), 
    N_ele, axis=1
)

E = (const['E_min'] + design_expanded[:, :, 0] * (const['E_max'] - const['E_min'])).T
nu = (const['poisson_min'] + design_expanded[:, :, 2] * (const['poisson_max'] - const['poisson_min'])).T
t = const['t']
rho = (const['rho_min'] + design_expanded[:, :, 1] * (const['rho_max'] - const['rho_min'])).T

AllLEle = get_element_stiffness_VEC(E.flatten(), nu.flatten(), t)
value_K = AllLEle.flatten().astype(np.float32)

# Check distribution of small values
abs_values = np.abs(value_K)
small_values = abs_values[abs_values < 1e-5]

print(f"Total values: {len(value_K)}")
print(f"Values < 1e-5: {len(small_values)}")
print(f"Values < 1e-10: {np.sum(abs_values < 1e-10)}")
print(f"Values < 1e-11: {np.sum(abs_values < 1e-11)}")
print(f"Values < 1e-12: {np.sum(abs_values < 1e-12)}")

if len(small_values) > 0:
    print(f"\nSmallest values (first 20):")
    print(np.sort(small_values)[:20])
    print(f"Max of small values: {np.max(small_values)}")

# Load MATLAB K to compare
import h5py
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
with h5py.File(matlab_mat_file, 'r') as f:
    K_DATA_ref = f['K_DATA'][0, 0]
    K_data_ml = np.array(f[K_DATA_ref]['data']).flatten()
    K_ir_ml = np.array(f[K_DATA_ref]['ir']).flatten()
    K_jc_ml = np.array(f[K_DATA_ref]['jc']).flatten()
    n = len(K_jc_ml) - 1
    K_ml = sp.csr_matrix((K_data_ml, K_ir_ml, K_jc_ml), shape=(n, n))

print(f"\nMATLAB K nnz: {K_ml.nnz}")
print(f"MATLAB K smallest absolute values (first 20):")
K_ml_coo = K_ml.tocoo()
K_ml_abs = np.abs(K_ml_coo.data)
print(np.sort(K_ml_abs)[:20])

