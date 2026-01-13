"""
Compare material property values (E, nu, rho, t) between Python and MATLAB.
"""
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import sys
import h5py

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC_simplified

# Configuration
python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
matlab_mat_file = Path(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat')
struct_idx = 0

print("="*70)
print("COMPARING MATERIAL PROPERTY VALUES")
print("="*70)

# Load design
if (python_data_dir / 'geometries_full.pt').exists():
    geometries = torch.load(python_data_dir / 'geometries_full.pt', map_location='cpu')
    if isinstance(geometries, torch.Tensor):
        geometries = geometries.numpy()
    design = geometries[struct_idx]
elif (python_data_dir / 'design_params_full.pt').exists():
    design_params = torch.load(python_data_dir / 'design_params_full.pt', map_location='cpu')
    if isinstance(design_params, torch.Tensor):
        design_params = design_params.numpy()
    design = design_params[struct_idx]
else:
    print("Error: Could not find design file")
    sys.exit(1)

if design.ndim == 2:
    N_pix = design.shape[0]
    design_3ch = np.stack([design, design, design], axis=-1)
else:
    N_pix = design.shape[0]
    design_3ch = design

print(f"\n1. Design loaded: shape={design.shape}, N_pix={N_pix}")

# Python const parameters
const_py = {
    'design': design_3ch,
    'N_pix': N_pix,
    'N_ele': 1,
    'a': 1.0,
    'E_min': 20e6,
    'E_max': 200e9,
    'rho_min': 400,
    'rho_max': 8000,
    'poisson_min': 0.0,
    'poisson_max': 0.5,
    't': 0.01,
    'design_scale': 'linear',
    'isUseImprovement': True,
    'isUseSecondImprovement': True,
}

print(f"\n2. Python const parameters:")
print(f"   E: [{const_py['E_min']:.6e}, {const_py['E_max']:.6e}]")
print(f"   rho: [{const_py['rho_min']:.2f}, {const_py['rho_max']:.2f}]")
print(f"   nu: [{const_py['poisson_min']:.2f}, {const_py['poisson_max']:.2f}]")
print(f"   t: {const_py['t']:.6f}")

# MATLAB const parameters (from ex_dispersion_batch_save.m)
print(f"\n3. MATLAB const parameters (from ex_dispersion_batch_save.m):")
print(f"   E_min = 20e6, E_max = 200e9 (line 55-56)")
print(f"   rho_min = 1200, rho_max = 8e3 (line 57-58)  ⚠️  DIFFERENT!")
print(f"   poisson_min = 0, poisson_max = 0.5 (line 59-60)")
print(f"   t = 1 (line 61)  ⚠️  DIFFERENT!")

# Extract material properties using Python's method
print(f"\n4. Extracting material properties (Python method):")
N_ele = const_py['N_ele']
design_expanded = np.repeat(
    np.repeat(const_py['design'], N_ele, axis=0), 
    N_ele, axis=1
)

if const_py['design_scale'] == 'linear':
    E_py = (const_py['E_min'] + design_expanded[:, :, 0] * (const_py['E_max'] - const_py['E_min'])).T
    nu_py = (const_py['poisson_min'] + design_expanded[:, :, 2] * (const_py['poisson_max'] - const_py['poisson_min'])).T
    rho_py = (const_py['rho_min'] + design_expanded[:, :, 1] * (const_py['rho_max'] - const_py['rho_min'])).T
    t_py = const_py['t']
else:
    E_py = np.exp(design_expanded[:, :, 0]).T
    nu_py = (const_py['poisson_min'] + design_expanded[:, :, 2] * (const_py['poisson_max'] - const_py['poisson_min'])).T
    rho_py = np.exp(design_expanded[:, :, 1]).T
    t_py = const_py['t']

print(f"   E_py: shape={E_py.shape}, min={np.min(E_py):.6e}, max={np.max(E_py):.6e}, mean={np.mean(E_py):.6e}")
print(f"   nu_py: shape={nu_py.shape}, min={np.min(nu_py):.6f}, max={np.max(nu_py):.6f}, mean={np.mean(nu_py):.6f}")
print(f"   rho_py: shape={rho_py.shape}, min={np.min(rho_py):.2f}, max={np.max(rho_py):.2f}, mean={np.mean(rho_py):.2f}")
print(f"   t_py: {t_py:.6f}")

# Check MATLAB material properties from saved .mat file
print(f"\n5. Checking MATLAB material properties from .mat file:")
try:
    with h5py.File(str(matlab_mat_file), 'r') as f:
        # Check for ELASTIC_MODULUS_DATA, DENSITY_DATA, POISSON_DATA
        if 'ELASTIC_MODULUS_DATA' in f:
            E_ml = np.array(f['ELASTIC_MODULUS_DATA'])[:, :, struct_idx]
            print(f"   E_ml: shape={E_ml.shape}, min={np.min(E_ml):.6e}, max={np.max(E_ml):.6e}, mean={np.mean(E_ml):.6e}")
        else:
            print("   ⚠️  ELASTIC_MODULUS_DATA not found")
        
        if 'DENSITY_DATA' in f:
            rho_ml = np.array(f['DENSITY_DATA'])[:, :, struct_idx]
            print(f"   rho_ml: shape={rho_ml.shape}, min={np.min(rho_ml):.2f}, max={np.max(rho_ml):.2f}, mean={np.mean(rho_ml):.2f}")
        else:
            print("   ⚠️  DENSITY_DATA not found")
        
        if 'POISSON_DATA' in f:
            nu_ml = np.array(f['POISSON_DATA'])[:, :, struct_idx]
            print(f"   nu_ml: shape={nu_ml.shape}, min={np.min(nu_ml):.6f}, max={np.max(nu_ml):.6f}, mean={np.mean(nu_ml):.6f}")
        else:
            print("   ⚠️  POISSON_DATA not found")
        
        # Check const.t
        if 'const' in f:
            const_ref = f['const']
            if isinstance(const_ref, h5py.Group):
                if 't' in const_ref:
                    t_ref = const_ref['t']
                    if isinstance(t_ref, h5py.Dataset):
                        t_ml = np.array(t_ref).item()
                    else:
                        t_ml = np.array(f[t_ref]).item()
                    print(f"   t_ml (from const): {t_ml:.6f}")
                else:
                    print("   ⚠️  const.t not found")
            else:
                print("   ⚠️  const is not a group")
        else:
            print("   ⚠️  const structure not found")
            
except Exception as e:
    print(f"   Error reading MATLAB data: {e}")
    import traceback
    traceback.print_exc()

# Compare if we have both
print(f"\n6. Comparison:")
if 'E_ml' in locals() and 'E_py' in locals():
    if E_ml.shape == E_py.shape:
        E_diff = np.abs(E_ml - E_py)
        print(f"   E difference: max={np.max(E_diff):.6e}, mean={np.mean(E_diff):.6e}")
        if np.allclose(E_ml, E_py, rtol=1e-5):
            print(f"   ✅ E values match")
        else:
            print(f"   ❌ E values differ")
    else:
        print(f"   ⚠️  E shapes don't match: MATLAB {E_ml.shape} vs Python {E_py.shape}")

if 'rho_ml' in locals() and 'rho_py' in locals():
    if rho_ml.shape == rho_py.shape:
        rho_diff = np.abs(rho_ml - rho_py)
        print(f"   rho difference: max={np.max(rho_diff):.2f}, mean={np.mean(rho_diff):.2f}")
        if np.allclose(rho_ml, rho_py, rtol=1e-5):
            print(f"   ✅ rho values match")
        else:
            print(f"   ❌ rho values differ")
            print(f"      Python range: [{const_py['rho_min']:.2f}, {const_py['rho_max']:.2f}]")
            print(f"      MATLAB range: [1200, 8000] (from ex_dispersion_batch_save.m)")
    else:
        print(f"   ⚠️  rho shapes don't match: MATLAB {rho_ml.shape} vs Python {rho_py.shape}")

if 'nu_ml' in locals() and 'nu_py' in locals():
    if nu_ml.shape == nu_py.shape:
        nu_diff = np.abs(nu_ml - nu_py)
        print(f"   nu difference: max={np.max(nu_diff):.6f}, mean={np.mean(nu_diff):.6f}")
        if np.allclose(nu_ml, nu_py, rtol=1e-5):
            print(f"   ✅ nu values match")
        else:
            print(f"   ❌ nu values differ")
    else:
        print(f"   ⚠️  nu shapes don't match: MATLAB {nu_ml.shape} vs Python {nu_py.shape}")

if 't_ml' in locals() and 't_py' in locals():
    t_diff = abs(t_ml - t_py)
    print(f"   t difference: {t_diff:.6f}")
    if np.isclose(t_ml, t_py, rtol=1e-5):
        print(f"   ✅ t values match")
    else:
        print(f"   ❌ t values differ: Python={t_py:.6f}, MATLAB={t_ml:.6f}")
        print(f"      Ratio (MATLAB/Python) = {t_ml/t_py:.2f}")

print("\n" + "="*70)

