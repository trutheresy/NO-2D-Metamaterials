#!/usr/bin/env python3
"""
Diagnostic script to trace K matrix construction in Python with float64 precision
and save all intermediate steps for comparison.
"""

import numpy as np
import scipy.sparse as sp
import h5py
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
from elements_vec import get_element_stiffness_VEC, get_element_mass_VEC

def apply_steel_rubber_paradigm(design, const):
    """Apply steel-rubber paradigm to single-channel design."""
    from scipy.interpolate import interp1d
    
    design_in_polymer = 0.0
    design_in_steel = 1.0
    
    E_polymer = 100e6
    E_steel = 200e9
    rho_polymer = 1200
    rho_steel = 8e3
    nu_polymer = 0.45
    nu_steel = 0.3
    
    E_min = const['E_min']
    E_max = const['E_max']
    rho_min = const['rho_min']
    rho_max = const['rho_max']
    poisson_min = const['poisson_min']
    poisson_max = const['poisson_max']
    
    design_out_polymer_E = (E_polymer - E_min) / (E_max - E_min)
    design_out_polymer_rho = (rho_polymer - rho_min) / (rho_max - rho_min)
    design_out_polymer_nu = (nu_polymer - poisson_min) / (poisson_max - poisson_min)
    
    design_out_steel_E = (E_steel - E_min) / (E_max - E_min)
    design_out_steel_rho = (rho_steel - rho_min) / (rho_max - rho_min)
    design_out_steel_nu = (nu_steel - poisson_min) / (poisson_max - poisson_min)
    
    design_vals = np.array([
        [design_out_polymer_E, design_out_steel_E],
        [design_out_polymer_rho, design_out_steel_rho],
        [design_out_polymer_nu, design_out_steel_nu]
    ])
    
    N_pix = design.shape[0]
    design_out = np.zeros((N_pix, N_pix, 3), dtype=np.float64)
    
    x_points = np.array([design_in_polymer, design_in_steel])
    for prop_idx in range(3):
        dvs = design_vals[prop_idx, :]
        interp_func = interp1d(x_points, dvs, kind='linear', 
                               bounds_error=False, fill_value=(dvs[0], dvs[1]))
        design_out[:, :, prop_idx] = interp_func(design)
    
    return design_out

def main():
    # Paths
    pt_input_path = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/pt")
    output_dir = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/python_intermediates_float64")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Tracing K Matrix Construction in Python (float64 precision)")
    print("=" * 80)
    print(f"Input:  {pt_input_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load design from PT file
    print("1. Loading data...")
    import torch
    geometries = torch.load(pt_input_path / "geometries_full.pt", map_location='cpu', weights_only=False)
    if isinstance(geometries, torch.Tensor):
        geometries = geometries.numpy()
    
    struct_idx = 0
    design_param = geometries[struct_idx]
    
    # Apply steel-rubber paradigm
    const_for_paradigm = {
        'E_min': 20e6,
        'E_max': 200e9,
        'rho_min': 1200,
        'rho_max': 8000,
        'poisson_min': 0.0,
        'poisson_max': 0.5
    }
    design_3ch = apply_steel_rubber_paradigm(design_param.astype(np.float64), const_for_paradigm)
    
    print(f"   Structure {struct_idx + 1}: design size = {design_3ch.shape}")
    print()
    
    # Create const dict
    const = {
        'design': design_3ch,
        'N_pix': 32,
        'N_ele': 1,
        'a': 1.0,
        'E_min': 20e6,
        'E_max': 200e9,
        'rho_min': 1200,
        'rho_max': 8000,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 1.0,
        'design_scale': 'linear',
        'isUseImprovement': True,
        'isUseSecondImprovement': False
    }
    
    # Compute K and M matrices
    print("2. Computing K and M matrices with float64 precision...")
    K, M = get_system_matrices_VEC(const)
    
    print(f"   K: shape={K.shape}, nnz={K.nnz}, dtype={K.dtype}")
    print(f"   M: shape={M.shape}, nnz={M.nnz}, dtype={M.dtype}")
    
    # Save matrices
    print()
    print("3. Saving matrices...")
    sp.save_npz(output_dir / 'step10_K_float64.npz', K)
    sp.save_npz(output_dir / 'step10_M_float64.npz', M)
    
    print(f"   Saved: step10_K_float64.npz")
    print(f"   Saved: step10_M_float64.npz")
    
    print()
    print("=" * 80)
    print(f"K and M matrices computed with float64 precision and saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

