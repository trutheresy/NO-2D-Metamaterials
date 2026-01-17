#!/usr/bin/env python3
"""
Diagnostic script to reconstruct K, M, T matrices using Python and save them for comparison.

This script:
1. Loads PT dataset
2. Computes K, M, T matrices using Python functions
3. Saves matrices for comparison with MATLAB
"""

import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path
import sys
import time
import h5py

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
from system_matrices import get_transformation_matrix

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


def reconstruct_KMT_python(pt_input_path, output_dir):
    """
    Reconstruct K, M, T matrices using Python functions.
    
    Parameters:
    -----------
    pt_input_path : Path
        Path to PT dataset directory
    output_dir : Path
        Directory to save K, M, T matrices
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Reconstructing K, M, T Matrices using Python")
    print("=" * 80)
    print(f"Input: {pt_input_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load PT files
    print("1. Loading PT dataset...")
    geometries = torch.load(pt_input_path / "geometries_full.pt", map_location='cpu', weights_only=False)
    wavevectors = torch.load(pt_input_path / "wavevectors_full.pt", map_location='cpu', weights_only=False)
    
    geometries_np = geometries.numpy()
    wavevectors_np = wavevectors.numpy()
    
    n_designs = geometries_np.shape[0]
    design_res = geometries_np.shape[1]
    n_wavevectors = wavevectors_np.shape[1]
    
    print(f"   n_designs: {n_designs}")
    print(f"   design_res: {design_res}")
    print(f"   n_wavevectors: {n_wavevectors}")
    
    # Constants
    E_min = 20000000
    E_max = 200000000000
    rho_min = 1200
    rho_max = 8000
    nu_min = 0.0
    nu_max = 0.5
    t_val = 1.0
    a_val = 1.0
    N_ele = 1
    
    # Storage for K, M, T matrices
    K_list = []
    M_list = []
    T_list_all = []  # List of lists: T_list_all[struct_idx][wv_idx] = T
    
    print()
    print("2. Computing K, M, T matrices for each structure...")
    
    for struct_idx in range(n_designs):
        print(f"\n   Structure {struct_idx + 1}/{n_designs}...")
        
        # Get design
        design_param = geometries_np[struct_idx]  # (design_res, design_res)
        
        # Apply steel-rubber paradigm
        const_for_paradigm = {
            'E_min': E_min,
            'E_max': E_max,
            'rho_min': rho_min,
            'rho_max': rho_max,
            'poisson_min': nu_min,
            'poisson_max': nu_max
        }
        design_3ch = apply_steel_rubber_paradigm(design_param.astype(np.float64), const_for_paradigm)
        
        # Create const dict for matrix computation
        const_for_km = {
            'design': design_3ch,
            'N_pix': design_res,
            'N_ele': N_ele,
            'a': a_val,
            'E_min': E_min,
            'E_max': E_max,
            'rho_min': rho_min,
            'rho_max': rho_max,
            'poisson_min': nu_min,
            'poisson_max': nu_max,
            't': t_val,
            'design_scale': 'linear',
            'isUseImprovement': True,
            'isUseSecondImprovement': False
        }
        
        # Compute K and M matrices
        print(f"      Computing K and M...")
        K, M = get_system_matrices_VEC(const_for_km)
        K_list.append(K)
        M_list.append(M)
        print(f"        K: shape={K.shape}, nnz={K.nnz}")
        print(f"        M: shape={M.shape}, nnz={M.nnz}")
        
        # Get wavevectors for this structure
        wavevectors_struct = wavevectors_np[struct_idx, :, :]  # (n_wavevectors, 2)
        
        # Compute T matrices
        print(f"      Computing T matrices for {n_wavevectors} wavevectors...")
        T_list_struct = []
        for wv_idx, wv in enumerate(wavevectors_struct):
            T = get_transformation_matrix(wv.astype(np.float32), const_for_km)
            if T is None:
                raise ValueError(f"Failed to compute T matrix for wavevector {wv_idx}")
            T_list_struct.append(T)
            if wv_idx == 0:
                print(f"        T[{wv_idx}]: shape={T.shape}, dtype={T.dtype}, sparse={sp.issparse(T)}")
        
        T_list_all.append(T_list_struct)
    
    # Save K, M, T matrices
    print()
    print("3. Saving K, M, T matrices...")
    
    # Save as PyTorch format
    torch.save(K_list, output_dir / "K_data_python.pt")
    torch.save(M_list, output_dir / "M_data_python.pt")
    torch.save(T_list_all, output_dir / "T_data_python.pt")
    print(f"   Saved: K_data_python.pt ({len(K_list)} structures)")
    print(f"   Saved: M_data_python.pt ({len(M_list)} structures)")
    print(f"   Saved: T_data_python.pt ({len(T_list_all)} structures × {n_wavevectors} wavevectors)")
    
    # Also save summary statistics
    print()
    print("4. Matrix Statistics:")
    for struct_idx in range(n_designs):
        K = K_list[struct_idx]
        M = M_list[struct_idx]
        print(f"   Structure {struct_idx + 1}:")
        print(f"     K: shape={K.shape}, nnz={K.nnz}, dtype={K.dtype}")
        print(f"     M: shape={M.shape}, nnz={M.nnz}, dtype={M.dtype}")
        if struct_idx == 0:
            T_sample = T_list_all[0][0]
            print(f"     T[0]: shape={T_sample.shape}, dtype={T_sample.dtype}, sparse={sp.issparse(T_sample)}")
    
    print()
    print("=" * 80)
    print("Python K, M, T reconstruction complete!")
    print("=" * 80)
    
    return K_list, M_list, T_list_all

if __name__ == "__main__":
    pt_input_path = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/pt")
    output_dir = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test")
    
    reconstruct_KMT_python(pt_input_path, output_dir)

