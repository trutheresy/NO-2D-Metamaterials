#!/usr/bin/env python3
"""
Diagnostic script to trace K matrix construction in Python and save all intermediate steps.
This will be compared with MATLAB to find where discrepancies occur.
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
    input_mat_file = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/mat/pt.mat")
    pt_input_path = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/pt")
    output_dir = Path("D:/Research/NO-2D-Metamaterials/data/KMT_py_matlab_reconstruction_test/python_intermediates")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Tracing K Matrix Construction in Python")
    print("=" * 80)
    print(f"Input:  {input_mat_file}")
    print(f"Output: {output_dir}")
    print()
    
    # Load design from PT file (to match what Python uses)
    print("1. Loading data...")
    import torch
    geometries = torch.load(pt_input_path / "geometries_full.pt", map_location='cpu', weights_only=False)
    if isinstance(geometries, torch.Tensor):
        geometries = geometries.numpy()
    
    struct_idx = 0
    design_param = geometries[struct_idx]  # (32, 32)
    
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
    
    # Now trace through get_system_matrices_VEC step by step
    print("2. Tracing K matrix construction...")
    
    # Step 1: N_ele_x, N_ele_y
    N_pix = const['N_pix']
    N_ele = const['N_ele']
    N_ele_x = N_pix * N_ele
    N_ele_y = N_pix * N_ele
    print(f"   Step 1: N_ele_x = {N_ele_x}, N_ele_y = {N_ele_y}")
    np.savez(output_dir / 'step1_N_ele.npz', N_ele_x=N_ele_x, N_ele_y=N_ele_y)
    
    # Step 2: Expand design (repelem)
    design_expanded = np.repeat(
        np.repeat(const['design'], N_ele, axis=0), 
        N_ele, axis=1
    )
    print(f"   Step 2: design_expanded size = {design_expanded.shape}")
    np.save(output_dir / 'step2_design_expanded.npy', design_expanded)
    
    # Step 3: Extract material properties
    design_ch0 = design_expanded[:, :, 0].astype(np.float64)
    design_ch1 = design_expanded[:, :, 1].astype(np.float64)
    design_ch2 = design_expanded[:, :, 2].astype(np.float64)
    E = (const['E_min'] + design_ch0 * (const['E_max'] - const['E_min'])).T.astype(np.float32)
    nu = (const['poisson_min'] + design_ch2 * (const['poisson_max'] - const['poisson_min'])).T.astype(np.float32)
    t = const['t']
    rho = (const['rho_min'] + design_ch1 * (const['rho_max'] - const['rho_min'])).T.astype(np.float32)
    print(f"   Step 3: Material properties")
    print(f"     E: size = {E.shape}, range = [{E.min():.6e}, {E.max():.6e}]")
    print(f"     nu: size = {nu.shape}, range = [{nu.min():.6e}, {nu.max():.6e}]")
    print(f"     rho: size = {rho.shape}, range = [{rho.min():.6e}, {rho.max():.6e}]")
    np.savez(output_dir / 'step3_material_props.npz', E=E, nu=nu, t=t, rho=rho)
    
    # Step 4: Node numbering
    nodenrs = np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1).reshape(1 + N_ele_y, 1 + N_ele_x, order='F')
    print(f"   Step 4: nodenrs size = {nodenrs.shape}, range = [{nodenrs.min()}, {nodenrs.max()}]")
    np.save(output_dir / 'step4_nodenrs.npy', nodenrs)
    
    # Step 5: edofVec
    edofVec = (2 * nodenrs[0:-1, 0:-1] - 1).reshape(N_ele_x * N_ele_y, 1, order='F').flatten()
    print(f"   Step 5: edofVec size = {edofVec.shape}, range = [{edofVec.min()}, {edofVec.max()}]")
    np.save(output_dir / 'step5_edofVec.npy', edofVec)
    
    # Step 6: edofMat
    offset_array = np.concatenate([
        2*(N_ele_y+1) + np.array([0, 1, 2, 3]),
        np.array([2, 3, 0, 1])
    ])
    edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(
        offset_array,
        (N_ele_x * N_ele_y, 1)
    )
    print(f"   Step 6: edofMat size = {edofMat.shape}")
    print(f"     First row: {edofMat[0, :]}")
    np.save(output_dir / 'step6_edofMat.npy', edofMat)
    
    # Step 7: row_idxs and col_idxs
    row_idxs_mat = np.kron(edofMat, np.ones((8, 1)))
    row_idxs = row_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()
    
    col_idxs_mat = np.kron(edofMat, np.ones((1, 8)))
    col_idxs = col_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()
    print(f"   Step 7: Indices")
    print(f"     row_idxs: size = {row_idxs.shape}, range = [{row_idxs.min()}, {row_idxs.max()}]")
    print(f"     col_idxs: size = {col_idxs.shape}, range = [{col_idxs.min()}, {col_idxs.max()}]")
    np.savez(output_dir / 'step7_indices.npz', row_idxs=row_idxs, col_idxs=col_idxs)
    
    # Step 8: Element matrices
    AllLEle = get_element_stiffness_VEC(E.flatten(order='F'), nu.flatten(order='F'), t)
    AllLMat = get_element_mass_VEC(rho.flatten(order='F'), t, const)
    print(f"   Step 8: Element matrices")
    print(f"     AllLEle: size = {AllLEle.shape}, range = [{AllLEle.min():.6e}, {AllLEle.max():.6e}]")
    print(f"     AllLMat: size = {AllLMat.shape}, range = [{AllLMat.min():.6e}, {AllLMat.max():.6e}]")
    np.savez(output_dir / 'step8_element_matrices.npz', AllLEle=AllLEle, AllLMat=AllLMat)
    
    # Step 9: Flattened values
    N_ele_total = N_ele_x * N_ele_y
    AllLEle_2d = AllLEle.reshape(N_ele_total, 64)
    AllLEle_transposed = AllLEle_2d.T
    value_K = AllLEle_transposed.flatten(order='F').astype(np.float32)
    
    AllLMat_2d = AllLMat.reshape(N_ele_total, 64)
    AllLMat_transposed = AllLMat_2d.T
    value_M = AllLMat_transposed.flatten(order='F').astype(np.float32)
    print(f"   Step 9: Flattened values")
    print(f"     value_K: size = {value_K.shape}, range = [{value_K.min():.6e}, {value_K.max():.6e}]")
    print(f"     value_M: size = {value_M.shape}, range = [{value_M.min():.6e}, {value_M.max():.6e}]")
    np.savez(output_dir / 'step9_values.npz', value_K=value_K, value_M=value_M)
    
    # Step 10: Final sparse matrices (convert to 0-based indices)
    row_idxs_0based = row_idxs - 1
    col_idxs_0based = col_idxs - 1
    N_nodes_x = N_ele_x + 1
    N_nodes_y = N_ele_y + 1
    N_dof = N_nodes_x * N_nodes_y * 2
    
    K = sp.coo_matrix((value_K, (row_idxs_0based, col_idxs_0based)), shape=(N_dof, N_dof), dtype=np.float32).tocsr()
    M = sp.coo_matrix((value_M, (row_idxs_0based, col_idxs_0based)), shape=(N_dof, N_dof), dtype=np.float32).tocsr()
    
    # Apply threshold
    threshold = 1e-10
    K.data[np.abs(K.data) < threshold] = 0
    M.data[np.abs(M.data) < threshold] = 0
    K.eliminate_zeros()
    M.eliminate_zeros()
    
    print(f"   Step 10: Final matrices")
    print(f"     K: size = {K.shape}, nnz = {K.nnz}")
    print(f"     M: size = {M.shape}, nnz = {M.nnz}")
    
    # Save as sparse matrix format
    sp.save_npz(output_dir / 'step10_K.npz', K)
    sp.save_npz(output_dir / 'step10_M.npz', M)
    
    print()
    print("=" * 80)
    print(f"All intermediate steps saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

