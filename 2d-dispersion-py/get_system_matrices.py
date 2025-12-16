"""
System matrix assembly function.

This module contains the function for assembling global system matrices,
equivalent to MATLAB's get_system_matrices.m.
"""

import numpy as np
from scipy.sparse import csr_matrix
from get_element_stiffness import get_element_stiffness
from get_element_mass import get_element_mass
from get_pixel_properties import get_pixel_properties
from get_global_idxs import get_global_idxs


def get_system_matrices(const, return_sensitivities=False):
    """
    Assemble global stiffness and mass matrices.
    
    This function matches MATLAB's get_system_matrices.m exactly.
    
    Parameters
    ----------
    const : dict
        Constants structure containing system parameters
    return_sensitivities : bool, optional
        Whether to return design sensitivities (default: False)
        
    Returns
    -------
    K : scipy.sparse matrix
        Global stiffness matrix
    M : scipy.sparse matrix
        Global mass matrix
    dKddesign : list of lists, optional
        Design sensitivities of stiffness matrix (if return_sensitivities=True)
    dMddesign : list of lists, optional
        Design sensitivities of mass matrix (if return_sensitivities=True)
    """
    
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_x, N_pix_y = N_pix[0], N_pix[1]
    else:
        N_pix_x = N_pix_y = N_pix
    
    N_ele_x = N_pix_x * const['N_ele']
    N_ele_y = N_pix_y * const['N_ele']
    N_nodes_x = N_ele_x + 1
    N_nodes_y = N_ele_y + 1
    N_dof = N_nodes_x * N_nodes_y * 2  # 2 DOF per node
    
    N_dof_per_element = 8
    total_elements = (const['N_ele'] * N_pix_x) * (const['N_ele'] * N_pix_y)
    
    # Preallocate arrays for sparse matrix construction
    row_idxs = np.zeros(N_dof_per_element**2 * total_elements, dtype=int)
    col_idxs = np.zeros(N_dof_per_element**2 * total_elements, dtype=int)
    value_K = np.zeros(N_dof_per_element**2 * total_elements, dtype=np.float32)
    value_M = np.zeros(N_dof_per_element**2 * total_elements, dtype=np.float32)
    
    if return_sensitivities:
        xpix_idxs = np.zeros(N_dof_per_element**2 * total_elements, dtype=int)
        ypix_idxs = np.zeros(N_dof_per_element**2 * total_elements, dtype=int)
        value_dKddesign = np.zeros(N_dof_per_element**2 * total_elements, dtype=np.float32)
        value_dMddesign = np.zeros(N_dof_per_element**2 * total_elements, dtype=np.float32)
    
    element_idx = 0
    for ele_idx_x in range(1, N_ele_x + 1):
        for ele_idx_y in range(1, N_ele_y + 1):
            pix_idx_x = int(np.ceil(ele_idx_x / const['N_ele'])) - 1
            pix_idx_y = int(np.ceil(ele_idx_y / const['N_ele'])) - 1
            
            # Get material properties
            E, nu, t, rho = get_pixel_properties(pix_idx_x, pix_idx_y, const)
            
            # Get element matrices
            k_ele = get_element_stiffness(E, nu, t, const)
            m_ele = get_element_mass(rho, t, const)
            
            # Get global DOF indices
            global_idxs = get_global_idxs(ele_idx_x, ele_idx_y, const)
            
            # Fill sparse matrix arrays
            start_idx = element_idx * N_dof_per_element**2
            end_idx = (element_idx + 1) * N_dof_per_element**2
            
            # Create index matrices
            global_idxs_mat = np.tile(global_idxs, (N_dof_per_element, 1))
            
            row_idxs[start_idx:end_idx] = global_idxs_mat.flatten()
            col_idxs[start_idx:end_idx] = global_idxs_mat.T.flatten()
            value_K[start_idx:end_idx] = k_ele.flatten().astype(np.float32)
            value_M[start_idx:end_idx] = m_ele.flatten().astype(np.float32)
            
            if return_sensitivities:
                from elements import get_element_stiffness_sensitivity, get_element_mass_sensitivity
                dk_eleddesign = get_element_stiffness_sensitivity(E, nu, t, const)
                dm_eleddesign = get_element_mass_sensitivity(rho, t, const)
                value_dKddesign[start_idx:end_idx] = dk_eleddesign.flatten()
                value_dMddesign[start_idx:end_idx] = dm_eleddesign.flatten()
                xpix_idxs[start_idx:end_idx] = pix_idx_x
                ypix_idxs[start_idx:end_idx] = pix_idx_y
            
            element_idx += 1
    
    # Assemble sparse matrices
    K = csr_matrix((value_K, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32)
    M = csr_matrix((value_M, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32)
    
    if return_sensitivities:
        dKddesign = []
        dMddesign = []
        for pix_idx_x in range(N_pix_x):
            dKddesign.append([])
            dMddesign.append([])
            for pix_idx_y in range(N_pix_y):
                mask = (xpix_idxs == pix_idx_x) & (ypix_idxs == pix_idx_y)
                dKddesign[pix_idx_x].append(
                    csr_matrix((value_dKddesign[mask], 
                              (row_idxs[mask], col_idxs[mask])), 
                             shape=(N_dof, N_dof), dtype=np.float32))
                dMddesign[pix_idx_x].append(
                    csr_matrix((value_dMddesign[mask], 
                              (row_idxs[mask], col_idxs[mask])), 
                             shape=(N_dof, N_dof), dtype=np.float32))
        
        return K, M, dKddesign, dMddesign
    
    return K, M

