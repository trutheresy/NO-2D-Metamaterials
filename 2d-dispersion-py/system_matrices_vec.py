"""
Vectorized system matrices functions for improved performance.

These functions provide vectorized versions of system matrix assembly,
equivalent to the MATLAB get_system_matrices_VEC function.
"""

import numpy as np
from scipy.sparse import coo_matrix
from elements_vec import get_element_stiffness_VEC, get_element_mass_VEC


def get_system_matrices_VEC(const):
    """
    Vectorized system matrices assembly.
    
    This is the exact translation of the MATLAB get_system_matrices_VEC function.
    
    Parameters
    ----------
    const : dict
        Constants structure containing design, material properties, etc.
        
    Returns
    -------
    K : scipy.sparse matrix
        Global stiffness matrix
    M : scipy.sparse matrix  
        Global mass matrix
    """
    
    # Total number of elements along x and y directions
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_ele_x = N_pix[0] * const['N_ele']
        N_ele_y = N_pix[1] * const['N_ele']
    else:
        N_ele_x = N_pix * const['N_ele']
        N_ele_y = N_pix * const['N_ele']
    
    # Replicate design for element-level resolution
    # MATLAB: const.design = repelem(const.design, const.N_ele, const.N_ele, 1)
    design_expanded = np.repeat(
        np.repeat(const['design'], const['N_ele'], axis=0), 
        const['N_ele'], axis=1
    )
    
    # Extract material properties based on design scale
    if const['design_scale'] == 'linear':
        E = (const['E_min'] + design_expanded[:, :, 0] * (const['E_max'] - const['E_min'])).T
        nu = (const['poisson_min'] + design_expanded[:, :, 2] * (const['poisson_max'] - const['poisson_min'])).T
        t = const['t']
        rho = (const['rho_min'] + design_expanded[:, :, 1] * (const['rho_max'] - const['rho_min'])).T
    elif const['design_scale'] == 'log':
        E = np.exp(design_expanded[:, :, 0]).T
        nu = (const['poisson_min'] + design_expanded[:, :, 2] * (const['poisson_max'] - const['poisson_min'])).T
        t = const['t']
        rho = np.exp(design_expanded[:, :, 1]).T
    else:
        raise ValueError("const['design_scale'] not recognized as 'log' or 'linear'")
    
    # Node numbering in a grid
    nodenrs = np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1).reshape(1 + N_ele_y, 1 + N_ele_x)
    
    # Element degree of freedom (in a vector) (global labeling)
    edofVec = (2 * nodenrs[0:-1, 0:-1] - 1).flatten()
    
    # Element degree of freedom matrix (exact MATLAB translation)
    # MATLAB: [2*(N_ele_y+1)+[0 1 2 3] 2 3 0 1]
    offset_array = np.concatenate([
        2*(N_ele_y+1) + np.array([0, 1, 2, 3]),  # First 4 elements
        np.array([2, 3, 0, 1])                    # Last 4 elements
    ])
    edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(
        offset_array,
        (N_ele_x * N_ele_y, 1)
    )
    
    # Row and column indices for sparse matrix assembly
    row_idxs = np.tile(edofMat, (8, 1)).T.flatten()
    col_idxs = np.tile(edofMat, (1, 8)).flatten()
    
    # Get element matrices (vectorized)
    AllLEle = get_element_stiffness_VEC(E.flatten(), nu.flatten(), t)
    AllLMat = get_element_mass_VEC(rho.flatten(), t, const)
    
    # Flatten element matrices for sparse assembly
    value_K = AllLEle.flatten()
    value_M = AllLMat.flatten()
    
    # Create sparse matrices
    K = coo_matrix((value_K, (row_idxs, col_idxs))).tocsr()
    M = coo_matrix((value_M, (row_idxs, col_idxs))).tocsr()
    
    return K, M


def get_system_matrices_VEC_simplified(const):
    """
    Simplified vectorized system matrices assembly.
    
    This function would be used when const.isUseSecondImprovement is True.
    For now, it calls the regular vectorized version.
    
    Parameters
    ----------
    const : dict
        Constants structure containing design, material properties, etc.
        
    Returns
    -------
    K : scipy.sparse matrix
        Global stiffness matrix
    M : scipy.sparse matrix  
        Global mass matrix
    """
    # For now, use the regular vectorized version
    # This could be optimized further if needed
    return get_system_matrices_VEC(const)
