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
        # Use float64 for intermediate calculation to avoid overflow with large E_max
        design_ch0 = design_expanded[:, :, 0].astype(np.float64)
        design_ch1 = design_expanded[:, :, 1].astype(np.float64)
        design_ch2 = design_expanded[:, :, 2].astype(np.float64)
        E = (const['E_min'] + design_ch0 * (const['E_max'] - const['E_min'])).T.astype(np.float32)
        nu = (const['poisson_min'] + design_ch2 * (const['poisson_max'] - const['poisson_min'])).T.astype(np.float32)
        t = const['t']
        rho = (const['rho_min'] + design_ch1 * (const['rho_max'] - const['rho_min'])).T.astype(np.float32)
    elif const['design_scale'] == 'log':
        E = np.exp(design_expanded[:, :, 0]).T
        nu = (const['poisson_min'] + design_expanded[:, :, 2] * (const['poisson_max'] - const['poisson_min'])).T
        t = const['t']
        rho = np.exp(design_expanded[:, :, 1]).T
    else:
        raise ValueError("const['design_scale'] not recognized as 'log' or 'linear'")
    
    # Node numbering in a grid
    # MATLAB: reshape(1:(1+N_ele_x)*(1+N_ele_y),1+N_ele_y,1+N_ele_x)
    # MATLAB reshape uses column-major (Fortran) order
    nodenrs = np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1).reshape(1 + N_ele_y, 1 + N_ele_x, order='F')
    
    # Element degree of freedom (in a vector) (global labeling)
    # MATLAB: reshape(2*nodenrs(1:end-1,1:end-1)-1,N_ele_x*N_ele_y,1)
    # MATLAB reshape to column vector is column-major
    edofVec = (2 * nodenrs[0:-1, 0:-1] - 1).reshape(N_ele_x * N_ele_y, 1, order='F').flatten()
    
    # Element degree of freedom matrix (exact MATLAB translation)
    # MATLAB: repmat(edofVec,1,8)+repmat([2*(N_ele_y+1)+[0 1 2 3] 2 3 0 1],N_ele_x*N_ele_y,1)
    offset_array = np.concatenate([
        2*(N_ele_y+1) + np.array([0, 1, 2, 3]),  # First 4 elements
        np.array([2, 3, 0, 1])                    # Last 4 elements
    ])
    edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(
        offset_array,
        (N_ele_x * N_ele_y, 1)
    )
    
    # Row and column indices for sparse matrix assembly
    # MATLAB: reshape(kron(edofMat,ones(8,1))',64*N_ele_x*N_ele_y,1)
    # MATLAB: reshape(kron(edofMat,ones(1,8))',64*N_ele_x*N_ele_y,1)
    # MATLAB's kron+transpose+reshape produces:
    #   row_idxs: DOFs cycling (matches kron(edofMat,ones(8,1))' after reshape)
    #   col_idxs: each DOF repeated 8 times (matches kron(edofMat,ones(1,8))' after reshape)
    # This indicates element matrix is stored COLUMN-MAJOR in MATLAB
    # Replicate MATLAB's exact behavior:
    row_idxs_mat = np.kron(edofMat, np.ones((8, 1)))
    row_idxs = row_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()
    
    col_idxs_mat = np.kron(edofMat, np.ones((1, 8)))
    col_idxs = col_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()
    
    # Get element matrices (vectorized)
    # MATLAB: E(:), nu(:), rho(:) use column-major (F-order) flattening
    # Python: E.flatten() uses row-major (C-order) by default, so use order='F' to match MATLAB
    AllLEle = get_element_stiffness_VEC(E.flatten(order='F'), nu.flatten(order='F'), t)
    AllLMat = get_element_mass_VEC(rho.flatten(order='F'), t, const)
    
    # Flatten element matrices for sparse assembly
    # MATLAB: AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)' then value_K = AllLEle(:)
    # MATLAB's get_element_stiffness_VEC returns (N_ele, 64) where each row is a flattened 8x8 matrix
    # The template is written row-by-row (64-element row vector), so flattening is C-order (row-major)
    # After transpose: (64, N_ele), then flatten with (:) uses column-major (F-order) to interleave elements
    N_ele_total = N_ele_x * N_ele_y
    # Reshape each 8x8 matrix to 64 with C-order (row-major) to match MATLAB's template order
    AllLEle_2d = AllLEle.reshape(N_ele_total, 64)  # (N_ele, 64) - C-order reshape preserves row-major
    AllLEle_transposed = AllLEle_2d.T  # (64, N_ele) - each column is one element's flattened matrix
    value_K = AllLEle_transposed.flatten(order='F').astype(np.float32)  # Column-major flatten (interleaves elements)
    
    AllLMat_2d = AllLMat.reshape(N_ele_total, 64)
    AllLMat_transposed = AllLMat_2d.T
    value_M = AllLMat_transposed.flatten(order='F').astype(np.float32)
    
    # Convert 1-based indices to 0-based for Python
    row_idxs = row_idxs - 1
    col_idxs = col_idxs - 1
    
    # Calculate explicit matrix dimensions
    N_nodes_x = N_ele_x + 1
    N_nodes_y = N_ele_y + 1
    N_dof = N_nodes_x * N_nodes_y * 2
    
    # Create sparse matrices with explicit shape (duplicate entries will be summed automatically)
    K = coo_matrix((value_K, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32).tocsr()
    M = coo_matrix((value_M, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32).tocsr()
    
    # Filter out very small values AFTER assembly to match MATLAB's sparse() behavior
    # MATLAB's sparse() automatically removes entries that are effectively zero after summing
    # Use a threshold of 1e-10 to match MATLAB's behavior
    threshold = 1e-10
    K.data[np.abs(K.data) < threshold] = 0
    M.data[np.abs(M.data) < threshold] = 0
    K.eliminate_zeros()
    M.eliminate_zeros()
    
    return K, M


def get_system_matrices_VEC_simplified(const):
    """
    Simplified vectorized system matrices assembly.
    
    This function matches the MATLAB get_system_matrices_VEC_simplified function.
    It uses a different edofMat offset pattern compared to the regular VEC version.
    
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
        # Use float64 for intermediate calculation to avoid overflow with large E_max
        design_ch0 = design_expanded[:, :, 0].astype(np.float64)
        design_ch1 = design_expanded[:, :, 1].astype(np.float64)
        design_ch2 = design_expanded[:, :, 2].astype(np.float64)
        E = (const['E_min'] + design_ch0 * (const['E_max'] - const['E_min'])).T.astype(np.float32)
        nu = (const['poisson_min'] + design_ch2 * (const['poisson_max'] - const['poisson_min'])).T.astype(np.float32)
        t = const['t']
        rho = (const['rho_min'] + design_ch1 * (const['rho_max'] - const['rho_min'])).T.astype(np.float32)
    elif const['design_scale'] == 'log':
        E = np.exp(design_expanded[:, :, 0]).T
        nu = (const['poisson_min'] + design_expanded[:, :, 2] * (const['poisson_max'] - const['poisson_min'])).T
        t = const['t']
        rho = np.exp(design_expanded[:, :, 1]).T
    else:
        raise ValueError("const['design_scale'] not recognized as 'log' or 'linear'")
    
    # Node numbering in a grid
    # MATLAB: reshape(1:(1+N_ele_x)*(1+N_ele_y),1+N_ele_y,1+N_ele_x)
    # MATLAB reshape uses column-major (Fortran) order
    nodenrs = np.arange(1, (1 + N_ele_x) * (1 + N_ele_y) + 1).reshape(1 + N_ele_y, 1 + N_ele_x, order='F')
    
    # Element degree of freedom (in a vector) (global labeling)
    # MATLAB: reshape(2*nodenrs(1:end-1,1:end-1)-1,N_ele_x*N_ele_y,1)
    # MATLAB reshape to column vector is column-major
    edofVec = (2 * nodenrs[0:-1, 0:-1] - 1).reshape(N_ele_x * N_ele_y, 1, order='F').flatten()
    
    # Element degree of freedom matrix (SIMPLIFIED VERSION - different from regular VEC)
    # MATLAB: repmat(edofVec,1,8)+repmat([2 3 2*(N_ele_x+1)+[2 3 0 1] 0 1],N_ele_x*N_ele_y,1)
    # NOTE: Simplified version uses N_ele_x instead of N_ele_y, and different offset pattern
    offset_array = np.concatenate([
        np.array([2, 3]),  # First 2 elements
        2*(N_ele_x+1) + np.array([2, 3, 0, 1]),  # Next 4 elements
        np.array([0, 1])   # Last 2 elements
    ])
    edofMat = np.tile(edofVec.reshape(-1, 1), (1, 8)) + np.tile(
        offset_array,
        (N_ele_x * N_ele_y, 1)
    )
    
    # Row and column indices for sparse matrix assembly
    # MATLAB: reshape(kron(edofMat,ones(8,1))',64*N_ele_x*N_ele_y,1)
    # MATLAB: reshape(kron(edofMat,ones(1,8))',64*N_ele_x*N_ele_y,1)
    row_idxs_mat = np.kron(edofMat, np.ones((8, 1)))
    row_idxs = row_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()
    
    col_idxs_mat = np.kron(edofMat, np.ones((1, 8)))
    col_idxs = col_idxs_mat.T.reshape(64 * N_ele_x * N_ele_y, 1, order='F').flatten()
    
    # Get element matrices (vectorized)
    # MATLAB: E(:), nu(:), rho(:) use column-major (F-order) flattening
    # Python: E.flatten() uses row-major (C-order) by default, so use order='F' to match MATLAB
    AllLEle = get_element_stiffness_VEC(E.flatten(order='F'), nu.flatten(order='F'), t)
    AllLMat = get_element_mass_VEC(rho.flatten(order='F'), t, const)
    
    # Flatten element matrices for sparse assembly
    # MATLAB: AllLEle = get_element_stiffness_VEC(E(:),nu(:),t)' then value_K = AllLEle(:)
    # MATLAB's get_element_stiffness_VEC returns (N_ele, 64) where each row is a flattened 8x8 matrix
    # The template is written row-by-row (64-element row vector), so flattening is C-order (row-major)
    # After transpose: (64, N_ele), then flatten with (:) uses column-major (F-order) to interleave elements
    N_ele_total = N_ele_x * N_ele_y
    # Reshape each 8x8 matrix to 64 with C-order (row-major) to match MATLAB's template order
    AllLEle_2d = AllLEle.reshape(N_ele_total, 64)  # (N_ele, 64) - C-order reshape preserves row-major
    AllLEle_transposed = AllLEle_2d.T  # (64, N_ele) - each column is one element's flattened matrix
    value_K = AllLEle_transposed.flatten(order='F').astype(np.float32)  # Column-major flatten (interleaves elements)
    
    AllLMat_2d = AllLMat.reshape(N_ele_total, 64)
    AllLMat_transposed = AllLMat_2d.T
    value_M = AllLMat_transposed.flatten(order='F').astype(np.float32)
    
    # Convert 1-based indices to 0-based for Python
    row_idxs = row_idxs - 1
    col_idxs = col_idxs - 1
    
    # Calculate explicit matrix dimensions
    N_nodes_x = N_ele_x + 1
    N_nodes_y = N_ele_y + 1
    N_dof = N_nodes_x * N_nodes_y * 2
    
    # Create sparse matrices with explicit shape (duplicate entries will be summed automatically)
    K = coo_matrix((value_K, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32).tocsr()
    M = coo_matrix((value_M, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float32).tocsr()
    
    # Filter out very small values AFTER assembly to match MATLAB's sparse() behavior
    # MATLAB's sparse() automatically removes entries that are effectively zero after summing
    # Use a threshold of 1e-10 to match MATLAB's behavior
    threshold = 1e-10
    K.data[np.abs(K.data) < threshold] = 0
    M.data[np.abs(M.data) < threshold] = 0
    K.eliminate_zeros()
    M.eliminate_zeros()
    
    return K, M
