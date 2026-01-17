"""
System matrix assembly and transformation functions.

This module contains functions for assembling global system matrices
and computing transformation matrices for periodic boundary conditions.
"""

import numpy as np
from scipy.sparse import csr_matrix
from elements import get_element_stiffness, get_element_mass, get_pixel_properties
from get_global_idxs import get_global_idxs


def get_system_matrices(const, use_vectorized=False, return_sensitivities=False):
    """
    Assemble global stiffness and mass matrices.
    
    Parameters
    ----------
    const : dict
        Constants structure containing system parameters
    use_vectorized : bool, optional
        Whether to use vectorized assembly (default: False)
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
    # Use float64 to match MATLAB precision
    value_K = np.zeros(N_dof_per_element**2 * total_elements, dtype=np.float64)
    value_M = np.zeros(N_dof_per_element**2 * total_elements, dtype=np.float64)
    
    if return_sensitivities:
        xpix_idxs = np.zeros(N_dof_per_element**2 * total_elements, dtype=int)
        ypix_idxs = np.zeros(N_dof_per_element**2 * total_elements, dtype=int)
        # Use float64 to match MATLAB precision
        value_dKddesign = np.zeros(N_dof_per_element**2 * total_elements, dtype=np.float64)
        value_dMddesign = np.zeros(N_dof_per_element**2 * total_elements, dtype=np.float64)
    
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
            value_K[start_idx:end_idx] = k_ele.flatten().astype(np.float64)  # Use float64 to match MATLAB
            value_M[start_idx:end_idx] = m_ele.flatten().astype(np.float64)  # Use float64 to match MATLAB
            
            if return_sensitivities:
                from .elements import get_element_stiffness_sensitivity, get_element_mass_sensitivity
                dk_eleddesign = get_element_stiffness_sensitivity(E, nu, t, const)
                dm_eleddesign = get_element_mass_sensitivity(rho, t, const)
                value_dKddesign[start_idx:end_idx] = dk_eleddesign.flatten()
                value_dMddesign[start_idx:end_idx] = dm_eleddesign.flatten()
                xpix_idxs[start_idx:end_idx] = pix_idx_x
                ypix_idxs[start_idx:end_idx] = pix_idx_y
            
            element_idx += 1
    
    # Assemble sparse matrices
    # Use float64 to match MATLAB precision
    K = csr_matrix((value_K, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float64)
    M = csr_matrix((value_M, (row_idxs, col_idxs)), shape=(N_dof, N_dof), dtype=np.float64)
    
    if return_sensitivities:
        dKddesign = []
        dMddesign = []
        for pix_idx_x in range(N_pix_x):
            dKddesign.append([])
            dMddesign.append([])
            for pix_idx_y in range(N_pix_y):
                mask = (xpix_idxs == pix_idx_x) & (ypix_idxs == pix_idx_y)
                # Use float64 to match MATLAB precision
                dKddesign[pix_idx_x].append(
                    csr_matrix((value_dKddesign[mask], 
                              (row_idxs[mask], col_idxs[mask])), 
                             shape=(N_dof, N_dof), dtype=np.float64))
                dMddesign[pix_idx_x].append(
                    csr_matrix((value_dMddesign[mask], 
                              (row_idxs[mask], col_idxs[mask])), 
                             shape=(N_dof, N_dof), dtype=np.float64))
        
        return K, M, dKddesign, dMddesign
    
    return K, M


def get_transformation_matrix(wavevector, const, return_derivatives=False):
    """
    Compute transformation matrix for periodic boundary conditions.
    
    Parameters
    ----------
    wavevector : array_like
        2D wavevector [kx, ky]
    const : dict
        Constants structure containing system parameters
    return_derivatives : bool, optional
        Whether to return derivatives with respect to wavevector (default: False)
        
    Returns
    -------
    T : scipy.sparse matrix
        Transformation matrix
    dTdwavevector : list of scipy.sparse matrices, optional
        Derivatives of transformation matrix with respect to wavevector components
    """
    
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_val = N_pix[0]
    else:
        N_pix_val = N_pix
    
    N_node = const['N_ele'] * N_pix_val + 1
    
    # Compute phase factors
    # Use float64 to match MATLAB precision
    r_x = np.array([const['a'], 0], dtype=np.float64)
    r_y = np.array([0, -const['a']], dtype=np.float64)
    r_corner = np.array([const['a'], -const['a']], dtype=np.float64)
    
    xphase = np.exp(1j * np.dot(wavevector, r_x)).astype(np.complex128)
    yphase = np.exp(1j * np.dot(wavevector, r_y)).astype(np.complex128)
    cornerphase = np.exp(1j * np.dot(wavevector, r_corner)).astype(np.complex128)
    
    # Compute derivatives if requested
    if return_derivatives:
        dxphasedwavevector = (1j * r_x * xphase).astype(np.complex128)
        dyphasedwavevector = (1j * r_y * yphase).astype(np.complex128)
        dcornerphasedwavevector = (1j * r_corner * cornerphase).astype(np.complex128)
    
    # Generate node indices (exact MATLAB translation using 1-based indexing)
    # MATLAB: reshape(meshgrid(1:(N_node-1),1:(N_node-1)),[],1)'
    # Note: MATLAB meshgrid uses matrix indexing, numpy defaults to Cartesian
    temp_x, temp_y = np.meshgrid(np.arange(1, N_node), np.arange(1, N_node), indexing='ij')
    node_idx_x = np.concatenate([
        temp_x.flatten(order='F'),  # Flatten in column-major (Fortran) order like MATLAB
        np.full(N_node - 1, N_node),  # Right boundary
        np.arange(1, N_node),  # Top boundary
        [N_node]  # Corner
    ])
    node_idx_y = np.concatenate([
        temp_y.flatten(order='F'),  # Flatten in column-major (Fortran) order like MATLAB
        np.arange(1, N_node),  # Right boundary
        np.full(N_node - 1, N_node),  # Top boundary
        [N_node]  # Corner
    ])
    
    # Convert to global node and DOF indices using MATLAB 1-based formulas
    global_node_idx = (node_idx_y - 1) * N_node + node_idx_x
    global_dof_idxs = np.concatenate([
        2 * global_node_idx - 1,  # x-displacements
        2 * global_node_idx       # y-displacements
    ])
    
    # Define slices for different node groups
    # MATLAB uses 1-based indexing:
    # unch_idxs = 1:((N_node-1)^2)
    # x_idxs = (((N_node-1)^2) + 1):(((N_node-1)^2) + 1 + N_node - 2)
    # y_idxs = (((N_node-1)^2) + N_node):(((N_node-1)^2) + N_node + N_node - 2)
    n_interior = (N_node - 1)**2
    n_right = N_node - 1
    n_top = N_node - 1
    
    unch_idxs = slice(0, n_interior)  # 0 to (N_node-1)^2 - 1 (0-based)
    # MATLAB: ((N_node-1)^2) + 1 to ((N_node-1)^2) + 1 + N_node - 2 (1-based, inclusive)
    # For N_node=6: 26:30 (1-based) = [25:30) (0-based, Python slice excludes end)
    # Convert to 0-based: (N_node-1)^2 to (N_node-1)^2 + N_node - 1
    x_idxs = slice(n_interior, n_interior + n_right)  # Python slice excludes endpoint, so this matches MATLAB's inclusive range
    # MATLAB: ((N_node-1)^2) + N_node to ((N_node-1)^2) + N_node + N_node - 2 (1-based, inclusive)
    # For N_node=6: 31:35 (1-based) = [30:35) (0-based, Python slice excludes end)
    # Convert to 0-based: (N_node-1)^2 + N_node - 1 to (N_node-1)^2 + 2*N_node - 2
    y_idxs = slice(n_interior + n_right, n_interior + n_right + n_top)
    
    # Reduced global node indices (MATLAB formulas)
    reduced_global_node_idx = np.concatenate([
        (node_idx_y[unch_idxs] - 1) * (N_node - 1) + node_idx_x[unch_idxs],
        (node_idx_y[x_idxs] - 1) * (N_node - 1) + node_idx_x[x_idxs] - (N_node - 1),
        node_idx_x[y_idxs],
        [1]  # Corner node
    ])
    
    reduced_global_dof_idxs = np.concatenate([
        2 * reduced_global_node_idx - 1,  # x-displacements
        2 * reduced_global_node_idx       # y-displacements
    ])
    
    # Build transformation matrix - convert all indices to 0-based for Python
    row_idxs = (global_dof_idxs - 1).astype(int)
    col_idxs = (reduced_global_dof_idxs - 1).astype(int)
    
    
    # Phase factors for each node type
    phase_factors = np.concatenate([
        np.ones((N_node - 1)**2),  # Interior nodes
        np.full(N_node - 1, xphase),  # Right boundary nodes
        np.full(N_node - 1, yphase),  # Top boundary nodes
        [cornerphase]  # Corner node
    ])
    
    value_T = np.tile(phase_factors, 2).astype(np.complex128)  # Repeat for both x and y DOF - use complex128 to match MATLAB
    
    # Calculate explicit dimensions
    # Full DOF: 2 * N_node^2 (all nodes with x and y DOF)
    N_dof_full = 2 * N_node * N_node
    # Reduced DOF: 2 * (N_node-1)^2 (after periodic BC reduction)
    N_dof_reduced = 2 * (N_node - 1) * (N_node - 1)
    
    # Explicitly set shape to ensure correct dimensions
    # Use complex128 to match MATLAB precision
    T = csr_matrix((value_T, (row_idxs, col_idxs)), 
                   shape=(N_dof_full, N_dof_reduced), 
                   dtype=np.complex128)
    
    if return_derivatives:
        dTdwavevector = []
        for wv_comp_idx in range(2):
            if wv_comp_idx == 0:  # x-component
                dphase_factors = np.concatenate([
                    np.zeros((N_node - 1)**2),
                    np.full(N_node - 1, dxphasedwavevector[0]),
                    np.full(N_node - 1, dyphasedwavevector[0]),
                    [dcornerphasedwavevector[0]]
                ])
            else:  # y-component
                dphase_factors = np.concatenate([
                    np.zeros((N_node - 1)**2),
                    np.full(N_node - 1, dxphasedwavevector[1]),
                    np.full(N_node - 1, dyphasedwavevector[1]),
                    [dcornerphasedwavevector[1]]
                ])
            
            value_dTdwavevector = np.tile(dphase_factors, 2).astype(np.complex128)  # Use complex128 to match MATLAB
            # row_idxs and col_idxs are already converted to 0-based above
            # Use same explicit shape as T
            dTdwavevector.append(
                csr_matrix((value_dTdwavevector, (row_idxs, col_idxs)), 
                          shape=(N_dof_full, N_dof_reduced), 
                          dtype=np.complex128))  # Use complex128 to match MATLAB
        
        return T, dTdwavevector
    
    return T

