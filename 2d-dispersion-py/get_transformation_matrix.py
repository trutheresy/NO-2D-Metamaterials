"""
Transformation matrix function.

This module contains the function for computing transformation matrices
for periodic boundary conditions, equivalent to MATLAB's get_transformation_matrix.m.
"""

import numpy as np
from scipy.sparse import csr_matrix


def get_transformation_matrix(wavevector, const, return_derivatives=False):
    """
    Compute transformation matrix for periodic boundary conditions.
    
    This function matches MATLAB's get_transformation_matrix.m exactly.
    
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
    r_x = np.array([const['a'], 0], dtype=np.float32)
    r_y = np.array([0, -const['a']], dtype=np.float32)
    r_corner = np.array([const['a'], -const['a']], dtype=np.float32)
    
    xphase = np.exp(1j * np.dot(wavevector, r_x)).astype(np.complex64)
    yphase = np.exp(1j * np.dot(wavevector, r_y)).astype(np.complex64)
    cornerphase = np.exp(1j * np.dot(wavevector, r_corner)).astype(np.complex64)
    
    # Compute derivatives if requested
    if return_derivatives:
        dxphasedwavevector = (1j * r_x * xphase).astype(np.complex64)
        dyphasedwavevector = (1j * r_y * yphase).astype(np.complex64)
        dcornerphasedwavevector = (1j * r_corner * cornerphase).astype(np.complex64)
    
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
    
    value_T = np.tile(phase_factors, 2).astype(np.complex64)  # Repeat for both x and y DOF
    
    # Calculate explicit dimensions
    # Full DOF: 2 * N_node^2 (all nodes with x and y DOF)
    N_dof_full = 2 * N_node * N_node
    # Reduced DOF: 2 * (N_node-1)^2 (after periodic BC reduction)
    N_dof_reduced = 2 * (N_node - 1) * (N_node - 1)
    
    # Explicitly set shape to ensure correct dimensions
    T = csr_matrix((value_T, (row_idxs, col_idxs)), 
                   shape=(N_dof_full, N_dof_reduced), 
                   dtype=np.complex64)
    
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
            
            value_dTdwavevector = np.tile(dphase_factors, 2).astype(np.complex64)
            # row_idxs and col_idxs are already converted to 0-based above
            # Use same explicit shape as T
            dTdwavevector.append(
                csr_matrix((value_dTdwavevector, (row_idxs, col_idxs)), 
                          shape=(N_dof_full, N_dof_reduced), 
                          dtype=np.complex64))
        
        return T, dTdwavevector
    
    return T

