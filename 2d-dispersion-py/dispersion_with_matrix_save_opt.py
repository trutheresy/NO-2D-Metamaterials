"""
Dispersion calculation with matrix saving optimization.

This is the exact translation of the MATLAB dispersion_with_matrix_save_opt.m function,
which is the actual function used by the generation script.
"""

import numpy as np
from scipy.sparse.linalg import eigs
from system_matrices import get_system_matrices
from system_matrices_vec import get_system_matrices_VEC, get_system_matrices_VEC_simplified
from system_matrices import get_transformation_matrix


def dispersion_with_matrix_save_opt(const, wavevectors):
    """
    Calculate dispersion with optional matrix saving.
    
    This is the exact translation of the MATLAB function with the same name.
    
    Parameters
    ----------
    const : dict
        Constants structure containing system parameters
    wavevectors : array_like
        Array of wavevectors (N_k x 2)
        
    Returns
    -------
    wv : array_like
        Wavevectors (same as input)
    fr : array_like
        Frequencies (N_k x N_eig)
    ev : array_like or None
        Eigenvectors if isSaveEigenvectors is True, otherwise None
    mesh : array_like or None
        Mesh information if isSaveMesh is True, otherwise None
    K_out : scipy.sparse matrix
        Global stiffness matrix
    M_out : scipy.sparse matrix
        Global mass matrix
    T_out : list
        List of transformation matrices for each wavevector
    """
    
    # Total number of degrees of freedom
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_val = N_pix[0]
    else:
        N_pix_val = N_pix
    
    N_dof = ((const['N_ele'] * N_pix_val)**2) * 2
    
    # Get mesh if requested
    if const.get('isSaveMesh', False):
        mesh = get_mesh(const)
    else:
        mesh = None
    
    # Initialize frequency array
    n_wavevectors = wavevectors.shape[0]
    fr = np.zeros((n_wavevectors, const['N_eig']))
    
    # Initialize eigenvector array if requested
    if const.get('isSaveEigenvectors', False):
        ev = np.zeros((N_dof, n_wavevectors, const['N_eig']), dtype=complex)
    else:
        ev = None
    
    # Get system matrices based on improvement flags
    if const.get('isUseSecondImprovement', False):
        K, M = get_system_matrices_VEC_simplified(const)
    elif const.get('isUseImprovement', True):
        K, M = get_system_matrices_VEC(const)
    else:
        K, M = get_system_matrices(const)
    
    # Initialize storage for matrices if requested
    if const.get('isSaveKandM', False):
        M_out = M
        K_out = K
        T_out = [None] * n_wavevectors  # To be populated during loop
    else:
        K_out = None
        M_out = None
        T_out = None
    
    # Process each wavevector
    for k_idx in range(n_wavevectors):
        wavevector = wavevectors[k_idx, :]
        T = get_transformation_matrix(wavevector, const)
        
        # Transform matrices to reduced space
        Kr = T.conj().T @ K @ T
        Mr = T.conj().T @ M @ T
        
        # Store transformation matrix if requested
        if const.get('isSaveKandM', False):
            T_out[k_idx] = T
        
        if not const.get('isUseGPU', False):
            # Solve generalized eigenvalue problem
            eig_vals, eig_vecs = eigs(Kr, M=Mr, k=const['N_eig'], 
                                     sigma=const['sigma_eig'])
            
            # Sort eigenvalues and eigenvectors
            idxs = np.argsort(eig_vals)
            eig_vals = eig_vals[idxs]
            eig_vecs = eig_vecs[:, idxs]
            
            # Store eigenvectors if requested
            if const.get('isSaveEigenvectors', False):
                # Normalize eigenvectors (exact MATLAB translation)
                # MATLAB: ev(:,k_idx,:) = (eig_vecs./vecnorm(eig_vecs,2,1)).*exp(-1i*angle(eig_vecs(1,:)))
                norms = np.linalg.norm(eig_vecs, axis=0)
                eig_vecs_normalized = eig_vecs / norms
                phase_align = np.exp(-1j * np.angle(eig_vecs[0, :]))
                ev[:, k_idx, :] = (eig_vecs_normalized * phase_align).T
            
            # Convert to frequencies (exact MATLAB translation)
            fr[k_idx, :] = np.sqrt(np.real(eig_vals)) / (2 * np.pi)
            
        elif const.get('isUseGPU', False):
            raise NotImplementedError('GPU use is not currently developed')
    
    return wavevectors, fr, ev, mesh, K_out, M_out, T_out


def get_mesh(const):
    """
    Get mesh information (placeholder function).
    
    This function would return mesh information if needed.
    For now, it returns None as mesh saving is typically disabled.
    """
    # This would need to be implemented if mesh saving is required
    return None
