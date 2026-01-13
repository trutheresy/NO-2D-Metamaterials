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
    fr = np.zeros((n_wavevectors, const['N_eig']), dtype=np.float32)
    
    # Initialize eigenvector array if requested (matching Han's eigenvector_dtype support)
    # Get transformation matrix to determine reduced DOF size
    if const.get('isSaveEigenvectors', False):
        # For first wavevector to determine dimensions
        T_temp = get_transformation_matrix(wavevectors[0, :], const)
        N_dof_reduced = T_temp.shape[1]
        eigenvector_dtype = const.get('eigenvector_dtype', 'double')
        if eigenvector_dtype == 'single':
            ev = np.zeros((N_dof_reduced, n_wavevectors, const['N_eig']), dtype=np.complex64)
        elif eigenvector_dtype == 'double':
            ev = np.zeros((N_dof_reduced, n_wavevectors, const['N_eig']), dtype=np.complex128)
        else:
            raise ValueError(f"eigenvector_dtype must be 'single' or 'double', got '{eigenvector_dtype}'")
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
        T = get_transformation_matrix(wavevector.astype(np.float32), const)
        
        # Transform matrices to reduced space
        Kr = (T.conj().T @ K @ T).astype(np.complex64)
        Mr = (T.conj().T @ M @ T).astype(np.complex64)
        
        # Store transformation matrix if requested
        if const.get('isSaveKandM', False):
            T_out[k_idx] = T
        
        if not const.get('isUseGPU', False):
            # Solve generalized eigenvalue problem
            # For small problems or when finding smallest eigenvalues, use full eig for accuracy
            # This is especially important for rigid body modes at k=0
            N_dof_reduced = Kr.shape[0]
            use_full_eig = (N_dof_reduced <= 1000) or (const.get('sigma_eig', '') == 'SM')
            
            if use_full_eig:
                # Use full eigenvalue decomposition for small problems or when finding smallest eigenvalues
                # This ensures we capture rigid body modes (zero eigenvalues) correctly
                Kr_dense = Kr.toarray() if hasattr(Kr, 'toarray') else Kr
                Mr_dense = Mr.toarray() if hasattr(Mr, 'toarray') else Mr
                eig_vals, eig_vecs = np.linalg.eig(np.linalg.solve(Mr_dense, Kr_dense))
                # Take only the first N_eig eigenvalues (sorted by real part)
                idxs = np.argsort(np.real(eig_vals))[:const['N_eig']]
                eig_vals = eig_vals[idxs]
                eig_vecs = eig_vecs[:, idxs]
                
                # Match MATLAB's behavior: for real matrices (k=0), eigenvectors should be real
                # Check if matrices are effectively real (imaginary parts negligible)
                # At k=0, matrices should be real, but may be stored as complex arrays
                is_real_matrices = (np.allclose(wavevector, 0.0) and 
                                   np.max(np.abs(Kr_dense.imag)) < 1e-10 and 
                                   np.max(np.abs(Mr_dense.imag)) < 1e-10)
                if is_real_matrices:
                    # Force eigenvectors to be real (MATLAB's eigs returns real eigenvectors for real matrices)
                    eig_vecs = np.real(eig_vecs)
                    eig_vals = np.real(eig_vals)
            else:
                # Use sparse eigs for large problems
                # Handle sigma_eig: if string use as 'which', if number use as 'sigma'
                sigma_eig = const.get('sigma_eig', 0)
                eigs_kwargs = {'k': const['N_eig']}
                if isinstance(sigma_eig, str):
                    eigs_kwargs['which'] = sigma_eig
                else:
                    eigs_kwargs['sigma'] = sigma_eig
                
                eig_vals, eig_vecs = eigs(Kr, M=Mr, **eigs_kwargs)
            # Sort eigenvalues and eigenvectors by real part (matching MATLAB)
            idxs = np.argsort(np.real(eig_vals))
            eig_vals = eig_vals[idxs]
            eig_vecs = eig_vecs[:, idxs]
            
            # Convert to complex64 for consistency
            eig_vals = eig_vals.astype(np.complex64)
            eig_vecs = eig_vecs.astype(np.complex64)
            
            # Store eigenvectors if requested (matching Han's eigenvector_dtype)
            if const.get('isSaveEigenvectors', False):
                # Normalize eigenvectors (exact MATLAB translation)
                # MATLAB: ev(:,k_idx,:) = (eig_vecs./vecnorm(eig_vecs,2,1)).*exp(-1i*angle(eig_vecs(1,:)))
                norms = np.linalg.norm(eig_vecs, axis=0)
                eig_vecs_normalized = eig_vecs / norms
                
                # Phase alignment: for real eigenvectors, angle is 0 or π, so phase_align is ±1 (real)
                # For complex eigenvectors, phase_align is complex
                if np.isrealobj(eig_vecs):
                    # Real eigenvectors: phase alignment is just sign alignment (±1)
                    phase_align = np.sign(eig_vecs[0, :]).astype(np.float64)
                    eig_vecs_aligned = eig_vecs_normalized * phase_align
                else:
                    # Complex eigenvectors: use complex phase alignment
                phase_align = np.exp(-1j * np.angle(eig_vecs[0, :]))
                    eig_vecs_aligned = eig_vecs_normalized * phase_align
                
                # Store eigenvectors: eig_vecs_normalized is (N_dof_reduced, N_eig)
                # ev[:, k_idx, :] expects (N_dof_reduced, N_eig)
                eigenvector_dtype = const.get('eigenvector_dtype', 'double')
                if np.isrealobj(eig_vecs_aligned):
                    # Real eigenvectors: store as float64 (matching MATLAB)
                    if eigenvector_dtype == 'single':
                        ev[:, k_idx, :] = eig_vecs_aligned.astype(np.float32)
                    else:  # double
                        ev[:, k_idx, :] = eig_vecs_aligned.astype(np.float64)
                else:
                    # Complex eigenvectors: store as complex
                if eigenvector_dtype == 'single':
                        ev[:, k_idx, :] = eig_vecs_aligned.astype(np.complex64)
                else:  # double
                        ev[:, k_idx, :] = eig_vecs_aligned.astype(np.complex128)
            
            # Convert to frequencies (exact MATLAB translation)
            # Handle negative eigenvalues (shouldn't happen but can due to numerical issues)
            real_eig_vals = np.real(eig_vals)
            # Set negative or very small eigenvalues to zero (rigid body modes or numerical errors)
            real_eig_vals = np.maximum(real_eig_vals, 0.0)
            fr[k_idx, :] = (np.sqrt(real_eig_vals).astype(np.float32) / (2 * np.pi)).astype(np.float32)
            
        elif const.get('isUseGPU', False):
            raise NotImplementedError('GPU use is not currently developed')
    
    # Match MATLAB's data type behavior: convert to real if all eigenvectors are effectively real
    # MATLAB stores real eigenvectors as float64 (real arrays), not complex arrays
    if ev is not None:
        max_imag_all = np.max(np.abs(ev.imag))
        if max_imag_all < 1e-10:  # Within numerical precision
            # Convert to real array (matching MATLAB's float64 real arrays)
            ev = np.real(ev).astype(np.float64)
    
    return wavevectors, fr, ev, mesh, K_out, M_out, T_out


def get_mesh(const):
    """
    Get mesh information (placeholder function).
    
    This function would return mesh information if needed.
    For now, it returns None as mesh saving is typically disabled.
    """
    # This would need to be implemented if mesh saving is required
    return None
