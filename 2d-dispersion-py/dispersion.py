"""
Core dispersion calculation functions.

This module contains the main dispersion calculation functions that solve
the eigenvalue problem for periodic structures.
"""

import numpy as np
from scipy.sparse.linalg import eigs
from system_matrices import get_system_matrices, get_transformation_matrix


def dispersion(const, wavevectors):
    """
    Calculate dispersion relations for given wavevectors.
    
    This function matches the MATLAB 2D-dispersion-Han version exactly.
    
    Parameters
    ----------
    const : dict
        Constants structure containing system parameters
    wavevectors : array_like
        Array of wavevectors (N x 2)
        
    Returns
    -------
    wv : array_like
        Wavevectors (same as input)
    fr : array_like
        Frequencies (N x N_eig)
    ev : array_like or None
        Eigenvectors if isSaveEigenvectors is True, otherwise None
    mesh : array_like or None
        Mesh information if isSaveMesh is True, otherwise None
    """
    from system_matrices_vec import get_system_matrices_VEC, get_system_matrices_VEC_simplified
    
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_val = N_pix[0]
    else:
        N_pix_val = N_pix
    
    N_dof = ((const['N_ele'] * N_pix_val)**2) * 2
    
    # Get mesh if requested (matching Han's version)
    if const.get('isSaveMesh', False):
        mesh = get_mesh(const)
    else:
        mesh = None
    
    n_wavevectors = wavevectors.shape[0]
    # MATLAB: fr = zeros(size(wavevectors,2),const.N_eig);
    # But wavevectors is (n_wv, 2), so size(wavevectors,2) = 2 (WRONG in MATLAB!)
    # MATLAB code has a bug - it should use size(wavevectors,1) = n_wv
    # We'll use the correct dimension: n_wavevectors
    fr = np.zeros((n_wavevectors, const['N_eig']))
    
    if const.get('isSaveEigenvectors', False):
        # Get transformation matrix to determine reduced DOF size
        # For first wavevector to determine dimensions
        T_temp = get_transformation_matrix(wavevectors[0, :], const)
        N_dof_reduced = T_temp.shape[1]
        # MATLAB: ev = zeros(N_dof,size(wavevectors,2),const.N_eig);
        # MATLAB code uses size(wavevectors,2) which would be 2 if wavevectors is (n_wv, 2)
        # This is a MATLAB bug - it should use size(wavevectors,1) = n_wv
        # We'll use the correct dimension: (N_dof_reduced, n_wavevectors, N_eig)
        # This matches what MATLAB INTENDS to do, even though the code has a bug
        ev = np.zeros((N_dof_reduced, n_wavevectors, const['N_eig']), dtype=complex)
    else:
        ev = None
    
    # Get system matrices (matching Han's version with isUseSecondImprovement)
    if const.get('isUseSecondImprovement', False):
        K, M = get_system_matrices_VEC_simplified(const)
    elif const.get('isUseImprovement', True):
        K, M = get_system_matrices_VEC(const)
    else:
        K, M = get_system_matrices(const, use_vectorized=False)
    
    # Process each wavevector
    for k_idx in range(n_wavevectors):
        wavevector = wavevectors[k_idx, :]
        T = get_transformation_matrix(wavevector, const)
        
        # Transform matrices to reduced space
        Kr = T.conj().T @ K @ T
        Mr = T.conj().T @ M @ T
        
        if not const.get('isUseGPU', False):
            # Solve generalized eigenvalue problem
            try:
                eig_vals, eig_vecs = eigs(Kr, M=Mr, k=const['N_eig'], 
                                         sigma=const['sigma_eig'])
            except Exception as e:
                # Convert to dense matrices for debugging if sparse solve fails
                # Sometimes the sparse solver has issues, but dense might work
                try:
                    Kr_dense = Kr.toarray() if hasattr(Kr, 'toarray') else Kr
                    Mr_dense = Mr.toarray() if hasattr(Mr, 'toarray') else Mr
                    # Check for singularity
                    if np.linalg.cond(Mr_dense) > 1e12:
                        raise ValueError(f"Mass matrix is singular (cond={np.linalg.cond(Mr_dense):.2e}) at wavevector {wavevector}")
                    eig_vals, eig_vecs = eigs(Kr_dense, M=Mr_dense, k=const['N_eig'], 
                                             sigma=const['sigma_eig'])
                except Exception as e2:
                    raise RuntimeError(f"Failed to solve eigenvalue problem at wavevector {wavevector}: {e2}")
            
            # Sort eigenvalues and eigenvectors
            idxs = np.argsort(eig_vals)
            eig_vals = eig_vals[idxs]
            eig_vecs = eig_vecs[:, idxs]
            
            # Normalize eigenvectors (exact MATLAB translation)
            if const.get('isSaveEigenvectors', False):
                # MATLAB: ev(:,k_idx,:) = (eig_vecs./vecnorm(eig_vecs,2,1)).*exp(-1i*angle(eig_vecs(1,:)))
                # eig_vecs is (N_dof_reduced, N_eig), after normalization and phase alignment it should be (N_dof_reduced, N_eig)
                # ev[:, k_idx, :] expects (N_dof_reduced, N_eig), so we store transpose: (N_eig, N_dof_reduced).T = (N_dof_reduced, N_eig)
                norms = np.linalg.norm(eig_vecs, axis=0)
                eig_vecs_normalized = eig_vecs / norms
                phase_align = np.exp(-1j * np.angle(eig_vecs[0, :]))
                # Store eigenvectors: each column is an eigenvector, so transpose to get (N_dof_reduced, N_eig)
                ev[:, k_idx, :] = (eig_vecs_normalized * phase_align)
            
            # Convert to frequencies (exact MATLAB translation)
            fr[k_idx, :] = np.sqrt(np.real(eig_vals)) / (2 * np.pi)
            
        elif const.get('isUseGPU', False):
            raise NotImplementedError('GPU use is not currently developed')
    
    return wavevectors, fr, ev, mesh


def get_mesh(const):
    """
    Get mesh information (placeholder function).
    
    This function would return mesh information if needed.
    For now, it returns None as mesh saving is typically disabled.
    """
    # This would need to be implemented if mesh saving is required
    return None


def dispersion2(const, wavevectors):
    """
    Enhanced dispersion calculation with group velocity and sensitivity analysis.
    
    Parameters
    ----------
    const : dict
        Constants structure containing system parameters
    wavevectors : array_like
        Array of wavevectors (N x 2)
        
    Returns
    -------
    wv : array_like
        Original wavevectors
    fr : array_like
        Frequencies (N x N_eig)
    ev : array_like or None
        Eigenvectors if isSaveEigenvectors is True
    cg : array_like or None
        Group velocities if isComputeGroupVelocity is True
    dfrddesign : array_like or None
        Frequency design sensitivities if isComputeFrequencyDesignSensitivity is True
    dcgddesign : array_like or None
        Group velocity design sensitivities if isComputeGroupVelocityDesignSensitivity is True
    """
    
    orig_wavevectors = wavevectors.copy()
    
    # Remove duplicate wavevectors for efficiency
    unique_wavevectors, unique_idxs, inverse_idxs = np.unique(
        wavevectors, axis=0, return_index=True, return_inverse=True)
    
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_val = N_pix[0]
    else:
        N_pix_val = N_pix
    
    n_unique = unique_wavevectors.shape[0]
    fr = np.zeros((n_unique, const['N_eig']))
    
    if const['isSaveEigenvectors']:
        ev = np.zeros((((const['N_ele'] * N_pix_val)**2) * 2, 
                      n_unique, const['N_eig']), dtype=complex)
    else:
        ev = None
    
    if const['isComputeGroupVelocity']:
        cg = np.zeros((n_unique, 2, const['N_eig']))
    else:
        cg = None
    
    # Initialize design sensitivity arrays
    design_size = list(const['design'].shape)
    design_size[2] = design_size[2] - 1  # Poisson's ratio is not a design variable
    
    if const['isComputeFrequencyDesignSensitivity'] or const['isComputeGroupVelocityDesignSensitivity']:
        dfrddesign = np.zeros([n_unique, const['N_eig']] + design_size, dtype=complex)
    else:
        dfrddesign = None
    
    if const['isComputeGroupVelocityDesignSensitivity']:
        dcgddesign = np.zeros([n_unique, 2, const['N_eig']] + design_size, dtype=complex)
    else:
        dcgddesign = None
    
    # Get system matrices
    if const['isUseImprovement']:
        K, M = get_system_matrices(const, use_vectorized=True)
    else:
        if const['isComputeFrequencyDesignSensitivity'] or const['isComputeGroupVelocityDesignSensitivity']:
            K, M, dKddesign, dMddesign = get_system_matrices(const, use_vectorized=False, 
                                                           return_sensitivities=True)
        else:
            K, M = get_system_matrices(const, use_vectorized=False)
    
    # Process each unique wavevector
    for k_idx in range(n_unique):
        wavevector = unique_wavevectors[k_idx, :]
        
        # Get transformation matrix and its derivatives
        if const['isComputeGroupVelocity'] or const['isComputeGroupVelocityDesignSensitivity']:
            T, dTdwavevector = get_transformation_matrix(wavevector, const, return_derivatives=True)
        else:
            T = get_transformation_matrix(wavevector, const, return_derivatives=False)
        
        # Transform matrices to reduced space
        Kr = T.conj().T @ K @ T
        Mr = T.conj().T @ M @ T
        
        # Compute derivatives with respect to wavevector if needed
        if const['isComputeGroupVelocity'] or const['isComputeGroupVelocityDesignSensitivity']:
            dKrdwavevector = []
            dMrdwavevector = []
            for wv_comp_idx in range(2):
                dKrdwv = (dTdwavevector[wv_comp_idx].conj().T @ K @ T + 
                         T.conj().T @ K @ dTdwavevector[wv_comp_idx])
                dMrdwv = (dTdwavevector[wv_comp_idx].conj().T @ M @ T + 
                         T.conj().T @ M @ dTdwavevector[wv_comp_idx])
                dKrdwavevector.append(dKrdwv)
                dMrdwavevector.append(dMrdwv)
        
        # Solve generalized eigenvalue problem
        eig_vals, eig_vecs = eigs(Kr, M=Mr, k=const['N_eig'], 
                                 sigma=const['sigma_eig'])
        
        # Sort eigenvalues and eigenvectors
        idxs = np.argsort(eig_vals)
        eig_vals = eig_vals[idxs]
        eig_vecs = eig_vecs[:, idxs]
        
        # Normalize eigenvectors by mass matrix
        mass_norms = np.sqrt(np.diag(eig_vecs.conj().T @ Mr @ eig_vecs))
        eig_vecs = eig_vecs / mass_norms
        eig_vecs = eig_vecs * np.exp(-1j * np.angle(eig_vecs[0, :]))
        
        if const['isSaveEigenvectors']:
            ev[:, k_idx, :] = eig_vecs.T
        
        # Check for negative eigenvalues
        if np.any(np.real(eig_vals) < -1e-1):
            print('Warning: large negative eigenvalue')
        
        # Convert to frequencies (angular frequency, not Hz)
        fr[k_idx, :] = np.sqrt(np.abs(np.real(eig_vals)))
        
        # Compute group velocities
        if const['isComputeGroupVelocity']:
            for wv_comp_idx in range(2):
                for eig_idx in range(const['N_eig']):
                    omega = fr[k_idx, eig_idx]
                    u = eig_vecs[:, eig_idx]
                    
                    numerator = (u.conj().T @ (dKrdwavevector[wv_comp_idx] - 
                                             omega**2 * dMrdwavevector[wv_comp_idx]) @ u)
                    denominator = eig_vecs[:, eig_idx].conj().T @ Mr @ eig_vecs[:, eig_idx]
                    
                    cg[k_idx, wv_comp_idx, eig_idx] = np.real(numerator / (2 * omega * denominator))
        
        # Compute frequency design sensitivities
        if const['isComputeFrequencyDesignSensitivity'] or const['isComputeGroupVelocityDesignSensitivity']:
            for i in range(design_size[0]):
                for j in range(design_size[1]):
                    for k in range(design_size[2]):
                        if k == 0:  # Elastic modulus parameter
                            dKrddesign_ij = T.conj().T @ dKddesign[i][j] @ T
                            for eig_idx in range(const['N_eig']):
                                omega = fr[k_idx, eig_idx]
                                u = eig_vecs[:, eig_idx]
                                dfrddesign[k_idx, eig_idx, i, j, k] = (
                                    u.conj().T @ dKrddesign_ij @ u / (2 * omega))
                                
                        elif k == 1:  # Density parameter
                            dMrddesign_ij = T.conj().T @ dMddesign[i][j] @ T
                            for eig_idx in range(const['N_eig']):
                                omega = fr[k_idx, eig_idx]
                                u = eig_vecs[:, eig_idx]
                                dfrddesign[k_idx, eig_idx, i, j, k] = (
                                    u.conj().T @ (-omega**2 * dMrddesign_ij) @ u / (2 * omega))
            
            # Take real part
            dfrddesign[k_idx, :, :, :, :] = np.real(dfrddesign[k_idx, :, :, :, :])
        
        # Compute group velocity design sensitivities
        if const['isComputeGroupVelocityDesignSensitivity']:
            from .get_duddesign import get_duddesign
            
            for wv_comp_idx in range(2):
                for eig_idx in range(const['N_eig']):
                    for i in range(design_size[0]):
                        for j in range(design_size[1]):
                            for k in range(design_size[2]):
                                omega = fr[k_idx, eig_idx]
                                u = eig_vecs[:, eig_idx]
                                
                                A = dKrdwavevector[wv_comp_idx] - omega**2 * dMrdwavevector[wv_comp_idx]
                                domegadtheta = dfrddesign[k_idx, eig_idx, i, j, k]
                                dTdgamma = dTdwavevector[wv_comp_idx]
                                
                                if k == 0:  # Elastic modulus parameter
                                    dKrdtheta = dKrddesign[i][j]
                                    dMrdtheta = np.zeros_like(dMrddesign[i][j])
                                    dKdtheta = dKddesign[i][j]
                                    d2Krdthetadgamma = (dTdgamma.conj().T @ dKdtheta @ T + 
                                                      T.conj().T @ dKdtheta @ dTdgamma)
                                    dAdtheta = d2Krdthetadgamma - 2 * omega * domegadtheta * dMrdwavevector[wv_comp_idx]
                                    
                                elif k == 1:  # Density parameter
                                    dKrdtheta = np.zeros_like(dKrddesign[i][j])
                                    dMrdtheta = dMddesign[i][j]
                                    dMdtheta = dMddesign[i][j]
                                    d2Mrdthetadgamma = (dTdgamma.conj().T @ dMdtheta @ T + 
                                                      T.conj().T @ dMdtheta @ dTdgamma)
                                    dAdtheta = (-2 * omega * domegadtheta * dMrdwavevector[wv_comp_idx] - 
                                              omega**2 * d2Mrdthetadgamma)
                                
                                # Get eigenvector derivative
                                duddesign = get_duddesign(Kr, Mr, omega, u, dKrdtheta, dMrdtheta)
                                
                                # Compute group velocity sensitivity
                                term1 = -1/(2*omega**2) * domegadtheta * u.conj().T @ A @ u
                                term2 = 1/(2*omega) * duddesign.conj().T @ A @ u
                                term3 = 1/(2*omega) * u.conj().T @ dAdtheta @ u
                                term4 = 1/(2*omega) * u.conj().T @ A @ duddesign
                                
                                dcgddesign[k_idx, wv_comp_idx, eig_idx, i, j, k] = np.real(
                                    term1 + term2 + term3 + term4)
    
    # Map results back to original wavevector order
    if const['isComputeGroupVelocity']:
        cg = cg[inverse_idxs, :, :]
    if const['isComputeFrequencyDesignSensitivity']:
        dfrddesign = dfrddesign[inverse_idxs, :, :, :, :]
    if const['isComputeGroupVelocityDesignSensitivity']:
        dcgddesign = dcgddesign[inverse_idxs, :, :, :, :, :]
    
    fr = fr[inverse_idxs, :]
    if const['isSaveEigenvectors']:
        ev = ev[:, inverse_idxs, :]
    
    return orig_wavevectors, fr, ev, cg, dfrddesign, dcgddesign

