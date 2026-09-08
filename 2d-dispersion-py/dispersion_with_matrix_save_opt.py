"""
Dispersion calculation with matrix saving optimization.

This is the exact translation of the MATLAB dispersion_with_matrix_save_opt.m function,
which is the actual function used by the generation script.
"""

import os
import json
import numpy as np
from scipy.sparse.linalg import eigs
from system_matrices import get_system_matrices
from system_matrices_vec import get_system_matrices_VEC, get_system_matrices_VEC_simplified
from system_matrices import get_transformation_matrix


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _raise_eigensolve_failure(exc, *, k_idx, wavevector, solver, sigma_eig, Kr, Mr):
    """
    Raise a structured runtime error for robust downstream logging.
    """
    diag = {
        "event": "eigensolve_failure",
        "k_idx_0based": int(k_idx),
        "k_idx_1based": int(k_idx) + 1,
        "wavevector": [float(wavevector[0]), float(wavevector[1])],
        "solver": str(solver),
        "sigma_eig": str(sigma_eig),
        "Kr_shape": tuple(int(x) for x in Kr.shape),
        "Mr_shape": tuple(int(x) for x in Mr.shape),
        "Kr_dtype": str(Kr.dtype),
        "Mr_dtype": str(Mr.dtype),
        "Kr_nnz": int(getattr(Kr, "nnz", np.count_nonzero(np.asarray(Kr)))),
        "Mr_nnz": int(getattr(Mr, "nnz", np.count_nonzero(np.asarray(Mr)))),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    raise RuntimeError("EIGENSOLVE_DIAG " + json.dumps(diag, sort_keys=True)) from exc


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
    parity_fr_float64 = _env_flag("PARITY_FR_FLOAT64", default=False)
    fr_dtype = np.float64 if parity_fr_float64 else np.float32
    fr = np.zeros((n_wavevectors, const['N_eig']), dtype=fr_dtype)

    parity_wavevector_float64 = _env_flag("PARITY_WAVEVECTOR_FLOAT64", default=False)
    parity_use_complex128 = _env_flag("PARITY_USE_COMPLEX128", default=False)
    parity_force_sparse_eigs = _env_flag("PARITY_FORCE_SPARSE_EIGS", default=False)
    parity_disable_neg_clamp = _env_flag("PARITY_DISABLE_NEG_CLAMP", default=False)
    wavevector_dtype = np.float64 if parity_wavevector_float64 else np.float32
    wavevectors_for_T = np.asarray(wavevectors, dtype=wavevector_dtype)
    
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
    
    # Optional precomputed transformation matrices.
    precomputed_wavevectors = const.get('precomputed_wavevectors', None)
    precomputed_T_data = const.get('precomputed_T_data', None)
    use_precomputed_by_index = False
    precomputed_lookup = None
    if precomputed_wavevectors is not None and precomputed_T_data is not None:
        precomputed_wavevectors = np.asarray(precomputed_wavevectors, dtype=np.float64)
        use_precomputed_by_index = (
            precomputed_wavevectors.shape == np.asarray(wavevectors).shape
            and len(precomputed_T_data) == n_wavevectors
            and np.allclose(precomputed_wavevectors, np.asarray(wavevectors, dtype=np.float64), rtol=0.0, atol=1e-12)
        )
        # Fallback lookup by wavevector key if order differs.
        precomputed_lookup = {
            (round(float(wv[0]), 12), round(float(wv[1]), 12)): Tm
            for wv, Tm in zip(precomputed_wavevectors, precomputed_T_data)
        }

    # Process each wavevector
    for k_idx in range(n_wavevectors):
        wavevector = wavevectors[k_idx, :]
        if use_precomputed_by_index:
            T = precomputed_T_data[k_idx]
        elif precomputed_lookup is not None:
            key = (round(float(wavevector[0]), 12), round(float(wavevector[1]), 12))
            T = precomputed_lookup.get(key, None)
            if T is None:
                T = get_transformation_matrix(wavevectors_for_T[k_idx, :], const)
        else:
            T = get_transformation_matrix(wavevectors_for_T[k_idx, :], const)
        
        # Transform matrices to reduced space
        matrix_dtype = np.complex128 if parity_use_complex128 else np.complex64
        Kr = T.conj().T @ K @ T
        Mr = T.conj().T @ M @ T
        if Kr.dtype != matrix_dtype:
            Kr = Kr.astype(matrix_dtype, copy=False)
        if Mr.dtype != matrix_dtype:
            Mr = Mr.astype(matrix_dtype, copy=False)
        
        # Store transformation matrix if requested
        if const.get('isSaveKandM', False):
            T_out[k_idx] = T
        
        if not const.get('isUseGPU', False):
            # Solve generalized eigenvalue problem
            # For small problems or when finding smallest eigenvalues, use full eig for accuracy
            # This is especially important for rigid body modes at k=0
            N_dof_reduced = Kr.shape[0]
            use_full_eig = (N_dof_reduced <= 1000) or (const.get('sigma_eig', '') == 'SM')
            if parity_force_sparse_eigs:
                use_full_eig = False
            
            if use_full_eig:
                # Use full eigenvalue decomposition for small problems or when finding smallest eigenvalues
                # This ensures we capture rigid body modes (zero eigenvalues) correctly
                Kr_dense = Kr.toarray() if hasattr(Kr, 'toarray') else Kr
                Mr_dense = Mr.toarray() if hasattr(Mr, 'toarray') else Mr
                try:
                    eig_vals, eig_vecs = np.linalg.eig(np.linalg.solve(Mr_dense, Kr_dense))
                except Exception as e:
                    _raise_eigensolve_failure(
                        e,
                        k_idx=k_idx,
                        wavevector=wavevector,
                        solver="dense_np.linalg",
                        sigma_eig=const.get("sigma_eig", 0),
                        Kr=Kr,
                        Mr=Mr,
                    )
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
                
                try:
                    eig_vals, eig_vecs = eigs(Kr, M=Mr, **eigs_kwargs)
                except Exception as e:
                    _raise_eigensolve_failure(
                        e,
                        k_idx=k_idx,
                        wavevector=wavevector,
                        solver="sparse_scipy_eigs",
                        sigma_eig=sigma_eig,
                        Kr=Kr,
                        Mr=Mr,
                    )
            # Sort eigenvalues and eigenvectors by real part (matching MATLAB)
            idxs = np.argsort(np.real(eig_vals))
            eig_vals = eig_vals[idxs]
            eig_vecs = eig_vecs[:, idxs]
            
            # Convert to controlled dtype for consistency
            eig_dtype = np.complex128 if parity_use_complex128 else np.complex64
            if eig_vals.dtype != eig_dtype:
                eig_vals = np.asarray(eig_vals, dtype=eig_dtype)
            if eig_vecs.dtype != eig_dtype:
                eig_vecs = np.asarray(eig_vecs, dtype=eig_dtype)
            
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
            if not parity_disable_neg_clamp:
                real_eig_vals = np.maximum(real_eig_vals, 0.0)
            fr[k_idx, :] = np.sqrt(real_eig_vals) / (2 * np.pi)
            
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
