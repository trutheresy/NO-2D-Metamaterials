import numpy as np
from scipy.sparse.linalg import eigs
from numpy.linalg import norm

def dispersion(wavevectors, const):
    """
    Compute the dispersion relation (frequency vs wavevector) 
    for a periodic medium using finite element matrices.
    
    Parameters:
    - wavevectors: ndarray of shape (N_k, 2), the wavevectors in the IBZ
    - const: dict with constants and flags (see details below)

    Returns:
    - wv: same as wavevectors
    - fr: frequencies of shape (N_k, N_eig)
    - ev: eigenvectors of shape (N_dof, N_k, N_eig) if saved, else None
    """
    N_k = wavevectors.shape[0]
    N_eig = const['N_eig']
    N_pix = const['N_pix']
    N_ele = const['N_ele']
    is_save_ev = const['isSaveEigenvectors']

    # Preallocate output arrays
    fr = np.zeros((N_k, N_eig))
    if is_save_ev:
        N_dof = (N_ele * N_pix)**2 * 2  # total number of degrees of freedom
        ev = np.zeros((N_dof, N_k, N_eig), dtype=np.complex128)
    else:
        ev = None

    # Get global system matrices (K and M)
    if const['isUseImprovement']:
        K, M = get_system_matrices_VEC(const)
    else:
        K, M = get_system_matrices(const)

    # Loop over all wavevectors
    for k_idx in range(N_k):
        kvec = wavevectors[k_idx, :]
        
        # Get Bloch transformation matrix for current wavevector
        T = get_transformation_matrix(kvec, const)

        # Apply the Bloch-periodic transformation
        Kr = T.conj().T @ K @ T
        Mr = T.conj().T @ M @ T

        # Solve the generalized eigenvalue problem Kr u = λ Mr u
        eig_vals, eig_vecs = eigs(Kr, M=Mr, k=N_eig, sigma=const['sigma_eig'])

        # Sort eigenvalues and eigenvectors by ascending eigenvalue
        idx = np.argsort(np.real(eig_vals))
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        # Normalize and align eigenvectors (optional)
        if is_save_ev:
            for j in range(N_eig):
                v = eig_vecs[:, j]
                v = v / norm(v)  # normalize by L2 norm
                v = v * np.exp(-1j * np.angle(v[0]))  # align complex phase
                ev[:, k_idx, j] = v

        # Convert eigenvalues to real frequencies (Hz)
        fr[k_idx, :] = np.sqrt(np.real(eig_vals)) / (2 * np.pi)

    return wavevectors, fr, ev
