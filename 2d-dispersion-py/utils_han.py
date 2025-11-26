"""
Utility functions matching 2D-dispersion-Han MATLAB library.

These functions provide utilities for batch processing and storage initialization
that match the MATLAB implementation exactly.
"""

import numpy as np


def make_chunks(N, M):
    """
    Split 1:N into consecutive chunks of length <= M.
    
    This is the exact translation of MATLAB's make_chunks.m function.
    
    Parameters
    ----------
    N : int
        Total number of items
    M : int
        Maximum chunk size
        
    Returns
    -------
    ranges : ndarray
        K-by-2 array where each row is [startIdx, endIdx] for that chunk.
        Chunks cover 1:N without overlap.
        
    Example
    -------
    >>> make_chunks(10, 3)
    array([[ 1,  3],
           [ 4,  6],
           [ 7,  9],
           [10, 10]])
    """
    if N <= 0 or M <= 0:
        return np.zeros((0, 2), dtype=int)
    
    # Chunk start indices (1-based like MATLAB)
    starts = np.arange(1, N + 1, M)
    
    # Chunk end indices (cap at N)
    ends = starts + M - 1
    ends[ends > N] = N
    
    ranges = np.column_stack([starts, ends])
    return ranges


def init_storage(const, N_struct_batch):
    """
    Initialize storage arrays for batch processing.
    
    This is the exact translation of MATLAB's init_storage.m function.
    
    Parameters
    ----------
    const : dict
        Constants structure containing system parameters
    N_struct_batch : int
        Number of structures in this batch
        
    Returns
    -------
    designs : ndarray
        Design array (N_pix x N_pix x 3 x N_struct_batch)
    WAVEVECTOR_DATA : ndarray
        Wavevector data (N_wv x 2 x N_struct_batch)
    EIGENVALUE_DATA : ndarray
        Eigenvalue data (N_wv x N_eig x N_struct_batch)
    N_dof : int
        Number of degrees of freedom
    DESIGN_NUMBERS : ndarray
        Design numbers (N_struct_batch,)
    EIGENVECTOR_DATA : ndarray
        Eigenvector data (N_dof x N_wv x N_eig x N_struct_batch)
    ELASTIC_MODULUS_DATA : ndarray
        Elastic modulus data (N_pix x N_pix x N_struct_batch)
    DENSITY_DATA : ndarray
        Density data (N_pix x N_pix x N_struct_batch)
    POISSON_DATA : ndarray
        Poisson's ratio data (N_pix x N_pix x N_struct_batch)
    K_DATA : list
        List of stiffness matrices (one per structure)
    M_DATA : list
        List of mass matrices (one per structure)
    T_DATA : list or None
        Transformation matrices (shared across structures)
    """
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_val = N_pix[0]
    else:
        N_pix_val = N_pix
    
    N_dof = 2 * (N_pix_val * const['N_ele'])**2
    n_wv = const['wavevectors'].shape[0]
    
    # Initialize arrays
    designs = np.zeros((N_pix_val, N_pix_val, 3, N_struct_batch))
    WAVEVECTOR_DATA = np.zeros((n_wv, 2, N_struct_batch))
    EIGENVALUE_DATA = np.zeros((n_wv, const['N_eig'], N_struct_batch))
    DESIGN_NUMBERS = np.arange(1, N_struct_batch + 1)  # 1-based like MATLAB
    
    # Eigenvector data with dtype support
    eigenvector_dtype = const.get('eigenvector_dtype', 'double')
    if eigenvector_dtype == 'single':
        ev_dtype = np.complex64
    else:  # double
        ev_dtype = np.complex128
    
    EIGENVECTOR_DATA = np.zeros((N_dof, n_wv, const['N_eig'], N_struct_batch), dtype=ev_dtype)
    
    ELASTIC_MODULUS_DATA = np.zeros((N_pix_val, N_pix_val, N_struct_batch))
    DENSITY_DATA = np.zeros((N_pix_val, N_pix_val, N_struct_batch))
    POISSON_DATA = np.zeros((N_pix_val, N_pix_val, N_struct_batch))
    
    # Initialize cell arrays for matrices
    K_DATA = [None] * N_struct_batch
    M_DATA = [None] * N_struct_batch
    T_DATA = None  # Will be set on first structure
    
    return (designs, WAVEVECTOR_DATA, EIGENVALUE_DATA, N_dof, DESIGN_NUMBERS,
            EIGENVECTOR_DATA, ELASTIC_MODULUS_DATA, DENSITY_DATA, POISSON_DATA,
            K_DATA, M_DATA, T_DATA)

