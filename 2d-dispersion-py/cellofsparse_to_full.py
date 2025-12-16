"""
Cell of sparse to full conversion function.

This module contains the function for converting cell arrays of sparse matrices
to full arrays, equivalent to MATLAB's cellofsparse_to_full.m.
"""

import numpy as np
import scipy.sparse as sp


def cellofsparse_to_full(system_matrix_data):
    """
    Convert cell array of sparse matrices to full array.
    
    This function matches MATLAB's cellofsparse_to_full.m exactly.
    
    Parameters
    ----------
    system_matrix_data : list
        List of sparse matrices (cell array equivalent)
        
    Returns
    -------
    out : ndarray
        Full array of shape (N, M, M) where N is number of matrices
        and M is the size of each matrix
    """
    
    N = len(system_matrix_data)
    
    # Get size of first matrix
    first_matrix = system_matrix_data[0]
    if sp.issparse(first_matrix):
        first_matrix_full = first_matrix.toarray()
    else:
        first_matrix_full = first_matrix
    
    matrix_shape = first_matrix_full.shape
    
    # Preallocate output array
    out = np.zeros((N,) + matrix_shape, dtype=np.float32)
    
    # Convert each sparse matrix to full and store
    for i in range(N):
        matrix = system_matrix_data[i]
        if sp.issparse(matrix):
            out[i, :, :] = matrix.toarray().astype(np.float32)
        else:
            out[i, :, :] = matrix.astype(np.float32)
    
    return out

