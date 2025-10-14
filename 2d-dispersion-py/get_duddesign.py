"""
Eigenvector derivative calculation.

This module contains functions for computing derivatives of eigenvectors
with respect to design parameters.
"""

import numpy as np
from scipy.linalg import solve


def get_duddesign(Kr, Mr, omega, u, dKrddesign, dMrddesign):
    """
    Compute derivative of eigenvector with respect to design parameter.
    
    Parameters
    ----------
    Kr : array_like
        Reduced stiffness matrix
    Mr : array_like
        Reduced mass matrix
    omega : float
        Angular frequency
    u : array_like
        Eigenvector
    dKrddesign : array_like
        Derivative of reduced stiffness matrix with respect to design parameter
    dMrddesign : array_like
        Derivative of reduced mass matrix with respect to design parameter
        
    Returns
    -------
    duddesign : array_like
        Derivative of eigenvector with respect to design parameter
    domegaddesign : complex
        Derivative of frequency with respect to design parameter
    """
    
    matrix_size = 2 * Kr.shape[0] + 2
    
    # Build the augmented system matrix
    A_sub1 = Kr - omega**2 * Mr
    A_sub2 = -Mr @ u
    A_sub3 = u.conj().T @ Mr
    z1 = np.zeros((1, u.shape[0]))
    z2 = np.zeros((1, u.shape[0] - 1))
    
    if matrix_size < 2000:
        # Use dense matrices for small systems
        A = np.block([
            [np.real(A_sub1), -np.imag(A_sub1), np.real(A_sub2), -np.imag(A_sub2)],
            [np.imag(A_sub1),  np.real(A_sub1), np.imag(A_sub2),  np.real(A_sub2)],
            [np.real(A_sub3), -np.imag(A_sub3),      0,             0],
            [z1,              1e3,            z2,             0]
        ])
    else:
        # Use sparse matrices for large systems
        from scipy.sparse import bmat
        A = bmat([
            [np.real(A_sub1), -np.imag(A_sub1), np.real(A_sub2), -np.imag(A_sub2)],
            [np.imag(A_sub1),  np.real(A_sub1), np.imag(A_sub2),  np.real(A_sub2)],
            [np.real(A_sub3), -np.imag(A_sub3),      0,             0],
            [z1,              1e3,            z2,             0]
        ])
    
    # Build right-hand side
    b_sub1 = -(dKrddesign - omega**2 * dMrddesign) @ u
    b_sub2 = -1/2 * u.conj().T @ dMrddesign @ u
    
    b = np.concatenate([
        np.real(b_sub1),
        np.imag(b_sub1),
        np.real(b_sub2),
        [0]
    ])
    
    # Solve the system
    x = solve(A, b)
    
    # Extract eigenvector and frequency derivatives
    n = u.shape[0]
    duddesign = x[:n] + 1j * x[n:2*n]
    domegaddesign = x[-2] + 1j * x[-1]
    
    return duddesign, domegaddesign

