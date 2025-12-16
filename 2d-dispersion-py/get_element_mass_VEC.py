"""
Vectorized element mass matrix function.

This module contains the vectorized function for computing element mass matrices,
equivalent to MATLAB's get_element_mass_VEC.m.
"""

import numpy as np


def get_element_mass_VEC(rho, t, const):
    """
    Vectorized element mass matrix calculation.
    
    This function matches MATLAB's get_element_mass_VEC.m exactly.
    
    Parameters
    ----------
    rho : array_like
        Density values for each element
    t : array_like
        Thickness values for each element
    const : dict
        Constants structure containing N_ele, N_pix, a
        
    Returns
    -------
    m_ele : ndarray
        Element mass matrices stacked (N_elements, 8, 8)
    """
    
    # Convert inputs to numpy arrays
    rho = np.asarray(rho)
    t = np.asarray(t)
    
    # Ensure all inputs have the same shape
    if t.ndim == 0:
        t = np.full_like(rho, t)
    
    # Reshape for vectorized computation
    rho = rho.flatten()
    t = t.flatten()
    
    n_elements = len(rho)
    
    # Calculate element mass (exact MATLAB translation)
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_val = N_pix[0]
    else:
        N_pix_val = N_pix
    
    m = rho * t * (const['a'] / (const['N_ele'] * N_pix_val))**2
    
    # Preallocate output array
    m_ele = np.zeros((n_elements, 8, 8), dtype=np.float32)
    
    # Define the 8x8 mass matrix template (exact MATLAB translation)
    mass_template = np.array([
        [4, 0, 2, 0, 1, 0, 2, 0],
        [0, 4, 0, 2, 0, 1, 0, 2],
        [2, 0, 4, 0, 2, 0, 1, 0],
        [0, 2, 0, 4, 0, 2, 0, 1],
        [1, 0, 2, 0, 4, 0, 2, 0],
        [0, 1, 0, 2, 0, 4, 0, 2],
        [2, 0, 1, 0, 2, 0, 4, 0],
        [0, 2, 0, 1, 0, 2, 0, 4]
    ], dtype=np.float32)
    
    # Apply the mass template and coefficient to all elements
    m_ele = ((1/36) * m[:, np.newaxis, np.newaxis] * mass_template[np.newaxis, :, :]).astype(np.float32)
    
    return m_ele.astype(np.float32)

