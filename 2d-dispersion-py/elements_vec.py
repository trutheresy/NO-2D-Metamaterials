"""
Vectorized element functions for improved performance.

These functions provide vectorized versions of element stiffness and mass
matrix calculations, equivalent to the MATLAB _VEC functions.
"""

import numpy as np


def get_element_stiffness_VEC(E, nu, t):
    """
    Vectorized element stiffness matrix calculation.
    
    Parameters
    ----------
    E : array_like
        Young's modulus values for each element
    nu : array_like  
        Poisson's ratio values for each element
    t : array_like
        Thickness values for each element
        
    Returns
    -------
    k_ele : ndarray
        Element stiffness matrices stacked (N_elements, 8, 8)
    """
    
    # Convert inputs to numpy arrays
    E = np.asarray(E)
    nu = np.asarray(nu)
    t = np.asarray(t)
    
    # Ensure all inputs have the same shape
    if E.ndim == 0:
        E = np.full_like(nu, E)
    if t.ndim == 0:
        t = np.full_like(nu, t)
    
    # Reshape for vectorized computation
    E = E.flatten()
    nu = nu.flatten()
    t = t.flatten()
    
    n_elements = len(E)
    
    # Preallocate output array
    # Use float64 to match MATLAB precision
    k_ele = np.zeros((n_elements, 8, 8), dtype=np.float64)
    
    # Vectorized stiffness matrix calculation
    # This is the exact translation of the MATLAB code
    coeff = ((1/48) * E * t / (1 - nu**2)).astype(np.float64)  # Use float64 to match MATLAB
    
    # Define the 8x8 stiffness matrix template
    # MATLAB: [24-8*nu , 6*nu+6  , -12-4*nu, 18*nu-6 , -12+4*nu, -6*nu-6 , 8*nu    , -18*nu+6,...]
    k_ele[:, 0, 0] = 24 - 8*nu
    k_ele[:, 0, 1] = 6*nu + 6
    k_ele[:, 0, 2] = -12 - 4*nu
    k_ele[:, 0, 3] = 18*nu - 6
    k_ele[:, 0, 4] = -12 + 4*nu
    k_ele[:, 0, 5] = -6*nu - 6
    k_ele[:, 0, 6] = 8*nu
    k_ele[:, 0, 7] = -18*nu + 6
    
    k_ele[:, 1, 0] = 6*nu + 6
    k_ele[:, 1, 1] = 24 - 8*nu
    k_ele[:, 1, 2] = -18*nu + 6
    k_ele[:, 1, 3] = 8*nu
    k_ele[:, 1, 4] = -6*nu - 6
    k_ele[:, 1, 5] = -12 + 4*nu
    k_ele[:, 1, 6] = 18*nu - 6
    k_ele[:, 1, 7] = -12 - 4*nu
    
    k_ele[:, 2, 0] = -12 - 4*nu
    k_ele[:, 2, 1] = -18*nu + 6
    k_ele[:, 2, 2] = 24 - 8*nu
    k_ele[:, 2, 3] = -6*nu - 6
    k_ele[:, 2, 4] = 8*nu
    k_ele[:, 2, 5] = 18*nu - 6
    k_ele[:, 2, 6] = -12 + 4*nu
    k_ele[:, 2, 7] = 6*nu + 6
    
    k_ele[:, 3, 0] = 18*nu - 6
    k_ele[:, 3, 1] = 8*nu
    k_ele[:, 3, 2] = -6*nu - 6
    k_ele[:, 3, 3] = 24 - 8*nu
    k_ele[:, 3, 4] = -18*nu + 6
    k_ele[:, 3, 5] = -12 - 4*nu
    k_ele[:, 3, 6] = 6*nu + 6
    k_ele[:, 3, 7] = -12 + 4*nu
    
    k_ele[:, 4, 0] = -12 + 4*nu
    k_ele[:, 4, 1] = -6*nu - 6
    k_ele[:, 4, 2] = 8*nu
    k_ele[:, 4, 3] = -18*nu + 6
    k_ele[:, 4, 4] = 24 - 8*nu
    k_ele[:, 4, 5] = 6*nu + 6
    k_ele[:, 4, 6] = -12 - 4*nu
    k_ele[:, 4, 7] = 18*nu - 6
    
    k_ele[:, 5, 0] = -6*nu - 6
    k_ele[:, 5, 1] = -12 + 4*nu
    k_ele[:, 5, 2] = 18*nu - 6
    k_ele[:, 5, 3] = -12 - 4*nu
    k_ele[:, 5, 4] = 6*nu + 6
    k_ele[:, 5, 5] = 24 - 8*nu
    k_ele[:, 5, 6] = -18*nu + 6
    k_ele[:, 5, 7] = 8*nu
    
    k_ele[:, 6, 0] = 8*nu
    k_ele[:, 6, 1] = 18*nu - 6
    k_ele[:, 6, 2] = -12 + 4*nu
    k_ele[:, 6, 3] = 6*nu + 6
    k_ele[:, 6, 4] = -12 - 4*nu
    k_ele[:, 6, 5] = -18*nu + 6
    k_ele[:, 6, 6] = 24 - 8*nu
    k_ele[:, 6, 7] = -6*nu - 6
    
    k_ele[:, 7, 0] = -18*nu + 6
    k_ele[:, 7, 1] = -12 - 4*nu
    k_ele[:, 7, 2] = 6*nu + 6
    k_ele[:, 7, 3] = -12 + 4*nu
    k_ele[:, 7, 4] = 18*nu - 6
    k_ele[:, 7, 5] = 8*nu
    k_ele[:, 7, 6] = -6*nu - 6
    k_ele[:, 7, 7] = 24 - 8*nu
    
    # Apply the coefficient to all elements
    k_ele = k_ele * coeff[:, np.newaxis, np.newaxis]
    
    return k_ele.astype(np.float64)  # Use float64 to match MATLAB


def get_element_mass_VEC(rho, t, const):
    """
    Vectorized element mass matrix calculation.
    
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
    # Use float64 to match MATLAB precision
    m_ele = np.zeros((n_elements, 8, 8), dtype=np.float64)
    
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
    ], dtype=np.float64)  # Use float64 to match MATLAB
    
    # Apply the mass template and coefficient to all elements
    m_ele = ((1/36) * m[:, np.newaxis, np.newaxis] * mass_template[np.newaxis, :, :]).astype(np.float64)  # Use float64 to match MATLAB
    
    return m_ele.astype(np.float64)  # Use float64 to match MATLAB
