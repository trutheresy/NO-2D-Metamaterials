"""
Element mass matrix function.

This module contains the function for computing element mass matrices,
equivalent to MATLAB's get_element_mass.m.
"""

import numpy as np


def get_element_mass(rho, t, const):
    """
    Compute element mass matrix for Q4 bilinear quadrilateral element.
    
    This function matches MATLAB's get_element_mass.m exactly.
    
    Parameters
    ----------
    rho : float
        Density
    t : float
        Thickness
    const : dict
        Constants structure containing system parameters
        
    Returns
    -------
    m_ele : array_like
        8x8 element mass matrix
        DOF order: [u1, v1, u2, v2, u3, v3, u4, v4]
        (counterclockwise from lower left node)
    """
    
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_val = N_pix[0]
    else:
        N_pix_val = N_pix
    
    # Element area
    element_size = const['a'] / (const['N_ele'] * N_pix_val)
    element_area = element_size**2
    m = rho * t * element_area  # Total mass of element
    
    # Consistent mass matrix for Q4 element
    m_ele = (1/36) * m * np.array([
        [4, 0, 2, 0, 1, 0, 2, 0],
        [0, 4, 0, 2, 0, 1, 0, 2],
        [2, 0, 4, 0, 2, 0, 1, 0],
        [0, 2, 0, 4, 0, 2, 0, 1],
        [1, 0, 2, 0, 4, 0, 2, 0],
        [0, 1, 0, 2, 0, 4, 0, 2],
        [2, 0, 1, 0, 2, 0, 4, 0],
        [0, 2, 0, 1, 0, 2, 0, 4]
    ], dtype=np.float32)
    
    return m_ele.astype(np.float32)

