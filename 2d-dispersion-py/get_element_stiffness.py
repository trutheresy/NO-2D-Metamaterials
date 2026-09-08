"""
Element stiffness matrix function.

This module contains the function for computing element stiffness matrices,
equivalent to MATLAB's get_element_stiffness.m.
"""

import numpy as np


def get_element_stiffness(E, nu, t, const):
    """
    Compute element stiffness matrix for Q4 bilinear quadrilateral element.
    
    This function matches MATLAB's get_element_stiffness.m exactly.
    
    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    t : float
        Thickness
    const : dict
        Constants structure containing system parameters
        
    Returns
    -------
    k_ele : array_like
        8x8 element stiffness matrix
        DOF order: [u1, v1, u2, v2, u3, v3, u4, v4]
        (counterclockwise from lower left node)
    """
    
    # Q4 bilinear quadrilateral element stiffness matrix (plane stress)
    # Full integration, analytical form
    k_ele = (1/48) * E * t / (1 - nu**2) * np.array([
        [24-8*nu,  6*nu+6,   -12-4*nu, 18*nu-6,  -12+4*nu, -6*nu-6,   8*nu,    -18*nu+6],
        [6*nu+6,   24-8*nu,  -18*nu+6, 8*nu,     -6*nu-6,  -12+4*nu,  18*nu-6, -12-4*nu],
        [-12-4*nu, -18*nu+6, 24-8*nu,  -6*nu-6,  8*nu,     18*nu-6,  -12+4*nu, 6*nu+6],
        [18*nu-6,  8*nu,     -6*nu-6,  24-8*nu, -18*nu+6, -12-4*nu,  6*nu+6,  -12+4*nu],
        [-12+4*nu, -6*nu-6,  8*nu,     -18*nu+6, 24-8*nu,  6*nu+6,   -12-4*nu, 18*nu-6],
        [-6*nu-6,  -12+4*nu, 18*nu-6,  -12-4*nu, 6*nu+6,   24-8*nu,  -18*nu+6, 8*nu],
        [8*nu,     18*nu-6,  -12+4*nu, 6*nu+6,   -12-4*nu, -18*nu+6, 24-8*nu,  -6*nu-6],
        [-18*nu+6, -12-4*nu, 6*nu+6,   -12+4*nu, 18*nu-6,  8*nu,     -6*nu-6,  24-8*nu]
    ], dtype=np.float32)
    
    return k_ele.astype(np.float32)

