"""
Element-level calculations for finite element matrices.

This module contains functions for computing element stiffness and mass matrices,
as well as their sensitivities with respect to design parameters.
"""

import numpy as np


def get_element_stiffness(E, nu, t, const):
    """
    Compute element stiffness matrix for Q4 bilinear quadrilateral element.
    
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


def get_element_mass(rho, t, const):
    """
    Compute element mass matrix for Q4 bilinear quadrilateral element.
    
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


def get_pixel_properties(pix_idx_x, pix_idx_y, const):
    """
    Get material properties for a specific pixel.
    
    Parameters
    ----------
    pix_idx_x : int
        Pixel index in x-direction
    pix_idx_y : int
        Pixel index in y-direction
    const : dict
        Constants structure containing design and material parameters
        
    Returns
    -------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    t : float
        Thickness
    rho : float
        Density
    """
    
    if const['design_scale'] == 'linear':
        E = const['E_min'] + const['design'][pix_idx_y, pix_idx_x, 0] * (const['E_max'] - const['E_min'])
        nu = const['poisson_min'] + const['design'][pix_idx_y, pix_idx_x, 2] * (const['poisson_max'] - const['poisson_min'])
        t = const['t']
        rho = const['rho_min'] + const['design'][pix_idx_y, pix_idx_x, 1] * (const['rho_max'] - const['rho_min'])
        
    elif const['design_scale'] == 'log':
        E = np.exp(const['design'][pix_idx_y, pix_idx_x, 0])
        nu = const['poisson_min'] + const['design'][pix_idx_y, pix_idx_x, 2] * (const['poisson_max'] - const['poisson_min'])
        t = const['t']
        rho = np.exp(const['design'][pix_idx_y, pix_idx_x, 1])
        
    else:
        raise ValueError("const.design_scale not recognized as 'log' or 'linear'")
    
    return E, nu, t, rho


def get_element_stiffness_sensitivity(E, nu, t, const):
    """
    Compute sensitivity of element stiffness matrix with respect to design parameters.
    
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
    dk_eleddesign : array_like
        8x8 sensitivity matrix (derivative with respect to elastic modulus)
    """
    
    # Sensitivity with respect to elastic modulus (design parameter k=0)
    # dK/dE = (1/E) * K
    k_ele = get_element_stiffness(E, nu, t, const)
    
    if const['design_scale'] == 'linear':
        dk_eleddesign = k_ele * (const['E_max'] - const['E_min'])
    elif const['design_scale'] == 'log':
        dk_eleddesign = k_ele  # d/d(ln(E)) = E * d/dE, so d/dE = (1/E) * d/d(ln(E))
    else:
        raise ValueError("const.design_scale not recognized")
    
    return dk_eleddesign.astype(np.float32)


def get_element_mass_sensitivity(rho, t, const):
    """
    Compute sensitivity of element mass matrix with respect to design parameters.
    
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
    dm_eleddesign : array_like
        8x8 sensitivity matrix (derivative with respect to density)
    """
    
    # Sensitivity with respect to density (design parameter k=1)
    # dM/drho = (1/rho) * M
    m_ele = get_element_mass(rho, t, const)
    
    if const['design_scale'] == 'linear':
        dm_eleddesign = m_ele * (const['rho_max'] - const['rho_min'])
    elif const['design_scale'] == 'log':
        dm_eleddesign = m_ele  # d/d(ln(rho)) = rho * d/drho, so d/drho = (1/rho) * d/d(ln(rho))
    else:
        raise ValueError("const.design_scale not recognized")
    
    return dm_eleddesign.astype(np.float32)

