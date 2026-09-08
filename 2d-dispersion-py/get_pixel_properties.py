"""
Pixel properties function.

This module contains the function for getting material properties from design,
equivalent to MATLAB's get_pixel_properties.m.
"""

import numpy as np


def get_pixel_properties(pix_idx_x, pix_idx_y, const):
    """
    Get material properties for a specific pixel.
    
    This function matches MATLAB's get_pixel_properties.m exactly.
    
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

