"""
Gaussian process kernel functions.

This module provides various kernel functions for generating correlated
random designs using Gaussian processes.
"""

import numpy as np
from scipy.stats import multivariate_normal


def matern52_kernel(points_i, points_j, sigma_f, sigma_l):
    """
    Matern 5/2 kernel function.
    
    Parameters
    ----------
    points_i : array_like
        First set of points (N_i x 2)
    points_j : array_like
        Second set of points (N_j x 2)
    sigma_f : float
        Signal standard deviation
    sigma_l : float
        Length scale
        
    Returns
    -------
    C : array_like
        Covariance matrix (N_i x N_j)
    """
    
    # Reshape for broadcasting
    points_i = points_i[:, np.newaxis, :]  # (N_i, 1, 2)
    points_j = points_j[np.newaxis, :, :]  # (1, N_j, 2)
    
    # Compute distances
    displacements = points_i - points_j  # (N_i, N_j, 2)
    r = np.linalg.norm(displacements, axis=2)  # (N_i, N_j)
    
    # Matern 5/2 kernel
    sqrt5_r_over_l = np.sqrt(5) * r / sigma_l
    C = sigma_f**2 * (1 + sqrt5_r_over_l + 5 * r**2 / (3 * sigma_l**2)) * np.exp(-sqrt5_r_over_l)
    
    return C


def periodic_kernel(points_i, points_j, sigma_f, sigma_l, period):
    """
    Periodic kernel function.
    
    Parameters
    ----------
    points_i : array_like
        First set of points (N_i x 2)
    points_j : array_like
        Second set of points (N_j x 2)
    sigma_f : float
        Signal standard deviation
    sigma_l : float
        Length scale
    period : array_like
        Period in each dimension (2,)
        
    Returns
    -------
    C : array_like
        Covariance matrix (N_i x N_j)
    """
    
    # Reshape for broadcasting
    points_i = points_i[:, np.newaxis, :]  # (N_i, 1, 2)
    points_j = points_j[np.newaxis, :, :]  # (1, N_j, 2)
    
    # Compute displacements
    displacements = points_i - points_j  # (N_i, N_j, 2)
    
    # Periodic kernel in each dimension
    sin_arg1 = np.pi * np.abs(displacements[:, :, 0]) / period[0]
    C1 = sigma_f**2 * np.exp(-2 * np.sin(sin_arg1)**2 / sigma_l**2)
    
    sin_arg2 = np.pi * np.abs(displacements[:, :, 1]) / period[1]
    C2 = sigma_f**2 * np.exp(-2 * np.sin(sin_arg2)**2 / sigma_l**2)
    
    # Combine dimensions
    C = C1 * C2
    
    return C


def periodic_kernel_not_squared(points_i, points_j, sigma_f, sigma_l, period):
    """
    Periodic kernel function (not squared version).
    
    This is the exact translation of MATLAB's periodic_kernel_not_squared.m.
    Uses abs(sin) instead of sin^2 in the exponent.
    
    Parameters
    ----------
    points_i : array_like
        First set of points (N_i x 2)
    points_j : array_like
        Second set of points (N_j x 2)
    sigma_f : float
        Signal standard deviation
    sigma_l : float
        Length scale
    period : array_like
        Period in each dimension (2,)
        
    Returns
    -------
    C : array_like
        Covariance matrix (N_i x N_j)
    """
    
    # Reshape for broadcasting (matching MATLAB permute operations)
    points_i = points_i[:, np.newaxis, :]  # (N_i, 1, 2)
    points_j = points_j[np.newaxis, :, :]  # (1, N_j, 2)
    
    # Compute displacements
    displacements = points_i - points_j  # (N_i, N_j, 2)
    
    # Periodic kernel not squared in each dimension
    # MATLAB: sin_arg1 = pi*abs(displacements(:,:,1)/period(1))
    # MATLAB: C1 = sigma_f^2*exp(-2*abs(sin(sin_arg1))/sigma_l)
    sin_arg1 = np.pi * np.abs(displacements[:, :, 0]) / period[0]
    C1 = sigma_f**2 * np.exp(-2 * np.abs(np.sin(sin_arg1)) / sigma_l)
    
    sin_arg2 = np.pi * np.abs(displacements[:, :, 1]) / period[1]
    C2 = sigma_f**2 * np.exp(-2 * np.abs(np.sin(sin_arg2)) / sigma_l)
    
    # Combine dimensions
    C = C1 * C2
    
    return C


def kernel_prop(kernel, N_pix, design_options):
    """
    Generate property using kernel-based Gaussian process.
    
    This is the exact translation of the MATLAB kernel_prop function.
    
    Parameters
    ----------
    kernel : str
        Kernel type ('matern52', 'periodic')
    N_pix : int or tuple
        Number of pixels
    design_options : dict
        Kernel parameters
        
    Returns
    -------
    prop : ndarray
        Property distribution (N_pix x N_pix)
    """
    
    if isinstance(N_pix, int):
        N_pix = (N_pix, N_pix)
    
    # Physical space
    xx = np.linspace(0, 1, N_pix[0])
    yy = np.linspace(0, 1, N_pix[1])
    X, Y = np.meshgrid(xx, yy)
    points = np.column_stack([X.flatten(), Y.flatten()])
    
    # Generate covariance matrix based on kernel type
    if kernel == 'matern52':
        C = matern52_kernel(points, points, design_options['sigma_f'], design_options['sigma_l'])
    elif kernel == 'periodic':
        period = [1, 1]
        C = periodic_kernel(points, points, design_options['sigma_f'], design_options['sigma_l'], period)
    elif kernel == 'periodic - not squared':
        period = [1, 1]
        C = periodic_kernel_not_squared(points, points, design_options['sigma_f'], design_options['sigma_l'], period)
    else:
        raise ValueError(f'Kernel name "{kernel}" not recognized')
    
    # Sample from multivariate Gaussian
    mu = 0.5 * np.ones(points.shape[0])
    prop = np.random.multivariate_normal(mu, C)
    
    # Reshape and threshold
    prop = prop.reshape(N_pix)
    prop[prop < 0] = 0  # thresholding bottom
    prop[prop > 1] = 1  # thresholding top
    
    return prop


def matern52_prop(kernel, N_pix, design_options):
    """
    Generate material property field using Matern 5/2 kernel.
    
    Parameters
    ----------
    kernel : str
        Kernel type ('matern52')
    N_pix : array_like
        Number of pixels in each direction (2,)
    design_options : dict
        Design options containing kernel parameters
        
    Returns
    -------
    prop : array_like
        Generated property field (N_pix[0] x N_pix[1])
    """
    
    # Create coordinate grid
    xx = np.linspace(0, 1, N_pix[0])
    yy = np.linspace(0, 1, N_pix[1])
    X, Y = np.meshgrid(xx, yy)
    points = np.column_stack([X.flatten(), Y.flatten()])
    
    if kernel == 'matern52':
        C = matern52_kernel(points, points, 
                           design_options['sigma_f'], 
                           design_options['sigma_l'])
    elif kernel == 'periodic':
        period = [1, 1]
        C = periodic_kernel(points, points, 
                           design_options['sigma_f'], 
                           design_options['sigma_l'], 
                           period)
    else:
        raise ValueError(f'kernel name "{kernel}" not recognized')
    
    # Sample from multivariate Gaussian
    mu = 0.5 * np.ones(points.shape[0])
    prop = multivariate_normal.rvs(mu, C)
    
    # Reshape and threshold
    prop = prop.reshape(N_pix)
    prop = np.clip(prop, 0, 1)  # Threshold to [0, 1]
    
    return prop


def periodic_with_matern_kernel(points_i, points_j, sigma_f, sigma_l, period, matern_weight=0.5):
    """
    Combined periodic and Matern kernel.
    
    Parameters
    ----------
    points_i : array_like
        First set of points (N_i x 2)
    points_j : array_like
        Second set of points (N_j x 2)
    sigma_f : float
        Signal standard deviation
    sigma_l : float
        Length scale
    period : array_like
        Period in each dimension (2,)
    matern_weight : float, optional
        Weight for Matern component (default: 0.5)
        
    Returns
    -------
    C : array_like
        Combined covariance matrix (N_i x N_j)
    """
    
    # Compute individual kernels
    C_periodic = periodic_kernel(points_i, points_j, sigma_f, sigma_l, period)
    C_matern = matern52_kernel(points_i, points_j, sigma_f, sigma_l)
    
    # Combine kernels
    C = matern_weight * C_matern + (1 - matern_weight) * C_periodic
    
    return C


def generate_correlated_design(N_pix, kernel_type='matern52', design_options=None):
    """
    Generate a correlated design using specified kernel.
    
    Parameters
    ----------
    N_pix : array_like
        Number of pixels in each direction (2,)
    kernel_type : str, optional
        Type of kernel to use (default: 'matern52')
    design_options : dict, optional
        Design options. If None, uses default values.
        
    Returns
    -------
    design : array_like
        Generated design (N_pix[0] x N_pix[1] x 3)
        Third dimension: [0] = elastic modulus, [1] = density, [2] = Poisson's ratio
    """
    
    if design_options is None:
        design_options = {
            'sigma_f': 1.0,
            'sigma_l': 0.5,
            'symmetry': 'none',
            'N_value': 3
        }
    
    # Generate property fields
    E_field = matern52_prop(kernel_type, N_pix, design_options)
    rho_field = matern52_prop(kernel_type, N_pix, design_options)
    nu_field = 0.6 * np.ones(N_pix)
    
    # Combine into design array
    design = np.stack([E_field, rho_field, nu_field], axis=2)
    
    return design

