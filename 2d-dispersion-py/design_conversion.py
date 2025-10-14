"""
Design format conversion utilities.

This module provides functions for converting between different
design formats (linear, logarithmic, explicit).
"""

import numpy as np


def convert_design(design, initial_format, target_format, E_min=2e9, E_max=200e9,
                  rho_min=1e3, rho_max=8e3, poisson_min=0.0, poisson_max=0.5):
    """
    Convert design between different formats.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    initial_format : str
        Initial format ('linear', 'log', 'explicit')
    target_format : str
        Target format ('linear', 'log', 'explicit')
    E_min, E_max : float, optional
        Young's modulus bounds
    rho_min, rho_max : float, optional
        Density bounds
    poisson_min, poisson_max : float, optional
        Poisson's ratio bounds
        
    Returns
    -------
    converted_design : array_like
        Converted design array
    """
    
    if initial_format == target_format:
        return design.copy()
    
    # Convert to explicit format first
    if initial_format == 'linear':
        explicit_design = np.stack([
            E_min + (E_max - E_min) * design[:, :, 0],  # E
            rho_min + (rho_max - rho_min) * design[:, :, 1],  # rho
            poisson_min + (poisson_max - poisson_min) * design[:, :, 2]  # nu
        ], axis=2)
    elif initial_format == 'log':
        explicit_design = np.stack([
            np.exp(design[:, :, 0]),  # E
            np.exp(design[:, :, 1]),  # rho
            poisson_min + (poisson_max - poisson_min) * design[:, :, 2]  # nu
        ], axis=2)
    elif initial_format == 'explicit':
        explicit_design = design.copy()
    else:
        raise ValueError(f"Unknown initial format: {initial_format}")
    
    # Convert from explicit to target format
    if target_format == 'linear':
        converted_design = np.stack([
            (explicit_design[:, :, 0] - E_min) / (E_max - E_min),  # E
            (explicit_design[:, :, 1] - rho_min) / (rho_max - rho_min),  # rho
            (explicit_design[:, :, 2] - poisson_min) / (poisson_max - poisson_min)  # nu
        ], axis=2)
    elif target_format == 'log':
        converted_design = np.stack([
            np.log(explicit_design[:, :, 0]),  # E
            np.log(explicit_design[:, :, 1]),  # rho
            (explicit_design[:, :, 2] - poisson_min) / (poisson_max - poisson_min)  # nu
        ], axis=2)
    elif target_format == 'explicit':
        converted_design = explicit_design
    else:
        raise ValueError(f"Unknown target format: {target_format}")
    
    return converted_design


def design_to_explicit(design, design_format='linear', E_min=2e9, E_max=200e9,
                      rho_min=1e3, rho_max=8e3, poisson_min=0.0, poisson_max=0.5):
    """
    Convert design to explicit material properties.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    design_format : str, optional
        Design format ('linear', 'log') (default: 'linear')
    E_min, E_max : float, optional
        Young's modulus bounds
    rho_min, rho_max : float, optional
        Density bounds
    poisson_min, poisson_max : float, optional
        Poisson's ratio bounds
        
    Returns
    -------
    explicit_properties : dict
        Dictionary containing explicit material properties
    """
    
    if design_format == 'linear':
        E = E_min + (E_max - E_min) * design[:, :, 0]
        rho = rho_min + (rho_max - rho_min) * design[:, :, 1]
        nu = poisson_min + (poisson_max - poisson_min) * design[:, :, 2]
    elif design_format == 'log':
        E = np.exp(design[:, :, 0])
        rho = np.exp(design[:, :, 1])
        nu = poisson_min + (poisson_max - poisson_min) * design[:, :, 2]
    else:
        raise ValueError(f"Unknown design format: {design_format}")
    
    return {
        'E': E,
        'rho': rho,
        'nu': nu
    }


def explicit_to_design(E, rho, nu, design_format='linear', E_min=2e9, E_max=200e9,
                      rho_min=1e3, rho_max=8e3, poisson_min=0.0, poisson_max=0.5):
    """
    Convert explicit material properties to design format.
    
    Parameters
    ----------
    E : array_like
        Young's modulus array
    rho : array_like
        Density array
    nu : array_like
        Poisson's ratio array
    design_format : str, optional
        Design format ('linear', 'log') (default: 'linear')
    E_min, E_max : float, optional
        Young's modulus bounds
    rho_min, rho_max : float, optional
        Density bounds
    poisson_min, poisson_max : float, optional
        Poisson's ratio bounds
        
    Returns
    -------
    design : array_like
        Design array (N_pix x N_pix x 3)
    """
    
    if design_format == 'linear':
        design = np.stack([
            (E - E_min) / (E_max - E_min),
            (rho - rho_min) / (rho_max - rho_min),
            (nu - poisson_min) / (poisson_max - poisson_min)
        ], axis=2)
    elif design_format == 'log':
        design = np.stack([
            np.log(E),
            np.log(rho),
            (nu - poisson_min) / (poisson_max - poisson_min)
        ], axis=2)
    else:
        raise ValueError(f"Unknown design format: {design_format}")
    
    return design


def validate_design_bounds(design, design_format='linear', E_min=2e9, E_max=200e9,
                          rho_min=1e3, rho_max=8e3, poisson_min=0.0, poisson_max=0.5):
    """
    Validate that design values are within specified bounds.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    design_format : str, optional
        Design format ('linear', 'log') (default: 'linear')
    E_min, E_max : float, optional
        Young's modulus bounds
    rho_min, rho_max : float, optional
        Density bounds
    poisson_min, poisson_max : float, optional
        Poisson's ratio bounds
        
    Returns
    -------
    is_valid : bool
        True if all values are within bounds
    violations : dict
        Dictionary containing information about violations
    """
    
    violations = {
        'E': {'min': [], 'max': []},
        'rho': {'min': [], 'max': []},
        'nu': {'min': [], 'max': []}
    }
    
    # Convert to explicit format for validation
    explicit_props = design_to_explicit(design, design_format, E_min, E_max,
                                      rho_min, rho_max, poisson_min, poisson_max)
    
    # Check Young's modulus
    E = explicit_props['E']
    if np.any(E < E_min):
        violations['E']['min'] = np.where(E < E_min)
    if np.any(E > E_max):
        violations['E']['max'] = np.where(E > E_max)
    
    # Check density
    rho = explicit_props['rho']
    if np.any(rho < rho_min):
        violations['rho']['min'] = np.where(rho < rho_min)
    if np.any(rho > rho_max):
        violations['rho']['max'] = np.where(rho > rho_max)
    
    # Check Poisson's ratio
    nu = explicit_props['nu']
    if np.any(nu < poisson_min):
        violations['nu']['min'] = np.where(nu < poisson_min)
    if np.any(nu > poisson_max):
        violations['nu']['max'] = np.where(nu > poisson_max)
    
    # Check if any violations occurred
    is_valid = all(len(violations[prop][bound]) == 0 
                  for prop in violations 
                  for bound in violations[prop])
    
    return is_valid, violations


def clip_design_to_bounds(design, design_format='linear', E_min=2e9, E_max=200e9,
                         rho_min=1e3, rho_max=8e3, poisson_min=0.0, poisson_max=0.5):
    """
    Clip design values to specified bounds.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    design_format : str, optional
        Design format ('linear', 'log') (default: 'linear')
    E_min, E_max : float, optional
        Young's modulus bounds
    rho_min, rho_max : float, optional
        Density bounds
    poisson_min, poisson_max : float, optional
        Poisson's ratio bounds
        
    Returns
    -------
    clipped_design : array_like
        Design array with values clipped to bounds
    """
    
    # Convert to explicit format
    explicit_props = design_to_explicit(design, design_format, E_min, E_max,
                                      rho_min, rho_max, poisson_min, poisson_max)
    
    # Clip values
    E_clipped = np.clip(explicit_props['E'], E_min, E_max)
    rho_clipped = np.clip(explicit_props['rho'], rho_min, rho_max)
    nu_clipped = np.clip(explicit_props['nu'], poisson_min, poisson_max)
    
    # Convert back to design format
    clipped_design = explicit_to_design(E_clipped, rho_clipped, nu_clipped,
                                      design_format, E_min, E_max,
                                      rho_min, rho_max, poisson_min, poisson_max)
    
    return clipped_design

