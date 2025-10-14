"""
Property generation functions.

This module contains functions for generating material property distributions
based on design parameters, equivalent to the MATLAB get_prop.m function.
"""

import numpy as np
from symmetry import apply_p4mm_symmetry
from kernels import kernel_prop


def get_prop(design_parameters, prop_idx):
    """
    Generate material property distribution based on design parameters.
    
    This is the exact translation of the MATLAB get_prop function.
    
    Parameters
    ----------
    design_parameters : DesignParameters
        Design parameters object containing style and options
    prop_idx : int
        Property index (1-based indexing)
        
    Returns
    -------
    prop : ndarray
        Property distribution (N_pix x N_pix)
    """
    
    # Extract parameters for this property
    design_style = design_parameters.design_style[prop_idx - 1]  # Convert to 0-based
    design_options = design_parameters.design_options[prop_idx - 1]
    N_pix = design_parameters.N_pix
    
    # Get design number if available
    if hasattr(design_parameters, 'design_number') and design_parameters.design_number is not None:
        design_number = design_parameters.design_number[prop_idx - 1]
    else:
        design_number = None
    
    # Generate property based on design style
    if design_style == 'constant':
        prop = design_options['constant_value'] * np.ones((N_pix, N_pix))
        
    elif design_style == 'uncorrelated':
        if design_number is not None:
            np.random.seed(design_number)
        prop = np.random.rand(N_pix, N_pix)
        
    elif design_style == 'kernel':
        if design_number is not None:
            np.random.seed(design_number)
        prop = kernel_prop(design_options['kernel'], N_pix, design_options)
        
    elif design_style == 'diagonal-band':
        prop = np.eye(N_pix)
        for i in range(1, design_options['feature_size'] + 1):
            prop += np.diag(np.ones(N_pix - i), i)
            prop += np.diag(np.ones(N_pix - i), -i)
            
    elif design_style == 'dispersive-tetragonal':
        # Dispersive cell - Tetragonal
        N_pix_inclusion = design_options['feature_size']
        prop = np.zeros((N_pix, N_pix))
        mask = np.zeros(N_pix)
        mask[:N_pix_inclusion] = 1
        mask = np.roll(mask, round((N_pix - N_pix_inclusion) / 2))
        idxs = np.where(mask)[0]
        prop[np.ix_(idxs, idxs)] = 1
        
    elif design_style == 'dispersive-tetragonal-negative':
        # Dispersive cell - Tetragonal (negative)
        prop = np.zeros((N_pix, N_pix))
        mask = slice(round(N_pix/4), round(3*N_pix/4))
        prop[mask, mask] = 1
        prop = ~prop.astype(bool)  # negative!
        
    elif design_style == 'dispersive-orthotropic':
        # Dispersive cell - Orthotropic
        prop = np.zeros((N_pix, N_pix))
        mask = slice(round(N_pix/4), round(3*N_pix/4))
        prop[:, mask] = 1
        
    elif design_style == 'homogeneous':
        # Homogeneous cell
        prop = np.ones((N_pix, N_pix))
        
    elif design_style == 'quasi-1D':
        # Quasi-1D cell
        prop = np.ones((N_pix, N_pix))
        prop[:, 0::2] = 0
        
    elif design_style == 'rotationally-symmetric':
        prop = np.zeros((N_pix, N_pix))
        mask1 = slice(round(N_pix/4), round(2*N_pix/4))
        prop[mask1, mask1] = 1
        mask2 = slice(round(2*N_pix/4), round(3*N_pix/4))
        prop[mask2, mask2] = 1
        
    elif design_style == 'sierpinski':
        ratio = N_pix / 30
        prop = np.ones((N_pix, N_pix))
        
        # First mask
        mask = np.zeros_like(prop, dtype=bool)
        cols = slice(int(ratio*2), int(ratio*14))
        rows = slice(int(ratio*(30-14)), int(ratio*(30-2)))
        mask[rows, cols] = np.triu(np.ones((int(ratio*12), int(ratio*12))), k=0).astype(bool)
        prop[mask] = 0
        
        # Second mask
        mask = np.zeros_like(prop, dtype=bool)
        cols = slice(int(ratio*4), int(ratio*28))
        rows = slice(int(ratio*(30-28)), int(ratio*(30-4)))
        mask[rows, cols] = np.triu(np.ones((int(ratio*24), int(ratio*24))), k=0).astype(bool)
        prop[mask] = 0
        
    else:
        raise ValueError(f'Design not recognized: {design_style}')
    
    # Apply symmetry if specified
    if 'symmetry_type' in design_options:
        if design_options['symmetry_type'] == 'c1m1':
            orig_min = np.min(prop)
            orig_range = np.ptp(prop)
            prop = 0.5 * prop + 0.5 * prop.T
            new_range = np.ptp(prop)
            prop = orig_range / new_range * prop
            new_min = np.min(prop)
            prop = prop - new_min + orig_min
            
        elif design_options['symmetry_type'] == 'p4mm':
            prop = apply_p4mm_symmetry(prop)
            
        elif design_options['symmetry_type'] == 'none':
            # do nothing
            pass
        else:
            raise ValueError('symmetry_type not recognized')
    
    # Apply discretization if specified
    if 'N_value' in design_options and design_options['N_value'] != np.inf:
        prop = np.round((design_options['N_value'] - 1) * prop) / (design_options['N_value'] - 1)
    
    return prop
