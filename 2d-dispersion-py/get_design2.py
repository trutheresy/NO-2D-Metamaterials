"""
Alternative design generation function.

This module contains the get_design2 function which is an alternative
to get_design, equivalent to the MATLAB get_design2.m function.
"""

import numpy as np
from get_prop import get_prop


def get_design2(design_parameters):
    """
    Generate design using alternative method.
    
    This is the exact translation of the MATLAB get_design2 function.
    
    Parameters
    ----------
    design_parameters : DesignParameters
        Design parameters object
        
    Returns
    -------
    design : ndarray
        Design array of shape (N_pix, N_pix, 3)
    """
    
    # Handle both scalar and list N_pix (matching MATLAB behavior)
    if isinstance(design_parameters.N_pix, (list, tuple)):
        N_pix_val = design_parameters.N_pix[0]
    else:
        N_pix_val = design_parameters.N_pix
    
    # Initialize design array
    design = np.zeros((N_pix_val, N_pix_val, 3))
    
    # Generate each property separately
    for prop_idx in range(1, 4):  # 1-based indexing as in MATLAB
        design[:, :, prop_idx - 1] = get_prop(design_parameters, prop_idx)
    
    return design
