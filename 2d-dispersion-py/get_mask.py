"""
Get mask function.

This module contains the function for getting masks based on symmetry type,
equivalent to MATLAB's get_mask.m.
"""

import numpy as np


def get_mask(symmetry_type, N_wv):
    """
    Get mask for symmetry type.
    
    This function matches MATLAB's get_mask.m exactly.
    
    Parameters
    ----------
    symmetry_type : str
        Type of symmetry ('c1m1', etc.)
    N_wv : array_like
        Number of wavevectors in each direction [N_wv_x, N_wv_y]
        
    Returns
    -------
    mask : array_like or None
        Boolean mask array, or None if symmetry type not supported
    """
    
    if symmetry_type == 'c1m1':
        # MATLAB: mask = triu(true([N_wv(1) (N_wv(2)+1)/2]));
        mask = np.triu(np.ones((N_wv[0], (N_wv[1] + 1) // 2), dtype=bool))
        # MATLAB: mask = [flipud(mask(2:end,:)); mask];
        mask = np.vstack([np.flipud(mask[1:, :]), mask])
        return mask
    else:
        # MATLAB function only handles 'c1m1', returns nothing for others
        return None

