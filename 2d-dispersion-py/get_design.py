"""
Design generation functions.

This module provides functions for generating various types of metamaterial
designs including predefined patterns and random designs.
"""

import numpy as np


def get_design(design_name, N_pix):
    """
    Generate a design based on the specified name and pixel dimensions.
    
    Parameters
    ----------
    design_name : str or int
        Name of the design pattern or random seed number
    N_pix : int or array_like
        Number of pixels in each direction
        
    Returns
    -------
    design : array_like
        3D array containing the design (N_pix x N_pix x 3)
        Third dimension: [0] = elastic modulus, [1] = density, [2] = Poisson's ratio
    """
    
    # Handle case where design_name is a number (random seed)
    try:
        seed = int(design_name)
        np.random.seed(seed)
        design = np.zeros((N_pix, N_pix, 3))
        design[:, :, 0] = np.round(np.random.rand(N_pix, N_pix))  # Elastic modulus
        design[:, :, 1] = design[:, :, 0]  # Density (same as elastic modulus)
        design[:, :, 2] = 0.6 * np.ones((N_pix, N_pix))  # Poisson's ratio
        return design
    except (ValueError, TypeError):
        pass
    
    # Handle named designs
    design = np.zeros((N_pix, N_pix, 3))
    
    if design_name == 'dispersive-tetragonal':
        # Dispersive cell - Tetragonal
        design[:, :, 0] = np.zeros((N_pix, N_pix))  # Elastic modulus
        idxs = slice(N_pix//4, 3*N_pix//4)
        design[idxs, idxs, 0] = 1
        design[:, :, 1] = design[:, :, 0]  # Density
        design[:, :, 2] = 0.6 * np.ones((N_pix, N_pix))  # Poisson's ratio
        
    elif design_name == 'dispersive-tetragonal-negative':
        # Dispersive cell - Tetragonal (negative)
        design[:, :, 0] = np.zeros((N_pix, N_pix))
        idxs = slice(N_pix//4, 3*N_pix//4)
        design[idxs, idxs, 0] = 1
        design[:, :, 1] = design[:, :, 0]
        design[:, :, 2] = 0.6 * np.ones((N_pix, N_pix))
        design[:, :, 0] = 1 - design[:, :, 0]  # Invert
        design[:, :, 1] = 1 - design[:, :, 1]  # Invert
        
    elif design_name == 'dispersive-orthotropic':
        # Dispersive cell - Orthotropic
        design[:, :, 0] = np.zeros((N_pix, N_pix))
        idxs = slice(N_pix//4, 3*N_pix//4)
        design[:, idxs, 0] = 1
        design[:, :, 1] = design[:, :, 0]
        design[:, :, 2] = 0.6 * np.ones((N_pix, N_pix))
        
    elif design_name == 'homogeneous':
        # Homogeneous cell
        design[:, :, 0] = np.ones((N_pix, N_pix))  # Elastic modulus
        design[:, :, 1] = design[:, :, 0]  # Density
        design[:, :, 2] = 0.6 * np.ones((N_pix, N_pix))  # Poisson's ratio
        
    elif design_name == 'quasi-1D':
        # Quasi-1D cell
        design[:, :, 0] = np.ones((N_pix, N_pix))
        design[:, ::2, 0] = 0  # Every other column
        design[:, :, 1] = design[:, :, 0]
        design[:, :, 2] = 0.6 * np.ones((N_pix, N_pix))
        
    elif design_name == 'rotationally-symmetric':
        # Rotationally symmetric design
        design[:, :, 0] = np.zeros((N_pix, N_pix))
        idxs1 = slice(N_pix//4, N_pix//2)
        idxs2 = slice(N_pix//2, 3*N_pix//4)
        design[idxs1, idxs1, 0] = 1
        design[idxs2, idxs2, 0] = 1
        design[:, :, 1] = design[:, :, 0]
        design[:, :, 2] = 0.6 * np.ones((N_pix, N_pix))
        
    elif design_name == 'dirac?':
        # Special Dirac-like design (5x5)
        design[:, :, 0] = np.zeros((5, 5))
        dirac_indices = [1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 24]
        for idx in dirac_indices:
            i, j = divmod(idx, 5)
            design[i, j, 0] = 1
        design[:, :, 1] = design[:, :, 0]
        design[:, :, 2] = 0.6 * np.ones((5, 5))
        
    elif design_name == 'correlated':
        # Load correlated design from external files
        try:
            # Note: These paths would need to be updated for the actual file locations
            # For now, we'll create a placeholder
            print("Warning: Correlated design requires external data files")
            design[:, :, 0] = np.random.rand(N_pix, N_pix)
            design[:, :, 1] = np.random.rand(N_pix, N_pix)
            design[:, :, 2] = 0.6 * np.ones((N_pix, N_pix))
        except:
            raise ValueError("Could not load correlated design data")
    
    else:
        raise ValueError(f'Design not recognized: {design_name}')
    
    return design

