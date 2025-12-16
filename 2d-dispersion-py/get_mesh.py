"""
Get mesh function.

This module contains the function for generating mesh information,
equivalent to MATLAB's get_mesh.m.
"""

import numpy as np


def get_mesh(const):
    """
    Get mesh information.
    
    This function matches MATLAB's get_mesh.m exactly.
    
    Parameters
    ----------
    const : dict
        Constants structure containing system parameters
        
    Returns
    -------
    mesh : dict
        Mesh structure containing:
        - dim: dimensionality of physical space
        - node_coords: cell array of node coordinates
    """
    
    # Dimensionality of physical space
    dim = 2  # Hard code 2-dimensional for 2D-dispersion
    
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_val = N_pix[0]
    else:
        N_pix_val = N_pix
    
    # Compute number of nodes along each side length of the unit cell
    N_node = const['N_ele'] * N_pix_val + 1
    N_node = [N_node] * dim
    
    # Compute size of unit cell
    unit_cell_size = const['a']
    unit_cell_size = [unit_cell_size] * dim
    
    # Compute node locations
    node_loc_grid_vec = []
    for i in range(dim):
        node_loc_grid_vec.append(np.linspace(0, unit_cell_size[i], N_node[i]))
    
    # Create grid using numpy's meshgrid (equivalent to MATLAB's ndgrid)
    # MATLAB's ndgrid returns column-major ordering, numpy's meshgrid with indexing='ij' does the same
    node_coords_grid = np.meshgrid(*node_loc_grid_vec, indexing='ij')
    
    # Convert to list of arrays (like MATLAB cell array)
    node_coords = [node_coords_grid[i] for i in range(dim)]
    
    # Attach to mesh dict
    mesh = {
        'dim': dim,
        'node_coords': node_coords
    }
    
    return mesh

