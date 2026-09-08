"""
Global DOF indexing functions.

This module provides functions for mapping local element DOF indices
to global system DOF indices.
"""

import numpy as np


def get_global_idxs(ele_idx_x, ele_idx_y, const):
    """
    Get global DOF indices for an element.
    
    Parameters
    ----------
    ele_idx_x : int
        Element index in x-direction
    ele_idx_y : int
        Element index in y-direction
    const : dict
        Constants structure containing system parameters
        
    Returns
    -------
    global_idxs : array_like
        Array of 8 global DOF indices for the element
        Order: [u1, v1, u2, v2, u3, v3, u4, v4]
        (counterclockwise from lower left node)
    """
    
    # Handle both scalar and list N_pix
    N_pix = const['N_pix']
    if isinstance(N_pix, (list, tuple)):
        N_pix_x, N_pix_y = N_pix[0], N_pix[1]
    else:
        N_pix_x = N_pix_y = N_pix
    
    N_ele_x = N_pix_x * const['N_ele']
    N_ele_y = N_pix_y * const['N_ele']
    N_nodes_x = N_ele_x + 1
    N_nodes_y = N_ele_y + 1
    
    # NOTE: This function receives 1-based indices from MATLAB-style loops
    # Local node numbering (counterclockwise from lower left in MATLAB convention)
    # Node 1: upper left
    # Node 2: upper right  
    # Node 3: lower right
    # Node 4: lower left
    
    N_node_y = N_ele_y + 1
    
    # Compute all four global node indices in MATLAB node order.
    # Node 1: upper left
    g1 = ((ele_idx_y + 1) - 1) * N_node_y + ele_idx_x
    # Node 2: upper right
    g2 = ((ele_idx_y + 1) - 1) * N_node_y + (ele_idx_x + 1)
    # Node 3: lower right
    g3 = (ele_idx_y - 1) * N_node_y + (ele_idx_x + 1)
    # Node 4: lower left
    g4 = (ele_idx_y - 1) * N_node_y + ele_idx_x

    # Fixed-size DOF output avoids repeated np.append reallocations.
    return np.array(
        [
            2 * g1 - 2, 2 * g1 - 1,
            2 * g2 - 2, 2 * g2 - 1,
            2 * g3 - 2, 2 * g3 - 1,
            2 * g4 - 2, 2 * g4 - 1,
        ],
        dtype=np.int64,
    )

