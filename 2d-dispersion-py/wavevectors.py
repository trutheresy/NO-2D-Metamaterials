"""
Wavevector generation functions.

This module provides functions for generating wavevectors in the
irreducible Brillouin zone (IBZ) for different symmetry types.
"""

import numpy as np

# Handle both package and script imports
try:
    from .utils import linspaceNDim
except ImportError:
    from utils import linspaceNDim


def get_IBZ_wavevectors(N_wv, a, symmetry_type='none', N_tesselations=1):
    """
    Generate wavevectors in the irreducible Brillouin zone.
    
    Parameters
    ----------
    N_wv : int or array_like
        Number of wavevectors in each direction
    a : float
        Lattice parameter
    symmetry_type : str, optional
        Type of symmetry to apply (default: 'none')
        Options: 'none', 'omit', 'p4mm', 'c1m1', 'p2mm'
    N_tesselations : int, optional
        Number of tesselations (default: 1)
        
    Returns
    -------
    wavevectors : array_like
        Array of wavevectors (N x 2)
    """
    
    if np.isscalar(N_wv):
        N_wv = [N_wv, N_wv]
    
    if symmetry_type == 'omit':
        # Square centered at origin
        X, Y = np.meshgrid(np.linspace(-np.pi/a, np.pi/a, N_wv[0]),
                          np.linspace(-np.pi/a, np.pi/a, N_wv[1]))
        gamma_x = X.flatten()
        gamma_y = Y.flatten()
        
    elif symmetry_type == 'none':
        # Asymmetric IBZ (rectangle)
        X, Y = np.meshgrid(np.linspace(-np.pi/a, np.pi/a, N_wv[0]),
                          np.linspace(0, np.pi/a, N_wv[1]))
        gamma_x = X.flatten()
        gamma_y = Y.flatten()
        
    elif symmetry_type == 'p4mm':
        # P4mm symmetry (triangular IBZ)
        X, Y = np.meshgrid(np.linspace(0, np.pi/a, N_wv[0]),
                          np.linspace(0, np.pi/a, N_wv[1]))
        # Use upper triangular mask
        mask = np.triu(np.ones_like(X, dtype=bool))
        gamma_x = X[mask]
        gamma_y = Y[mask]
        
    elif symmetry_type == 'c1m1':
        # C1m1 symmetry
        if N_wv[1] % 2 == 0:
            raise ValueError('For symmetry type c1m1, N_wv[1] must be an odd integer')
        
        X, Y = np.meshgrid(np.linspace(0, np.pi/a, N_wv[0]),
                          np.linspace(-np.pi/a, np.pi/a, N_wv[1]))
        
        # Create mask for c1m1 symmetry
        half_size = (N_wv[1] + 1) // 2
        mask_upper = np.triu(np.ones([N_wv[0], half_size], dtype=bool))
        mask_lower = np.flipud(mask_upper[1:, :])  # Remove center row
        mask = np.vstack([mask_lower, mask_upper])
        
        gamma_x = X[mask]
        gamma_y = Y[mask]
        
    elif symmetry_type == 'p2mm':
        # P2mm symmetry (quarter of full BZ)
        X, Y = np.meshgrid(np.linspace(0, np.pi/a, N_wv[0]),
                          np.linspace(0, np.pi/a, N_wv[1]))
        gamma_x = X.flatten()
        gamma_y = Y.flatten()
        
    else:
        raise ValueError(f'symmetry_type "{symmetry_type}" not recognized')
    
    # Apply tesselations
    wavevectors = N_tesselations * np.column_stack([gamma_x, gamma_y])
    
    return wavevectors


def get_IBZ_contour_wavevectors(N_k, a, symmetry_type='none'):
    """
    Generate wavevectors along the boundary of the irreducible Brillouin zone.
    
    Parameters
    ----------
    N_k : int or array_like
        Number of wavevectors along each contour segment
        If array, only first element is used
    a : float
        Lattice parameter
    symmetry_type : str, optional
        Type of symmetry to apply (default: 'none')
        Options: 'p4mm', 'c1m1', 'p6mm', 'none', 'all contour segments'
        
    Returns
    -------
    wavevectors : ndarray
        Array of wavevectors along the IBZ boundary (N x 2)
    contour_info : dict
        Dictionary containing:
        - N_segment: Number of contour segments
        - vertex_labels: Labels for high-symmetry points
        - vertices: Coordinates of vertices
        - wavevector_parameter: Parameter along contour (0 to N_segment)
    """
    
    # Handle N_k as vector
    if hasattr(N_k, '__len__') and len(N_k) > 1:
        import warnings
        warnings.warn('received N_k as a vector, using first element')
        N_k = N_k[0]
    
    vertex_labels = []
    
    def get_contour_from_vertices(vertices, N_k):
        """Helper function to create contour from vertices."""
        wavevectors = np.empty((0, 2))
        for vertex_idx in range(len(vertices) - 1):
            # Generate points between consecutive vertices
            segment = linspaceNDim(vertices[vertex_idx], vertices[vertex_idx + 1], N_k)
            # Remove duplicate point (except for first segment)
            if vertex_idx > 0:
                segment = segment[1:]
            wavevectors = np.vstack([wavevectors, segment])
        return wavevectors
    
    if symmetry_type == 'p4mm':
        # Gamma -> X -> M -> Gamma
        vertices = np.array([
            [0, 0],           # Gamma
            [np.pi/a, 0],     # X
            [np.pi/a, np.pi/a],  # M
            [0, 0]            # Gamma
        ])
        wavevectors = get_contour_from_vertices(vertices, N_k)
        vertex_labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$']
        
    elif symmetry_type == 'c1m1':
        # Gamma -> X -> M -> Gamma -> O_bar -> X
        vertices = np.array([
            [0, 0],              # Gamma
            [np.pi/a, 0],        # X
            [np.pi/a, np.pi/a],  # M
            [0, 0],              # Gamma
            [np.pi/a, -np.pi/a], # O_bar
            [np.pi/a, 0]         # X
        ])
        wavevectors = get_contour_from_vertices(vertices, N_k)
        vertex_labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$', r'$\bar{O}$', r'$X$']
        
    elif symmetry_type == 'p6mm':
        # Gamma -> K -> M -> Gamma (for hexagonal symmetry)
        cos30 = np.cos(np.pi/6)
        sin30 = np.sin(np.pi/6)
        vertices = np.array([
            [0, 0],                                   # Gamma
            [np.pi/a * cos30 * cos30, -np.pi/a * cos30 * sin30],  # K
            [np.pi/a, 0],                             # M
            [0, 0]                                    # Gamma
        ])
        wavevectors = get_contour_from_vertices(vertices, N_k)
        vertex_labels = [r'$\Gamma$', r'$K$', r'$M$', r'$\Gamma$']
        
    elif symmetry_type == 'none':
        # Gamma -> X -> M -> Gamma -> Y -> O -> Gamma
        vertices = np.array([
            [0, 0],              # Gamma
            [np.pi/a, 0],        # X
            [np.pi/a, np.pi/a],  # M
            [0, 0],              # Gamma
            [0, np.pi/a],        # Y
            [-np.pi/a, np.pi/a], # O
            [0, 0]               # Gamma
        ])
        wavevectors = get_contour_from_vertices(vertices, N_k)
        vertex_labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$', r'$Y$', r'$O$', r'$\Gamma$']
        
    elif symmetry_type == 'all contour segments':
        # All possible contour segments from origin and perimeter
        vertices = np.array([
            [0, 0], [np.pi/a, 0],
            [0, 0], [np.pi/a, np.pi/a],
            [0, 0], [0, np.pi/a],
            [0, 0], [-np.pi/a, np.pi/a],
            [0, 0], [-np.pi/a, 0],
            [0, 0], [-np.pi/a, -np.pi/a],
            [0, 0], [0, -np.pi/a],
            [0, 0], [np.pi/a, -np.pi/a],
            [np.pi/a, 0], [np.pi/a, np.pi/a],
            [np.pi/a, np.pi/a], [0, np.pi/a],
            [0, np.pi/a], [-np.pi/a, np.pi/a],
            [-np.pi/a, np.pi/a], [-np.pi/a, 0],
            [-np.pi/a, 0], [-np.pi/a, -np.pi/a],
            [-np.pi/a, -np.pi/a], [0, -np.pi/a],
            [0, -np.pi/a], [np.pi/a, -np.pi/a],
            [np.pi/a, -np.pi/a], [np.pi/a, 0]
        ])
        
        wavevectors = np.empty((0, 2))
        for vertex_idx in range(0, len(vertices) - 1, 2):
            segment = linspaceNDim(vertices[vertex_idx], vertices[vertex_idx + 1], N_k)
            if vertex_idx <= 16:
                wavevectors = np.vstack([wavevectors, segment])
            else:
                # Remove duplicate point for later segments
                wavevectors = np.vstack([wavevectors[:-1], segment])
        vertex_labels = []  # Too many to label
        
    else:
        raise ValueError(f'symmetry_type "{symmetry_type}" not recognized')
    
    # Create contour info dictionary
    N_segment = len(vertices) - 1
    contour_info = {
        'N_segment': N_segment,
        'vertex_labels': vertex_labels,
        'vertices': vertices,
        'wavevector_parameter': np.linspace(0, N_segment, len(wavevectors))
    }
    
    if not vertex_labels:
        import warnings
        warnings.warn('critical_point_labels not yet defined for this symmetry_type')
    
    return wavevectors, contour_info


def apply_p4mm_symmetry(wavevectors, a):
    """
    Apply P4mm symmetry operations to expand wavevectors to full Brillouin zone.
    
    Parameters
    ----------
    wavevectors : array_like
        Wavevectors in irreducible Brillouin zone (N x 2)
    a : float
        Lattice parameter
        
    Returns
    -------
    full_wavevectors : array_like
        Wavevectors expanded to full Brillouin zone (4N x 2)
    """
    
    # P4mm symmetry operations:
    # 1. Identity
    # 2. 90-degree rotation
    # 3. 180-degree rotation  
    # 4. 270-degree rotation
    
    full_wavevectors = []
    
    for wv in wavevectors:
        kx, ky = wv[0], wv[1]
        
        # Apply all symmetry operations
        symmetry_ops = [
            [kx, ky],           # Identity
            [-ky, kx],          # 90-degree rotation
            [-kx, -ky],         # 180-degree rotation
            [ky, -kx]           # 270-degree rotation
        ]
        
        full_wavevectors.extend(symmetry_ops)
    
    return np.array(full_wavevectors)

