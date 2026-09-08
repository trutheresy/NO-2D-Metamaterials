"""
Symmetry operations for metamaterial designs.

This module provides functions for applying various symmetry operations
to metamaterial designs and dispersion data.
"""

import numpy as np


def apply_p4mm_symmetry(A):
    """
    Apply P4mm symmetry operations to a 2D array.
    
    Parameters
    ----------
    A : array_like
        2D array to symmetrize
        
    Returns
    -------
    A_sym : array_like
        Symmetrized array with P4mm symmetry
    """
    
    # Measure original data range
    orig_range = [np.min(A), np.max(A)]
    
    # Apply symmetry operations
    # 1. Left-right symmetry
    A = 0.5 * (A + np.fliplr(A))
    
    # 2. Top-bottom symmetry
    A = 0.5 * (A + np.flipud(A))
    
    # 3. Diagonal symmetry
    A = 0.5 * (A + A.T)
    
    # 4. Anti-diagonal symmetry
    A = 0.5 * (np.fliplr(np.flipud(A)) + np.fliplr(np.flipud(A)).T)
    
    A_sym = A
    
    # Normalize to preserve original range
    A_sym = A_sym - np.min(A_sym)
    A_sym = A_sym / np.max(A_sym)
    A_sym = A_sym * (orig_range[1] - orig_range[0])
    A_sym = A_sym + orig_range[0]
    
    return A_sym


def apply_rotational_symmetry(design, n_fold=4):
    """
    Apply n-fold rotational symmetry to a design.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    n_fold : int, optional
        Number of fold symmetry (default: 4)
        
    Returns
    -------
    symmetric_design : array_like
        Design with rotational symmetry applied
    """
    
    symmetric_design = design.copy()
    
    for i in range(3):  # Apply to each property channel
        prop = design[:, :, i]
        
        # Create symmetric version by averaging rotated versions
        symmetric_prop = np.zeros_like(prop)
        
        for k in range(n_fold):
            angle = 2 * np.pi * k / n_fold
            rotated = rotate_2d_array(prop, angle)
            symmetric_prop += rotated
        
        symmetric_prop /= n_fold
        symmetric_design[:, :, i] = symmetric_prop
    
    return symmetric_design


def rotate_2d_array(A, angle):
    """
    Rotate a 2D array by a given angle.
    
    Parameters
    ----------
    A : array_like
        2D array to rotate
    angle : float
        Rotation angle in radians
        
    Returns
    -------
    rotated_A : array_like
        Rotated array
    """
    
    from scipy.ndimage import rotate
    
    # Convert angle to degrees
    angle_deg = np.degrees(angle)
    
    # Rotate array
    rotated_A = rotate(A, angle_deg, reshape=False, order=1)
    
    return rotated_A


def apply_mirror_symmetry(design, axis='both'):
    """
    Apply mirror symmetry to a design.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    axis : str, optional
        Axis for mirror symmetry ('x', 'y', or 'both') (default: 'both')
        
    Returns
    -------
    symmetric_design : array_like
        Design with mirror symmetry applied
    """
    
    symmetric_design = design.copy()
    
    for i in range(3):  # Apply to each property channel
        prop = design[:, :, i]
        
        if axis == 'x' or axis == 'both':
            prop = 0.5 * (prop + np.fliplr(prop))
        
        if axis == 'y' or axis == 'both':
            prop = 0.5 * (prop + np.flipud(prop))
        
        symmetric_design[:, :, i] = prop
    
    return symmetric_design


def check_symmetry(design, symmetry_type='p4mm', tolerance=1e-6):
    """
    Check if a design has the specified symmetry.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    symmetry_type : str, optional
        Type of symmetry to check (default: 'p4mm')
    tolerance : float, optional
        Tolerance for symmetry check (default: 1e-6)
        
    Returns
    -------
    is_symmetric : bool
        True if design has the specified symmetry
    max_error : float
        Maximum error in symmetry
    """
    
    if symmetry_type == 'p4mm':
        # Check P4mm symmetry
        prop = design[:, :, 0]  # Use first property channel
        
        # Check left-right symmetry
        lr_error = np.max(np.abs(prop - np.fliplr(prop)))
        
        # Check top-bottom symmetry
        tb_error = np.max(np.abs(prop - np.flipud(prop)))
        
        # Check diagonal symmetry
        diag_error = np.max(np.abs(prop - prop.T))
        
        # Check anti-diagonal symmetry
        anti_diag_error = np.max(np.abs(prop - np.fliplr(np.flipud(prop)).T))
        
        max_error = max(lr_error, tb_error, diag_error, anti_diag_error)
        is_symmetric = max_error < tolerance
        
    elif symmetry_type == 'rotational':
        # Check rotational symmetry
        prop = design[:, :, 0]
        rotated = rotate_2d_array(prop, np.pi/2)
        max_error = np.max(np.abs(prop - rotated))
        is_symmetric = max_error < tolerance
        
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry_type}")
    
    return is_symmetric, max_error


def enforce_symmetry(design, symmetry_type='p4mm'):
    """
    Enforce symmetry on a design.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    symmetry_type : str, optional
        Type of symmetry to enforce (default: 'p4mm')
        
    Returns
    -------
    symmetric_design : array_like
        Design with enforced symmetry
    """
    
    if symmetry_type == 'p4mm':
        symmetric_design = design.copy()
        for i in range(3):
            symmetric_design[:, :, i] = apply_p4mm_symmetry(design[:, :, i])
    
    elif symmetry_type == 'rotational':
        symmetric_design = apply_rotational_symmetry(design)
    
    elif symmetry_type == 'mirror':
        symmetric_design = apply_mirror_symmetry(design)
    
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry_type}")
    
    return symmetric_design

