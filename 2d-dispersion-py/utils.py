"""
Utility functions for 2D dispersion analysis.

This module provides various utility functions for data processing,
validation, and other common operations.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve
import warnings


def linspaceNDim(d1, d2, n=100):
    """
    Linearly spaced multidimensional data set.
    
    Generates N points linearly spaced between each element of arrays d1 and d2.
    This is an N-dimensional generalization of numpy's linspace function.
    
    Parameters
    ----------
    d1 : array_like
        Starting point (1D array or list)
    d2 : array_like
        Ending point (1D array or list)
    n : int, optional
        Number of points to generate (default: 100)
        
    Returns
    -------
    Y : ndarray
        Array of shape (n, len(d1)) with points linearly spaced between d1 and d2
        
    Examples
    --------
    >>> d1 = [0, 0]
    >>> d2 = [1, 2]
    >>> Y = linspaceNDim(d1, d2, 5)
    # Y will be a 5x2 matrix with points linearly spaced between d1 and d2
    
    Notes
    -----
    Based on the linspace_NDim function by Steeve AMBROISE.
    See license in fileshare_licenses/linspace_NDim license.txt
    """
    
    d1 = np.atleast_1d(d1).flatten()
    d2 = np.atleast_1d(d2).flatten()
    
    if len(d1) != len(d2):
        raise ValueError('d1 and d2 must have the same number of elements')
    
    # Create linearly spaced points for each dimension
    Y = np.zeros((n, len(d1)))
    
    for i in range(len(d1)):
        Y[:, i] = np.linspace(d1[i], d2[i], n)
    
    return Y


def validate_constants(const):
    """
    Validate constants structure for required fields.
    
    Parameters
    ----------
    const : dict
        Constants structure to validate
        
    Returns
    -------
    is_valid : bool
        True if constants are valid
    missing_fields : list
        List of missing required fields
    """
    
    required_fields = [
        'a', 'N_ele', 'N_pix', 'N_eig', 'design', 'design_scale',
        'E_min', 'E_max', 'rho_min', 'rho_max', 'poisson_min', 'poisson_max',
        't', 'sigma_eig'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in const:
            missing_fields.append(field)
    
    is_valid = len(missing_fields) == 0
    
    if not is_valid:
        warnings.warn(f"Missing required fields: {missing_fields}")
    
    return is_valid, missing_fields


def check_contour_analysis(wavevectors, frequencies, tolerance=1e-6):
    """
    Check for contour analysis issues in dispersion data.
    
    Parameters
    ----------
    wavevectors : array_like
        Wavevectors (N x 2)
    frequencies : array_like
        Frequencies (N x N_eig)
    tolerance : float, optional
        Tolerance for checking issues (default: 1e-6)
        
    Returns
    -------
    issues : dict
        Dictionary containing detected issues
    """
    
    issues = {
        'negative_frequencies': False,
        'complex_frequencies': False,
        'nan_frequencies': False,
        'infinite_frequencies': False,
        'duplicate_wavevectors': False
    }
    
    # Check for negative frequencies
    if np.any(frequencies < -tolerance):
        issues['negative_frequencies'] = True
        warnings.warn("Negative frequencies detected")
    
    # Check for complex frequencies
    if np.any(np.imag(frequencies) > tolerance):
        issues['complex_frequencies'] = True
        warnings.warn("Complex frequencies detected")
    
    # Check for NaN frequencies
    if np.any(np.isnan(frequencies)):
        issues['nan_frequencies'] = True
        warnings.warn("NaN frequencies detected")
    
    # Check for infinite frequencies
    if np.any(np.isinf(frequencies)):
        issues['infinite_frequencies'] = True
        warnings.warn("Infinite frequencies detected")
    
    # Check for duplicate wavevectors
    unique_wavevectors = np.unique(wavevectors, axis=0)
    if len(unique_wavevectors) < len(wavevectors):
        issues['duplicate_wavevectors'] = True
        warnings.warn("Duplicate wavevectors detected")
    
    return issues


def convert_design_scale(design, from_scale, to_scale, E_min=2e9, E_max=200e9, 
                        rho_min=1e3, rho_max=8e3):
    """
    Convert design between linear and logarithmic scales.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    from_scale : str
        Source scale ('linear' or 'log')
    to_scale : str
        Target scale ('linear' or 'log')
    E_min, E_max : float, optional
        Young's modulus bounds
    rho_min, rho_max : float, optional
        Density bounds
        
    Returns
    -------
    converted_design : array_like
        Converted design array
    """
    
    if from_scale == to_scale:
        return design.copy()
    
    converted_design = design.copy()
    
    if from_scale == 'linear' and to_scale == 'log':
        # Convert from linear to log scale
        # E: linear -> log
        E_linear = E_min + design[:, :, 0] * (E_max - E_min)
        converted_design[:, :, 0] = np.log(E_linear)
        
        # rho: linear -> log
        rho_linear = rho_min + design[:, :, 1] * (rho_max - rho_min)
        converted_design[:, :, 1] = np.log(rho_linear)
        
    elif from_scale == 'log' and to_scale == 'linear':
        # Convert from log to linear scale
        # E: log -> linear
        E_log = design[:, :, 0]
        E_linear = np.exp(E_log)
        converted_design[:, :, 0] = (E_linear - E_min) / (E_max - E_min)
        
        # rho: log -> linear
        rho_log = design[:, :, 1]
        rho_linear = np.exp(rho_log)
        converted_design[:, :, 1] = (rho_linear - rho_min) / (rho_max - rho_min)
    
    else:
        raise ValueError(f"Invalid scale conversion: {from_scale} -> {to_scale}")
    
    return converted_design


def get_mask(design, threshold=0.5):
    """
    Get binary mask from design.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    threshold : float, optional
        Threshold for binary mask (default: 0.5)
        
    Returns
    -------
    mask : array_like
        Binary mask (N_pix x N_pix)
    """
    
    # Use elastic modulus channel for mask
    return (design[:, :, 0] > threshold).astype(int)


def slicing_data(data, slice_indices, axis=0):
    """
    Slice data along specified axis.
    
    Parameters
    ----------
    data : array_like
        Data array to slice
    slice_indices : array_like
        Indices to slice
    axis : int, optional
        Axis along which to slice (default: 0)
        
    Returns
    -------
    sliced_data : array_like
        Sliced data array
    """
    
    # Create slice object
    slice_obj = [slice(None)] * data.ndim
    slice_obj[axis] = slice_indices
    
    return data[tuple(slice_obj)]


def get_prop(design, prop_type='modulus'):
    """
    Extract specific property from design.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix x N_pix x 3)
    prop_type : str, optional
        Property type ('modulus', 'density', 'poisson') (default: 'modulus')
        
    Returns
    -------
    prop : array_like
        Property array (N_pix x N_pix)
    """
    
    prop_map = {
        'modulus': 0,
        'density': 1,
        'poisson': 2
    }
    
    if prop_type not in prop_map:
        raise ValueError(f"Unknown property type: {prop_type}")
    
    return design[:, :, prop_map[prop_type]]


def compute_band_gap(frequencies, gap_threshold=0.01):
    """
    Compute band gap from frequency data.
    
    Parameters
    ----------
    frequencies : array_like
        Frequency data (N_wavevectors x N_eig)
    gap_threshold : float, optional
        Minimum gap size to consider (default: 0.01)
        
    Returns
    -------
    band_gaps : array_like
        Band gaps (N_wavevectors,)
    gap_locations : array_like
        Locations of band gaps (N_wavevectors,)
    """
    
    # Compute gaps between consecutive bands
    band_gaps = np.diff(frequencies, axis=1)
    
    # Find significant gaps
    significant_gaps = band_gaps > gap_threshold
    
    # Get gap locations and sizes
    gap_locations = np.argmax(significant_gaps, axis=1)
    max_gaps = np.max(band_gaps, axis=1)
    
    return max_gaps, gap_locations


def normalize_eigenvectors(eigenvectors, method='mass'):
    """
    Normalize eigenvectors.
    
    Parameters
    ----------
    eigenvectors : array_like
        Eigenvectors (N_dof x N_wavevectors x N_eig)
    method : str, optional
        Normalization method ('mass', 'max', 'norm') (default: 'mass')
        
    Returns
    -------
    normalized_eigenvectors : array_like
        Normalized eigenvectors
    """
    
    if method == 'mass':
        # Mass normalization (requires mass matrix - simplified here)
        norms = np.linalg.norm(eigenvectors, axis=0)
        return eigenvectors / norms
    
    elif method == 'max':
        # Maximum value normalization
        max_vals = np.max(np.abs(eigenvectors), axis=0)
        return eigenvectors / max_vals
    
    elif method == 'norm':
        # L2 norm normalization
        norms = np.linalg.norm(eigenvectors, axis=0)
        return eigenvectors / norms
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_group_velocity_approximation(frequencies, wavevectors, delta_k=1e-6):
    """
    Compute group velocity using finite difference approximation.
    
    Parameters
    ----------
    frequencies : array_like
        Frequency data (N_wavevectors x N_eig)
    wavevectors : array_like
        Wavevector data (N_wavevectors x 2)
    delta_k : float, optional
        Small perturbation for finite difference (default: 1e-6)
        
    Returns
    -------
    group_velocities : array_like
        Group velocities (N_wavevectors x 2 x N_eig)
    """
    
    N_wv, N_eig = frequencies.shape
    group_velocities = np.zeros((N_wv, 2, N_eig))
    
    for i in range(N_wv):
        for j in range(2):  # x and y components
            # Find nearby wavevectors
            k_perturbed = wavevectors[i].copy()
            k_perturbed[j] += delta_k
            
            # Find closest wavevector in the set
            distances = np.linalg.norm(wavevectors - k_perturbed, axis=1)
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < delta_k * 2:  # Within reasonable distance
                # Compute finite difference
                df_dk = (frequencies[closest_idx] - frequencies[i]) / delta_k
                group_velocities[i, j, :] = df_dk
    
    return group_velocities

