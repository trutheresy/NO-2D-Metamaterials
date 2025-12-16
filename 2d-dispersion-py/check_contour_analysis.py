"""
Check contour analysis function.

This module contains the function for checking contour analysis,
equivalent to MATLAB's check_contour_analysis.m script functionality.
Note: The MATLAB version is a script, this is a function version.
"""

import numpy as np
from scipy.interpolate import griddata


def check_contour_analysis(data_fn, full_bg_thresh=100, contour_bg_thresh=100, 
                           bg_width_error_thresh=50):
    """
    Check contour analysis for dispersion data.
    
    This function provides functionality equivalent to MATLAB's check_contour_analysis.m script.
    
    Parameters
    ----------
    data_fn : str
        Path to data file (MATLAB .mat file)
    full_bg_thresh : float, optional
        Threshold for full band gap detection (default: 100)
    contour_bg_thresh : float, optional
        Threshold for contour band gap detection (default: 100)
    bg_width_error_thresh : float, optional
        Threshold for band gap width error (default: 50)
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - contour_bg: boolean array of contour band gaps
        - full_bg: boolean array of full band gaps
        - contour_bg_width: contour band gap widths
        - full_bg_width: full band gap widths
        - bg_width_error: band gap width errors
    """
    
    # Load data (would need mat73_loader or scipy.io.loadmat)
    try:
        from mat73_loader import load_matlab_v73
        data = load_matlab_v73(data_fn)
    except:
        import scipy.io as sio
        data = sio.loadmat(data_fn)
    
    # Extract variables (MATLAB script uses unpack_struct)
    WAVEVECTOR_DATA = data['WAVEVECTOR_DATA']
    EIGENVALUE_DATA = data['EIGENVALUE_DATA']
    const = data['const']
    design_parameters = data.get('design_parameters', {})
    
    # Get symmetry type
    if isinstance(design_parameters, dict):
        design_options = design_parameters.get('design_options', {})
        if isinstance(design_options, dict):
            symmetry_type = design_options.get('symmetry_type', 'none')
        else:
            symmetry_type = 'none'
    else:
        symmetry_type = 'none'
    
    # Create list of wavevectors that define the IBZ contour
    from get_IBZ_contour_wavevectors import get_IBZ_contour_wavevectors
    N_wv = const['N_wv'] if isinstance(const, dict) else const.N_wv
    a = const['a'] if isinstance(const, dict) else const.a
    contour_wavevectors, contour_info = get_IBZ_contour_wavevectors(N_wv[0], a, symmetry_type)
    
    N_struct = EIGENVALUE_DATA.shape[2] if len(EIGENVALUE_DATA.shape) > 2 else 1
    N_eig = const['N_eig'] if isinstance(const, dict) else const.N_eig
    
    wv_x = WAVEVECTOR_DATA[:, 0, 0]
    wv_y = WAVEVECTOR_DATA[:, 1, 0]
    
    contour_bg = np.zeros((N_eig - 1, N_struct), dtype=bool)
    full_bg = np.zeros((N_eig - 1, N_struct), dtype=bool)
    contour_bg_width = np.zeros((N_eig - 1, N_struct))
    full_bg_width = np.zeros((N_eig - 1, N_struct))
    bg_width_error = np.zeros((N_eig - 1, N_struct))
    
    for struct_idx in range(N_struct):
        for eig_idx in range(N_eig - 1):
            # Lower band
            full_frequencies = EIGENVALUE_DATA[:, eig_idx, struct_idx]
            
            # Interpolate to contour
            contour_frequencies = griddata(
                (wv_x, wv_y), full_frequencies,
                (contour_wavevectors[:, 0], contour_wavevectors[:, 1]),
                method='linear'
            )
            
            # Eval max of lower band
            full_max = np.max(full_frequencies)
            contour_max = np.max(contour_frequencies)
            
            # Upper band
            full_frequencies = EIGENVALUE_DATA[:, eig_idx + 1, struct_idx]
            
            # Interpolate to contour
            contour_frequencies = griddata(
                (wv_x, wv_y), full_frequencies,
                (contour_wavevectors[:, 0], contour_wavevectors[:, 1]),
                method='linear'
            )
            
            # Eval min of upper band
            full_min = np.min(full_frequencies)
            contour_min = np.min(contour_frequencies)
            
            # Process mins and maxes
            if (contour_min - contour_max) > contour_bg_thresh:
                contour_bg[eig_idx, struct_idx] = True
            contour_bg_width[eig_idx, struct_idx] = max(contour_min - contour_max, 0)
            
            if full_min - full_max > full_bg_thresh:
                full_bg[eig_idx, struct_idx] = True
            full_bg_width[eig_idx, struct_idx] = max(full_min - full_max, 0)
            
            bg_width_error[eig_idx, struct_idx] = (contour_bg_width[eig_idx, struct_idx] - 
                                                   full_bg_width[eig_idx, struct_idx])
            if bg_width_error[eig_idx, struct_idx] > bg_width_error_thresh:
                print(f'Contour analysis fails for struct_idx = {struct_idx}, '
                      f'eig_idxs = [{eig_idx} {eig_idx+1}]')
    
    N_full_bg = np.sum(full_bg)
    N_contour_bg = np.sum(contour_bg)
    
    results = {
        'contour_bg': contour_bg,
        'full_bg': full_bg,
        'contour_bg_width': contour_bg_width,
        'full_bg_width': full_bg_width,
        'bg_width_error': bg_width_error,
        'N_full_bg': N_full_bg,
        'N_contour_bg': N_contour_bg
    }
    
    return results

