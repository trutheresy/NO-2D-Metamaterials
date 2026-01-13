#!/usr/bin/env python3
"""
Reduced PyTorch to MATLAB Dataset Converter (Inverse Function)

This script converts reduced PyTorch `.pt` format back to MATLAB `.mat` format.

Features:
- Loads reduced PyTorch dataset from a folder
- Reconstructs full MATLAB structure from reduced data
- Reconstructs EIGENVECTOR_DATA from reduced displacements using indices
- Reconstructs EIGENVALUE_DATA from eigenvectors using K, M, T matrices
- Reconstructs designs with all panes
- Saves as MATLAB v7.3 `.mat` file

Note: This is the inverse function of matlab_to_reduced_pt.py
"""

import numpy as np
import torch
import h5py
import scipy.io
from pathlib import Path
import os
import time
import argparse
import sys
import scipy.sparse as sp

# Custom utilities
try:
    import NO_utils
    import NO_utils_multiple
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Please ensure NO_utils.py and NO_utils_multiple.py are in the same directory or PYTHONPATH")
    sys.exit(1)

# Import functions for computing K, M, T matrices and reconstructing frequencies
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC, get_system_matrices_VEC_simplified
from system_matrices import get_transformation_matrix
from scipy.interpolate import interp1d


def apply_steel_rubber_paradigm(design, const):
    """
    Python implementation of MATLAB's apply_steel_rubber_paradigm function.
    
    This function takes a single-channel design array and applies the steel-rubber
    paradigm to create a 3-channel design with proper material property mapping.
    
    Parameters
    ----------
    design : ndarray
        Single-channel design array (N_pix, N_pix) with values in [0, 1]
    const : dict
        Constants dictionary with material property bounds:
        - E_min, E_max: Young's modulus bounds
        - rho_min, rho_max: Density bounds
        - poisson_min, poisson_max: Poisson's ratio bounds
        
    Returns
    -------
    design_out : ndarray
        3-channel design array (N_pix, N_pix, 3)
    """
    # Hardcoded material property values from apply_steel_rubber_paradigm.m
    design_in_polymer = 0.0
    design_in_steel = 1.0
    
    E_polymer = 100e6
    E_steel = 200e9
    
    rho_polymer = 1200.0
    rho_steel = 8e3
    
    nu_polymer = 0.45
    nu_steel = 0.3
    
    # Extract const values
    E_min = const['E_min']
    E_max = const['E_max']
    rho_min = const['rho_min']
    rho_max = const['rho_max']
    poisson_min = const['poisson_min']
    poisson_max = const['poisson_max']
    
    # Compute design output values for polymer and steel
    design_out_polymer_E = (E_polymer - E_min) / (E_max - E_min)
    design_out_polymer_rho = (rho_polymer - rho_min) / (rho_max - rho_min)
    design_out_polymer_nu = (nu_polymer - poisson_min) / (poisson_max - poisson_min)
    
    design_out_steel_E = (E_steel - E_min) / (E_max - E_min)
    design_out_steel_rho = (rho_steel - rho_min) / (rho_max - rho_min)
    design_out_steel_nu = (nu_steel - poisson_min) / (poisson_max - poisson_min)
    
    # Prepare design values array
    design_vals = np.array([
        [design_out_polymer_E, design_out_steel_E],
        [design_out_polymer_rho, design_out_steel_rho],
        [design_out_polymer_nu, design_out_steel_nu]
    ])
    
    # Initialize output design (3 channels)
    N_pix = design.shape[0]
    design_out = np.zeros((N_pix, N_pix, 3), dtype=np.float64)
    
    # Interpolate for each property (pane) using linear interpolation
    # This matches MATLAB's interp1([design_in_polymer design_in_steel], dvs, design)
    x_points = np.array([design_in_polymer, design_in_steel])
    for prop_idx in range(3):
        dvs = design_vals[prop_idx, :]
        # Use linear interpolation (MATLAB's interp1 with 'linear' method)
        interp_func = interp1d(x_points, dvs, kind='linear', 
                               bounds_error=False, fill_value=(dvs[0], dvs[1]))
        design_out[:, :, prop_idx] = interp_func(design)
    
    return design_out


def interleave_arrays(arr1, arr2, dim):
    """
    Interleave two arrays along a specified dimension (inverse of split_array).
    
    Parameters:
    -----------
    arr1 : np.ndarray
        First array (elements at even indices)
    arr2 : np.ndarray
        Second array (elements at odd indices)
    dim : int
        Dimension along which to interleave
    
    Returns:
    --------
    np.ndarray : Interleaved array
    """
    if arr1.shape != arr2.shape:
        raise ValueError(f"Arrays must have the same shape. Got {arr1.shape} and {arr2.shape}")
    
    shape = list(arr1.shape)
    if shape[dim] % 2 != 0:
        raise ValueError(f"Dimension {dim} must be even for interleaving")
    
    # Create output shape with doubled dimension
    output_shape = shape.copy()
    output_shape[dim] = shape[dim] * 2
    
    # Create output array
    output = np.zeros(output_shape, dtype=arr1.dtype)
    
    # Create slices for interleaving
    slices_even = [slice(None)] * len(shape)
    slices_odd = [slice(None)] * len(shape)
    slices_even[dim] = slice(None, None, 2)
    slices_odd[dim] = slice(1, None, 2)
    
    output[tuple(slices_even)] = arr1
    output[tuple(slices_odd)] = arr2
    
    return output


def create_full_indices(n_designs, n_wavevectors, n_bands):
    """
    Create full indices for all combinations of designs, wavevectors, and bands.
    Optimized with vectorized operations using meshgrid.
    
    Parameters:
    -----------
    n_designs : int
        Number of designs
    n_wavevectors : int
        Number of wavevectors
    n_bands : int
        Number of bands
    
    Returns:
    --------
    torch.Tensor : Tensor of shape (n_designs * n_wavevectors * n_bands, 3) with indices
    """
    # Use meshgrid for efficient generation of all combinations
    d_indices = np.arange(n_designs, dtype=np.int64)
    w_indices = np.arange(n_wavevectors, dtype=np.int64)
    b_indices = np.arange(n_bands, dtype=np.int64)
    
    # Create meshgrid and flatten
    D, W, B = np.meshgrid(d_indices, w_indices, b_indices, indexing='ij')
    
    # Stack and reshape to (n_designs * n_wavevectors * n_bands, 3)
    indices = np.stack([D.flatten(), W.flatten(), B.flatten()], axis=1)
    
    # Convert to torch tensor
    return torch.from_numpy(indices)


def convert_reduced_pt_to_matlab(pt_input_path, matlab_output_path, use_predictions=True):
    """
    Convert reduced PyTorch dataset back to MATLAB .mat format.
    
    Parameters:
    -----------
    pt_input_path : Path
        Path to the folder containing .pt files
    matlab_output_path : Path
        Path to output .mat file
    use_predictions : bool
        If True, look for predictions*.pt file instead of displacements_dataset.pt
        and create full indices for all combinations
    
    Returns:
    --------
    dict : Information about the conversion
    """
    print("\n" + "=" * 80)
    print("Converting Reduced PT Dataset to MATLAB Format")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Load Reduced PT Dataset
    print("\nStep 1: Loading Reduced PT Dataset")
    
    # Determine which displacements file to use
    displacements_file = None
    if use_predictions:
        # Look for .pt files starting with "predictions"
        prediction_files = list(pt_input_path.glob("predictions*.pt"))
        
        if len(prediction_files) == 0:
            raise FileNotFoundError(
                f"No predictions*.pt file found in {pt_input_path}. "
                f"Set --no_use_predictions to use displacements_dataset.pt instead."
            )
        elif len(prediction_files) > 1:
            raise ValueError(
                f"Multiple predictions*.pt files found in {pt_input_path}:\n" +
                "\n".join(f"  - {f.name}" for f in prediction_files) +
                "\nPlease ensure only one predictions file exists, or set --no_use_predictions."
            )
        else:
            displacements_file = prediction_files[0]
            print(f"  Using predictions file: {displacements_file.name}")
    else:
        displacements_file = pt_input_path / "displacements_dataset.pt"
        if not displacements_file.exists():
            raise FileNotFoundError(f"displacements_dataset.pt not found in {pt_input_path}")
        print(f"  Using displacements_dataset.pt")
    
    # Load all required files
    displacements_dataset = torch.load(displacements_file, map_location='cpu', weights_only=False)
    
    # Load geometries and other files first to get dimensions
    geometries = torch.load(pt_input_path / "geometries_full.pt", map_location='cpu', weights_only=False)
    waveforms = torch.load(pt_input_path / "waveforms_full.pt", map_location='cpu', weights_only=False)
    wavevectors = torch.load(pt_input_path / "wavevectors_full.pt", map_location='cpu', weights_only=False)
    bands_fft = torch.load(pt_input_path / "band_fft_full.pt", map_location='cpu', weights_only=False)
    design_params = torch.load(pt_input_path / "design_params_full.pt", map_location='cpu', weights_only=False)
    
    # Convert to numpy for dimension extraction
    geometries_np = geometries.numpy()
    wavevectors_np = wavevectors.numpy()
    
    # Get dimensions
    n_designs = geometries_np.shape[0]
    design_res = geometries_np.shape[1]
    n_wavevectors = wavevectors_np.shape[1]
    n_bands = bands_fft.shape[0]
    
    # Create or load indices
    if use_predictions:
        print(f"  Creating full indices for all combinations...")
        reduced_indices = create_full_indices(n_designs, n_wavevectors, n_bands)
        print(f"  Created {len(reduced_indices)} indices for {n_designs} designs × {n_wavevectors} wavevectors × {n_bands} bands")
    else:
        reduced_indices = torch.load(pt_input_path / "reduced_indices.pt", map_location='cpu', weights_only=False)
        print(f"  Loaded reduced_indices.pt with {len(reduced_indices)} indices")
    geometries = torch.load(pt_input_path / "geometries_full.pt", map_location='cpu', weights_only=False)
    waveforms = torch.load(pt_input_path / "waveforms_full.pt", map_location='cpu', weights_only=False)
    wavevectors = torch.load(pt_input_path / "wavevectors_full.pt", map_location='cpu', weights_only=False)
    bands_fft = torch.load(pt_input_path / "band_fft_full.pt", map_location='cpu', weights_only=False)
    design_params = torch.load(pt_input_path / "design_params_full.pt", map_location='cpu', weights_only=False)
    
    # Convert to numpy (efficient batch conversion)
    eigenvector_x_real = displacements_dataset.tensors[0].numpy()
    eigenvector_x_imag = displacements_dataset.tensors[1].numpy()
    eigenvector_y_real = displacements_dataset.tensors[2].numpy()
    eigenvector_y_imag = displacements_dataset.tensors[3].numpy()
    
    # Convert other arrays needed later
    geometries_np = geometries.numpy()
    wavevectors_np = wavevectors.numpy()
    design_params_np = design_params.numpy()
    
    print(f"  Loaded dataset dimensions:")
    print(f"    n_designs: {n_designs}")
    print(f"    design_res: {design_res}")
    print(f"    n_wavevectors: {n_wavevectors}")
    print(f"    n_bands: {n_bands}")
    print(f"    n_samples: {len(reduced_indices)}")
    
    # Step 2: Reconstruct Full EIGENVECTOR_DATA
    print("\nStep 2: Reconstructing Full EIGENVECTOR_DATA")
    
    # Initialize full eigenvector arrays (fill with zeros for missing entries)
    # Use complex128 (float64) for computation to match MATLAB's double precision
    EIGENVECTOR_DATA_x_full = np.zeros((n_designs, n_wavevectors, n_bands, design_res, design_res), 
                                        dtype=np.complex128)
    EIGENVECTOR_DATA_y_full = np.zeros((n_designs, n_wavevectors, n_bands, design_res, design_res), 
                                        dtype=np.complex128)
    
    # Place reduced eigenvectors at correct indices (optimized with vectorized operations where possible)
    # Convert indices to numpy for faster indexing
    if isinstance(reduced_indices, torch.Tensor):
        indices_np = reduced_indices.numpy()
    else:
        # Handle list of tuples - convert efficiently without list comprehension
        n_indices = len(reduced_indices)
        indices_np = np.zeros((n_indices, 3), dtype=np.int64)
        for i, (d, w, b) in enumerate(reduced_indices):
            indices_np[i, 0] = int(d) if isinstance(d, torch.Tensor) else int(d)
            indices_np[i, 1] = int(w) if isinstance(w, torch.Tensor) else int(w)
            indices_np[i, 2] = int(b) if isinstance(b, torch.Tensor) else int(b)
    
    n_samples = len(indices_np)
    
    # Pre-allocate complex arrays for eigenvectors (more efficient than creating in loop)
    # Convert to complex128 (float64) for computation to match MATLAB's double precision
    eigenvector_x_complex = (eigenvector_x_real + 1j * eigenvector_x_imag).astype(np.complex128)
    eigenvector_y_complex = (eigenvector_y_real + 1j * eigenvector_y_imag).astype(np.complex128)
    
    # Place eigenvectors at correct indices (vectorized where possible)
    for sample_idx in range(n_samples):
        d_idx, w_idx, b_idx = indices_np[sample_idx]
        # Direct assignment (faster than creating new arrays)
        EIGENVECTOR_DATA_x_full[d_idx, w_idx, b_idx, :, :] = eigenvector_x_complex[sample_idx]
        EIGENVECTOR_DATA_y_full[d_idx, w_idx, b_idx, :, :] = eigenvector_y_complex[sample_idx]
    
    print(f"  Reconstructed EIGENVECTOR_DATA_x shape: {EIGENVECTOR_DATA_x_full.shape}")
    print(f"  Reconstructed EIGENVECTOR_DATA_y shape: {EIGENVECTOR_DATA_y_full.shape}")
    
    # Step 3: Combine x and y eigenvectors into single array
    print("\nStep 3: Combining x and y eigenvectors")
    
    # Reshape to (n_designs, n_wavevectors, n_bands, 2*design_res*design_res)
    EIGENVECTOR_DATA_x_flat = EIGENVECTOR_DATA_x_full.reshape(n_designs, n_wavevectors, n_bands, -1)
    EIGENVECTOR_DATA_y_flat = EIGENVECTOR_DATA_y_full.reshape(n_designs, n_wavevectors, n_bands, -1)
    
    # Interleave x and y components
    n_dof = 2 * design_res * design_res
    # Use complex128 (float64) for computation to match MATLAB's double precision
    EIGENVECTOR_DATA_combined = np.zeros((n_designs, n_wavevectors, n_bands, n_dof), dtype=np.complex128)
    
    # Interleave manually
    EIGENVECTOR_DATA_combined[:, :, :, 0::2] = EIGENVECTOR_DATA_x_flat
    EIGENVECTOR_DATA_combined[:, :, :, 1::2] = EIGENVECTOR_DATA_y_flat
    
    # Transpose to match MATLAB format: (n_designs, n_eig, n_wv, n_dof)
    EIGENVECTOR_DATA = EIGENVECTOR_DATA_combined.transpose(0, 2, 1, 3)
    
    print(f"  Combined EIGENVECTOR_DATA shape: {EIGENVECTOR_DATA.shape}")
    
    # Step 4: Reconstruct EIGENVALUE_DATA from eigenvectors using K, M, T matrices
    print("\nStep 4: Reconstructing EIGENVALUE_DATA from eigenvectors")
    
    # Initialize EIGENVALUE_DATA: MATLAB format is (n_designs, n_bands, n_wavevectors)
    # But we'll store as (n_designs, n_wavevectors, n_bands) during computation, then transpose
    EIGENVALUE_DATA = np.zeros((n_designs, n_wavevectors, n_bands), dtype=np.float64)
    
    # Required functions must be available - no fallback
    # Extract const parameters for matrix computation
    # Use values from const dict that will be created later
    E_min = 20000000
    E_max = 200000000000
    rho_min = 1200
    rho_max = 8000
    nu_min = 0.0
    nu_max = 0.5
    t_val = 1.0
    a_val = 1.0
    N_ele = 1  # Match MATLAB: N_ele=1
    
    # Reconstruct frequencies for each design
    for struct_idx in range(n_designs):
        print(f"  Reconstructing frequencies for structure {struct_idx + 1}/{n_designs}...")
        
        # Get design for this structure (single channel, values in [0, 1])
        design_param = geometries_np[struct_idx]  # (design_res, design_res)
        
        # Apply steel-rubber paradigm to get 3-channel design
        # This matches MATLAB's apply_steel_rubber_paradigm function
        const_for_paradigm = {
            'E_min': E_min,
            'E_max': E_max,
            'rho_min': rho_min,
            'rho_max': rho_max,
            'poisson_min': nu_min,
            'poisson_max': nu_max
        }
        design_3ch = apply_steel_rubber_paradigm(design_param.astype(np.float64), const_for_paradigm)
        
        # Create const dict for matrix computation
        const_for_km = {
            'design': design_3ch,
            'N_pix': design_res,
            'N_ele': N_ele,
            'a': a_val,
            'E_min': E_min,
            'E_max': E_max,
            'rho_min': rho_min,
            'rho_max': rho_max,
            'poisson_min': nu_min,
            'poisson_max': nu_max,
            't': t_val,
            'design_scale': 'linear',
            'isUseImprovement': True,
            'isUseSecondImprovement': False
        }
        
        # Compute K and M matrices
        print(f"    Computing K and M matrices...")
        K, M = get_system_matrices_VEC(const_for_km)
        print(f"      K: shape={K.shape}, nnz={K.nnz}")
        print(f"      M: shape={M.shape}, nnz={M.nnz}")
        
        # Get wavevectors for this structure
        wavevectors_struct = wavevectors_np[struct_idx, :, :]  # (n_wavevectors, 2)
        
        # Compute T matrices for each wavevector
        print(f"    Computing T matrices for {len(wavevectors_struct)} wavevectors...")
        T_data = []
        for wv_idx, wv in enumerate(wavevectors_struct):
            T = get_transformation_matrix(wv.astype(np.float32), const_for_km)
            if T is None:
                raise ValueError(f"Failed to compute T matrix for wavevector {wv_idx} in structure {struct_idx + 1}")
            T_data.append(T)
        
        # Extract eigenvectors for this structure
        # EIGENVECTOR_DATA is in format: (n_designs, n_eig, n_wv, n_dof)
        # We need (n_dof, n_wv, n_eig) for reconstruction function
        eigenvectors_struct = EIGENVECTOR_DATA[struct_idx, :, :, :]  # (n_eig, n_wv, n_dof)
        eigenvectors_struct = eigenvectors_struct.transpose(2, 1, 0)  # (n_dof, n_wv, n_eig)
        
        # Reconstruct frequencies using the same method as plot_dispersion_infer_eigenfrequencies.py
        frequencies_recon = np.zeros((n_wavevectors, n_bands), dtype=np.float64)
        
        for wv_idx in range(n_wavevectors):
            # Get transformation matrix for this wavevector
            T = T_data[wv_idx]
            if T is None:
                raise ValueError(f"T matrix is None for wavevector {wv_idx} in structure {struct_idx + 1}")
            
            # Convert matrices to sparse format for efficiency
            T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.float32))
            K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
            M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
            
            # Transform to reduced space: Kr = T' * K * T, Mr = T' * M * T
            Kr = T_sparse.conj().T @ K_sparse @ T_sparse
            Mr = T_sparse.conj().T @ M_sparse @ T_sparse
            
            # Process each eigenvalue band
            for band_idx in range(n_bands):
                # Extract eigenvector for this wavevector and band
                eigvec = eigenvectors_struct[:, wv_idx, band_idx].astype(np.complex128)
                
                # Reconstruct eigenvalue: eigval = norm(Kr*eigvec)/norm(Mr*eigvec)
                # This is equivalent to: Kr*eigvec = eigval*Mr*eigvec
                Kr_eigvec = Kr @ eigvec
                Mr_eigvec = Mr @ eigvec
                
                # Convert sparse results to dense for norm calculation
                if sp.issparse(Kr_eigvec):
                    Kr_eigvec = Kr_eigvec.toarray().flatten()
                if sp.issparse(Mr_eigvec):
                    Mr_eigvec = Mr_eigvec.toarray().flatten()
                
                # Compute eigenvalue and convert to frequency
                # NOTE: Use Rayleigh quotient, not the older norm-ratio formula.
                # The norm-based formula assumes Kr*v == eigval*Mr*v exactly,
                # which is very sensitive when eigenvectors are approximate
                # (float16→float32 conversions). The Rayleigh quotient is the
                # standard, numerically stable choice:
                #   eigval = (v^H Kr v) / (v^H Mr v)
                eigval = (eigvec.conj() @ (Kr @ eigvec)) / (eigvec.conj() @ (Mr @ eigvec))
                frequencies_recon[wv_idx, band_idx] = np.sqrt(np.real(eigval)) / (2 * np.pi)
        
        # Store in EIGENVALUE_DATA (n_designs, n_wavevectors, n_bands)
        EIGENVALUE_DATA[struct_idx, :, :] = frequencies_recon
        print(f"    Successfully reconstructed frequencies for structure {struct_idx + 1}")
    
    # Transpose to MATLAB format: (n_designs, n_bands, n_wavevectors)
    EIGENVALUE_DATA = EIGENVALUE_DATA.transpose(0, 2, 1)
    
    print(f"  EIGENVALUE_DATA shape: {EIGENVALUE_DATA.shape}")
    print(f"  Successfully reconstructed all frequencies")
    print(f"  Frequency range: [{np.min(EIGENVALUE_DATA):.6e}, {np.max(EIGENVALUE_DATA):.6e}] Hz")
    
    # Step 5: Reconstruct designs with all panes
    print("\nStep 5: Reconstructing designs with all panes")
    
    # Apply steel-rubber paradigm to reconstruct 3-channel designs
    # This matches MATLAB's apply_steel_rubber_paradigm function
    n_panes = 3
    designs_full = np.zeros((n_designs, n_panes, design_res, design_res), dtype=np.float64)
    
    const_for_paradigm = {
        'E_min': E_min,
        'E_max': E_max,
        'rho_min': rho_min,
        'rho_max': rho_max,
        'poisson_min': nu_min,
        'poisson_max': nu_max
    }
    
    for struct_idx in range(n_designs):
        design_param = geometries_np[struct_idx].astype(np.float64)  # (design_res, design_res)
        design_3ch = apply_steel_rubber_paradigm(design_param, const_for_paradigm)  # (design_res, design_res, 3)
        # Transpose to match MATLAB format: (n_panes, design_res, design_res)
        designs_full[struct_idx, :, :, :] = np.transpose(design_3ch, (2, 0, 1))
    
    print(f"  Reconstructed designs shape: {designs_full.shape}")
    print(f"  Pane 0 (modulus) range: [{np.min(designs_full[:, 0, :, :]):.6e}, {np.max(designs_full[:, 0, :, :]):.6e}]")
    print(f"  Pane 1 (density) range: [{np.min(designs_full[:, 1, :, :]):.6e}, {np.max(designs_full[:, 1, :, :]):.6e}]")
    print(f"  Pane 2 (poisson) range: [{np.min(designs_full[:, 2, :, :]):.6e}, {np.max(designs_full[:, 2, :, :]):.6e}]")
    
    # Step 6: Reconstruct WAVEVECTOR_DATA
    print("\nStep 6: Reconstructing WAVEVECTOR_DATA")
    
    # Transpose to match MATLAB format: (n_designs, 2, n_wv)
    WAVEVECTOR_DATA = wavevectors_np.transpose(0, 2, 1)
    
    print(f"  Reconstructed WAVEVECTOR_DATA shape: {WAVEVECTOR_DATA.shape}")
    
    # Step 7: Reconstruct WAVEFORM_DATA
    print("\nStep 7: Reconstructing WAVEFORM_DATA")
    
    # WAVEFORM_DATA should be (n_designs, n_wavevectors, design_res, design_res)
    # We have waveforms for the first design, duplicate for all designs
    WAVEFORM_DATA = np.zeros((n_designs, n_wavevectors, design_res, design_res), dtype=np.float32)
    
    # Use waveforms from first design (they're the same for all designs)
    waveforms_np = waveforms.numpy()
    for d_idx in range(n_designs):
        WAVEFORM_DATA[d_idx, :, :, :] = waveforms_np
    
    print(f"  Reconstructed WAVEFORM_DATA shape: {WAVEFORM_DATA.shape}")
    
    # Step 8: Reconstruct const dictionary
    print("\nStep 8: Reconstructing const dictionary")
    
    # Hardcoded project-specific values extracted from original .mat files
    # N_wv is [25, 13] for the full half-plane grid (91 wavevectors are a subset in the IBZ)
    const = {
        'N_pix': np.array([[float(design_res)]], dtype=np.float64),  # (1, 1)
        'N_ele': np.array([[1.0]], dtype=np.float64),  # (1, 1)
        'N_eig': np.array([[float(n_bands)]], dtype=np.float64),  # (1, 1)
        'N_wv': np.array([[25.0], [13.0]], dtype=np.float64),  # (2, 1) - full grid dimensions (hardcoded)
        'a': np.array([[1.0]], dtype=np.float64),  # Hardcoded from original
        'E_max': np.array([[200000000000.0]], dtype=np.float64),  # Hardcoded from original
        'E_min': np.array([[20000000.0]], dtype=np.float64),  # Hardcoded from original
        'poisson_max': np.array([[0.5]], dtype=np.float64),  # Hardcoded from original
        'poisson_min': np.array([[0.0]], dtype=np.float64),  # Hardcoded from original
        'rho_max': np.array([[8000.0]], dtype=np.float64),  # Hardcoded from original
        'rho_min': np.array([[1200.0]], dtype=np.float64),  # Hardcoded from original
        't': np.array([[1.0]], dtype=np.float64),  # Hardcoded from original
        'sigma_eig': np.array([[0.01]], dtype=np.float64),  # Hardcoded from original
        # Character arrays: MATLAB stores strings as character arrays (one char per element)
        # Hardcoded project-specific values extracted from original .mat files
        'design_scale': np.array([list('linear')], dtype='U1').T,  # (6, 1) character array
        'symmetry_type': np.array([list('p4mm')], dtype='U1').T,  # (4, 1) character array - project fixed value
        'eigenvector_dtype': np.array([list('single')], dtype='U1').T,  # (6, 1) character array - project fixed value
        'isSaveEigenvectors': np.array([[1.0]], dtype=np.float64),
        # Note: isSaveKandM removed - will be missing in reconstructed file
        'isSaveMesh': np.array([[0.0]], dtype=np.float64),
        'isUseGPU': np.array([[0.0]], dtype=np.float64),
        'isUseImprovement': np.array([[1.0]], dtype=np.float64),
        'isUseParallel': np.array([[1.0]], dtype=np.float64),
        'isUseSecondImprovement': np.array([[0.0]], dtype=np.float64),
        'design': np.transpose(apply_steel_rubber_paradigm(
            geometries_np[0].astype(np.float64),
            {
                'E_min': E_min,
                'E_max': E_max,
                'rho_min': rho_min,
                'rho_max': rho_max,
                'poisson_min': nu_min,
                'poisson_max': nu_max
            }
        ), (2, 0, 1)).astype(np.float64),  # (3, 32, 32)
        'wavevectors': wavevectors_np[0, :, :].T  # Wavevectors from first design
    }
    
    print(f"  Reconstructed const dictionary with {len(const)} keys")
    
    # Step 9: Prepare other metadata
    print("\nStep 9: Preparing metadata")
    
    N_struct = np.array([[float(n_designs)]], dtype=np.float64)
    N_batch = np.array([[1.0]], dtype=np.float64)  # Default: single batch
    N_struct_batch = np.array([[float(n_designs)]], dtype=np.float64)  # Number of geometries per batch (equals total geometries for single batch)
    imag_tol = np.array([[0.001]], dtype=np.float64)  # Hardcoded from original (1e-3)
    # Note: rng_seed_offset removed - will be missing in reconstructed file
    
    # Step 10: Save as MATLAB v7.3 format
    print("\nStep 10: Saving as MATLAB v7.3 format")
    
    # Prepare dataset dictionary
    # Note: Use float64 for most arrays to match original MATLAB format (double precision)
    dataset = {
        'WAVEVECTOR_DATA': WAVEVECTOR_DATA.astype(np.float64),  # Match original: double
        'EIGENVALUE_DATA': EIGENVALUE_DATA.astype(np.float64),  # Reconstructed from eigenvectors (match original precision)
        'EIGENVECTOR_DATA': EIGENVECTOR_DATA.astype(np.complex64),  # Complex single (float32 real/imag)
        'designs': designs_full.astype(np.float64),  # Match original: double
        'design_params': design_params_np.astype(np.float64),
        'N_struct': N_struct.astype(np.float64),
        'N_batch': N_batch.astype(np.float64),
        'N_struct_batch': N_struct_batch.astype(np.float64),
        'imag_tol': imag_tol.astype(np.float64),
        # Note: rng_seed_offset removed - will be missing in reconstructed file
    }
    
    # Save const as a struct
    # Note: MATLAB v7.3 files created by MATLAB have a 128-byte header, but HDF5 requires
    # userblock_size >= 512. We'll create a pure HDF5 file that can be read with matfile or h5read
    # Use libver='earliest' for maximum MATLAB compatibility (MATLAB uses older HDF5 versions)
    with h5py.File(matlab_output_path, 'w', libver='earliest') as f:
        # Create MATLAB-specific HDF5 groups for v7.3 compatibility
        # These groups are required for MATLAB to recognize the file as a valid v7.3 file
        refs_grp = f.create_group('#refs#')
        subsystem_grp = f.create_group('#subsystem#')
        # Save regular arrays
        for key, value in dataset.items():
            if key == 'EIGENVECTOR_DATA':
                # Save complex array as structured array (compound dtype) - MATLAB v7.3 format
                # Convert to float32 (single precision) to match original MATLAB format
                EIGENVECTOR_DATA_real = value.real.astype(np.float32)
                EIGENVECTOR_DATA_imag = value.imag.astype(np.float32)
                
                # Create structured array with compound dtype (matches MATLAB format)
                structured_dtype = np.dtype([('real', np.float32), ('imag', np.float32)])
                EIGENVECTOR_DATA_structured = np.empty(value.shape, dtype=structured_dtype)
                EIGENVECTOR_DATA_structured['real'] = EIGENVECTOR_DATA_real
                EIGENVECTOR_DATA_structured['imag'] = EIGENVECTOR_DATA_imag
                
                # Create dataset with structured dtype (compound datatype)
                dset = f.create_dataset(
                    'EIGENVECTOR_DATA',
                    data=EIGENVECTOR_DATA_structured,
                    dtype=structured_dtype
                )
                # Add MATLAB_class attribute to indicate it's a single-precision complex array
                dset.attrs['MATLAB_class'] = np.bytes_(b'single')
            else:
                dset = f.create_dataset(key, data=value)
                # Add MATLAB_class attribute based on dtype
                if value.dtype == np.float32:
                    dset.attrs['MATLAB_class'] = np.bytes_(b'single')
                elif value.dtype == np.float64:
                    dset.attrs['MATLAB_class'] = np.bytes_(b'double')
                elif value.dtype == np.complex64:
                    dset.attrs['MATLAB_class'] = np.bytes_(b'single')
                elif value.dtype == np.complex128:
                    dset.attrs['MATLAB_class'] = np.bytes_(b'double')
                elif np.issubdtype(value.dtype, np.integer):
                    if value.dtype.itemsize <= 4:
                        dset.attrs['MATLAB_class'] = np.bytes_(b'int32')
                    else:
                        dset.attrs['MATLAB_class'] = np.bytes_(b'int64')
        
        # Save const as a struct group
        const_grp = f.create_group('const')
        for key, value in const.items():
            if isinstance(value, np.ndarray):
                if value.dtype == object:
                    # Handle string arrays
                    dt = h5py.special_dtype(vlen=str)
                    dset = const_grp.create_dataset(key, value.shape, dtype=dt)
                    dset[:] = value.astype(str)
                elif value.dtype.kind == 'U':
                    # Handle Unicode character arrays - convert to uint16 (ASCII codes) like MATLAB
                    # Convert each character to its ASCII code
                    char_codes = np.zeros(value.shape, dtype=np.uint16)
                    for idx in np.ndindex(value.shape):
                        char_codes[idx] = ord(value[idx])
                    const_grp.create_dataset(key, data=char_codes)
                else:
                    const_grp.create_dataset(key, data=value)
            else:
                const_grp.attrs[key] = value
    
    elapsed_time = time.time() - start_time
    file_size = matlab_output_path.stat().st_size / (1024 * 1024)
    
    print(f"  Saved to: {matlab_output_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Conversion completed in {elapsed_time:.2f} seconds")
    
    return {
        'output_path': matlab_output_path,
        'file_size_mb': file_size,
        'elapsed_time': elapsed_time,
        'n_designs': n_designs,
        'n_wavevectors': n_wavevectors,
        'n_bands': n_bands
    }


def main():
    """Main function to convert PyTorch datasets to MATLAB format."""
    parser = argparse.ArgumentParser(
        description='Convert reduced PyTorch .pt files back to MATLAB .mat format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single PT dataset folder to MATLAB
  python reduced_pt_to_matlab.py --input_folder "path/to/pt/dataset" --output_file "output.mat"
  
  # Convert with auto-generated output filename
  python reduced_pt_to_matlab.py -i "path/to/pt/dataset" -o "output_folder"
        """
    )
    
    parser.add_argument(
        '--input_folder', '-i',
        type=str,
        required=True,
        help='Path to folder containing .pt files (reduced PT dataset)'
    )
    
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        required=True,
        help='Path to output .mat file or output directory. If directory, filename is auto-generated from input folder name. If file path (ends with .mat), uses that exact filename.'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the output file can be loaded with NO_utils.extract_data()'
    )
    
    parser.add_argument(
        '--use_predictions',
        action='store_true',
        default=True,
        help='Look for predictions*.pt file instead of displacements_dataset.pt (default: True)'
    )
    
    parser.add_argument(
        '--no_use_predictions',
        dest='use_predictions',
        action='store_false',
        help='Use displacements_dataset.pt instead of looking for predictions*.pt'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    pt_input_path = Path(args.input_folder)
    output_path = Path(args.output_file)
    
    # Determine if output_path is a file path or directory path
    # If it has .mat extension, treat as file path; otherwise treat as directory path
    if output_path.suffix == '.mat':
        # Explicit file path provided
        matlab_output_path = output_path
        # Add "predictions" suffix if using predictions and not explicitly specified
        if args.use_predictions and matlab_output_path.stem and not matlab_output_path.stem.endswith('_predictions'):
            matlab_output_path = matlab_output_path.parent / f"{matlab_output_path.stem}_predictions{matlab_output_path.suffix}"
        # Create parent directory if it doesn't exist
        matlab_output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Directory path provided (or path without .mat extension) - auto-generate filename
        # Create directory if it doesn't exist
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from input folder name
        input_name = pt_input_path.name
        # Add "predictions" suffix if using predictions
        if args.use_predictions:
            matlab_output_path = output_dir / f"{input_name}_predictions.mat"
        else:
            matlab_output_path = output_dir / f"{input_name}.mat"
    
    print(f"PT input folder: {pt_input_path}")
    print(f"MATLAB output file: {matlab_output_path}")
    
    # Validate input
    if not pt_input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {pt_input_path}")
    
    if not pt_input_path.is_dir():
        raise ValueError(f"Input path must be a directory: {pt_input_path}")
    
    # Check for required files (displacements_dataset.pt or predictions file handled in convert function)
    required_files = [
        "geometries_full.pt",
        "waveforms_full.pt",
        "wavevectors_full.pt",
        "band_fft_full.pt",
        "design_params_full.pt"
    ]
    
    if not args.use_predictions:
        # If not using predictions, require displacements_dataset.pt and reduced_indices.pt
        required_files.extend(["displacements_dataset.pt", "reduced_indices.pt"])
    
    missing_files = [f for f in required_files if not (pt_input_path / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {pt_input_path}:\n" + 
            "\n".join(f"  - {f}" for f in missing_files)
        )
    
    # Run conversion
    result = convert_reduced_pt_to_matlab(pt_input_path, matlab_output_path, use_predictions=args.use_predictions)
    
    # Summary
    print("\n" + "=" * 80)
    print("Conversion Summary")
    print("=" * 80)
    print(f"Output file: {result['output_path']}")
    print(f"File size: {result['file_size_mb']:.2f} MB")
    print(f"Conversion time: {result['elapsed_time']:.2f} seconds")
    print(f"n_designs: {result['n_designs']}")
    print(f"n_wavevectors: {result['n_wavevectors']}")
    print(f"n_bands: {result['n_bands']}")
    print("=" * 80)
    
    # Verification if requested
    if args.verify:
        print("\n" + "=" * 80)
        print("Verifying Reconstructed MATLAB File")
        print("=" * 80)
        
        try:
            import tempfile
            import shutil
            
            # Create a temporary directory with the reconstructed file
            temp_dir = tempfile.mkdtemp(prefix="verify_reconstructed_")
            temp_mat_path = Path(temp_dir) / matlab_output_path.name
            
            # Copy the reconstructed file to temp directory
            shutil.copy2(matlab_output_path, temp_mat_path)
            
            # Try to extract data using NO_utils
            (designs, design_params, n_designs, n_panes, design_res,
             WAVEVECTOR_DATA, WAVEFORM_DATA, n_dim, n_wavevectors,
             EIGENVALUE_DATA, n_bands, EIGENVECTOR_DATA_x,
             EIGENVECTOR_DATA_y, const, N_struct,
             imag_tol, rng_seed_offset) = NO_utils.extract_data(temp_dir)
            
            print("\n✓ Successfully loaded with NO_utils.extract_data()!")
            print(f"  n_designs: {n_designs}")
            print(f"  n_panes: {n_panes}")
            print(f"  design_res: {design_res}")
            print(f"  n_wavevectors: {n_wavevectors}")
            print(f"  n_bands: {n_bands}")
            print(f"  EIGENVECTOR_DATA_x shape: {EIGENVECTOR_DATA_x.shape}")
            print(f"  EIGENVECTOR_DATA_y shape: {EIGENVECTOR_DATA_y.shape}")
            
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            print("\n" + "=" * 80)
            print("✓ Verification successful! The reconstructed file is compatible.")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n✗ Error during verification: {e}")
            import traceback
            traceback.print_exc()
            # Clean up
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

