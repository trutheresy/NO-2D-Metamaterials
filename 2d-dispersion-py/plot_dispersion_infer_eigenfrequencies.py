"""
Plot Dispersion Script with Eigenfrequency Reconstruction

This script mirrors the MATLAB plot_dispersion.m functionality by:
- Loading PyTorch format datasets
- Computing or loading K, M, and T matrices
- Reconstructing frequencies from eigenvectors using K, M, T matrices
- Creating comparison plots (original vs reconstructed frequencies)

This provides validation that stored eigenvectors are correct.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import sys

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

# Import local modules
try:
    from plotting import plot_design
    from wavevectors import get_IBZ_contour_wavevectors
    from system_matrices import get_system_matrices, get_transformation_matrix
    from system_matrices_vec import get_system_matrices_VEC, get_system_matrices_VEC_simplified
except ImportError:
    import plotting
    import wavevectors
    import system_matrices
    import system_matrices_vec
    plot_design = plotting.plot_design
    get_IBZ_contour_wavevectors = wavevectors.get_IBZ_contour_wavevectors
    get_system_matrices = system_matrices.get_system_matrices
    get_transformation_matrix = system_matrices.get_transformation_matrix
    get_system_matrices_VEC = system_matrices_vec.get_system_matrices_VEC
    get_system_matrices_VEC_simplified = system_matrices_vec.get_system_matrices_VEC_simplified


def load_pt_dataset(data_dir: Path, original_data_dir: Path = None, require_eigenvalue_data: bool = True):
    """
    Load PyTorch dataset from the .pt format directory.
    
    All required data should be in the data_dir. If original_data_dir is provided
    and require_eigenvalue_data is True, will attempt to load eigenvalue data from it.
    
    Also attempts to load K, M, T matrices if available.
    
    Parameters
    ----------
    data_dir : Path
        Path to PyTorch dataset directory (contains all required data)
    original_data_dir : Path, optional
        Path to original dataset (for eigenvalue data, only used if require_eigenvalue_data=True)
    require_eigenvalue_data : bool, optional
        Whether to require eigenvalue data (default: True)
        Set to False when using infer mode
    """
    data = {}
    
    # Check if this is a reduced dataset (has reduced_indices.pt)
    is_reduced = (data_dir / 'reduced_indices.pt').exists()
    
    # Load PyTorch files
    data['geometries'] = torch.load(data_dir / 'geometries_full.pt', map_location='cpu')
    data['wavevectors'] = torch.load(data_dir / 'wavevectors_full.pt', map_location='cpu')
    
    # Convert to NumPy for easier manipulation
    geometries_np = data['geometries'].numpy()
    wavevectors_np = data['wavevectors'].numpy()
    
    # Try to load K, M, T matrices if available
    if (data_dir / 'K_data.pt').exists():
        print("  Loading K matrices from PyTorch format...")
        data['K_data'] = torch.load(data_dir / 'K_data.pt', map_location='cpu')
        if isinstance(data['K_data'], torch.Tensor):
            data['K_data'] = data['K_data'].numpy()
    if (data_dir / 'M_data.pt').exists():
        print("  Loading M matrices from PyTorch format...")
        data['M_data'] = torch.load(data_dir / 'M_data.pt', map_location='cpu')
        if isinstance(data['M_data'], torch.Tensor):
            data['M_data'] = data['M_data'].numpy()
    if (data_dir / 'T_data.pt').exists():
        print("  Loading T matrices from PyTorch format...")
        data['T_data'] = torch.load(data_dir / 'T_data.pt', map_location='cpu')
        if isinstance(data['T_data'], torch.Tensor):
            data['T_data'] = data['T_data'].numpy()
    
    # Try to load eigenvectors if available (check multiple possible filenames)
    # Priority: displacements_dataset.pt (most common), then eigenvectors_full.pt, then eigenvectors.pt
    eigenvector_file = None
    for filename in ['displacements_dataset.pt', 'eigenvectors_full.pt', 'eigenvectors.pt']:
        if (data_dir / filename).exists():
            eigenvector_file = data_dir / filename
            break
    
    if eigenvector_file is not None:
        print(f"  Loading eigenvectors from: {eigenvector_file.name}")
        eigenvectors_loaded = torch.load(eigenvector_file, map_location='cpu')
        
        # Handle different data structures
        if isinstance(eigenvectors_loaded, torch.Tensor):
            data['eigenvectors'] = eigenvectors_loaded.numpy()
        elif hasattr(eigenvectors_loaded, 'tensors'):
            # Handle TensorDataset - extract the tensor data
            # TensorDataset typically contains (x_real, x_imag, y_real, y_imag) for displacements
            tensors = eigenvectors_loaded.tensors
            print(f"    TensorDataset with {len(tensors)} tensors")
            if len(tensors) >= 4:
                # Extract x and y components (real and imaginary)
                x_real = tensors[0].numpy()
                x_imag = tensors[1].numpy()
                y_real = tensors[2].numpy()
                y_imag = tensors[3].numpy()
                # Combine into complex: x + i*y (x and y are separate components)
                # Format: (N_samples, H, W) for each component
                # We need to stack them properly
                # For now, store as separate components - we'll combine later
                data['eigenvectors_x'] = (x_real + 1j * x_imag).astype(np.complex64)
                data['eigenvectors_y'] = (y_real + 1j * y_imag).astype(np.complex64)
                # Store combined as field format (will need to be expanded to DOF later)
                data['eigenvectors'] = np.stack([data['eigenvectors_x'], data['eigenvectors_y']], axis=1)  # (N_samples, 2, H, W)
                print(f"    Extracted x and y components, shape: {data['eigenvectors'].shape}")
            else:
                # Just take first tensor
                data['eigenvectors'] = tensors[0].numpy()
        elif isinstance(eigenvectors_loaded, (list, tuple)):
            # Handle list/tuple of tensors
            data['eigenvectors'] = np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in eigenvectors_loaded])
        else:
            # Try to convert to numpy directly
            try:
                data['eigenvectors'] = np.array(eigenvectors_loaded)
            except:
                print(f"    WARNING: Could not convert eigenvectors to numpy array")
                data['eigenvectors'] = eigenvectors_loaded
    
    if is_reduced:
        # Load reduced indices
        data['reduced_indices'] = torch.load(data_dir / 'reduced_indices.pt', map_location='cpu')
        reduced_indices_np = np.array(data['reduced_indices'])
        
        print(f"Detected reduced PyTorch dataset.")
        print(f"  Reduced indices shape: {reduced_indices_np.shape}")
        
        # Only try to load eigenvalue data if required and original_data_dir is provided
        if require_eigenvalue_data and original_data_dir is not None:
            # Load original eigenvalue data from NumPy format
            print(f"Loading original eigenvalue data from: {original_data_dir}")
            # Try to load from NumPy format first
            if (original_data_dir / 'eigenvalue_data.npy').exists():
                eigenvalue_data_orig = np.load(original_data_dir / 'eigenvalue_data.npy')
                data['eigenvalue_data'] = eigenvalue_data_orig
                print(f"  Merged with original eigenvalue data: {eigenvalue_data_orig.shape}")
            # Try to load from MATLAB .mat file
            elif original_data_dir.suffix == '.mat' or (original_data_dir.parent / f'{original_data_dir.name}.mat').exists():
                mat_file = original_data_dir if original_data_dir.suffix == '.mat' else (original_data_dir.parent / f'{original_data_dir.name}.mat')
                print(f"Loading eigenvalue data from MATLAB file: {mat_file}")
                try:
                    import h5py
                    with h5py.File(mat_file, 'r') as f:
                        eigenvalue_data_orig = np.array(f['EIGENVALUE_DATA'], dtype=np.float32)
                        print(f"  Original MATLAB eigenvalue shape: {eigenvalue_data_orig.shape}")
                        # MATLAB format: h5py loads as (N_struct, N_eig, N_wv) or (N_wv, N_eig, N_struct)
                        # Need to transpose to (N_struct, N_wv, N_eig)
                        if len(eigenvalue_data_orig.shape) == 3:
                            if eigenvalue_data_orig.shape[0] > eigenvalue_data_orig.shape[2]:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(0, 2, 1)
                            elif eigenvalue_data_orig.shape[2] > eigenvalue_data_orig.shape[0]:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(2, 0, 1)
                            else:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(1, 0, 2)
                        print(f"  Transposed eigenvalue shape: {eigenvalue_data_orig.shape}")
                    data['eigenvalue_data'] = eigenvalue_data_orig
                    print(f"  Merged with original eigenvalue data: {eigenvalue_data_orig.shape}")
                except Exception as e:
                    print(f"  ERROR: Could not load from MATLAB file: {e}")
                    raise FileNotFoundError("Cannot plot dispersion without eigenvalue data")
            else:
                print(f"  WARNING: Original eigenvalue data not found at {original_data_dir}")
                print(f"  Cannot plot dispersion without eigenvalue data.")
                raise FileNotFoundError("Cannot plot dispersion without eigenvalue data")
        else:
            if require_eigenvalue_data:
                print(f"  WARNING: Reduced dataset detected but no original_data_dir provided.")
                print(f"  Cannot plot dispersion without eigenvalue data.")
                raise ValueError("For reduced datasets, original_data_dir must be provided")
            else:
                print(f"  NOTE: Reduced dataset detected but eigenvalue data not required (infer mode)")
    else:
        # Unreduced dataset
        print(f"Detected unreduced PyTorch dataset.")
        
        # Only try to load eigenvalue data if required and original_data_dir is provided
        if require_eigenvalue_data and original_data_dir is not None:
            # Try to load from NumPy format first
            if (original_data_dir / 'eigenvalue_data.npy').exists():
                print(f"Loading eigenvalue data from NumPy format: {original_data_dir}")
                eigenvalue_data_orig = np.load(original_data_dir / 'eigenvalue_data.npy')
                data['eigenvalue_data'] = eigenvalue_data_orig
                print(f"  Loaded eigenvalue data: {eigenvalue_data_orig.shape}")
            # Try to load from MATLAB .mat file
            elif original_data_dir.suffix == '.mat' or (original_data_dir.parent / f'{original_data_dir.name}.mat').exists():
                mat_file = original_data_dir if original_data_dir.suffix == '.mat' else (original_data_dir.parent / f'{original_data_dir.name}.mat')
                print(f"Loading eigenvalue data from MATLAB file: {mat_file}")
                try:
                    import h5py
                    with h5py.File(mat_file, 'r') as f:
                        eigenvalue_data_orig = np.array(f['EIGENVALUE_DATA'], dtype=np.float32)
                        print(f"  Original MATLAB eigenvalue shape: {eigenvalue_data_orig.shape}")
                        if len(eigenvalue_data_orig.shape) == 3:
                            if eigenvalue_data_orig.shape[0] > eigenvalue_data_orig.shape[2]:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(0, 2, 1)
                            elif eigenvalue_data_orig.shape[2] > eigenvalue_data_orig.shape[0]:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(2, 0, 1)
                            else:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(1, 0, 2)
                        print(f"  Transposed eigenvalue shape: {eigenvalue_data_orig.shape}")
                    data['eigenvalue_data'] = eigenvalue_data_orig
                    print(f"  Loaded eigenvalue data: {eigenvalue_data_orig.shape}")
                except Exception as e:
                    print(f"  ERROR: Could not load from MATLAB file: {e}")
                    raise FileNotFoundError("Cannot plot dispersion without eigenvalue data")
            else:
                print(f"  WARNING: Eigenvalue data not found at {original_data_dir}")
                raise FileNotFoundError("Cannot plot dispersion without eigenvalue data")
        else:
            if require_eigenvalue_data:
                print(f"  WARNING: Eigenvalue data not found.")
                raise FileNotFoundError("Cannot plot dispersion without eigenvalue data")
            else:
                print(f"  NOTE: Eigenvalue data not required (infer mode)")
    
    # Optional components
    if (data_dir / 'waveforms_full.pt').exists():
        data['waveforms'] = torch.load(data_dir / 'waveforms_full.pt', map_location='cpu').numpy()
    if (data_dir / 'band_fft_full.pt').exists():
        data['bands_fft'] = torch.load(data_dir / 'band_fft_full.pt', map_location='cpu').numpy()
    if (data_dir / 'design_params_full.pt').exists():
        data['design_params'] = torch.load(data_dir / 'design_params_full.pt', map_location='cpu').numpy()
    
    # Store as NumPy arrays for consistency
    data['designs'] = geometries_np
    data['wavevectors'] = wavevectors_np
    
    return data


def create_const_dict(design, N_pix, N_ele=4, a=1.0, 
                       E_min=20e6, E_max=200e9,
                       rho_min=400, rho_max=8000,
                       nu_min=0.05, nu_max=0.3,
                       t=0.01, design_scale='linear',
                       isUseImprovement=True, isUseSecondImprovement=False):
    """
    Create const dictionary for computing K, M, T matrices.
    
    Parameters
    ----------
    design : array_like
        Design array (N_pix, N_pix) with values in [0, 1]
    N_pix : int
        Number of pixels per dimension
    N_ele : int, optional
        Number of elements per pixel (default: 4)
    a : float, optional
        Lattice parameter (default: 1.0)
    E_min, E_max : float, optional
        Young's modulus range (default: 20e6 to 200e9)
    rho_min, rho_max : float, optional
        Density range (default: 400 to 8000)
    nu_min, nu_max : float, optional
        Poisson's ratio range (default: 0.05 to 0.3)
    t : float, optional
        Thickness (default: 0.01)
    design_scale : str, optional
        Design scaling ('linear' or 'log', default: 'linear')
    isUseImprovement : bool, optional
        Use vectorized assembly (default: True)
    isUseSecondImprovement : bool, optional
        Use simplified vectorized assembly (default: False)
        
    Returns
    -------
    const : dict
        Constants dictionary for matrix computation
    """
    # Convert design to 3-channel format (E, rho, nu)
    if design.ndim == 2:
        # Single channel design - replicate for all properties
        design_3ch = np.zeros((N_pix, N_pix, 3))
        design_3ch[:, :, 0] = design  # Elastic modulus
        design_3ch[:, :, 1] = design  # Density (coupled)
        design_3ch[:, :, 2] = design  # Poisson's ratio (coupled)
    else:
        design_3ch = design
    
    const = {
        'design': design_3ch,
        'N_pix': N_pix,
        'N_ele': N_ele,
        'a': a,
        'E_min': E_min,
        'E_max': E_max,
        'rho_min': rho_min,
        'rho_max': rho_max,
        'poisson_min': nu_min,
        'poisson_max': nu_max,
        't': t,
        'design_scale': design_scale,
        'isUseImprovement': isUseImprovement,
        'isUseSecondImprovement': isUseSecondImprovement,
        'N_eig': 6,  # Default number of eigenvalues
        'sigma_eig': 0,  # Default eigenvalue shift
    }
    
    return const


def compute_K_M_matrices(const):
    """
    Compute K and M matrices using the appropriate method.
    
    Parameters
    ----------
    const : dict
        Constants dictionary
        
    Returns
    -------
    K : scipy.sparse matrix
        Global stiffness matrix
    M : scipy.sparse matrix
        Global mass matrix
    """
    if const.get('isUseSecondImprovement', False):
        K, M = get_system_matrices_VEC_simplified(const)
    elif const.get('isUseImprovement', True):
        K, M = get_system_matrices_VEC(const)
    else:
        K, M = get_system_matrices(const, use_vectorized=False)
    
    return K, M


def reconstruct_frequencies_from_eigenvectors(K, M, T_data, eigenvectors, wavevectors, N_eig, struct_idx=0, N_struct=1):
    """
    Reconstruct frequencies from eigenvectors using K, M, T matrices.
    
    This mirrors the MATLAB plot_dispersion.m functionality (lines 100-116).
    
    Parameters
    ----------
    K : scipy.sparse matrix
        Global stiffness matrix
    M : scipy.sparse matrix
        Global mass matrix
    T_data : list or array
        Transformation matrices (one per wavevector)
    eigenvectors : array_like
        Eigenvectors - supported formats:
        - (N_dof, N_wv, N_eig) - MATLAB style DOF format
        - (N_struct, N_wv, N_eig, N_dof) - PyTorch style DOF format
        - (N_samples, 2, H, W) - Field format with x/y components (N_samples = N_struct × N_wv × N_eig)
    wavevectors : array_like
        Wavevectors (N_wv, 2)
    N_eig : int
        Number of eigenvalues
    struct_idx : int, optional
        Structure index (for multi-structure formats)
    N_struct : int, optional
        Total number of structures (for multi-structure formats)
        
    Returns
    -------
    frequencies_recon : array_like or None
        Reconstructed frequencies (N_wv, N_eig), or None if reconstruction fails
    """
    n_wavevectors = len(wavevectors)
    frequencies_recon = np.zeros((n_wavevectors, N_eig))
    
    # Handle different eigenvector formats
    print(f"    Eigenvectors shape: {eigenvectors.shape}")
    
    if eigenvectors.ndim == 3:
        # Format: (N_dof, N_wv, N_eig) - MATLAB style DOF format
        ev_format = 'matlab'
        print(f"    Detected MATLAB-style DOF format: (N_dof={eigenvectors.shape[0]}, N_wv={eigenvectors.shape[1]}, N_eig={eigenvectors.shape[2]})")
        
    elif eigenvectors.ndim == 4:
        # Check if second dimension is 2 (x/y components) - field format
        if eigenvectors.shape[1] == 2 and eigenvectors.shape[2] == eigenvectors.shape[3]:
            # Format: (N_samples, 2, H, W) - field format with x/y components
            # where N_samples = N_struct × N_wv × N_eig
            print(f"    Detected field format with x/y components: (N_samples={eigenvectors.shape[0]}, 2, H={eigenvectors.shape[2]}, W={eigenvectors.shape[3]})")
            H, W = eigenvectors.shape[2], eigenvectors.shape[3]
            
            # Flatten spatial dimensions: (N_samples, 2*H*W)
            eigenvectors_flat = eigenvectors.reshape(eigenvectors.shape[0], 2 * H * W)
            
            # Reshape to (N_struct, N_wv, N_eig, 2*H*W)
            samples_per_struct = n_wavevectors * N_eig
            total_samples = eigenvectors.shape[0]
            
            if total_samples % samples_per_struct != 0:
                raise ValueError(
                    f"Field format: total samples {total_samples} not divisible by "
                    f"samples per struct {samples_per_struct} (N_wv={n_wavevectors}, N_eig={N_eig})"
                )
            
            inferred_N_struct = total_samples // samples_per_struct
            print(f"    Reshaped to (N_struct={inferred_N_struct}, N_wv={n_wavevectors}, N_eig={N_eig}, field_size={2*H*W})")
            
            eigenvectors_reshaped = eigenvectors_flat.reshape(inferred_N_struct, n_wavevectors, N_eig, 2 * H * W)
            
            # Extract for this structure: (N_wv, N_eig, 2*H*W)
            eigenvectors_struct = eigenvectors_reshaped[struct_idx]
            
            # Transpose to (2*H*W, N_wv, N_eig) for consistent indexing
            eigenvectors = eigenvectors_struct.transpose(2, 0, 1)  # (2*H*W, N_wv, N_eig)
            ev_format = 'field_xy'
            field_size = 2 * H * W
            print(f"    Final shape: (field_size={field_size}, N_wv={n_wavevectors}, N_eig={N_eig})")
            
        else:
            # Format: (N_struct, N_wv, N_eig, N_dof) - PyTorch style DOF format
            ev_format = 'pytorch'
            print(f"    Detected PyTorch-style DOF format: (N_struct={eigenvectors.shape[0]}, N_wv={eigenvectors.shape[1]}, N_eig={eigenvectors.shape[2]}, N_dof={eigenvectors.shape[3]})")
            eigenvectors = eigenvectors[struct_idx]  # Take this structure
            eigenvectors = eigenvectors.transpose(2, 0, 1)  # (N_dof, N_wv, N_eig)
            print(f"    Extracted structure {struct_idx}, reshaped to: (N_dof={eigenvectors.shape[0]}, N_wv={eigenvectors.shape[1]}, N_eig={eigenvectors.shape[2]})")
    else:
        raise ValueError(
            f"Unsupported eigenvector shape: {eigenvectors.shape}. "
            f"Supported formats: (N_dof, N_wv, N_eig), (N_struct, N_wv, N_eig, N_dof), "
            f"or (N_samples, 2, H, W)"
        )
    
    # Check dimension compatibility before processing
    import scipy.sparse as sp
    
    # Get expected DOF size from first T matrix
    T_sample = None
    if isinstance(T_data, list) and len(T_data) > 0:
        T_sample = T_data[0]
    elif isinstance(T_data, np.ndarray):
        T_sample = T_data[0] if T_data.ndim == 3 else T_data
    
    if T_sample is not None:
        if sp.issparse(T_sample):
            expected_dof = T_sample.shape[0]  # Reduced DOF space
        else:
            expected_dof = T_sample.shape[0]
        
        actual_dof = eigenvectors.shape[0]
        
        if expected_dof != actual_dof:
            print(f"    ERROR: Dimension mismatch - matrices expect {expected_dof} DOF, eigenvectors have {actual_dof}")
            print(f"    Field format eigenvectors may not be directly usable for matrix reconstruction")
            print(f"    Consider using full DOF format eigenvectors")
            return None
    
    # Process each wavevector
    for wv_idx in range(n_wavevectors):
        # Get transformation matrix for this wavevector
        T = None
        if isinstance(T_data, list):
            if len(T_data) > wv_idx and T_data[wv_idx] is not None:
                T = T_data[wv_idx]
            else:
                continue
        elif isinstance(T_data, np.ndarray):
            if T_data.ndim == 3:
                T = T_data[wv_idx]
            else:
                T = T_data  # Single T matrix for all wavevectors
        else:
            continue
        
        if T is None:
            continue
        
        # Convert matrices to sparse format for efficiency
        T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.float32))
        K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
        M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
        
        # Transform to reduced space: Kr = T' * K * T, Mr = T' * M * T
        Kr = T_sparse.conj().T @ K_sparse @ T_sparse
        Mr = T_sparse.conj().T @ M_sparse @ T_sparse
        
        # Process each eigenvalue band
        for band_idx in range(N_eig):
            # Extract eigenvector for this wavevector and band
            eigvec = eigenvectors[:, wv_idx, band_idx].astype(np.complex128)
            
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
            eigval = np.linalg.norm(Kr_eigvec) / np.linalg.norm(Mr_eigvec)
            frequencies_recon[wv_idx, band_idx] = np.sqrt(np.real(eigval)) / (2 * np.pi)
    
    return frequencies_recon


def create_grid_interpolators(wavevectors, frequencies, N_wv):
    """
    Create grid interpolators for each eigenvalue band.
    """
    x_unique = np.sort(np.unique(wavevectors[:, 0]))
    y_unique = np.sort(np.unique(wavevectors[:, 1]))

    # Auto-detect grid size if N_wv is None
    if N_wv is None:
        N_x, N_y = len(x_unique), len(y_unique)
    elif isinstance(N_wv, int) or np.isscalar(N_wv):
        N_x, N_y = int(N_wv), int(N_wv)
        if len(x_unique) != N_x or len(y_unique) != N_y:
            N_x, N_y = len(x_unique), len(y_unique)
    else:
        N_x, N_y = int(N_wv[0]), int(N_wv[1])
        if len(x_unique) != N_x or len(y_unique) != N_y:
            N_x, N_y = len(x_unique), len(y_unique)

    expected_total = N_x * N_y
    actual_total = len(wavevectors)
    if expected_total != actual_total:
        return None, None  # Signal to use scattered interpolation

    # MATLAB column-major equivalent reshape
    frequencies_grid = frequencies.reshape((N_y, N_x, frequencies.shape[1]), order='F')

    # Create interpolators for each band
    grid_interp = []
    for eig_idx in range(frequencies.shape[1]):
        interp = RegularGridInterpolator(
            (y_unique, x_unique),
            frequencies_grid[:, :, eig_idx],
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        grid_interp.append(interp)

    wavevectors_grid = (x_unique, y_unique)
    return grid_interp, wavevectors_grid


def extract_grid_points_on_contour(wavevectors, frequencies, contour_info, a, tolerance=1e-10):
    """Extract actual grid points that lie on the IBZ contour path (no interpolation)."""
    vertices = contour_info['vertices']
    contour_points_list = []
    contour_freqs_list = []
    contour_param_list = []
    total_distance = 0.0

    for seg_idx in range(len(vertices) - 1):
        v_start = vertices[seg_idx]
        v_end = vertices[seg_idx + 1]
        direction = v_end - v_start
        segment_length = np.linalg.norm(direction)
        if segment_length < tolerance:
            continue
        direction_unit = direction / segment_length

        segment_points = []
        segment_freqs = []
        segment_distances = []
        for i, wv in enumerate(wavevectors):
            to_point = wv - v_start
            projection = np.dot(to_point, direction_unit)
            if -tolerance <= projection <= segment_length + tolerance:
                perpendicular = to_point - projection * direction_unit
                if np.linalg.norm(perpendicular) < tolerance:
                    segment_points.append(wv)
                    segment_freqs.append(frequencies[i])
                    segment_distances.append(projection)

        if len(segment_points) > 0:
            sort_idx = np.argsort(segment_distances)
            segment_points = np.array(segment_points)[sort_idx]
            segment_freqs = np.array(segment_freqs)[sort_idx]
            segment_distances = np.array(segment_distances)[sort_idx]
            start_idx = 1 if seg_idx > 0 else 0
            contour_points_list.append(segment_points[start_idx:])
            contour_freqs_list.append(segment_freqs[start_idx:])
            contour_param_list.append(total_distance + segment_distances[start_idx:])

        total_distance += segment_length

    contour_points = np.vstack(contour_points_list) if contour_points_list else np.empty((0, 2))
    contour_freqs = np.vstack(contour_freqs_list) if contour_freqs_list else np.empty((0, frequencies.shape[1]))
    contour_param = np.concatenate(contour_param_list) if contour_param_list else np.array([])
    if len(contour_param) > 0 and total_distance > 0:
        contour_param = contour_param / total_distance * contour_info['N_segment']
    return contour_points, contour_freqs, contour_param


def plot_dispersion_on_contour(ax, contour_info, frequencies_contour, contour_param=None, 
                               title='Dispersion', mark_points=False, frequencies_recon_contour=None):
    """Plot dispersion relation on IBZ contour, optionally with reconstruction overlay."""
    if contour_param is None:
        x_param = contour_info['wavevector_parameter']
    else:
        x_param = contour_param

    # Plot original frequencies
    for band_idx in range(frequencies_contour.shape[1]):
        ax.plot(x_param, 
               frequencies_contour[:, band_idx],
               linewidth=2, label='original' if band_idx == 0 else None)
        
        if mark_points:
            ax.plot(x_param, 
                   frequencies_contour[:, band_idx],
                   'o', markersize=4, markeredgewidth=0.5, markeredgecolor='white')
    
    # Plot reconstructed frequencies if provided
    if frequencies_recon_contour is not None:
        for band_idx in range(frequencies_recon_contour.shape[1]):
            ax.plot(x_param,
                   frequencies_recon_contour[:, band_idx],
                   '--', linewidth=1, alpha=0.7, 
                   label='reconstructed' if band_idx == 0 else None)
    
    # Add vertical lines at segment boundaries
    for i in range(contour_info['N_segment'] + 1):
        ax.axvline(i, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add vertex labels
    if 'vertex_labels' in contour_info and contour_info['vertex_labels']:
        vertex_positions = np.linspace(0, contour_info['N_segment'], 
                                      len(contour_info['vertex_labels']))
        ax.set_xticks(vertex_positions)
        ax.set_xticklabels(contour_info['vertex_labels'])
    
    ax.set_xlabel('Wavevector Contour Parameter', fontsize=12)
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    if frequencies_recon_contour is not None:
        ax.legend()


def main(cli_data_dir=None, cli_original_dir=None, n_structs=None, infer=True):
    """
    Main script execution.
    
    Parameters
    ----------
    cli_data_dir : str, optional
        Path to PyTorch dataset directory
    cli_original_dir : str, optional
        Path to original dataset (for eigenvalue data, only used if infer=False)
    n_structs : int, optional
        Number of structures to plot
    infer : bool, optional
        If True, reconstruct frequencies from eigenvectors using K, M, T (default: True)
        If False, use eigenvalue_data directly (requires original_dir)
    """
    print("=" * 70)
    print("Plot Dispersion Script with Eigenfrequency Reconstruction")
    print("=" * 70)
    
    # Configuration
    if cli_data_dir is not None:
        data_dir = Path(cli_data_dir)
    else:
        data_dir = Path(r"D:\Research\NO-2D-Metamaterials\data\out_binarized_1")
    
    original_data_dir = None
    if cli_original_dir is not None:
        original_data_dir = Path(cli_original_dir)
    
    if not data_dir.exists():
        print(f"\nERROR: Dataset directory not found: {data_dir}")
        return
    
    # Flags
    isExportPng = True
    png_resolution = 150
    mark_points = True
    use_interpolation = False
    N_k_interp = 120
    
    # Load data
    print(f"\nLoading PyTorch dataset from: {data_dir}")
    if infer:
        print("  Mode: INFER frequencies from eigenvectors (infer=True)")
        print("  Will compute K, M, T matrices and reconstruct frequencies")
        print("  All data should be in the dataset directory")
    else:
        print("  Mode: USE eigenvalue_data directly (infer=False)")
        print("  Will load eigenvalue data from original_dir")
        if original_data_dir is None:
            print("\nERROR: original_dir is required when infer=False")
            print("  Please provide path to eigenvalue data (NumPy .npy or MATLAB .mat file)")
            return
    
    try:
        # Only load eigenvalue data if infer=False
        # Pass require_eigenvalue_data=False when in infer mode
        data = load_pt_dataset(data_dir, original_data_dir if not infer else None, 
                             require_eigenvalue_data=not infer)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        return
    
    # Print dataset structure
    print(f"\n{'='*70}")
    print("Dataset Structure:")
    print(f"{'='*70}")
    for key in ['designs', 'wavevectors', 'eigenvalue_data', 'K_data', 'M_data', 'T_data', 'eigenvectors']:
        if key in data:
            item = data[key]
            if hasattr(item, 'shape'):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, list):
                print(f"  {key}: list with {len(item)} elements")
    print(f"{'='*70}\n")
    
    # Create output directory
    import os
    current_dir = Path(os.getcwd())
    dataset_name = data_dir.name
    output_dir = current_dir / 'plots' / f'{dataset_name}_recon'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Extract shapes
    designs = data['designs']
    wavevectors_all = data['wavevectors']
    
    if infer:
        # Mode: Infer frequencies from eigenvectors
        # Check if we have eigenvectors
        has_eigenvectors = 'eigenvectors' in data and data['eigenvectors'] is not None
        
        if not has_eigenvectors:
            print("\nERROR: Eigenvectors are required when infer=True")
            print("  Please provide eigenvectors_full.pt in the dataset directory")
            return
        
        # Check if we have K, M, T data (optional - will compute if missing)
        has_K_data = 'K_data' in data and data['K_data'] is not None
        has_M_data = 'M_data' in data and data['M_data'] is not None
        has_T_data = 'T_data' in data and data['T_data'] is not None
        
        if has_K_data and has_M_data and has_T_data:
            print("  Found K, M, T matrices in dataset - will use them")
        else:
            print("  K, M, T matrices not found - will compute them on-the-fly")
        
        # Try to load eigenvalue_data as fallback (in case reconstruction fails)
        # This allows us to still generate plots even if eigenvectors are in field format
        eigenvalues_all = None
        if original_data_dir is not None:
            try:
                # Try to load eigenvalue_data from original_dir as fallback
                if (original_data_dir / 'eigenvalue_data.npy').exists():
                    eigenvalues_all = np.load(original_data_dir / 'eigenvalue_data.npy')
                    print("  Loaded eigenvalue_data as fallback (for use if reconstruction fails)")
                elif original_data_dir.suffix == '.mat' or (original_data_dir.parent / f'{original_data_dir.name}.mat').exists():
                    mat_file = original_data_dir if original_data_dir.suffix == '.mat' else (original_data_dir.parent / f'{original_data_dir.name}.mat')
                    if mat_file.exists():
                        import h5py
                        with h5py.File(mat_file, 'r') as f:
                            eigenvalue_data_orig = np.array(f['EIGENVALUE_DATA'], dtype=np.float32)
                            if len(eigenvalue_data_orig.shape) == 3:
                                if eigenvalue_data_orig.shape[0] > eigenvalue_data_orig.shape[2]:
                                    eigenvalue_data_orig = eigenvalue_data_orig.transpose(0, 2, 1)
                                elif eigenvalue_data_orig.shape[2] > eigenvalue_data_orig.shape[0]:
                                    eigenvalue_data_orig = eigenvalue_data_orig.transpose(2, 0, 1)
                                else:
                                    eigenvalue_data_orig = eigenvalue_data_orig.transpose(1, 0, 2)
                            eigenvalues_all = eigenvalue_data_orig
                            print("  Loaded eigenvalue_data from MATLAB file as fallback")
            except Exception as e:
                print(f"  Could not load eigenvalue_data as fallback: {e}")
        
        can_reconstruct = True
        
    else:
        # Mode: Use eigenvalue_data directly
        if 'eigenvalue_data' not in data:
            print("\nERROR: eigenvalue_data is required when infer=False")
            print("  Please provide original_dir pointing to eigenvalue data")
            return
        
        eigenvalues_all = data['eigenvalue_data']
        can_reconstruct = False  # No reconstruction needed
    
    # Infer parameters
    N_pix = designs.shape[1]  # Assume square designs
    
    if infer:
        # Get N_eig from eigenvectors shape
        eigenvectors = data['eigenvectors']
        n_wv_per_struct = wavevectors_all.shape[1]  # Number of wavevectors per structure
        
        if eigenvectors.ndim == 4:
            if eigenvectors.shape[1] == 2:
                # Format: (N_samples, 2, H, W) - field format with x/y components
                # Infer N_eig from total samples
                total_samples = eigenvectors.shape[0]
                N_eig = 6  # Default
                
                for test_N_eig in [6, 8, 10, 12]:
                    samples_per_struct = n_wv_per_struct * test_N_eig
                    if total_samples % samples_per_struct == 0:
                        inferred_N_struct = total_samples // samples_per_struct
                        if inferred_N_struct == designs.shape[0]:
                            N_eig = test_N_eig
                            print(f"  Inferred N_eig = {N_eig} from field format eigenvectors")
                            break
            else:
                # Format: (N_struct, N_wv, N_eig, N_dof) - PyTorch style DOF format
                N_eig = eigenvectors.shape[2]
                print(f"  Detected N_eig = {N_eig} from PyTorch-style DOF format")
                
        elif eigenvectors.ndim == 3:
            # Format: (N_dof, N_wv, N_eig) - MATLAB style DOF format
            N_eig = eigenvectors.shape[2]
            print(f"  Detected N_eig = {N_eig} from MATLAB-style DOF format")
        else:
            N_eig = 6  # Default
            print(f"  Using default N_eig = {N_eig}")
    else:
        # Use eigenvalue_data to infer N_eig
        N_eig = eigenvalues_all.shape[2] if eigenvalues_all.ndim == 3 else 6
        print(f"  Detected N_eig = {N_eig} from eigenvalue_data")
    a = 1.0  # Default lattice parameter
    
    # Material parameter ranges (default values)
    E_min, E_max = 20e6, 200e9
    rho_min, rho_max = 400, 8000
    nu_min, nu_max = 0.05, 0.3
    
    # Number of structures/geometries to plot
    # If n_structs is None, default to 5 (or all if dataset has fewer than 5)
    # If n_structs is specified, limit to that number (or all if dataset has fewer)
    if n_structs is None:
        n_structs_to_plot = min(5, designs.shape[0])
        print(f"\nNumber of geometries to process: {n_structs_to_plot} (default: 5, dataset has {designs.shape[0]} total)")
    else:
        n_structs_to_plot = min(int(n_structs), designs.shape[0])
        if n_structs_to_plot < designs.shape[0]:
            print(f"\nNumber of geometries to process: {n_structs_to_plot} (truncated from {designs.shape[0]} total)")
        else:
            print(f"\nNumber of geometries to process: {n_structs_to_plot} (all geometries in dataset)")
    
    struct_idxs = range(n_structs_to_plot)
    
    print(f"Plotting {len(struct_idxs)} structures...")
    if infer:
        print("  Mode: Reconstructing frequencies from eigenvectors")
    else:
        print("  Mode: Using eigenvalue_data directly")
    
    for struct_idx in struct_idxs:
        print(f"\n{'='*70}")
        print(f"Processing structure {struct_idx + 1}/{len(struct_idxs)}")
        print(f"{'='*70}")
        
        # Get design and material properties
        design_param = designs[struct_idx, :, :]
        elastic_modulus = E_min + (E_max - E_min) * design_param
        density = rho_min + (rho_max - rho_min) * design_param
        poisson_ratio = nu_min + (nu_max - nu_min) * design_param
        geometry = np.stack([elastic_modulus, density, poisson_ratio], axis=-1)
        
        # Plot design
        print("  Plotting design...")
        fig_design, _ = plot_design(geometry)
        if isExportPng:
            png_path = output_dir / 'design' / f'{struct_idx}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_design.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
            plt.close(fig_design)
        
        # Extract wavevector data
        wavevectors = wavevectors_all[struct_idx, :, :]
        
        # Get frequencies based on mode
        if infer:
            # Mode: Reconstruct from eigenvectors
            frequencies = None  # Will be computed
            frequencies_recon = None
            print("  Reconstructing frequencies from eigenvectors...")
            try:
                # Get or compute K, M matrices
                if has_K_data and has_M_data:
                    print("    Using K, M matrices from dataset...")
                    K = data['K_data'][struct_idx] if isinstance(data['K_data'], (list, np.ndarray)) and len(data['K_data']) > struct_idx else data['K_data']
                    M = data['M_data'][struct_idx] if isinstance(data['M_data'], (list, np.ndarray)) and len(data['M_data']) > struct_idx else data['M_data']
                else:
                    print("    Computing K, M matrices...")
                    const = create_const_dict(design_param, N_pix, a=a, 
                                            E_min=E_min, E_max=E_max,
                                            rho_min=rho_min, rho_max=rho_max,
                                            nu_min=nu_min, nu_max=nu_max)
                    K, M = compute_K_M_matrices(const)
                
                # Get T matrices
                if has_T_data:
                    print("    Using T matrices from dataset...")
                    T_data = data['T_data']
                else:
                    print("    Computing T matrices...")
                    const = create_const_dict(design_param, N_pix, a=a,
                                            E_min=E_min, E_max=E_max,
                                            rho_min=rho_min, rho_max=rho_max,
                                            nu_min=nu_min, nu_max=nu_max)
                    T_data = []
                    for wv in wavevectors:
                        T = get_transformation_matrix(wv.astype(np.float32), const)
                        T_data.append(T)
                
                # Get eigenvectors
                eigenvectors = data['eigenvectors']
                
                # Reconstruct frequencies
                frequencies_recon = reconstruct_frequencies_from_eigenvectors(
                    K, M, T_data, eigenvectors, wavevectors, N_eig,
                    struct_idx=struct_idx, N_struct=designs.shape[0]
                )
                
                if frequencies_recon is None:
                    print("    Reconstruction failed - field format eigenvectors incompatible with matrix DOF space")
                    print("    Falling back to eigenvalue_data if available...")
                    
                    # Fallback to eigenvalue_data if available
                    if eigenvalues_all is not None:
                        frequencies = eigenvalues_all[struct_idx, :, :]
                        print(f"    Using eigenvalue_data: {frequencies.shape[0]} wavevectors × {frequencies.shape[1]} bands")
                        # If eigenvalue_data has fewer wavevectors than the dataset, it's likely IBZ subset
                        if frequencies.shape[0] != len(wavevectors):
                            print(f"    eigenvalue_data has {frequencies.shape[0]} wavevectors (IBZ subset), dataset has {len(wavevectors)} (full grid)")
                            print(f"    Matching IBZ wavevectors from dataset...")
                            
                            # Generate IBZ wavevectors using same parameters as MATLAB
                            # MATLAB uses: N_wv = [25, ceil(25/2)] = [25, 13], symmetry_type = 'p4mm'
                            # We need to determine the actual parameters used, but for now try to match by finding closest wavevectors
                            try:
                                # Generate IBZ wavevectors to match
                                # Try common parameters: N_wv = [25, 13] for p4mm gives ~325 points, but IBZ subset is smaller
                                # Actually, let's match by finding the closest wavevectors in the dataset
                                from scipy.spatial.distance import cdist
                                
                                # Load the actual IBZ wavevectors from MATLAB data if available
                                # For now, find matching wavevectors by comparing coordinates
                                # The eigenvalue_data corresponds to IBZ wavevectors, so we need to find which dataset wavevectors match
                                
                                # Try to load wavevectors from MATLAB file to get exact IBZ set
                                if original_data_dir is not None:
                                    mat_file = original_data_dir if original_data_dir.suffix == '.mat' else (original_data_dir.parent / f'{original_data_dir.name}.mat')
                                    if mat_file.exists():
                                        try:
                                            import h5py
                                            with h5py.File(mat_file, 'r') as f:
                                                if 'WAVEVECTOR_DATA' in f:
                                                    # MATLAB format: WAVEVECTOR_DATA is (N_wv, 2, N_struct)
                                                    wv_matlab = np.array(f['WAVEVECTOR_DATA'][:, :, struct_idx], dtype=np.float32)
                                                    print(f"    Loaded IBZ wavevectors from MATLAB: {wv_matlab.shape}")
                                                    
                                                    if wv_matlab.shape[0] == frequencies.shape[0]:
                                                        # Match wavevectors by finding closest in dataset
                                                        from scipy.spatial.distance import cdist
                                                        distances = cdist(wv_matlab, wavevectors)
                                                        matched_indices = np.argmin(distances, axis=1)
                                                        # Check if matches are close enough (within tolerance)
                                                        min_distances = np.min(distances, axis=1)
                                                        tolerance = 1e-5  # Increased tolerance for float16 precision
                                                        max_dist = np.max(min_distances)
                                                        
                                                        if np.all(min_distances < tolerance):
                                                            wavevectors = wavevectors[matched_indices, :]
                                                            print(f"    Matched {len(wavevectors)} IBZ wavevectors from dataset (max distance: {max_dist:.2e})")
                                                        else:
                                                            print(f"    WARNING: Some wavevector matches have distance > tolerance (max: {max_dist:.2e})")
                                                            # Still use the matches, but warn
                                                            wavevectors = wavevectors[matched_indices, :]
                                                            print(f"    Using matched wavevectors anyway")
                                                    else:
                                                        # Fallback: use first N wavevectors
                                                        wavevectors = wavevectors[:frequencies.shape[0], :]
                                                        print(f"    WARNING: MATLAB wavevectors ({wv_matlab.shape[0]}) don't match frequencies ({frequencies.shape[0]})")
                                                else:
                                                    # Fallback: use first N wavevectors
                                                    wavevectors = wavevectors[:frequencies.shape[0], :]
                                                    print(f"    WAVEVECTOR_DATA not in MATLAB file, using first {frequencies.shape[0]} wavevectors")
                                        except Exception as e:
                                            print(f"    WARNING: Could not load WAVEVECTOR_DATA from MATLAB: {e}")
                                            # Fallback: use first N wavevectors
                                            wavevectors = wavevectors[:frequencies.shape[0], :]
                                            print(f"    Using first {frequencies.shape[0]} wavevectors as fallback")
                                    else:
                                        # Fallback: use first N wavevectors
                                        wavevectors = wavevectors[:frequencies.shape[0], :]
                                        print(f"    MATLAB file not found, using first {frequencies.shape[0]} wavevectors")
                                else:
                                    # Fallback: use first N wavevectors
                                    wavevectors = wavevectors[:frequencies.shape[0], :]
                                    print(f"    No original_data_dir, using first {frequencies.shape[0]} wavevectors")
                            except Exception as e:
                                print(f"    WARNING: Could not match IBZ wavevectors: {e}")
                                # Fallback: use first N wavevectors
                                wavevectors = wavevectors[:frequencies.shape[0], :]
                                print(f"    Using first {frequencies.shape[0]} wavevectors as fallback")
                        frequencies_recon = None
                    else:
                        print("    No eigenvalue_data available as fallback")
                        frequencies = None
                        frequencies_recon = None
                else:
                    # Use reconstructed frequencies as the main frequencies
                    frequencies = frequencies_recon
                    print(f"    Reconstructed {frequencies.shape[0]} wavevectors × {frequencies.shape[1]} bands")
                
            except Exception as e:
                print(f"    ERROR during reconstruction: {e}")
                import traceback
                traceback.print_exc()
                frequencies = None
                frequencies_recon = None
        else:
            # Mode: Use eigenvalue_data directly
            frequencies = eigenvalues_all[struct_idx, :, :]
            frequencies_recon = None
        
        # Create interpolants (using griddata for scattered interpolation)
        # We'll evaluate on contour points directly
        
        # Check if we have frequencies
        if frequencies is None:
            print("  WARNING: No frequencies available, skipping dispersion plot")
            continue
        
        # Get IBZ contour
        try:
            # Use number of wavevectors to determine grid size for contour
            n_wv = len(wavevectors)
            # For IBZ contour, typically use a reasonable grid size (10 is common)
            # But we need to match the actual wavevector grid
            _, contour_info = get_IBZ_contour_wavevectors(10, a, 'p4mm')
            wavevectors_contour, frequencies_contour_grid, contour_param_grid = \
                extract_grid_points_on_contour(wavevectors, frequencies, contour_info, a, tolerance=2e-3)
            
            if len(frequencies_contour_grid) == 0:
                print(f"  WARNING: No contour points found, skipping dispersion plot")
                continue
        except Exception as e:
            print(f"  WARNING: Could not generate contour: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Evaluate on contour
        frequencies_contour = frequencies_contour_grid
        contour_param = contour_param_grid
        frequencies_recon_contour = None  # No overlay in infer mode (frequencies are already reconstructed)
        
        # Plot dispersion
        print("  Plotting dispersion...")
        fig_disp = plt.figure(figsize=(10, 6))
        ax_disp = fig_disp.add_subplot(111)
        
        if infer:
            title = 'Dispersion Relation (Reconstructed from Eigenvectors)'
        else:
            title = 'Dispersion Relation'
        
        plot_dispersion_on_contour(ax_disp, contour_info, frequencies_contour, contour_param,
                                 title=title, mark_points=mark_points,
                                 frequencies_recon_contour=frequencies_recon_contour)
        
        if isExportPng:
            if infer:
                png_path = output_dir / 'dispersion' / f'{struct_idx}_recon.png'
            else:
                png_path = output_dir / 'dispersion' / f'{struct_idx}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_disp.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
            print(f"    Saved: {png_path}")
            plt.close(fig_disp)
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    if isExportPng:
        print(f"Output saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot dispersion with eigenfrequency reconstruction")
    parser.add_argument("data_dir", nargs="?", help="Path to the PyTorch dataset directory")
    parser.add_argument("original_dir", nargs="?", help="Path to original dataset (for eigenvalue data, required if --no-infer)")
    parser.add_argument("-n", "--n-structs", "--max-geometries", type=int, default=None, 
                       dest="n_structs",
                       help="Maximum number of geometries to process (default: 5). If dataset has fewer geometries, all will be processed.")
    parser.add_argument("--no-infer", action="store_true", help="Use eigenvalue_data directly instead of reconstructing (requires original_dir)")
    args = parser.parse_args()
    main(args.data_dir, args.original_dir, args.n_structs, infer=not args.no_infer)

