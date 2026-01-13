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
    from scipy.interpolate import interp1d
except ImportError:
    import plotting
    import wavevectors
    import system_matrices
    import system_matrices_vec
    from scipy.interpolate import interp1d
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


def apply_steel_rubber_paradigm_single_channel(design_single, E_min, E_max, rho_min, rho_max, nu_min, nu_max):
    """
    Apply steel-rubber paradigm to single-channel design.
    
    This matches MATLAB's apply_steel_rubber_paradigm.m functionality.
    Takes a single-channel design (N_pix x N_pix) with values in [0, 1] and maps
    them to normalized property values for E, rho, nu using the steel-rubber paradigm.
    
    Parameters
    ----------
    design_single : ndarray
        Single-channel design (N_pix, N_pix) with values in [0, 1]
    E_min, E_max : float
        Young's modulus bounds
    rho_min, rho_max : float
        Density bounds
    nu_min, nu_max : float
        Poisson's ratio bounds
        
    Returns
    -------
    design_3ch : ndarray
        3-channel design (N_pix, N_pix, 3) with normalized property values
    """
    # Material properties for steel-rubber paradigm (matching MATLAB)
    design_in_polymer = 0.0
    design_in_steel = 1.0
    
    E_polymer = 100e6  # MATLAB: 100e6
    E_steel = 200e9
    
    rho_polymer = 1200
    rho_steel = 8000
    
    nu_polymer = 0.45
    nu_steel = 0.3
    
    # Convert material properties to normalized design space
    # Formula: design_out = (prop_value - prop_min) / (prop_max - prop_min)
    design_out_polymer_E = (E_polymer - E_min) / (E_max - E_min)
    design_out_polymer_rho = (rho_polymer - rho_min) / (rho_max - rho_min)
    design_out_polymer_nu = (nu_polymer - nu_min) / (nu_max - nu_min)
    
    design_out_steel_E = (E_steel - E_min) / (E_max - E_min)
    design_out_steel_rho = (rho_steel - rho_min) / (rho_max - rho_min)
    design_out_steel_nu = (nu_steel - nu_min) / (nu_max - nu_min)
    
    # Create 3-channel design
    N_pix = design_single.shape[0]
    design_3ch = np.zeros((N_pix, N_pix, 3))
    
    # Map design values to normalized property values using linear interpolation
    # For each property, interpolate between polymer (design=0) and steel (design=1) values
    design_flat = design_single.flatten()
    
    # Elastic modulus
    design_3ch[:, :, 0] = np.interp(
        design_flat,
        [design_in_polymer, design_in_steel],
        [design_out_polymer_E, design_out_steel_E]
    ).reshape(N_pix, N_pix)
    
    # Density
    design_3ch[:, :, 1] = np.interp(
        design_flat,
        [design_in_polymer, design_in_steel],
        [design_out_polymer_rho, design_out_steel_rho]
    ).reshape(N_pix, N_pix)
    
    # Poisson's ratio
    design_3ch[:, :, 2] = np.interp(
        design_flat,
        [design_in_polymer, design_in_steel],
        [design_out_polymer_nu, design_out_steel_nu]
    ).reshape(N_pix, N_pix)
    
    # Don't clip - let plot_design handle the actual value ranges
    # Some normalized values may exceed 1.0 if material properties are outside bounds
    # This is correct behavior matching MATLAB's apply_steel_rubber_paradigm.m
    # plot_design will now use individual colorbars and auto-scale for each property
    
    return design_3ch


def create_const_dict(design, N_pix, N_ele=1, a=1.0, 
                       E_min=20e6, E_max=200e9,
                       rho_min=1200, rho_max=8000,
                       nu_min=0.0, nu_max=0.5,
                       t=1.0, design_scale='linear',
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
        Density range (default: 1200 to 8000, matching MATLAB)
    nu_min, nu_max : float, optional
        Poisson's ratio range (default: 0.0 to 0.5, matching MATLAB)
    t : float, optional
        Thickness (default: 1.0, matching MATLAB)
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


def convert_field_to_dof_format(eigenvectors_field, N_pix, N_ele=1, reduced_space=True):
    """
    Convert field format eigenvectors (N_samples, 2, H, W) to DOF format (N_dof).
    
    Field format stores (N_pix, N_pix) for x and y components separately.
    If reduced_space=True: maps to reduced DOF space 2 * (N_ele * N_pix)^2
    If reduced_space=False: maps to full DOF space 2 * (N_ele * N_pix + 1)^2
    
    The field format eigenvectors are typically already in reduced space after T transformation.
    Each pixel (i, j) in field format maps to a block of (N_ele, N_ele) nodes in the reduced grid.
    
    Parameters
    ----------
    eigenvectors_field : ndarray
        Field format: (N_samples, 2, H, W) where H=W=N_pix
    N_pix : int
        Number of pixels per dimension
    N_ele : int, optional
        Number of elements per pixel (default: 4)
    reduced_space : bool, optional
        If True, convert to reduced DOF space (default: True)
        If False, convert to full DOF space
        
    Returns
    -------
    eigenvectors_dof : ndarray
        DOF format: (N_samples, N_dof)
        If reduced_space=True: N_dof = 2 * (N_ele * N_pix)^2
        If reduced_space=False: N_dof = 2 * (N_ele * N_pix + 1)^2
    """
    N_samples = eigenvectors_field.shape[0]
    H, W = eigenvectors_field.shape[2], eigenvectors_field.shape[3]
    
    if H != N_pix or W != N_pix:
        raise ValueError(f"Field format H={H}, W={W} does not match N_pix={N_pix}")
    
    # Extract x and y components
    x_field = eigenvectors_field[:, 0, :, :]  # (N_samples, H, W)
    y_field = eigenvectors_field[:, 1, :, :]  # (N_samples, H, W)
    
    # Flatten spatial dimensions
    x_flat = x_field.reshape(N_samples, H * W)  # (N_samples, N_pix^2)
    y_flat = y_field.reshape(N_samples, H * W)  # (N_samples, N_pix^2)
    
    if reduced_space:
        # Reduced DOF space: N_dof_reduced = 2 * (N_ele * N_pix)^2
        # Field format (N_pix, N_pix) represents pixel-level data
        # Each pixel corresponds to (N_ele, N_ele) nodes in the reduced grid
        # So we need to expand: pixel (i, j) -> nodes (i*N_ele:(i+1)*N_ele, j*N_ele:(j+1)*N_ele)
        N_nodes_reduced = N_ele * N_pix
        N_dof_reduced = 2 * N_nodes_reduced * N_nodes_reduced
        
        # Vectorized conversion: expand each pixel to N_ele x N_ele block of nodes
        # Reshape x_flat and y_flat to (N_samples, N_pix, N_pix)
        x_field_reshaped = x_flat.reshape(N_samples, N_pix, N_pix)
        y_field_reshaped = y_flat.reshape(N_samples, N_pix, N_pix)
        
        # Expand each pixel to N_ele x N_ele block using np.repeat
        # For x: (N_samples, N_pix, N_pix) -> (N_samples, N_pix, N_pix, 1, 1) -> (N_samples, N_pix, N_ele, N_pix, N_ele)
        x_expanded = np.repeat(np.repeat(x_field_reshaped[:, :, :, np.newaxis, np.newaxis], 
                                         N_ele, axis=3), N_ele, axis=4)
        y_expanded = np.repeat(np.repeat(y_field_reshaped[:, :, :, np.newaxis, np.newaxis], 
                                         N_ele, axis=3), N_ele, axis=4)
        
        # Reshape to (N_samples, N_nodes_reduced, N_nodes_reduced)
        x_nodes = x_expanded.reshape(N_samples, N_nodes_reduced, N_nodes_reduced)
        y_nodes = y_expanded.reshape(N_samples, N_nodes_reduced, N_nodes_reduced)
        
        # Flatten node grid: (N_samples, N_nodes_reduced, N_nodes_reduced) -> (N_samples, N_nodes_reduced^2)
        x_nodes_flat = x_nodes.reshape(N_samples, N_nodes_reduced * N_nodes_reduced)
        y_nodes_flat = y_nodes.reshape(N_samples, N_nodes_reduced * N_nodes_reduced)
        
        # Interleave x and y: DOF[2*i] = x, DOF[2*i+1] = y
        eigenvectors_dof = np.zeros((N_samples, N_dof_reduced), dtype=eigenvectors_field.dtype)
        eigenvectors_dof[:, 0::2] = x_nodes_flat  # x components at even indices
        eigenvectors_dof[:, 1::2] = y_nodes_flat  # y components at odd indices
    else:
        # Full DOF space: N_dof_full = 2 * (N_ele * N_pix + 1)^2
        N_nodes_full = N_ele * N_pix + 1
        N_dof_full = 2 * N_nodes_full * N_nodes_full
        
        # Create DOF format array
        eigenvectors_dof = np.zeros((N_samples, N_dof_full), dtype=eigenvectors_field.dtype)
        
        # Field format maps to [:N_pix, :N_pix] in the full grid
        for i in range(N_samples):
            for j in range(H * W):
                row = j // W
                col = j % W
                node_idx = row * N_nodes_full + col
                eigenvectors_dof[i, 2 * node_idx] = x_flat[i, j]
                eigenvectors_dof[i, 2 * node_idx + 1] = y_flat[i, j]
    
    return eigenvectors_dof


def reconstruct_frequencies_from_eigenvectors(K, M, T_data, eigenvectors, wavevectors, N_eig, struct_idx=0, N_struct=1, N_pix=None, N_ele=1):
    """
    Reconstruct frequencies from eigenvectors using K, M, T matrices.
    
    This mirrors the MATLAB plot_dispersion.m functionality (lines 100-116).
    Always reconstructs frequencies - never uses pre-computed eigenvalue_data.
    
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
    N_pix : int, optional
        Number of pixels per dimension (required for field format conversion)
    N_ele : int, optional
        Number of elements per pixel (default: 4, required for field format conversion)
        
    Returns
    -------
    frequencies_recon : ndarray
        Reconstructed frequencies (N_wv, N_eig)
    """
    import scipy.sparse as sp
    
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
            
            if N_pix is None:
                N_pix = H  # Assume H = N_pix
                print(f"    Inferred N_pix = {N_pix} from field format")
            
            # Convert field format to reduced DOF format (eigenvectors are already in reduced space)
            print(f"    Converting field format to reduced DOF format (N_pix={N_pix}, N_ele={N_ele})...")
            eigenvectors_dof = convert_field_to_dof_format(eigenvectors, N_pix, N_ele, reduced_space=True)
            print(f"    Converted to reduced DOF format: shape={eigenvectors_dof.shape}")
            
            # Reshape to (N_struct, N_wv, N_eig, N_dof)
            samples_per_struct = n_wavevectors * N_eig
            total_samples = eigenvectors_dof.shape[0]
            
            if total_samples % samples_per_struct != 0:
                raise ValueError(
                    f"Field format: total samples {total_samples} not divisible by "
                    f"samples per struct {samples_per_struct} (N_wv={n_wavevectors}, N_eig={N_eig})"
                )
            
            inferred_N_struct = total_samples // samples_per_struct
            N_dof = eigenvectors_dof.shape[1]
            print(f"    Reshaped to (N_struct={inferred_N_struct}, N_wv={n_wavevectors}, N_eig={N_eig}, N_dof={N_dof})")
            
            eigenvectors_reshaped = eigenvectors_dof.reshape(inferred_N_struct, n_wavevectors, N_eig, N_dof)
            
            # Extract for this structure: (N_wv, N_eig, N_dof)
            eigenvectors_struct = eigenvectors_reshaped[struct_idx]
            
            # Transpose to (N_dof, N_wv, N_eig) for consistent indexing
            eigenvectors = eigenvectors_struct.transpose(2, 0, 1)  # (N_dof, N_wv, N_eig)
            ev_format = 'field_xy'
            print(f"    Final shape: (N_dof={eigenvectors.shape[0]}, N_wv={n_wavevectors}, N_eig={N_eig})")
            
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
    
    # Get expected DOF size from first T matrix
    T_sample = None
    if isinstance(T_data, list) and len(T_data) > 0:
        T_sample = T_data[0]
    elif isinstance(T_data, np.ndarray):
        T_sample = T_data[0] if T_data.ndim == 3 else T_data
    
    if T_sample is None:
        raise ValueError("T_data is required for frequency reconstruction")
    
    if sp.issparse(T_sample):
        # T matrix shape: (full_dof, reduced_dof)
        # Eigenvectors should be in reduced space, so check against T.shape[1]
        expected_dof = T_sample.shape[1]  # Reduced DOF space (columns of T)
    else:
        expected_dof = T_sample.shape[1]  # Reduced DOF space (columns of T)
    
    actual_dof = eigenvectors.shape[0]
    
    if expected_dof != actual_dof:
        raise ValueError(
            f"Dimension mismatch - reduced matrices expect {expected_dof} DOF (reduced space), eigenvectors have {actual_dof}. "
            f"This may indicate incorrect field-to-DOF conversion or matrix computation."
        )
    
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


def main(cli_data_dir=None, cli_original_dir=None, n_structs=None, infer=True, save_km=True, save_t=False, save_plot_points=False):
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
    save_km : bool, optional
        If True, save computed K and M matrices (default: True)
    save_t : bool, optional
        If True, save computed T matrices (default: False)
    save_plot_points : bool, optional
        If True, save plot point locations (wavevectors_contour, frequencies_contour, contour_param) to .npz file (default: False)
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
        print("  NO FALLBACK: Frequencies are ALWAYS reconstructed from eigenvectors")
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

        # Prepare containers for saving K, M, and T if requested
        km_store_K = None
        km_store_M = None
        t_store = None
        if save_km:
            n_struct_total = designs.shape[0]
            # If existing data present, reuse; else initialize placeholders
            km_store_K = list(data['K_data']) if has_K_data else [None] * n_struct_total
            km_store_M = list(data['M_data']) if has_M_data else [None] * n_struct_total
        if save_t:
            n_struct_total = designs.shape[0]
            # T_data is a list of lists: outer list for structures, inner list for wavevectors
            # If existing data present, reuse; else initialize placeholders
            if has_T_data:
                # T_data might be a single list (for first structure) or list of lists
                if isinstance(data['T_data'], list) and len(data['T_data']) > 0:
                    if isinstance(data['T_data'][0], list):
                        t_store = list(data['T_data'])  # Already list of lists
                    else:
                        # Single structure's T_data - wrap in outer list
                        t_store = [list(data['T_data'])] + [None] * (n_struct_total - 1)
                else:
                    t_store = [None] * n_struct_total
            else:
                t_store = [None] * n_struct_total
        
        if has_K_data and has_M_data and has_T_data:
            print("  Found K, M, T matrices in dataset - will use them")
        else:
            print("  K, M, T matrices not found - will compute them on-the-fly")
        
        # NO FALLBACK: Always reconstruct frequencies from eigenvectors using K, M, T
        print("  Mode: ALWAYS reconstruct frequencies from eigenvectors (no fallback to eigenvalue_data)")
        
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
    
    # Material parameter ranges (default values - matching MATLAB ex_dispersion_batch_save.m)
    E_min, E_max = 20e6, 200e9
    rho_min, rho_max = 1200, 8000  # MATLAB: const.rho_min = 1200, const.rho_max = 8e3
    nu_min, nu_max = 0.0, 0.5  # MATLAB: const.poisson_min = 0, const.poisson_max = 0.5
    t_val = 1.0  # MATLAB: const.t = 1 (line 61)
    
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
    if save_plot_points:
        print("  Plot point locations will be saved to .npz file")
    
    # Store plot points for saving
    plot_points_data = {} if save_plot_points else None
    
    for struct_idx in struct_idxs:
        print(f"\n{'='*70}")
        print(f"Processing structure {struct_idx + 1}/{len(struct_idxs)}")
        print(f"{'='*70}")
        
        # Get design and material properties
        design_param = designs[struct_idx, :, :]  # Single channel design (N_pix, N_pix) with values in [0, 1]
        
        # Plot raw design (before normalization) - replicate single channel to 3 channels
        print("  Plotting raw design (before normalization)...")
        design_raw_3ch = np.zeros((design_param.shape[0], design_param.shape[1], 3))
        design_raw_3ch[:, :, 0] = design_param  # Modulus
        design_raw_3ch[:, :, 1] = design_param  # Density
        design_raw_3ch[:, :, 2] = design_param  # Poisson's ratio
        fig_design_raw, _ = plot_design(design_raw_3ch)
        if isExportPng:
            png_path_raw = output_dir / 'design_raw' / f'{struct_idx}.png'
            png_path_raw.parent.mkdir(parents=True, exist_ok=True)
            fig_design_raw.savefig(png_path_raw, dpi=png_resolution, bbox_inches='tight')
            plt.close(fig_design_raw)
        
        # For plotting: apply steel-rubber paradigm to get actual material property values
        # This matches MATLAB's apply_steel_rubber_paradigm.m functionality
        # The design values [0, 1] are mapped to actual material property values for E, rho, nu
        # First get normalized values, then convert back to actual property values for plotting
        design_normalized = apply_steel_rubber_paradigm_single_channel(
            design_param, E_min, E_max, rho_min, rho_max, nu_min, nu_max
        )
        
        # Convert normalized values back to actual material property values for plotting
        # This ensures the plot shows actual Poisson's ratio values (0.3-0.45) not normalized (0.6-0.9)
        design_for_plot = np.zeros_like(design_normalized)
        design_for_plot[:, :, 0] = E_min + (E_max - E_min) * design_normalized[:, :, 0]  # Elastic modulus
        design_for_plot[:, :, 1] = rho_min + (rho_max - rho_min) * design_normalized[:, :, 1]  # Density
        design_for_plot[:, :, 2] = nu_min + (nu_max - nu_min) * design_normalized[:, :, 2]  # Poisson's ratio
        
        # Plot design (using actual material property values)
        print("  Plotting design (after normalization)...")
        fig_design, _ = plot_design(design_for_plot)
        
        # For matrix computation: compute actual material properties
        elastic_modulus = E_min + (E_max - E_min) * design_param
        density = rho_min + (rho_max - rho_min) * design_param
        poisson_ratio = nu_min + (nu_max - nu_min) * design_param
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
                expected_full_dof = 2 * (1 * N_pix + 1) * (1 * N_pix + 1)  # N_ele=1 -> 2*(N_pix+1)^2
                use_loaded_KM = False
                if has_K_data and has_M_data:
                    K_loaded = data['K_data'][struct_idx] if isinstance(data['K_data'], (list, np.ndarray)) and len(data['K_data']) > struct_idx else data['K_data']
                    M_loaded = data['M_data'][struct_idx] if isinstance(data['M_data'], (list, np.ndarray)) and len(data['M_data']) > struct_idx else data['M_data']
                    try:
                        if K_loaded.shape[0] == expected_full_dof and M_loaded.shape[0] == expected_full_dof:
                            use_loaded_KM = True
                            K = K_loaded
                            M = M_loaded
                            print(f"    Using K, M matrices from dataset (matched expected DOF={expected_full_dof})...")
                        else:
                            print(f"    Loaded K/M shapes {K_loaded.shape}, {M_loaded.shape} do not match expected DOF={expected_full_dof}; recomputing.")
                    except Exception:
                        pass
                if not use_loaded_KM:
                    print("    Computing K, M matrices...")
                    const = create_const_dict(design_param, N_pix, a=a, 
                                            E_min=E_min, E_max=E_max,
                                            rho_min=rho_min, rho_max=rho_max,
                                            nu_min=nu_min, nu_max=nu_max,
                                            t=t_val)
                    K, M = compute_K_M_matrices(const)
                    # Store for saving later if requested
                    if save_km:
                        km_store_K[struct_idx] = K
                        km_store_M[struct_idx] = M
                
                # Get T matrices
                if has_T_data:
                    print("    Using T matrices from dataset...")
                    # T_data might be a list of lists (one per structure) or a single list
                    if isinstance(data['T_data'], list) and len(data['T_data']) > 0:
                        if isinstance(data['T_data'][0], list):
                            # List of lists - get this structure's T matrices
                            T_data = data['T_data'][struct_idx] if struct_idx < len(data['T_data']) else data['T_data'][0]
                        else:
                            # Single list - assume it's for the first structure
                            T_data = data['T_data'] if struct_idx == 0 else None
                    else:
                        T_data = None
                    
                    if T_data is None:
                        print("    T matrices not found for this structure, computing...")
                        has_T_data = False  # Force recomputation
                
                if not has_T_data or T_data is None:
                    print("    Computing T matrices...")
                    const = create_const_dict(design_param, N_pix, a=a,
                                            E_min=E_min, E_max=E_max,
                                            rho_min=rho_min, rho_max=rho_max,
                                            nu_min=nu_min, nu_max=nu_max,
                                            t=t_val)
                    T_data = []
                    for wv in wavevectors:
                        T = get_transformation_matrix(wv.astype(np.float32), const)
                        T_data.append(T)
                    
                    # Store for saving later if requested
                    if save_t:
                        t_store[struct_idx] = T_data
                
                # Get eigenvectors
                eigenvectors = data['eigenvectors']
                
                # Reconstruct frequencies (always - no fallback)
                frequencies_recon = reconstruct_frequencies_from_eigenvectors(
                    K, M, T_data, eigenvectors, wavevectors, N_eig,
                    struct_idx=struct_idx, N_struct=designs.shape[0],
                    N_pix=N_pix, N_ele=1  # Match MATLAB: N_ele=1, reduced DOF=2048, full DOF=2178
                )
                
                # Use reconstructed frequencies as the main frequencies
                frequencies = frequencies_recon
                print(f"    Reconstructed {frequencies.shape[0]} wavevectors × {frequencies.shape[1]} bands")
                
            except Exception as e:
                print(f"    ERROR during reconstruction: {e}")
                import traceback
                traceback.print_exc()
                print("    Reconstruction failed - cannot proceed without frequencies")
                print("    Check that eigenvectors are in correct format and K, M, T matrices are computed correctly")
                frequencies = None
                frequencies_recon = None
        else:
            # Mode: Use eigenvalue_data directly
            frequencies = eigenvalues_all[struct_idx, :, :]
            frequencies_recon = None
        
        # Create interpolants (using griddata for scattered interpolation)
        # We'll evaluate on contour points directly
        
        # Check if we have frequencies (required - no fallback)
        if frequencies is None:
            print("  ERROR: No frequencies available - reconstruction failed")
            print("  Cannot proceed without frequencies. Check eigenvectors and matrix computation.")
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
        
        # Save plot points if requested
        if save_plot_points:
            plot_points_data[f'struct_{struct_idx}_wavevectors_contour'] = wavevectors_contour
            plot_points_data[f'struct_{struct_idx}_frequencies_contour'] = frequencies_contour
            plot_points_data[f'struct_{struct_idx}_contour_param'] = contour_param
            plot_points_data[f'struct_{struct_idx}_use_interpolation'] = np.array([False])  # Always grid points only
        
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

    # Save computed K and M if requested and if we computed any
    if infer and save_km and km_store_K is not None and km_store_M is not None:
        try:
            import torch
            torch.save(km_store_K, data_dir / 'K_data.pt')
            torch.save(km_store_M, data_dir / 'M_data.pt')
            print(f"\nSaved K_data.pt and M_data.pt to {data_dir}")
        except Exception as e:
            print(f"\nWARNING: Failed to save K/M data: {e}")
    
    # Save computed T matrices if requested and if we computed any
    if infer and save_t and t_store is not None:
        try:
            import torch
            # Filter out None entries (structures that weren't processed)
            t_store_filtered = [t for t in t_store if t is not None]
            if len(t_store_filtered) > 0:
                torch.save(t_store_filtered, data_dir / 'T_data.pt')
                print(f"Saved T_data.pt to {data_dir} ({len(t_store_filtered)} structures)")
        except Exception as e:
            print(f"\nWARNING: Failed to save T data: {e}")
    
    # Save plot points if requested
    if save_plot_points and plot_points_data:
        plot_points_path = output_dir / 'plot_points.npz'
        np.savez_compressed(plot_points_path, **plot_points_data)
        print(f"\nPlot point locations saved to: {plot_points_path}")

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
    parser.add_argument("--no-save-km", action="store_true", help="Do not save computed K and M matrices")
    parser.add_argument("--save-t", action="store_true", help="Save computed T matrices (default: False)")
    parser.add_argument("--save-plot-points", action="store_true", help="Save plot point locations (wavevectors_contour, frequencies_contour, contour_param) to .npz file")
    args = parser.parse_args()
    main(args.data_dir, args.original_dir, args.n_structs, infer=not args.no_infer, save_km=not args.no_save_km, save_t=args.save_t, save_plot_points=args.save_plot_points)

