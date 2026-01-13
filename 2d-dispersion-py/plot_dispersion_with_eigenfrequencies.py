"""
Plot Dispersion Script with Eigenfrequencies (Consolidated)

This script loads datasets and creates comprehensive dispersion visualizations.
Supports multiple data formats:
- MATLAB .mat files (v7.3 HDF5 and older formats)
- PyTorch .pt files (directory with .pt files)
- NumPy .npy files (directory with .npy files)

This script consolidates functionality from:
- plot_dispersion_mat.py (MATLAB format)
- plot_dispersion_pt.py (PyTorch format)
- plot_dispersion_np.py (NumPy format)

Features:
- Automatic format detection
- Unit cell designs
- Dispersion relations along IBZ contours
- Frequency reconstruction from eigenvectors (for MATLAB format)
- Comparison of original vs reconstructed frequencies
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata, LinearNDInterpolator
import warnings
import sys

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

# Import local modules - handle both package and script modes
try:
    from plotting import plot_design
    from wavevectors import get_IBZ_contour_wavevectors
    from mat73_loader import load_matlab_v73
    from system_matrices import get_transformation_matrix
    from system_matrices_vec import get_system_matrices_VEC, get_system_matrices_VEC_simplified
except ImportError:
    # If relative imports fail, try absolute imports from current directory
    import plotting
    import wavevectors
    try:
        import mat73_loader
        load_matlab_v73 = mat73_loader.load_matlab_v73
    except ImportError:
        load_matlab_v73 = None
    try:
        import system_matrices
        import system_matrices_vec
        get_transformation_matrix = system_matrices.get_transformation_matrix
        get_system_matrices_VEC = system_matrices_vec.get_system_matrices_VEC
        get_system_matrices_VEC_simplified = system_matrices_vec.get_system_matrices_VEC_simplified
    except ImportError:
        get_transformation_matrix = None
        get_system_matrices_VEC = None
        get_system_matrices_VEC_simplified = None
    plot_design = plotting.plot_design
    get_IBZ_contour_wavevectors = wavevectors.get_IBZ_contour_wavevectors

try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# MATLAB .mat file loader
# ============================================================================

def load_dataset(data_path, verbose=False):
    """
    Load dataset from .mat file (handles both v7.3 HDF5 and older formats).
    Includes dereferencing for HDF5 object arrays (e.g., from convert_mat_precision).
    
    Parameters
    ----------
    data_path : str or Path
        Path to the .mat file
    verbose : bool, optional
        If True, print detailed loading information
        
    Returns
    -------
    data : dict
        Loaded dataset
    """
    print(f"Loading dataset: {data_path}")
    
    # Try to load with scipy first (for older MATLAB files)
    try:
        data = sio.loadmat(data_path, squeeze_me=False)
        # Remove MATLAB metadata keys
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        print("  Loaded using scipy.io.loadmat (MATLAB < v7.3)")
    except (NotImplementedError, ValueError) as e:
        # If that fails (v7.3 HDF5 format or unknown format), use our robust h5py loader
        if "Unknown mat file type" in str(e) or isinstance(e, NotImplementedError):
            print("  File is MATLAB v7.3 format, using robust h5py loader...")
        else:
            print(f"  scipy.io.loadmat failed: {e}, trying h5py loader...")
        try:
            data = load_matlab_v73(data_path, verbose=verbose)
        except ImportError:
            raise ImportError(
                "h5py is required to read MATLAB v7.3 files. "
                "Install it with: pip install h5py"
            )
        
        # Dereference HDF5 object/reference arrays (for f16.mat compatibility)
        try:
            import h5py
            with h5py.File(data_path, 'r') as f:
                # Check for #refs# group (MATLAB v7.3 object reference storage)
                if '#refs#' in f:
                    refs_group = f['#refs#']
                    print(f"  Found #refs# group with keys: {list(refs_group.keys())}")
                
                for key in list(data.keys()):
                    val = data[key]
                    # Case 1: dtype=object with h5py.Reference entries
                    if isinstance(val, np.ndarray) and val.dtype == np.object_:
                        try:
                            dereferenced = []
                            for ref in val.flat:
                                if isinstance(ref, h5py.Reference):
                                    dereferenced.append(f[ref][()])
                                else:
                                    dereferenced.append(ref)
                            data[key] = np.array(dereferenced).reshape(val.shape)
                            print(f"    Dereferenced {key}: new shape={data[key].shape}")
                        except Exception as e:
                            print(f"    Could not dereference {key}: {e}")
                    # Case 2: small uint32/uint64 arrays - these are indices into #refs#
                    elif isinstance(val, np.ndarray) and val.dtype in (np.uint32, np.uint64):
                        if val.ndim <= 2 and val.size < 100 and '#refs#' in f:
                            try:
                                # The uint32 values are indices into the #refs# group
                                refs_group = f['#refs#']
                                dereferenced = []
                                for idx in val.flat:
                                    ref_key = f'#{idx}'
                                    if ref_key in refs_group:
                                        dereferenced.append(refs_group[ref_key][()])
                                    else:
                                        dereferenced.append(None)
                                
                                if any(d is not None for d in dereferenced):
                                    # Reshape to match original dimensions
                                    data[key] = np.array(dereferenced).reshape(val.shape)
                                    print(f"    Dereferenced {key} from #refs#: {val.shape} -> {data[key].shape}")
                            except Exception as e:
                                print(f"    Could not dereference {key} from #refs#: {e}")
        except Exception as e:
            print(f"  Dereferencing step failed: {e}")
    
    return data


# ============================================================================
# PyTorch .pt file loader
# ============================================================================

def load_pt_dataset(data_dir: Path, original_data_dir: Path = None):
    """
    Load PyTorch dataset from the .pt format directory.
    
    Supports both reduced and unreduced datasets. If original_data_dir is provided
    and the dataset appears to be reduced, merges with original eigenvalue data.
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
    
    if is_reduced:
        # Load reduced indices
        data['reduced_indices'] = torch.load(data_dir / 'reduced_indices.pt', map_location='cpu')
        reduced_indices_np = np.array(data['reduced_indices'])
        
        print(f"Detected reduced PyTorch dataset.")
        print(f"  Reduced indices shape: {reduced_indices_np.shape}")
        
        if original_data_dir is not None:
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
                            # If first dimension is largest, likely (N_struct, N_eig, N_wv) -> (N_struct, N_wv, N_eig)
                            if eigenvalue_data_orig.shape[0] > eigenvalue_data_orig.shape[2]:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(0, 2, 1)
                            # If last dimension is largest, likely (N_wv, N_eig, N_struct) -> (N_struct, N_wv, N_eig)
                            elif eigenvalue_data_orig.shape[2] > eigenvalue_data_orig.shape[0]:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(2, 0, 1)
                            # If middle dimension is largest, likely (N_wv, N_struct, N_eig) -> (N_struct, N_wv, N_eig)
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
            print(f"  WARNING: Reduced dataset detected but no original_data_dir provided.")
            print(f"  Cannot plot dispersion without eigenvalue data.")
            raise ValueError("For reduced datasets, original_data_dir must be provided")
    else:
        # Unreduced dataset - check if eigenvalue data exists in PyTorch format
        # Note: convert_mat_to_pytorch.py doesn't save eigenvalue data, so we'll need
        # to load from original NumPy format or reconstruct
        print(f"Detected unreduced PyTorch dataset.")
        
        if original_data_dir is not None:
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
                        # MATLAB format: h5py loads as (N_struct, N_eig, N_wv) or (N_wv, N_eig, N_struct)
                        # Need to transpose to (N_struct, N_wv, N_eig)
                        if len(eigenvalue_data_orig.shape) == 3:
                            # If first dimension is largest, likely (N_struct, N_eig, N_wv) -> (N_struct, N_wv, N_eig)
                            if eigenvalue_data_orig.shape[0] > eigenvalue_data_orig.shape[2]:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(0, 2, 1)
                            # If last dimension is largest, likely (N_wv, N_eig, N_struct) -> (N_struct, N_wv, N_eig)
                            elif eigenvalue_data_orig.shape[2] > eigenvalue_data_orig.shape[0]:
                                eigenvalue_data_orig = eigenvalue_data_orig.transpose(2, 0, 1)
                            # If middle dimension is largest, likely (N_wv, N_struct, N_eig) -> (N_struct, N_wv, N_eig)
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
                print(f"  PyTorch format from convert_mat_to_pytorch.py does not include eigenvalue data.")
                print(f"  Please provide original_data_dir pointing to NumPy format dataset or MATLAB .mat file.")
                raise FileNotFoundError("Cannot plot dispersion without eigenvalue data")
        else:
            print(f"  WARNING: Eigenvalue data not found.")
            print(f"  PyTorch format from convert_mat_to_pytorch.py does not include eigenvalue data.")
            print(f"  Please provide original_data_dir pointing to NumPy format dataset or MATLAB .mat file.")
            raise FileNotFoundError("Cannot plot dispersion without eigenvalue data")
    
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



# ============================================================================
# NumPy .npy file loader
# ============================================================================

def load_np_dataset(data_dir: Path, original_data_dir: Path = None):
    """
    Load NumPy dataset from the py format directory.
    
    Supports both reduced and unreduced datasets. If original_data_dir is provided
    and the dataset appears to be reduced, merges with original eigenvalue data.
    """
    data = {}
    
    # Check if this is a reduced dataset (has reduced_indices.npy)
    is_reduced = (data_dir / 'reduced_indices.npy').exists()
    
    if is_reduced and original_data_dir is not None:
        # Load reduced dataset and merge with original eigenvalue data
        print(f"Detected reduced dataset. Loading original eigenvalue data from: {original_data_dir}")
        
        # Load reduced dataset components
        data['designs'] = np.load(data_dir / 'designs.npy')
        data['wavevectors'] = np.load(data_dir / 'wavevectors.npy')
        data['reduced_indices'] = np.load(data_dir / 'reduced_indices.npy')
        
        # Load original eigenvalue data
        if (original_data_dir / 'eigenvalue_data.npy').exists():
            eigenvalue_data_orig = np.load(original_data_dir / 'eigenvalue_data.npy')
            data['eigenvalue_data'] = eigenvalue_data_orig
            print(f"  Merged with original eigenvalue data: {eigenvalue_data_orig.shape}")
        else:
            print(f"  WARNING: Original eigenvalue data not found at {original_data_dir}")
            print(f"  Attempting to use reduced dataset's eigenvalue data if available...")
            if (data_dir / 'eigenvalue_data.npy').exists():
                data['eigenvalue_data'] = np.load(data_dir / 'eigenvalue_data.npy')
            else:
                raise FileNotFoundError("Cannot plot dispersion without eigenvalue data")
    else:
        # Load unreduced dataset (standard format)
        data['designs'] = np.load(data_dir / 'designs.npy')
        data['wavevectors'] = np.load(data_dir / 'wavevectors.npy')
        data['eigenvalue_data'] = np.load(data_dir / 'eigenvalue_data.npy')
    
    # Optional components
    if (data_dir / 'eigenvector_data_x.npy').exists():
        data['eigenvector_data_x'] = np.load(data_dir / 'eigenvector_data_x.npy')
    if (data_dir / 'eigenvector_data_y.npy').exists():
        data['eigenvector_data_y'] = np.load(data_dir / 'eigenvector_data_y.npy')
    if (data_dir / 'waveforms.npy').exists():
        data['waveforms'] = np.load(data_dir / 'waveforms.npy')
    if (data_dir / 'bands_fft.npy').exists():
        data['bands_fft'] = np.load(data_dir / 'bands_fft.npy')
    if (data_dir / 'design_params.npy').exists():
        data['design_params'] = np.load(data_dir / 'design_params.npy')
    
    return data



# ============================================================================
# Frequency reconstruction (MATLAB format only)
# ============================================================================

def compute_K_M_T_from_design(data, struct_idx, design, const):
    """
    Compute K, M, and T matrices from design and const (on-the-fly computation).
    
    This mirrors MATLAB's plot_dispersion.m behavior when K_DATA, M_DATA, T_DATA
    are not present in the dataset.
    
    Parameters
    ----------
    data : dict
        Dataset dictionary
    struct_idx : int
        Structure index (0-based)
    design : ndarray
        Design array (N_pix, N_pix, 3) or (N_pix, N_pix)
    const : dict
        Constants dictionary (from data['const'])
        
    Returns
    -------
    K : scipy.sparse matrix
        Global stiffness matrix
    M : scipy.sparse matrix
        Global mass matrix
    T_data : list
        List of transformation matrices (one per wavevector)
    """
    # Extract parameters from const
    def extract_scalar(val):
        """Extract scalar from various nested structures."""
        if np.isscalar(val):
            return val
        elif isinstance(val, np.ndarray):
            if val.ndim == 0:
                return val.item()
            else:
                return val.flatten()[0]
        else:
            try:
                return val[0, 0][0, 0] if hasattr(val, '__getitem__') else val
            except:
                return val
    
    # Extract const parameters
    if isinstance(const, dict):
        N_pix = int(extract_scalar(const.get('N_pix', 32)))
        N_ele = int(extract_scalar(const.get('N_ele', 1)))
        a = float(extract_scalar(const.get('a', 1.0)))
        E_min = float(extract_scalar(const.get('E_min', 20e6)))
        E_max = float(extract_scalar(const.get('E_max', 200e9)))
        rho_min = float(extract_scalar(const.get('rho_min', 1200)))
        rho_max = float(extract_scalar(const.get('rho_max', 8000)))
        nu_min = float(extract_scalar(const.get('poisson_min', const.get('nu_min', 0.0))))
        nu_max = float(extract_scalar(const.get('poisson_max', const.get('nu_max', 0.5))))
        t = float(extract_scalar(const.get('t', 1.0)))
        design_scale = const.get('design_scale', 'linear')
        isUseImprovement = bool(const.get('isUseImprovement', True))
        isUseSecondImprovement = bool(const.get('isUseSecondImprovement', False))
    else:
        # Handle scipy nested structure
        N_pix = int(extract_scalar(const['N_pix']))
        N_ele = int(extract_scalar(const['N_ele']))
        a = float(extract_scalar(const['a']))
        E_min = float(extract_scalar(const['E_min']))
        E_max = float(extract_scalar(const['E_max']))
        rho_min = float(extract_scalar(const['rho_min']))
        rho_max = float(extract_scalar(const['rho_max']))
        nu_min = float(extract_scalar(const.get('poisson_min', const.get('nu_min', 0.0))))
        nu_max = float(extract_scalar(const.get('poisson_max', const.get('nu_max', 0.5))))
        t = float(extract_scalar(const['t']))
        design_scale = 'linear'  # Default
        isUseImprovement = True
        isUseSecondImprovement = False
    
    # Ensure design is 3-channel
    if design.ndim == 2:
        # Single channel - replicate for all properties
        design_3ch = np.zeros((N_pix, N_pix, 3))
        design_3ch[:, :, 0] = design
        design_3ch[:, :, 1] = design
        design_3ch[:, :, 2] = design
    else:
        design_3ch = design
    
    # Create const dict for matrix computation
    const_for_km = {
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
        'isUseSecondImprovement': isUseSecondImprovement
    }
    
    # Compute K and M matrices
    print(f"    Computing K and M matrices from design...")
    if isUseSecondImprovement and get_system_matrices_VEC_simplified is not None:
        K, M = get_system_matrices_VEC_simplified(const_for_km)
    elif isUseImprovement and get_system_matrices_VEC is not None:
        K, M = get_system_matrices_VEC(const_for_km)
    else:
        raise ValueError("Cannot compute K, M matrices - required functions not available")
    
    # Get wavevectors for this structure
    wavevectors = data['WAVEVECTOR_DATA'][struct_idx, :, :].T  # (N_wv, 2)
    
    # Compute T matrices for each wavevector
    print(f"    Computing T matrices for {len(wavevectors)} wavevectors...")
    T_data = []
    if get_transformation_matrix is None:
        raise ValueError("Cannot compute T matrices - get_transformation_matrix not available")
    
    for wv_idx, wv in enumerate(wavevectors):
        T = get_transformation_matrix(wv.astype(np.float32), const_for_km)
        T_data.append(T)
    
    return K, M, T_data


def reconstruct_frequencies_from_eigenvectors(data, struct_idx):
    """
    Reconstruct frequencies from eigenvectors to verify correctness.
    
    Uses: eigval = ||K*v|| / ||M*v|| where Kr*v = Mr*v*eigval
    Then: frequency = sqrt(eigval) / (2*pi)
    
    Parameters
    ----------
    data : dict
        Dataset dictionary (properly loaded with mat73_loader)
    struct_idx : int
        Structure index (0-based)
        
    Returns
    -------
    frequencies_recon : ndarray
        Reconstructed frequencies
    """
    # Extract parameters - robust handling for both scipy and h5py formats
    def extract_scalar(val):
        """Extract scalar from various nested structures."""
        if np.isscalar(val):
            return int(val)
        elif isinstance(val, np.ndarray):
            if val.ndim == 0:
                return int(val.item())
            else:
                return int(val.flatten()[0])
        else:
            try:
                return int(val[0, 0][0, 0])  # scipy nested
            except:
                return int(val)
    
    # Get dimensions directly from EIGENVECTOR_DATA shape
    # EIGENVECTOR_DATA shape: (N_struct, N_eig, N_wv, N_dof)
    eigvec_shape = data['EIGENVECTOR_DATA'].shape
    N_struct_in_data = eigvec_shape[0]
    N_eig = eigvec_shape[1]
    N_wv = eigvec_shape[2]
    N_dof = eigvec_shape[3]
    
    frequencies_recon = np.zeros((N_wv, N_eig))
    
    # Check if K_DATA, M_DATA, T_DATA are present
    has_K_M_T = ('K_DATA' in data and 'M_DATA' in data and 'T_DATA' in data and
                 data['K_DATA'] is not None and data['M_DATA'] is not None and data['T_DATA'] is not None)
    
    if not has_K_M_T:
        # Compute K, M, T on-the-fly from design (like MATLAB's plot_dispersion.m)
        print(f"  K_DATA, M_DATA, T_DATA not found - computing from design...")
        
        # Get design for this structure
        designs = data['designs']  # (N_struct, 3, N_pix, N_pix) or (N_struct, N_pix, N_pix)
        if designs.ndim == 4:
            # 3-channel design: (N_struct, 3, N_pix, N_pix)
            design = designs[struct_idx, :, :, :].transpose(1, 2, 0)  # (N_pix, N_pix, 3)
        else:
            # Single-channel design: (N_struct, N_pix, N_pix)
            design = designs[struct_idx, :, :]  # (N_pix, N_pix)
        
        # Get const
        const = data.get('const', {})
        
        # Compute K, M, T
        K, M, T_data = compute_K_M_T_from_design(data, struct_idx, design, const)
        
        print(f"  Successfully computed K, M, T matrices")
        print(f"    K: {K.shape}, {K.nnz} non-zeros, dtype={K.dtype}")
        print(f"    M: {M.shape}, {M.nnz} non-zeros, dtype={M.dtype}")
        print(f"    T: {len(T_data)} matrices")
    else:
        # Use existing K_DATA, M_DATA, T_DATA
        # K_DATA and M_DATA have shape (1, N_struct) from HDF5 loading
        K_DATA_array = data['K_DATA']
        M_DATA_array = data['M_DATA']
        
        # Extract the specific structure's matrices
        if K_DATA_array.shape == (1, ):
            # Single structure case
            K = K_DATA_array.flat[0]
            M = M_DATA_array.flat[0]
        elif K_DATA_array.ndim == 2 and K_DATA_array.shape[0] == 1:
            # Shape (1, N_struct) from HDF5
            K = K_DATA_array[0, struct_idx]
            M = M_DATA_array[0, struct_idx]
        else:
            # Direct indexing
            K = K_DATA_array.flat[struct_idx]
            M = M_DATA_array.flat[struct_idx]
        
        # Verify they are sparse matrices
        if not sp.issparse(K):
            raise TypeError(f"K is not a sparse matrix! Type: {type(K)}, Shape: {getattr(K, 'shape', 'N/A')}")
        if not sp.issparse(M):
            raise TypeError(f"M is not a sparse matrix! Type: {type(M)}, Shape: {getattr(M, 'shape', 'N/A')}")
        
        # Handle T_DATA shape (1, N_wv) from HDF5
        T_DATA_array = data['T_DATA']
        T_data = []
        for wv_idx in range(N_wv):
            if T_DATA_array.ndim == 2 and T_DATA_array.shape[0] == 1:
                # Shape (1, N_wv) from HDF5
                T = T_DATA_array[0, wv_idx]
            else:
                T = T_DATA_array.flat[wv_idx]
            
            if not sp.issparse(T):
                raise TypeError(f"T[{wv_idx}] is not a sparse matrix! Type: {type(T)}")
            
            # Fix void dtype if present
            if T.dtype.kind == 'V':
                T = T.astype(np.complex128)
            
            T_data.append(T)
    
    print(f"  Reconstructing frequencies for structure {struct_idx}...")
    print(f"    K: {K.shape}, {K.nnz} non-zeros, dtype={K.dtype}")
    print(f"    M: {M.shape}, {M.nnz} non-zeros, dtype={M.dtype}")
    print(f"    Reconstructing {N_wv} wavevectors × {N_eig} bands")
    
    # Loop over wavevectors
    for wv_idx in range(N_wv):
        # Get T for this wavevector
        T = T_data[wv_idx]
        
        # Debug: print T info on first iteration
        if wv_idx == 0:
            print(f"    T[0]: type={type(T)}, sparse={sp.issparse(T)}")
            if sp.issparse(T):
                print(f"    T[0]: shape={T.shape}, dtype={T.dtype}, nnz={T.nnz}")
        
        # Create reduced matrices
        Kr = T.conj().T @ K @ T
        Mr = T.conj().T @ M @ T
        
        # Loop over eigenvalue bands
        for band_idx in range(N_eig):
            # Correct indexing: [struct_idx, band_idx, wv_idx, :]
            eigvec_raw = data['EIGENVECTOR_DATA'][struct_idx, band_idx, wv_idx, :]
            
            # Handle structured dtype with real and imag fields
            if eigvec_raw.dtype.names and 'real' in eigvec_raw.dtype.names:
                eigvec = eigvec_raw['real'] + 1j * eigvec_raw['imag']
            else:
                eigvec = eigvec_raw
            
            # Debug on first iteration
            if wv_idx == 0 and band_idx == 0:
                print(f"    eigvec[0,0]: shape={eigvec.shape}, dtype={eigvec.dtype}")
                print(f"    Kr shape={Kr.shape}, Mr shape={Mr.shape}")
                print(f"    OK: Dimensions match!" if eigvec.shape[0] == Kr.shape[0] else "ERROR: Dimension mismatch!")
            
            # Compute eigenvalue: eigval = ||Kr*v|| / ||Mr*v||
            Kr_v = Kr @ eigvec
            Mr_v = Mr @ eigvec
            eigval = np.linalg.norm(Kr_v) / np.linalg.norm(Mr_v)
            
            # Convert to frequency
            frequencies_recon[wv_idx, band_idx] = np.sqrt(eigval) / (2 * np.pi)
    
    return frequencies_recon



# ============================================================================
# Shared plotting function: create_grid_interpolators
# ============================================================================

def create_grid_interpolators(wavevectors, frequencies, N_wv):
    """
    Create grid interpolators for each eigenvalue band.
    
    Matches MATLAB behavior:
    - wavevectors_grid = {sort(unique(wavevectors(:,1))), sort(unique(wavevectors(:,2)))}
    - frequencies_grid = reshape(frequencies, [flip(N_wv) N_eig])
    - gridInterp = griddedInterpolant(flip(wavevectors_grid), frequencies_grid(:,:,eig_idx))
    
    Parameters
    ----------
    wavevectors : ndarray
        Wavevector data (N_wv × 2), column 0=x, column 1=y
    frequencies : ndarray
        Frequency data (N_wv × N_eig)
    N_wv : array_like
        Number of wavevectors in each direction [N_x, N_y]
        
    Returns
    -------
    grid_interp : list
        List of interpolators for each band
    wavevectors_grid : tuple
        Tuple of (x_grid, y_grid) coordinate arrays
    """
    # Get unique wavevector coordinates (matching MATLAB's sort(unique(...)))
    x_unique = np.sort(np.unique(wavevectors[:, 0]))
    y_unique = np.sort(np.unique(wavevectors[:, 1]))
    
    # Ensure N_wv elements are integers
    N_x, N_y = int(N_wv[0]), int(N_wv[1])
    
    # Verify we have the right number of points
    if len(x_unique) != N_x or len(y_unique) != N_y:
        print(f"    WARNING: Grid size mismatch!")
        print(f"      Expected: {N_x} × {N_y} = {N_x*N_y} points")
        print(f"      Got: {len(x_unique)} × {len(y_unique)} = {len(x_unique)*len(y_unique)} points")
        print(f"      Total wavevectors: {len(wavevectors)}")
        print(f"      Using detected grid size: {len(x_unique)} × {len(y_unique)}")
        N_x, N_y = len(x_unique), len(y_unique)
    
    # Verify total points match
    expected_total = N_x * N_y
    actual_total = len(wavevectors)
    if expected_total != actual_total:
        raise ValueError(f"Grid size mismatch: {N_x}×{N_y}={expected_total} but have {actual_total} wavevectors")
    
    # MATLAB: reshape(frequencies, [flip(N_wv) N_eig]) = reshape to [N_y, N_x, N_eig]
    # MATLAB uses column-major (Fortran) order for reshape
    # Python default is row-major (C) order, so we must specify order='F'
    # This ensures wavevectors at (i, j) grid position map to correct frequency
    frequencies_grid = frequencies.reshape((N_y, N_x, frequencies.shape[1]), order='F')
    
    # Debug: verify reshape makes sense by checking corner points
    print(f"    Grid: {N_x} × {N_y} = {N_x*N_y} points")
    print(f"    Wavevector range: x in [{x_unique[0]:.4f}, {x_unique[-1]:.4f}], y in [{y_unique[0]:.4f}, {y_unique[-1]:.4f}]")
    
    # Verify wavevectors actually form a rectangular grid (every x with every y)
    # Create expected grid
    expected_grid = np.array([[x, y] for y in y_unique for x in x_unique])  # Column-major order
    
    # Check if wavevectors match expected grid
    # They might not be in the same order, so sort both
    wv_sorted = wavevectors[np.lexsort((wavevectors[:, 0], wavevectors[:, 1]))]
    expected_sorted = expected_grid[np.lexsort((expected_grid[:, 0], expected_grid[:, 1]))]
    
    if not np.allclose(wv_sorted, expected_sorted, atol=1e-10):
        print(f"    WARNING: Wavevectors don't form a perfect rectangular grid!")
        print(f"    This may cause interpolation issues.")
    else:
        print(f"    OK: Wavevectors form a valid rectangular grid")
    
    # MATLAB uses flip(wavevectors_grid) for griddedInterpolant
    # flip({x_unique, y_unique}) = {y_unique, x_unique}
    # So MATLAB interpolant expects (y, x) order
    
    # Create interpolators for each band
    grid_interp = []
    for eig_idx in range(frequencies.shape[1]):
        # MATLAB: griddedInterpolant({y_grid, x_grid}, data(y, x))
        # Python RegularGridInterpolator: points=(y_grid, x_grid), values=data[y, x]
        interp = RegularGridInterpolator(
            (y_unique, x_unique),  # Flipped to match MATLAB's flip(wavevectors_grid)
            frequencies_grid[:, :, eig_idx],  # (N_y, N_x) - no transpose needed
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        grid_interp.append(interp)
    
    wavevectors_grid = (x_unique, y_unique)  # Return in original order for reference
    
    return grid_interp, wavevectors_grid



# ============================================================================
# Shared plotting function: extract_grid_points_on_contour
# ============================================================================

def extract_grid_points_on_contour(wavevectors, frequencies, contour_info, a, tolerance=1e-10):
    """
    Extract grid points that lie on the IBZ contour (no interpolation).
    
    This finds actual computed wavevectors that lie on the high-symmetry path
    and returns them in order along the contour.
    
    Parameters
    ----------
    wavevectors : ndarray
        All computed wavevector grid points (N_wv × 2)
    frequencies : ndarray
        Frequencies at grid points (N_wv × N_eig)
    contour_info : dict
        Contour information from get_IBZ_contour_wavevectors
    a : float
        Lattice parameter
    tolerance : float
        Tolerance for point matching (default: 1e-10)
        
    Returns
    -------
    contour_points : ndarray
        Wavevectors on contour (N_contour × 2)
    contour_freqs : ndarray
        Frequencies on contour (N_contour × N_eig)
    contour_param : ndarray
        Parameter along contour for x-axis
    """
    vertices = contour_info['vertices']
    contour_points_list = []
    contour_freqs_list = []
    contour_param_list = []
    
    total_distance = 0.0
    
    # Process each segment
    for seg_idx in range(len(vertices) - 1):
        v_start = vertices[seg_idx]
        v_end = vertices[seg_idx + 1]
        
        # Direction vector
        direction = v_end - v_start
        segment_length = np.linalg.norm(direction)
        
        if segment_length < tolerance:
            continue  # Skip zero-length segments
        
        direction_unit = direction / segment_length
        
        # Find grid points on this segment
        # A point is on the segment if it lies on the line and within bounds
        segment_points = []
        segment_freqs = []
        segment_distances = []
        
        for i, wv in enumerate(wavevectors):
            # Vector from start to this point
            to_point = wv - v_start
            
            # Project onto direction
            projection = np.dot(to_point, direction_unit)
            
            # Check if projection is within segment bounds
            if -tolerance <= projection <= segment_length + tolerance:
                # Check if point is actually on the line
                perpendicular = to_point - projection * direction_unit
                if np.linalg.norm(perpendicular) < tolerance:
                    segment_points.append(wv)
                    segment_freqs.append(frequencies[i])
                    segment_distances.append(projection)
        
        if len(segment_points) > 0:
            # Sort by distance along segment
            sort_idx = np.argsort(segment_distances)
            segment_points = np.array(segment_points)[sort_idx]
            segment_freqs = np.array(segment_freqs)[sort_idx]
            segment_distances = np.array(segment_distances)[sort_idx]
            
            # Skip first point if not first segment (avoid duplicates at vertices)
            start_idx = 1 if seg_idx > 0 else 0
            
            contour_points_list.append(segment_points[start_idx:])
            contour_freqs_list.append(segment_freqs[start_idx:])
            # Add segment offset to distances
            contour_param_list.append(total_distance + segment_distances[start_idx:])
        
        total_distance += segment_length
    
    # Concatenate all segments
    contour_points = np.vstack(contour_points_list)
    contour_freqs = np.vstack(contour_freqs_list)
    contour_param = np.concatenate(contour_param_list)
    
    # Normalize parameter to match contour_info (0 to N_segment)
    contour_param = contour_param / total_distance * contour_info['N_segment']
    
    return contour_points, contour_freqs, contour_param



# ============================================================================
# Shared plotting function: plot_dispersion_on_contour
# ============================================================================

def plot_dispersion_on_contour(ax, contour_info, frequencies_contour, contour_param=None, title='Dispersion', mark_points=False):
    """
    Plot dispersion relation on IBZ contour.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    contour_info : dict
        Contour information dictionary
    frequencies_contour : ndarray
        Frequencies evaluated on contour
    title : str, optional
        Plot title
    mark_points : bool, optional
        Whether to add markers to the points
    """
    # Choose x-axis parameter
    if contour_param is None:
        x_param = contour_info['wavevector_parameter']
    else:
        x_param = contour_param

    # Plot frequency bands
    for band_idx in range(frequencies_contour.shape[1]):
        ax.plot(x_param, 
               frequencies_contour[:, band_idx],
               linewidth=2)
        
        # Mark points if requested
        if mark_points:
            ax.plot(x_param, 
                   frequencies_contour[:, band_idx],
                   'o', markersize=4, markeredgewidth=0.5, markeredgecolor='white')
    
    # Add vertical lines at segment boundaries
    for i in range(contour_info['N_segment'] + 1):
        ax.axvline(i, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add vertex labels
    if 'vertex_labels' in contour_info and contour_info['vertex_labels']:
        # Get positions of vertices along parameter
        vertex_positions = np.linspace(0, contour_info['N_segment'], 
                                      len(contour_info['vertex_labels']))
        ax.set_xticks(vertex_positions)
        ax.set_xticklabels(contour_info['vertex_labels'])
    
    ax.set_xlabel('Wavevector Contour Parameter', fontsize=12)
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)



# ============================================================================
# Helper: Infer wavevector grid size
# ============================================================================

def get_N_wv_from_data(designs_shape, wavevectors_shape):
    """
    Infer N_wv grid size from data.
    
    For a rectangular grid, N_wv = N_x * N_y where N_x and N_y are determined
    from the wavevector data by finding unique x and y coordinates.
    """
    # Take first sample's wavevectors
    wv_sample = wavevectors_shape[1]  # N_wv total points
    # Try to infer N_x and N_y by assuming a rectangular grid
    # Common values: 25x13, 25x25, etc.
    n_wv = wavevectors_shape[1]
    
    # Try to find factors that make sense
    factors = []
    for i in range(1, int(np.sqrt(n_wv)) + 1):
        if n_wv % i == 0:
            factors.append((i, n_wv // i))
    
    # Choose reasonable aspect ratio
    if factors:
        # Prefer factor pairs that are closer to each other (less extreme ratios)
        # This helps avoid very wide or tall grids
        best_factors = factors[0]
        min_ratio_diff = abs(np.log(factors[0][0] / factors[0][1]))
        for fx, fy in factors:
            ratio_diff = abs(np.log(fx / fy))
            if ratio_diff < min_ratio_diff:
                min_ratio_diff = ratio_diff
                best_factors = (fx, fy)
        N_x, N_y = best_factors
        return [N_x, N_y]
    
    # Fallback: assume square grid
    n_per_dim = int(np.sqrt(n_wv))
    return [n_per_dim, n_per_dim]



# ============================================================================
# Format detection and unified loader
# ============================================================================

def detect_format_and_load(data_path, original_data_dir=None, verbose=False):
    """
    Detect data format and load dataset.
    
    Parameters
    ----------
    data_path : str or Path
        Path to dataset file or directory
    original_data_dir : str or Path, optional
        Path to original dataset (for PyTorch/NumPy reduced datasets)
    verbose : bool, optional
        Verbose loading output
        
    Returns
    -------
    data : dict
        Loaded dataset in unified format
    format_type : str
        Detected format type: 'mat', 'pt', or 'np'
    """
    data_path = Path(data_path)
    
    if data_path.is_file() and data_path.suffix == '.mat':
        # MATLAB .mat file
        print(f"Detected MATLAB .mat format")
        return load_dataset(data_path, verbose=verbose), 'mat'
    elif data_path.is_dir():
        # Check for PyTorch format (has geometries_full.pt)
        if (data_path / 'geometries_full.pt').exists():
            print(f"Detected PyTorch format")
            if torch is None:
                raise ImportError("PyTorch is required for .pt format. Install with: pip install torch")
            return load_pt_dataset(data_path, Path(original_data_dir) if original_data_dir else None), 'pt'
        # Check for NumPy format (has designs.npy)
        elif (data_path / 'designs.npy').exists():
            print(f"Detected NumPy format")
            return load_np_dataset(data_path, Path(original_data_dir) if original_data_dir else None), 'np'
        else:
            raise ValueError(f"Could not detect format for directory: {data_path}")
    else:
        raise ValueError(f"Unknown format: {data_path}")


# ============================================================================
# Unified main function (combines logic from all three scripts)
# ============================================================================
def main(cli_data_path=None, cli_original_dir=None, n_structs=None, save_plot_points=False):
    """
    Main script execution (unified for all formats).
    
    Parameters
    ----------
    cli_data_path : str, optional
        Path to dataset file or directory
    cli_original_dir : str, optional
        Path to original dataset (for PyTorch/NumPy reduced datasets)
    n_structs : int, optional
        Number of structures to plot
    save_plot_points : bool, optional
        If True, save plot point locations (wavevectors_contour, frequencies_contour, contour_param) to .npz file (default: False)
    """
    print("=" * 70)
    print("Plot Dispersion Script with Eigenfrequencies (Consolidated)")
    print("=" * 70)
    
    # Configuration
    if cli_data_path is not None:
        data_path = Path(cli_data_path)
    else:
        print("\nERROR: Please provide a path to the dataset.")
        print("Usage: python plot_dispersion_with_eigenfrequencies.py <data_path> [original_dir] [-n N_STRUCTS] [--save-plot-points]")
        print("\nSupported formats:")
        print("  - MATLAB .mat files")
        print("  - PyTorch dataset directories (with geometries_full.pt)")
        print("  - NumPy dataset directories (with designs.npy)")
        return
    
    if not data_path.exists():
        print(f"\nERROR: Dataset path not found: {data_path}")
        return
    
    # Flags
    isExportPng = True
    png_resolution = 150
    verbose_loading = False
    mark_points = True
    use_interpolation = True  # True: interpolate (smooth), False: grid points only (exact) - Set to True to match MATLAB
    N_k_interp = 10  # Match MATLAB's N_k = 10
    
    # Load data with format detection
    print(f"\nLoading dataset: {data_path}")
    try:
        data, format_type = detect_format_and_load(data_path, cli_original_dir, verbose_loading)
    except Exception as e:
        print(f"\nERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print dataset structure
    print(f"\n{'='*70}")
    print("Dataset Structure:")
    print(f"{'='*70}")
    for key in ['designs', 'WAVEVECTOR_DATA', 'EIGENVALUE_DATA', 'eigenvalue_data', 
                'wavevectors', 'WAVEVECTOR_DATA', 'EIGENVECTOR_DATA', 'K_DATA', 'M_DATA', 'T_DATA']:
        if key in data:
            item = data[key]
            if hasattr(item, 'shape'):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            elif hasattr(item, '__len__'):
                print(f"  {key}: length={len(item)}, type={type(item)}")
            else:
                print(f"  {key}: type={type(item)}")
    print(f"{'='*70}\n")
    
    # Create output directory
    import os
    from datetime import datetime
    current_dir = Path(os.getcwd())
    if format_type == 'mat':
        dataset_name = data_path.stem
        output_dir = current_dir / f'dispersion_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{format_type}'
    else:
        dataset_name = data_path.name
        output_dir = current_dir / 'plots' / f'{dataset_name}_{format_type}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Normalize data to unified format
    if format_type == 'mat':
        # MATLAB format: designs (N_struct, 3, N_pix, N_pix), WAVEVECTOR_DATA (N_struct, 2, N_wv), EIGENVALUE_DATA (N_struct, N_eig, N_wv)
        designs = data['designs']  # (N_struct, 3, N_pix, N_pix)
        wavevectors_all = data['WAVEVECTOR_DATA']  # (N_struct, 2, N_wv)
        eigenvalues_all = data['EIGENVALUE_DATA']  # (N_struct, N_eig, N_wv)
        design_is_3channel = True
        # Extract N_wv from const
        const = data.get('const', {})
        if isinstance(const, dict):
            N_wv_param = const.get('N_wv', [25, 25])
        else:
            try:
                N_wv_param = const['N_wv'][0, 0][0] if hasattr(const['N_wv'], '__getitem__') else const['N_wv']
            except:
                N_wv_param = [25, 25]
        if np.isscalar(N_wv_param):
            N_wv_param = [int(N_wv_param), int(N_wv_param)]
        elif hasattr(N_wv_param, 'flatten'):
            N_wv_param = [int(x) for x in N_wv_param.flatten()]
            if len(N_wv_param) == 1:
                N_wv_param = [N_wv_param[0], N_wv_param[0]]
        # Extract 'a' parameter
        if isinstance(const, dict):
            a_val = const.get('a', 1.0)
        else:
            try:
                a_val = const['a'][0, 0][0, 0] if hasattr(const['a'], '__getitem__') else const['a']
            except:
                a_val = 1.0
        a = float(a_val.flatten()[0]) if hasattr(a_val, 'flatten') else float(a_val)
        # Extract symmetry_type (matching MATLAB's logic)
        symmetry_type = 'p4mm'  # Default
        if isinstance(const, dict):
            sym_type_raw = const.get('symmetry_type', None)
        else:
            try:
                sym_type_raw = const['symmetry_type']
            except (KeyError, TypeError):
                sym_type_raw = None
        
        if sym_type_raw is not None:
            # Handle different formats (string, uint16 array, etc.)
            if isinstance(sym_type_raw, str):
                symmetry_type = sym_type_raw.strip()
            elif isinstance(sym_type_raw, np.ndarray):
                if sym_type_raw.dtype.kind == 'U' or sym_type_raw.dtype.kind == 'S':
                    # String array
                    symmetry_type = ''.join(sym_type_raw.flatten()).strip()
                elif sym_type_raw.dtype == np.uint16:
                    # Character codes (MATLAB format)
                    symmetry_type = ''.join([chr(int(x)) for x in sym_type_raw.flatten()]).strip()
                else:
                    # Try to convert
                    try:
                        symmetry_type = str(sym_type_raw).strip()
                    except:
                        pass
        
        print(f"  Extracted symmetry_type from dataset: '{symmetry_type}'")
        
        # has_reconstruction: True if we can reconstruct frequencies (either from existing K/M/T or by computing them)
        has_reconstruction = 'EIGENVECTOR_DATA' in data and (
            ('K_DATA' in data and 'M_DATA' in data and 'T_DATA' in data) or
            (get_system_matrices_VEC is not None and get_transformation_matrix is not None)
        )
    else:
        # PyTorch/NumPy format: designs (N_struct, N_pix, N_pix), wavevectors (N_struct, N_wv, 2), eigenvalue_data (N_struct, N_wv, N_eig)
        designs = data['designs']  # (N_struct, N_pix, N_pix)
        wavevectors_all = data['wavevectors']  # (N_struct, N_wv, 2)
        eigenvalues_all = data['eigenvalue_data']  # (N_struct, N_wv, N_eig)
        design_is_3channel = False
        N_wv_param = get_N_wv_from_data(designs.shape, wavevectors_all.shape)
        a = 1.0  # Default
        has_reconstruction = False
    
    # Make plots for specified number of structures
    if n_structs is None:
        n_structs_to_plot = min(5, designs.shape[0])
    else:
        n_structs_to_plot = min(int(n_structs), designs.shape[0])
    struct_idxs = range(n_structs_to_plot)
    
    print(f"\nPlotting {len(struct_idxs)} structures...")
    if save_plot_points:
        print("  Plot point locations will be saved to .npz files")
    
    # Store plot points for saving
    plot_points_data = {} if save_plot_points else None
    
    for struct_idx in struct_idxs:
        print(f"\n{'='*70}")
        print(f"Processing structure {struct_idx + 1}/{len(struct_idxs)}")
        print(f"{'='*70}")
        
        # Plot the design
        print("  Plotting design...")
        if design_is_3channel:
            # MATLAB format: (N_struct, 3, N_pix, N_pix) -> (N_pix, N_pix, 3)
            design = designs[struct_idx, :, :, :].transpose(1, 2, 0)
        else:
            # PyTorch/NumPy format: (N_struct, N_pix, N_pix) -> need to reconstruct 3-channel
            design_param = designs[struct_idx, :, :]  # (N_pix, N_pix)
            # Material parameter ranges
            E_min, E_max = 20e6, 200e9
            rho_min, rho_max = 400, 8000
            nu_min, nu_max = 0.05, 0.3
            # Compute actual material properties
            elastic_modulus = E_min + (E_max - E_min) * design_param
            density = rho_min + (rho_max - rho_min) * design_param
            poisson_ratio = nu_min + (nu_max - nu_min) * design_param
            design = np.stack([elastic_modulus, density, poisson_ratio], axis=-1)  # (N_pix, N_pix, 3)
        
        fig_design, _ = plot_design(design)
        if isExportPng:
            png_path = output_dir / 'design' / f'{struct_idx}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_design.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
            print(f"    Saved: {png_path}")
            plt.close(fig_design)
        
        # Extract wavevector and frequency data
        if format_type == 'mat':
            wavevectors = wavevectors_all[struct_idx, :, :].T  # (N_wv, 2)
            frequencies = eigenvalues_all[struct_idx, :, :].T  # (N_wv, N_eig)
        else:
            wavevectors = wavevectors_all[struct_idx, :, :]  # (N_wv, 2)
            frequencies = eigenvalues_all[struct_idx, :, :]  # (N_wv, N_eig)
        
        print(f"  Wavevectors shape: {wavevectors.shape}")
        print(f"  Frequencies shape: {frequencies.shape}")
        
        # Reconstruct frequencies from eigenvectors (MATLAB format only)
        # MATLAB's plot_dispersion.m reconstructs frequencies if EIGENVALUE_DATA is missing/invalid
        frequencies_recon = None
        if format_type == 'mat' and has_reconstruction:
            # Check if EIGENVALUE_DATA is valid (not all NaN)
            # MATLAB checks if EIGENVALUE_DATA is missing or empty, then reconstructs
            # Since the file has all NaN, we should reconstruct
            nan_count = np.isnan(frequencies).sum()
            total_count = frequencies.size
            print(f"  EIGENVALUE_DATA NaN check: {nan_count}/{total_count} NaN values")
            
            if nan_count == total_count or nan_count > total_count * 0.9:  # All or mostly NaN
                print("  WARNING: EIGENVALUE_DATA is all/mostly NaN, reconstructing from eigenvectors...")
                try:
                    frequencies = reconstruct_frequencies_from_eigenvectors(data, struct_idx)
                    print(f"  Successfully reconstructed frequencies from eigenvectors")
                    print(f"  Reconstructed frequencies shape: {frequencies.shape}")
                    print(f"  Reconstructed frequencies range: [{np.nanmin(frequencies):.6e}, {np.nanmax(frequencies):.6e}]")
                except Exception as e:
                    print(f"  ERROR: Reconstruction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                print("  Reconstructing frequencies from eigenvectors for verification...")
                try:
                    frequencies_recon = reconstruct_frequencies_from_eigenvectors(data, struct_idx)
                    error = np.max(np.abs(frequencies_recon - frequencies)) / np.max(np.abs(frequencies))
                    print(f"  Reconstruction error: {error:.6e}")
                except Exception as e:
                    print(f"  WARNING: Reconstruction failed: {e}")
                    frequencies_recon = None
        elif format_type == 'mat' and not has_reconstruction:
            # Check if EIGENVALUE_DATA is all NaN - if so, we need to reconstruct
            nan_count = np.isnan(frequencies).sum()
            total_count = frequencies.size
            if nan_count == total_count or nan_count > total_count * 0.9:
                print("  ERROR: EIGENVALUE_DATA is all/mostly NaN and cannot reconstruct - missing EIGENVECTOR_DATA or required functions")
            else:
                print("  WARNING: Cannot reconstruct frequencies - missing K_DATA, M_DATA, T_DATA, or EIGENVECTOR_DATA")
        
        # Create grid interpolators
        use_scattered = False  # Initialize for all formats
        grid_interp = None
        try:
            if format_type == 'mat':
                grid_interp, wavevectors_grid = create_grid_interpolators(wavevectors, frequencies, N_wv_param)
            else:
                grid_interp, wavevectors_grid = create_grid_interpolators(wavevectors, frequencies, None)
                use_scattered = (grid_interp is None)
        except (ValueError, AssertionError) as e:
            # Grid interpolation failed - use scattered interpolation instead
            print(f"  WARNING: Grid interpolation failed: {e}")
            print(f"  Falling back to scattered interpolation (like MATLAB)")
            use_scattered = True
            grid_interp = None
        
        if frequencies_recon is not None and format_type == 'mat':
            try:
                grid_interp_recon, _ = create_grid_interpolators(wavevectors, frequencies_recon, N_wv_param)
            except (ValueError, AssertionError):
                # Grid interpolation failed for reconstructed frequencies - will use scattered interpolation
                grid_interp_recon = None
        
        # Get IBZ contour (use symmetry_type from dataset, matching MATLAB)
        if use_interpolation:
            print(f"  Using interpolation mode...")
            print(f"    Symmetry type: {symmetry_type}")
            try:
                wavevectors_contour, contour_info = get_IBZ_contour_wavevectors(N_k_interp, a, symmetry_type)
                print(f"    Generated contour path with {len(wavevectors_contour)} interpolation points")
            except Exception as e:
                print(f"  WARNING: Could not generate contour: {e}")
                continue
        else:
            print(f"  Using grid-points-only mode...")
            print(f"    Symmetry type: {symmetry_type}")
            try:
                _, contour_info = get_IBZ_contour_wavevectors(10, a, symmetry_type)
                wavevectors_contour, frequencies_contour_grid, contour_param_grid = \
                    extract_grid_points_on_contour(wavevectors, frequencies, contour_info, a, tolerance=2e-3)
                print(f"    Found {len(wavevectors_contour)} grid points on contour path")
            except Exception as e:
                print(f"  WARNING: Could not extract grid points: {e}")
                continue
        
        # Plot IBZ contour wavevectors (only for first structure)
        if struct_idx == struct_idxs[0]:
            fig_contour = plt.figure(figsize=(8, 8))
            ax_contour = fig_contour.add_subplot(111)
            ax_contour.plot(wavevectors_contour[:, 0], wavevectors_contour[:, 1], 'k.', markersize=2)
            ax_contour.set_aspect('equal')
            ax_contour.set_xlabel('Wavevector x component [1/m]', fontsize=12)
            ax_contour.set_ylabel('Wavevector y component [1/m]', fontsize=12)
            ax_contour.set_title('IBZ Contour Wavevectors', fontsize=14)
            ax_contour.grid(True, alpha=0.3)
            if isExportPng:
                png_path = output_dir / 'contour' / f'{struct_idx}.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig_contour.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                print(f"    Saved: {png_path}")
                plt.close(fig_contour)
        
        # Evaluate frequencies on contour
        if use_scattered or (format_type == 'mat' and grid_interp is None):
            # Use scattered interpolation for non-grid wavevectors (PyTorch/NumPy)
            print("  Using scattered interpolation...")
            if not use_interpolation:
                frequencies_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
                for eig_idx in range(frequencies.shape[1]):
                    # Use LinearNDInterpolator with extrapolation (like MATLAB)
                    interp = LinearNDInterpolator(wavevectors, frequencies[:, eig_idx], fill_value=np.nan)
                    frequencies_contour[:, eig_idx] = interp(wavevectors_contour)
                    
                    # Handle extrapolation like MATLAB's extrap_method='linear'
                    nan_mask = np.isnan(frequencies_contour[:, eig_idx])
                    if np.any(nan_mask):
                        from scipy.spatial.distance import cdist
                        nan_indices = np.where(nan_mask)[0]
                        if len(nan_indices) > 0:
                            distances = cdist(wavevectors_contour[nan_indices], wavevectors)
                            nearest_idx = np.argmin(distances, axis=1)
                            frequencies_contour[nan_indices, eig_idx] = frequencies[nearest_idx, eig_idx]
                
                # Use grid points that were found
                contour_param = contour_param_grid if 'contour_param_grid' in locals() else contour_info['wavevector_parameter'][:len(wavevectors_contour)]
            else:
                frequencies_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
                for eig_idx in range(frequencies.shape[1]):
                    # Use LinearNDInterpolator with extrapolation (like MATLAB)
                    interp = LinearNDInterpolator(wavevectors, frequencies[:, eig_idx], fill_value=np.nan)
                    frequencies_contour[:, eig_idx] = interp(wavevectors_contour)
                    
                    # Handle extrapolation like MATLAB's extrap_method='linear'
                    # MATLAB's 'linear' extrapolation uses nearest neighbor for points outside convex hull
                    nan_mask = np.isnan(frequencies_contour[:, eig_idx])
                    if np.any(nan_mask):
                        from scipy.spatial.distance import cdist
                        nan_indices = np.where(nan_mask)[0]
                        if len(nan_indices) > 0:
                            # Find nearest neighbors for NaN points (matches MATLAB's extrapolation)
                            distances = cdist(wavevectors_contour[nan_indices], wavevectors)
                            nearest_idx = np.argmin(distances, axis=1)
                            frequencies_contour[nan_indices, eig_idx] = frequencies[nearest_idx, eig_idx]
                            print(f"      Extrapolated {len(nan_indices)} points using nearest neighbor for band {eig_idx+1}")
                
                contour_param = contour_info['wavevector_parameter']
        elif use_interpolation:
            print("  Interpolating frequencies to contour points...")
            if grid_interp is not None:
                # Use grid interpolation if available
                frequencies_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
                for eig_idx in range(frequencies.shape[1]):
                    points_yx = wavevectors_contour[:, [1, 0]]
                    frequencies_contour[:, eig_idx] = grid_interp[eig_idx](points_yx)
            else:
                # Fall back to scattered interpolation (like MATLAB's scatteredInterpolant)
                # MATLAB uses scatteredInterpolant with extrap_method='linear' which extrapolates
                # Python's griddata with method='linear' returns NaN outside convex hull
                # Use LinearNDInterpolator which can handle extrapolation like MATLAB
                print("    Using scattered interpolation (like MATLAB's scatteredInterpolant)...")
                frequencies_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
                for eig_idx in range(frequencies.shape[1]):
                    # Create interpolator (matches MATLAB's scatteredInterpolant)
                    interp = LinearNDInterpolator(wavevectors, frequencies[:, eig_idx], fill_value=np.nan)
                    # Evaluate on contour points
                    frequencies_contour[:, eig_idx] = interp(wavevectors_contour)
                    
                    # Handle extrapolation like MATLAB's extrap_method='linear'
                    # MATLAB extrapolates using nearest neighbor method when outside convex hull
                    nan_mask = np.isnan(frequencies_contour[:, eig_idx])
                    if np.any(nan_mask):
                        # For points outside convex hull, use nearest neighbor extrapolation
                        # Find nearest point in input data for each NaN point
                        from scipy.spatial.distance import cdist
                        nan_indices = np.where(nan_mask)[0]
                        if len(nan_indices) > 0:
                            # Find nearest neighbors for NaN points
                            distances = cdist(wavevectors_contour[nan_indices], wavevectors)
                            nearest_idx = np.argmin(distances, axis=1)
                            frequencies_contour[nan_indices, eig_idx] = frequencies[nearest_idx, eig_idx]
            contour_param = contour_info['wavevector_parameter']
        else:
            print("  Using exact grid point frequencies...")
            frequencies_contour = frequencies_contour_grid
            contour_param = contour_param_grid
        
        # Save plot points if requested
        if save_plot_points:
            plot_points_data[f'struct_{struct_idx}_wavevectors_contour'] = wavevectors_contour
            plot_points_data[f'struct_{struct_idx}_frequencies_contour'] = frequencies_contour
            plot_points_data[f'struct_{struct_idx}_contour_param'] = contour_param
            plot_points_data[f'struct_{struct_idx}_use_interpolation'] = np.array([use_interpolation])
        
        # Plot dispersion relation
        mode_str = "Interpolated" if use_interpolation else "Grid Points"
        print(f"  Plotting dispersion ({mode_str})...")
        fig_disp = plt.figure(figsize=(10, 6))
        ax_disp = fig_disp.add_subplot(111)
        plot_dispersion_on_contour(ax_disp, contour_info, frequencies_contour, contour_param,
                                   title=f'Dispersion Relation ({mode_str})', mark_points=mark_points)
        if isExportPng:
            png_path = output_dir / 'dispersion' / f'{struct_idx}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_disp.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
            print(f"    Saved: {png_path}")
            plt.close(fig_disp)
        
        # Plot reconstructed dispersion (MATLAB format only)
        if frequencies_recon is not None and format_type == 'mat':
            print("  Plotting reconstructed dispersion...")
            if use_interpolation:
                frequencies_recon_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
                if grid_interp_recon is not None:
                    # Use grid interpolation if available
                    for eig_idx in range(frequencies.shape[1]):
                        points_yx = wavevectors_contour[:, [1, 0]]
                        frequencies_recon_contour[:, eig_idx] = grid_interp_recon[eig_idx](points_yx)
                else:
                    # Use scattered interpolation (like MATLAB's scatteredInterpolant)
                    for eig_idx in range(frequencies.shape[1]):
                        interp = LinearNDInterpolator(wavevectors, frequencies_recon[:, eig_idx], fill_value=np.nan)
                        frequencies_recon_contour[:, eig_idx] = interp(wavevectors_contour)
                        # Handle extrapolation
                        nan_mask = np.isnan(frequencies_recon_contour[:, eig_idx])
                        if np.any(nan_mask):
                            from scipy.spatial.distance import cdist
                            nan_indices = np.where(nan_mask)[0]
                            if len(nan_indices) > 0:
                                distances = cdist(wavevectors_contour[nan_indices], wavevectors)
                                nearest_idx = np.argmin(distances, axis=1)
                                frequencies_recon_contour[nan_indices, eig_idx] = frequencies_recon[nearest_idx, eig_idx]
                recon_param = contour_info['wavevector_parameter']
            else:
                _, frequencies_recon_contour, recon_param = extract_grid_points_on_contour(
                    wavevectors, frequencies_recon, contour_info, a)
            
            fig_recon = plt.figure(figsize=(10, 6))
            ax_recon = fig_recon.add_subplot(111)
            plot_dispersion_on_contour(ax_recon, contour_info, frequencies_recon_contour, recon_param,
                                      title='Dispersion Relation (Reconstructed)', mark_points=mark_points)
            if isExportPng:
                png_path = output_dir / 'dispersion' / f'{struct_idx}_recon.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig_recon.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                print(f"    Saved: {png_path}")
                plt.close(fig_recon)
            
            # Plot difference
            print("  Plotting difference...")
            difference = frequencies_contour - frequencies_recon_contour
            max_abs_diff = np.max(np.abs(difference))
            max_freq = np.max(np.abs(frequencies_contour))
            rel_error_pct = 100 * max_abs_diff / max_freq if max_freq > 0 else 0
            
            fig_diff = plt.figure(figsize=(10, 6))
            ax_diff = fig_diff.add_subplot(111)
            x_axis_param = contour_info['wavevector_parameter'] if use_interpolation else contour_param
            if len(x_axis_param) != difference.shape[0]:
                min_len = min(len(x_axis_param), difference.shape[0])
                x_axis_param = x_axis_param[:min_len]
                difference = difference[:min_len, :]
            
            for band_idx in range(difference.shape[1]):
                ax_diff.plot(x_axis_param, difference[:, band_idx], linewidth=2)
            
            for i in range(contour_info['N_segment'] + 1):
                ax_diff.axvline(i, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax_diff.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Zero')
            ax_diff.set_xlabel('Wavevector Contour Parameter', fontsize=12)
            ax_diff.set_ylabel('Frequency Difference [Hz]', fontsize=12)
            ax_diff.set_title(f'Difference (Original - Reconstructed)\nMax abs diff = {max_abs_diff:.3e} Hz ({rel_error_pct:.3f}%)', fontsize=14)
            ax_diff.grid(True, alpha=0.3)
            ax_diff.legend()
            
            if isExportPng:
                png_path = output_dir / 'dispersion' / f'{struct_idx}_diff.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig_diff.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                print(f"    Saved: {png_path}")
                plt.close(fig_diff)
    
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
    parser = argparse.ArgumentParser(description="Plot dispersion from dataset (supports MATLAB .mat, PyTorch .pt, NumPy .npy)")
    parser.add_argument("data_path", nargs="?", help="Path to dataset file (.mat) or directory (.pt or .npy)")
    parser.add_argument("original_dir", nargs="?", help="Path to original dataset (for PyTorch/NumPy reduced datasets)")
    parser.add_argument("-n", "--n-structs", type=int, default=None, help="Number of structures to plot (default: 5)")
    parser.add_argument("--save-plot-points", action="store_true", help="Save plot point locations (wavevectors_contour, frequencies_contour, contour_param) to .npz file")
    args = parser.parse_args()
    main(args.data_path, args.original_dir, args.n_structs, save_plot_points=args.save_plot_points)

