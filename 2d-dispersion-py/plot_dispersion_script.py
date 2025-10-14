"""
Plot Dispersion Script

This script loads a dataset and creates comprehensive dispersion visualizations including:
- Unit cell designs
- Dispersion relations along IBZ contours
- Frequency reconstruction from eigenvectors
- Comparison of original vs reconstructed frequencies

This is the Python equivalent of the MATLAB plot_dispersion.m script.
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
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
except ImportError:
    # If relative imports fail, try absolute imports from current directory
    import plotting
    import wavevectors
    import mat73_loader
    plot_design = plotting.plot_design
    get_IBZ_contour_wavevectors = wavevectors.get_IBZ_contour_wavevectors
    load_matlab_v73 = mat73_loader.load_matlab_v73


def load_dataset(data_path, verbose=False):
    """
    Load dataset from .mat file (handles both v7.3 HDF5 and older formats).
    
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
        
    except NotImplementedError:
        # If that fails, use our robust h5py loader
        print("  File is MATLAB v7.3 format, using robust h5py loader...")
        try:
            data = load_matlab_v73(data_path, verbose=verbose)
        except ImportError:
            raise ImportError(
                "h5py is required to read MATLAB v7.3 files. "
                "Install it with: pip install h5py"
            )
    
    return data

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
    
    # Get K and M for this structure
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
    
    print(f"  Reconstructing frequencies for structure {struct_idx}...")
    print(f"    K: {K.shape}, {K.nnz} non-zeros, dtype={K.dtype}")
    print(f"    M: {M.shape}, {M.nnz} non-zeros, dtype={M.dtype}")
    print(f"    Reconstructing {N_wv} wavevectors × {N_eig} bands")
    
    # Handle T_DATA shape (1, N_wv) from HDF5
    T_DATA_array = data['T_DATA']
    
    # Loop over wavevectors
    for wv_idx in range(N_wv):
        # Get T for this wavevector
        if T_DATA_array.ndim == 2 and T_DATA_array.shape[0] == 1:
            # Shape (1, N_wv) from HDF5
            T = T_DATA_array[0, wv_idx]
        else:
            T = T_DATA_array.flat[wv_idx]
        
        # Debug: print T info on first iteration
        if wv_idx == 0:
            print(f"    T[0]: type={type(T)}, sparse={sp.issparse(T)}")
            if sp.issparse(T):
                print(f"    T[0]: shape={T.shape}, dtype={T.dtype}, nnz={T.nnz}")
        
        if not sp.issparse(T):
            raise TypeError(f"T[{wv_idx}] is not a sparse matrix! Type: {type(T)}")
        
        # Fix void dtype if present
        if T.dtype.kind == 'V':
            T = T.astype(np.complex128)
        
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
                print(f"    ✓ Dimensions match!" if eigvec.shape[0] == Kr.shape[0] else "✗ Dimension mismatch!")
            
            # Compute eigenvalue: eigval = ||Kr*v|| / ||Mr*v||
            Kr_v = Kr @ eigvec
            Mr_v = Mr @ eigvec
            eigval = np.linalg.norm(Kr_v) / np.linalg.norm(Mr_v)
            
            # Convert to frequency
            frequencies_recon[wv_idx, band_idx] = np.sqrt(eigval) / (2 * np.pi)
    
    return frequencies_recon


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
    print(f"    Wavevector range: x ∈ [{x_unique[0]:.4f}, {x_unique[-1]:.4f}], y ∈ [{y_unique[0]:.4f}, {y_unique[-1]:.4f}]")
    
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
        print(f"    ✓ Wavevectors form a valid rectangular grid")
    
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


def plot_dispersion_on_contour(ax, contour_info, frequencies_contour, contour_param=None, title='Dispersion'):
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


def main():
    """
    Main script execution.
    """
    print("=" * 70)
    print("Plot Dispersion Script")
    print("=" * 70)
    
    # Configuration
    # Update this path to point to your dataset
    data_fn_options = [
        "../generate_dispersion_dataset_Han/OUTPUT/output 13-Oct-2025 23-22-59/continuous 13-Oct-2025 23-22-59.mat",
    ]
    
    # Try to find dataset
    data_fn = None
    for option in data_fn_options:
        test_path = Path("../2D-dispersion_alex") / option
        if test_path.exists():
            data_fn = test_path
            break
    
    if data_fn is None:
        print("\nERROR: Could not find dataset file.")
        print("Please update the data_fn path in this script.")
        return
    
    # Flags
    isExportPng = True
    png_resolution = 150
    verbose_loading = False  # Set to True for detailed loading info
    
    # Contour plotting mode
    use_interpolation = False  # True: interpolate (smooth, N_k=120 points), False: grid points only (exact, ~25-50 points)
    N_k_interp = 25*25  # Number of interpolation points per segment (only used if use_interpolation=True)
    
    # Load data
    data = load_dataset(data_fn, verbose=verbose_loading)
    
    # Print dataset structure for debugging
    print(f"\n{'='*70}")
    print("Dataset Structure:")
    print(f"{'='*70}")
    for key in ['designs', 'WAVEVECTOR_DATA', 'EIGENVALUE_DATA', 'EIGENVECTOR_DATA', 
                'K_DATA', 'M_DATA', 'T_DATA']:
        if key in data:
            item = data[key]
            if hasattr(item, 'shape'):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            elif hasattr(item, '__len__'):
                print(f"  {key}: length={len(item)}, type={type(item)}")
            else:
                print(f"  {key}: type={type(item)}")
    print(f"{'='*70}\n")
    
    # Get filename for output directory
    fn = Path(data_fn).stem
    output_dir = Path('png') / fn
    
    # Make plots for one or multiple unit cells
    # designs shape: (N_struct, 3, N_pix, N_pix)
    struct_idxs = range(0, min(5, data['designs'].shape[0]))  # Up to 5 structures
    
    print(f"\nPlotting {len(struct_idxs)} structures...")
    
    for struct_idx in struct_idxs:
        print(f"\n{'='*70}")
        print(f"Processing structure {struct_idx + 1}/{len(struct_idxs)}")
        print(f"{'='*70}")
        
        # Plot the design
        print("  Plotting design...")
        # designs shape: (N_struct, 3, N_pix, N_pix) → need (N_pix, N_pix, 3)
        design = data['designs'][struct_idx, :, :, :].transpose(1, 2, 0)  # Reorder to (N_pix, N_pix, 3)
        fig_design, _ = plot_design(design)  # plot_design returns (fig, axes)
        
        if isExportPng:
            png_path = output_dir / 'design' / f'{struct_idx}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_design.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
            print(f"    Saved: {png_path}")
            plt.close(fig_design)  # Close to free memory
        
        # Extract wavevector and frequency data
        # WAVEVECTOR_DATA shape: (N_struct, 2, N_wv) → need (N_wv, 2)
        # EIGENVALUE_DATA shape: (N_struct, N_eig, N_wv) → need (N_wv, N_eig)
        wavevectors = data['WAVEVECTOR_DATA'][struct_idx, :, :].T  # Transpose to get (N_wv, 2)
        frequencies = data['EIGENVALUE_DATA'][struct_idx, :, :].T  # Transpose to get (N_wv, N_eig)
        
        print(f"  Wavevectors shape: {wavevectors.shape} (should be N_wv × 2)")
        print(f"  Frequencies shape: {frequencies.shape} (should be N_wv × N_eig)")
        
        # Reconstruct frequencies from eigenvectors
        if 'EIGENVECTOR_DATA' in data:
            print("  Reconstructing frequencies from eigenvectors...")
            frequencies_recon = reconstruct_frequencies_from_eigenvectors(data, struct_idx)
            
            # Check reconstruction error
            error = np.max(np.abs(frequencies_recon - frequencies)) / np.max(np.abs(frequencies))
            print(f"  Reconstruction error: {error:.6e}")
        else:
            frequencies_recon = None
            print("  No eigenvector data available for reconstruction")
        
        # Create grid interpolators
        const = data['const']
        if isinstance(const, dict):
            N_wv_param = const['N_wv']
        else:
            try:
                N_wv_param = const['N_wv'][0, 0][0]
            except:
                N_wv_param = const['N_wv']
        
        # Ensure N_wv_param is an array [N_x, N_y] of integers
        if np.isscalar(N_wv_param):
            N_wv_param = [int(N_wv_param), int(N_wv_param)]
        elif hasattr(N_wv_param, 'flatten'):
            N_wv_param = [int(x) for x in N_wv_param.flatten()]
            if len(N_wv_param) == 1:
                N_wv_param = [N_wv_param[0], N_wv_param[0]]
        else:
            # Try to convert to list of ints
            N_wv_param = [int(N_wv_param[0]), int(N_wv_param[1])]
        
        grid_interp, wavevectors_grid = create_grid_interpolators(wavevectors, frequencies, N_wv_param)
        
        if frequencies_recon is not None:
            grid_interp_recon, _ = create_grid_interpolators(wavevectors, frequencies_recon, N_wv_param)
        
        # Update interpolation density to match total grid points (e.g., 25*13)
        try:
            N_k_interp = int(N_wv_param[0]) * int(N_wv_param[1])
        except Exception:
            # Fallback to previous value if parsing fails
            pass

        # Extract 'a' parameter
        if isinstance(const, dict):
            a_val = const['a']
        else:
            try:
                a_val = const['a'][0, 0][0, 0]
            except:
                a_val = const['a']
        
        # Convert to float
        a = float(a_val.flatten()[0]) if hasattr(a_val, 'flatten') else float(a_val)
        
        # Get IBZ contour - method depends on mode
        # NOTE: Using 'p4mm' symmetry - adjust if your structures have different symmetry
        
        if use_interpolation:
            # Mode 1: Interpolation (smooth curves, more points)
            print(f"  Using interpolation mode (N_k={N_k_interp} points per segment)...")
            
            try:
                wavevectors_contour, contour_info = get_IBZ_contour_wavevectors(N_k_interp, a, 'p4mm')
                print(f"    Generated contour path with {len(wavevectors_contour)} interpolation points")
            except Exception as e:
                print(f"  WARNING: Could not generate contour: {e}")
                print("  Skipping contour plots for this structure")
                continue
        else:
            # Mode 2: Grid points only (exact values, fewer points)
            print(f"  Using grid-points-only mode (exact computed values)...")
            
            try:
                # Still need contour_info for plotting, but with dummy N_k
                _, contour_info = get_IBZ_contour_wavevectors(10, a, 'p4mm')
                # Extract actual grid points on contour
                wavevectors_contour, frequencies_contour_grid, contour_param_grid = \
                    extract_grid_points_on_contour(wavevectors, frequencies, contour_info, a)
                print(f"    Found {len(wavevectors_contour)} grid points on contour path")
            except Exception as e:
                print(f"  WARNING: Could not extract grid points: {e}")
                print("  Skipping contour plots for this structure")
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
        
        # Evaluate frequencies on contour - method depends on mode
        if use_interpolation:
            # Mode 1: Interpolate to contour points
            print("  Interpolating frequencies to contour points...")
            frequencies_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
            for eig_idx in range(frequencies.shape[1]):
                # Interpolator expects (y, x) order due to flip in MATLAB
                # wavevectors_contour is (N × 2) with [x, y] columns
                # Need to pass as [:, [1, 0]] to get [y, x] order
                points_yx = wavevectors_contour[:, [1, 0]]  # Swap columns: (y, x)
                frequencies_contour[:, eig_idx] = grid_interp[eig_idx](points_yx)
            
            # Create contour parameter for plotting
            contour_param = contour_info['wavevector_parameter']
            
            if frequencies_recon is not None:
                frequencies_recon_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
                for eig_idx in range(frequencies.shape[1]):
                    points_yx = wavevectors_contour[:, [1, 0]]  # Swap columns: (y, x)
                    frequencies_recon_contour[:, eig_idx] = grid_interp_recon[eig_idx](points_yx)
        else:
            # Mode 2: Use exact grid point values (already extracted)
            print("  Using exact grid point frequencies (no interpolation)...")
            frequencies_contour = frequencies_contour_grid
            contour_param = contour_param_grid
            
            # For reconstructed frequencies, extract grid points too
            if frequencies_recon is not None:
                _, frequencies_recon_contour, contour_param_recon = \
                    extract_grid_points_on_contour(wavevectors, frequencies_recon, contour_info, a)
                print(f"    Extracted {len(frequencies_recon_contour)} reconstructed grid points")
                # Ensure both use the same parameterization length
                # If different, fall back to using the original contour_param for plotting both
                if len(contour_param_recon) == len(contour_param):
                    contour_param_recon_use = contour_param_recon
                else:
                    contour_param_recon_use = contour_param
        
        # Plot dispersion relation (original)
        mode_str = "Interpolated" if use_interpolation else "Grid Points"
        print(f"  Plotting original dispersion ({mode_str})...")
        fig_disp = plt.figure(figsize=(10, 6))
        ax_disp = fig_disp.add_subplot(111)
        plot_dispersion_on_contour(ax_disp, contour_info, frequencies_contour, contour_param,
                                   title=f'Dispersion Relation (Original, {mode_str})')
        
        if isExportPng:
            png_path = output_dir / 'dispersion' / f'{struct_idx}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_disp.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
            print(f"    Saved: {png_path}")
            plt.close(fig_disp)
        
        # Plot dispersion relation (reconstructed)
        if frequencies_recon is not None:
            print("  Plotting reconstructed dispersion...")
            fig_recon = plt.figure(figsize=(10, 6))
            ax_recon = fig_recon.add_subplot(111)
            # In grid-only mode, ensure we use a matching contour_param for reconstructed data
            recon_param_to_use = contour_param
            if not use_interpolation:
                try:
                    recon_param_to_use = contour_param_recon_use
                except NameError:
                    recon_param_to_use = contour_param
            plot_dispersion_on_contour(ax_recon, contour_info, frequencies_recon_contour, recon_param_to_use,
                                      title='Dispersion Relation (Reconstructed from Eigenvectors)')
            
            if isExportPng:
                png_path = output_dir / 'dispersion' / f'{struct_idx}_recon.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig_recon.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                print(f"    Saved: {png_path}")
                plt.close(fig_recon)
            
            # Plot the difference between original and reconstructed
            print("  Plotting difference (original - reconstructed)...")
            fig_diff = plt.figure(figsize=(10, 6))
            ax_diff = fig_diff.add_subplot(111)
            
            difference = frequencies_contour - frequencies_recon_contour
            max_abs_diff = np.max(np.abs(difference))
            max_freq = np.max(np.abs(frequencies_contour))
            rel_error_pct = 100 * max_abs_diff / max_freq if max_freq > 0 else 0
            
            # Choose x-axis parameterization consistent with the mode
            if use_interpolation:
                x_axis_param = contour_info['wavevector_parameter']
            else:
                x_axis_param = contour_param
                # Safety: trim to match if slight mismatch persists
                if len(x_axis_param) != difference.shape[0]:
                    min_len = min(len(x_axis_param), difference.shape[0])
                    x_axis_param = x_axis_param[:min_len]
                    difference = difference[:min_len, :]
            
            # Plot difference for each band
            for band_idx in range(difference.shape[1]):
                ax_diff.plot(x_axis_param, 
                           difference[:, band_idx],
                           linewidth=2)
            
            # Add vertical lines at segment boundaries
            for i in range(contour_info['N_segment'] + 1):
                ax_diff.axvline(i, color='k', linestyle='--', alpha=0.3, linewidth=1)
            
            # Add horizontal zero reference line
            ax_diff.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Zero')
            
            ax_diff.set_xlabel('Wavevector Contour Parameter', fontsize=12)
            ax_diff.set_ylabel('Frequency Difference [Hz]', fontsize=12)
            ax_diff.set_title(f'Difference (Original - Reconstructed)\n' + 
                            f'Max abs diff = {max_abs_diff:.3e} Hz ({rel_error_pct:.3f}%)', 
                            fontsize=14)
            ax_diff.grid(True, alpha=0.3)
            ax_diff.legend()
            
            if isExportPng:
                png_path = output_dir / 'dispersion' / f'{struct_idx}_diff.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig_diff.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                print(f"    Saved: {png_path}")
                plt.close(fig_diff)
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    if isExportPng:
        print(f"Output saved to: {output_dir}")
    print(f"{'='*70}")
    
    # Show plots if not saving
    if not isExportPng:
        plt.show()


if __name__ == "__main__":
    main()

