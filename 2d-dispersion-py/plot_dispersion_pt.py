"""
Plot Dispersion Script for PyTorch Dataset Format

This script loads PyTorch format datasets (from convert_mat_to_pytorch.py) and creates dispersion visualizations:
- Unit cell designs
- Dispersion relations along IBZ contours
- Material property fields

Supports both reduced (randomly downsampled) and unreduced PyTorch datasets.
For reduced datasets, can optionally merge with original eigenvalue data from NumPy format.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import sys

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

# Import local modules
try:
    from plotting import plot_design
    from wavevectors import get_IBZ_contour_wavevectors
except ImportError:
    import plotting
    import wavevectors
    plot_design = plotting.plot_design
    get_IBZ_contour_wavevectors = wavevectors.get_IBZ_contour_wavevectors


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
        # Verify we have the right number of points; fallback to detected
        if len(x_unique) != N_x or len(y_unique) != N_y:
            print(f"    WARNING: Grid size mismatch!")
            print(f"      Expected: {N_x} × {N_y} = {N_x*N_y} points")
            print(f"      Got: {len(x_unique)} × {len(y_unique)} = {len(x_unique)*len(y_unique)} points")
            N_x, N_y = len(x_unique), len(y_unique)
    else:
        N_x, N_y = int(N_wv[0]), int(N_wv[1])
        # Verify we have the right number of points; fallback to detected
        if len(x_unique) != N_x or len(y_unique) != N_y:
            print(f"    WARNING: Grid size mismatch!")
            print(f"      Expected: {N_x} × {N_y} = {N_x*N_y} points")
            print(f"      Got: {len(x_unique)} × {len(y_unique)} = {len(x_unique)*len(y_unique)} points")
            N_x, N_y = len(x_unique), len(y_unique)

    expected_total = N_x * N_y
    actual_total = len(wavevectors)
    if expected_total != actual_total:
        # Wavevectors don't form a regular grid - use scattered interpolation instead
        print(f"    Note: Wavevectors don't form a regular grid ({actual_total} points, expected {expected_total} for {N_x}×{N_y})")
        print(f"    Using scattered interpolation instead of grid interpolation")
        return None, None  # Signal to use scattered interpolation

    # MATLAB column-major equivalent reshape
    frequencies_grid = frequencies.reshape((N_y, N_x, frequencies.shape[1]), order='F')

    # Create interpolators for each band (note flipped order y,x to match MATLAB)
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

    contour_points = np.vstack(contour_points_list)
    contour_freqs = np.vstack(contour_freqs_list)
    contour_param = np.concatenate(contour_param_list)
    contour_param = contour_param / total_distance * contour_info['N_segment']
    return contour_points, contour_freqs, contour_param


def plot_dispersion_on_contour(ax, contour_info, frequencies_contour, contour_param=None, title='Dispersion', mark_points=False):
    """Plot dispersion relation on IBZ contour."""
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
        vertex_positions = np.linspace(0, contour_info['N_segment'], 
                                      len(contour_info['vertex_labels']))
        ax.set_xticks(vertex_positions)
        ax.set_xticklabels(contour_info['vertex_labels'])
    
    ax.set_xlabel('Wavevector Contour Parameter', fontsize=12)
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)


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


def main(cli_data_dir=None, cli_original_dir=None, n_structs=None):
    """
    Main script execution.
    """
    print("=" * 70)
    print("Plot Dispersion Script (PyTorch Dataset Format)")
    print("=" * 70)
    
    # Configuration
    if cli_data_dir is not None:
        data_dir = Path(cli_data_dir)
    else:
        # Default to a PyTorch dataset
        data_dir = Path(r"D:\Research\NO-2D-Metamaterials\data\out_binarized_1")
    
    original_data_dir = None
    if cli_original_dir is not None:
        original_data_dir = Path(cli_original_dir)
    
    if not data_dir.exists():
        print(f"\nERROR: Dataset directory not found: {data_dir}")
        print("Please provide a valid path to the PyTorch dataset.")
        return
    
    # Flags
    isExportPng = True
    png_resolution = 150
    verbose_loading = False
    mark_points = True  # Add markers to dispersion plots
    
    # Contour plotting mode
    use_interpolation = False  # True: interpolate (smooth), False: grid points only (exact)
    N_k_interp = 120  # Number of interpolation points per segment (only used if use_interpolation=True)
    
    # Load data
    print(f"\nLoading PyTorch dataset from: {data_dir}")
    try:
        data = load_pt_dataset(data_dir, original_data_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("\nNote: PyTorch datasets from convert_mat_to_pytorch.py do not include eigenvalue data.")
        print("Please provide original_data_dir pointing to the NumPy format dataset with eigenvalue_data.npy")
        return
    
    # Print dataset structure
    print(f"\n{'='*70}")
    print("Dataset Structure:")
    print(f"{'='*70}")
    for key in ['designs', 'wavevectors', 'eigenvalue_data']:
        if key in data:
            item = data[key]
            if hasattr(item, 'shape'):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
    print(f"{'='*70}\n")
    
    # Create output directory: plots/<dataset_name>_pt/ in current directory
    import os
    current_dir = Path(os.getcwd())
    dataset_name = data_dir.name
    output_dir = current_dir / 'plots' / f'{dataset_name}_pt'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Extract shapes
    designs = data['designs']                 # (N_struct, N_pix, N_pix)
    wavevectors_all = data['wavevectors']     # (N_struct, N_wv, 2)
    eigenvalues_all = data['eigenvalue_data'] # (N_struct, N_wv, N_eig)
    
    # Infer N_wv grid size
    N_wv_param = get_N_wv_from_data(designs.shape, wavevectors_all.shape)
    print(f"Inferred wavevector grid size: {N_wv_param[0]} × {N_wv_param[1]}")
    
    # Assume lattice parameter (default to 1.0)
    a = 1.0
    
    # Make plots for specified number of structures (default: 5)
    if n_structs is None:
        n_structs_to_plot = min(5, designs.shape[0])
    else:
        n_structs_to_plot = min(int(n_structs), designs.shape[0])
    struct_idxs = range(n_structs_to_plot)
    
    print(f"\nPlotting {len(struct_idxs)} structures...")
    
    for struct_idx in struct_idxs:
        print(f"\n{'='*70}")
        print(f"Processing structure {struct_idx + 1}/{len(struct_idxs)}")
        print(f"{'='*70}")
        
        # Plot the design
        print("  Plotting design...")
        # designs shape: (N_struct, N_pix, N_pix) - single channel normalized [0,1]
        # Need to reconstruct 3-channel material properties
        
        design_param = designs[struct_idx, :, :]  # (N_pix, N_pix), values in [0, 1]
        
        # Material parameter ranges (binary dataset)
        E_min, E_max = 20e6, 200e9
        rho_min, rho_max = 400, 8000
        nu_min, nu_max = 0.05, 0.3
        
        # Compute actual material properties (coupled - all follow same design pattern)
        elastic_modulus = E_min + (E_max - E_min) * design_param
        density = rho_min + (rho_max - rho_min) * design_param
        poisson_ratio = nu_min + (nu_max - nu_min) * design_param
        
        geometry = np.stack([elastic_modulus, density, poisson_ratio], axis=-1)  # (N_pix, N_pix, 3)
        
        fig_design, _ = plot_design(geometry)
        
        if isExportPng:
            png_path = output_dir / 'design' / f'{struct_idx}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_design.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
            print(f"    Saved: {png_path}")
            plt.close(fig_design)
        
        # Extract wavevector and frequency data
        wavevectors = wavevectors_all[struct_idx, :, :]  # (N_wv, 2)
        frequencies = eigenvalues_all[struct_idx, :, :]  # (N_wv, N_eig)
        
        print(f"  Wavevectors shape: {wavevectors.shape}")
        print(f"  Frequencies shape: {frequencies.shape}")
        
        # Create grid interpolators (pass None to auto-detect from actual data)
        # If wavevectors don't form a regular grid, use scattered interpolation
        from scipy.interpolate import griddata
        grid_interp, wavevectors_grid = create_grid_interpolators(wavevectors, frequencies, None)
        use_scattered = (grid_interp is None)
        
        # Get IBZ contour - method depends on mode
        # NOTE: Using 'p4mm' symmetry - adjust if your structures have different symmetry
        
        if use_interpolation:
            # Mode 1: Interpolation (smooth curves, more points)
            print(f"  Using interpolation mode...")
            
            try:
                wavevectors_contour, contour_info = get_IBZ_contour_wavevectors(N_k_interp, a, 'p4mm')
                print(f"    Generated contour path with {len(wavevectors_contour)} interpolation points")
            except Exception as e:
                print(f"  WARNING: Could not generate contour: {e}")
                print("  Skipping contour plots for this structure")
                continue
        else:
            # Mode 2: Grid points only (exact values, fewer points)
            print(f"  Using grid-points-only mode...")
            
            try:
                # Still need contour_info for plotting
                _, contour_info = get_IBZ_contour_wavevectors(10, a, 'p4mm')
                # Extract actual grid points on contour
                wavevectors_contour, frequencies_contour_grid, contour_param_grid = \
                    extract_grid_points_on_contour(wavevectors, frequencies, contour_info, a, tolerance=2e-3)
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
        if use_scattered:
            # Use scattered interpolation for non-grid wavevectors
            print("  Using scattered interpolation for non-grid wavevectors...")
            # First, get the contour wavevectors
            if not use_interpolation:
                # We already have wavevectors_contour from extract_grid_points_on_contour
                # Use those directly with scattered interpolation
                frequencies_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
                for eig_idx in range(frequencies.shape[1]):
                    frequencies_contour[:, eig_idx] = griddata(
                        wavevectors, frequencies[:, eig_idx], wavevectors_contour,
                        method='linear', fill_value=np.nan
                    )
                # Remove NaN values if any
                valid_mask = ~np.isnan(frequencies_contour).any(axis=1)
                if np.any(valid_mask):
                    wavevectors_contour = wavevectors_contour[valid_mask]
                    frequencies_contour = frequencies_contour[valid_mask]
                    # Use the contour parameter from extract_grid_points_on_contour
                    contour_param = contour_param_grid[valid_mask] if 'contour_param_grid' in locals() else contour_info['wavevector_parameter'][:len(wavevectors_contour)]
                else:
                    print("  WARNING: No valid interpolated points, using grid points")
                    frequencies_contour = frequencies_contour_grid
                    contour_param = contour_param_grid
            else:
                # Generate contour and interpolate
                frequencies_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
                for eig_idx in range(frequencies.shape[1]):
                    frequencies_contour[:, eig_idx] = griddata(
                        wavevectors, frequencies[:, eig_idx], wavevectors_contour,
                        method='linear', fill_value=np.nan
                    )
                # Remove NaN values if any
                valid_mask = ~np.isnan(frequencies_contour).any(axis=1)
                if np.any(valid_mask):
                    wavevectors_contour = wavevectors_contour[valid_mask]
                    frequencies_contour = frequencies_contour[valid_mask]
                    contour_param = contour_info['wavevector_parameter'][valid_mask]
                else:
                    print("  WARNING: No valid interpolated points")
                    contour_param = contour_info['wavevector_parameter']
        elif use_interpolation:
            # Mode 1: Interpolate to contour points using grid interpolator
            print("  Interpolating frequencies to contour points...")
            frequencies_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
            for eig_idx in range(frequencies.shape[1]):
                points_yx = wavevectors_contour[:, [1, 0]]  # Swap columns: (y, x)
                frequencies_contour[:, eig_idx] = grid_interp[eig_idx](points_yx)
            
            contour_param = contour_info['wavevector_parameter']
        else:
            # Mode 2: Use exact grid point values (already extracted)
            print("  Using exact grid point frequencies (no interpolation)...")
            frequencies_contour = frequencies_contour_grid
            contour_param = contour_param_grid
        
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
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    if isExportPng:
        print(f"Output saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot dispersion from PyTorch dataset")
    parser.add_argument("data_dir", nargs="?", help="Path to the PyTorch dataset directory")
    parser.add_argument("original_dir", nargs="?", help="Path to original NumPy dataset (for eigenvalue data)")
    parser.add_argument("-n", "--n-structs", type=int, default=None, help="Number of structures to plot (default: 5)")
    args = parser.parse_args()
    main(args.data_dir, args.original_dir, args.n_structs)

# Process outline 
## Load PyTorch dataset
### Load bands (6)
### Load wavevectors (91,2)
### Load geometries (N, 32, 32)
### Load eigenvectors (N, WV, B, 32, 32)
### Load K matrices (N, 2000+, 2000+) 
### Load M matrices (N, 2000+, 2000+)
### Load T matrices (N, 2000+, 2000+)

## Reconstruct frequencies from eigenvectors (plot dispersion.m)
###