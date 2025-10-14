"""
Plot Dispersion Script (NumPy/PyTorch dataset variant)

This script loads the converted NumPy dataset (dataset_conversion_reduction format)
from 'py_complex128' or 'py_complex64' folders and renders figures equivalent to plot_dispersion_script.py:
- Design panels
- IBZ contour scatter
- Dispersion along IBZ contour (original)

If the original .mat (with K/M/T) is found next to the converted folder, the script
also reconstructs eigenvalues, and plots reconstructed and difference figures.
If not found, reconstructed/difference plots are skipped.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Local imports (package mode)
try:
    from plotting import plot_design
    from wavevectors import get_IBZ_contour_wavevectors
except ImportError:
    import plotting
    import wavevectors
    plot_design = plotting.plot_design
    get_IBZ_contour_wavevectors = wavevectors.get_IBZ_contour_wavevectors
    try:
        import mat73_loader
        load_matlab_v73 = mat73_loader.load_matlab_v73
    except Exception:
        load_matlab_v73 = None


def load_numpy_dataset(data_dir: Path):
    """Load converted NumPy dataset from the 'python' subfolder."""
    data = {}
    # Required
    data['designs'] = np.load(data_dir / 'designs.npy')                 # (N_struct, N_pix, N_pix)
    data['wavevectors'] = np.load(data_dir / 'wavevectors.npy')         # (N_struct, N_wv, 2)
    data['eigenvalue_data'] = np.load(data_dir / 'eigenvalue_data.npy') # (N_struct, N_wv, N_eig)
    # Optional
    if (data_dir / 'eigenvector_data_x.npy').exists():
        data['eigenvector_data_x'] = np.load(data_dir / 'eigenvector_data_x.npy')
    if (data_dir / 'eigenvector_data_y.npy').exists():
        data['eigenvector_data_y'] = np.load(data_dir / 'eigenvector_data_y.npy')
    if (data_dir / 'waveforms.npy').exists():
        data['waveforms'] = np.load(data_dir / 'waveforms.npy')
    if (data_dir / 'bands_fft.npy').exists():
        data['bands_fft'] = np.load(data_dir / 'bands_fft.npy')
    return data


def create_grid_interpolators(wavevectors, frequencies, N_wv):
    """
    Create grid interpolators for each eigenvalue band.
    Matches MATLAB/Python behavior in the original script.
    """
    x_unique = np.sort(np.unique(wavevectors[:, 0]))
    y_unique = np.sort(np.unique(wavevectors[:, 1]))

    # Ensure N_wv elements are integers
    N_x, N_y = int(N_wv[0]), int(N_wv[1])

    # Verify we have the right number of points; fallback to detected
    if len(x_unique) != N_x or len(y_unique) != N_y:
        N_x, N_y = len(x_unique), len(y_unique)

    expected_total = N_x * N_y
    actual_total = len(wavevectors)
    if expected_total != actual_total:
        raise ValueError(f"Grid size mismatch: {N_x}×{N_y}={expected_total} but have {actual_total} wavevectors")

    # MATLAB column-major equivalent reshape
    frequencies_grid = frequencies.reshape((N_y, N_x, frequencies.shape[1]), order='F')

    # Create interpolators for each band (note flipped order y,x)
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


def load_original_mat(mat_path: Path, verbose=False):
    """Load the original MATLAB dataset (for K/M/T reconstruction)."""
    print(f"Loading original .mat for reconstruction: {mat_path}")
    try:
        data = sio.loadmat(str(mat_path), squeeze_me=False)
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        print("  Loaded using scipy.io.loadmat")
        return data
    except NotImplementedError:
        if load_matlab_v73 is None:
            raise RuntimeError("h5py/mat73_loader required to read v7.3 files")
        print("  File is MATLAB v7.3 format, using h5py loader...")
        return load_matlab_v73(str(mat_path), verbose=verbose)


def reconstruct_frequencies_via_KMT(mat_data, struct_idx: int, N_eig: int):
    """
    Reconstruct frequencies by solving reduced eigenproblems Kr*phi = Mr*phi*lambda
    for each wavevector, where Kr = T^H K T and Mr = T^H M T.

    Returns frequencies as sqrt(lambda)/(2*pi) with shape (N_wv, N_eig), sorted ascending.
    """
    eigvec_shape = mat_data['EIGENVECTOR_DATA'].shape
    N_wv = eigvec_shape[2]

    K_DATA_array = mat_data['K_DATA']
    M_DATA_array = mat_data['M_DATA']

    if K_DATA_array.shape == (1, ):
        K = K_DATA_array.flat[0]
        M = M_DATA_array.flat[0]
    elif K_DATA_array.ndim == 2 and K_DATA_array.shape[0] == 1:
        K = K_DATA_array[0, struct_idx]
        M = M_DATA_array[0, struct_idx]
    else:
        K = K_DATA_array.flat[struct_idx]
        M = M_DATA_array.flat[struct_idx]

    if not sp.issparse(K) or not sp.issparse(M):
        raise TypeError("K/M are not sparse matrices; unexpected format")

    T_DATA_array = mat_data['T_DATA']

    frequencies_recon = np.zeros((N_wv, N_eig))

    for wv_idx in range(N_wv):
        if T_DATA_array.ndim == 2 and T_DATA_array.shape[0] == 1:
            T = T_DATA_array[0, wv_idx]
        else:
            T = T_DATA_array.flat[wv_idx]

        if not sp.issparse(T):
            raise TypeError(f"T[{wv_idx}] is not sparse")

        if T.dtype.kind == 'V':
            T = T.astype(np.complex128)

        Kr = T.conj().T @ K @ T
        Mr = T.conj().T @ M @ T

        try:
            vals, _ = spla.eigsh(Kr, k=N_eig, M=Mr, which='SM')
        except Exception:
            vals, _ = spla.eigsh(Kr.tocsr(), k=N_eig, M=Mr.tocsr(), which='SM')

        vals = np.real(vals)
        vals.sort()
        frequencies_recon[wv_idx, :] = np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi)

    return frequencies_recon


def main():
    print("=" * 70)
    print("Plot Dispersion Script (NumPy dataset)")
    print("=" * 70)

    # Configuration
    # Default base dir next to the .mat dataset; collect all converted folders
    base_dir = Path(r"D:\Research\NO-2D-Metamaterials\generate_dispersion_dataset_Han\OUTPUT\output 13-Oct-2025 23-22-59")
    data_dirs = []
    if (base_dir / 'py_complex128').exists():
        data_dirs.append(base_dir / 'py_complex128')
    if (base_dir / 'py_complex64').exists():
        data_dirs.append(base_dir / 'py_complex64')
    if not data_dirs:
        print("\nERROR: Could not find converted dataset folder (py_complex128 or py_complex64) in:")
        print(f"  {base_dir}")
        return

    # Flags
    isExportPng = True
    png_resolution = 150
    use_interpolation = False  # True: interpolate; False: exact grid points

    for data_dir in data_dirs:
        print(f"\n=== Processing dataset folder: {data_dir} ===")

        # Load NumPy data
        data = load_numpy_dataset(data_dir)

        # Unpack
        designs_np = data['designs']                 # (N_struct, N_pix, N_pix)
        wavevectors_all = data['wavevectors']        # (N_struct, N_wv, 2)
        eigenvalues_all = data['eigenvalue_data']    # (N_struct, N_wv, N_eig)

        N_struct = designs_np.shape[0]
        N_pix = designs_np.shape[1]
        N_wv = wavevectors_all.shape[1]
        N_eig = eigenvalues_all.shape[2]

        # Prepare output dir: save PNGs directly into the dataset folder
        output_dir = data_dir

        print(f"\nPlotting all {N_struct} structures...")
        struct_idxs = range(0, N_struct)

        for struct_idx in struct_idxs:
            print(f"-- Folder {data_dir.name}: struct {int(struct_idx)} / {N_struct-1}")
            print(f"\n{'='*70}")
            print(f"Processing structure {struct_idx + 1}/{len(struct_idxs)}")
            print(f"{'='*70}")

            # Plot the design
            print("  Plotting design...")
            # We only have elastic modulus; replicate into 3 channels for visualization
            design_single = designs_np[struct_idx]  # (N_pix, N_pix)
            design_3ch = np.stack([design_single, design_single, design_single], axis=2)
            fig_design, _ = plot_design(design_3ch)
            if isExportPng:
                png_path = output_dir / 'design' / f'{int(struct_idx)}.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig_design.canvas.draw()
                fig_design.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                print(f"    Saved: {png_path}")
                plt.close(fig_design)

            # Extract wavevector and frequency data for this structure
            wavevectors = wavevectors_all[struct_idx]      # (N_wv, 2)
            frequencies = eigenvalues_all[struct_idx]      # (N_wv, N_eig)

            # Prepare contour info (p4mm default)
            a = 1.0
            if use_interpolation:
                print("  Using interpolation mode...")
                # Choose dense contour path points per segment proportional to grid size
                N_k_interp = int(max(25, N_wv))
                wavevectors_contour, contour_info = get_IBZ_contour_wavevectors(N_k_interp, a, 'p4mm')
            else:
                print("  Using grid-points-only mode (exact computed values)...")
                # Obtain contour info (dummy N)
                _, contour_info = get_IBZ_contour_wavevectors(10, a, 'p4mm')

            # Attempt reconstruction via original .mat if present
            mat_candidates = list(base_dir.glob('*.mat'))
            mat_path = mat_candidates[0] if mat_candidates else None
            do_reconstruct = mat_path is not None
            if do_reconstruct:
                try:
                    mat_data = load_original_mat(mat_path, verbose=False)
                    freqs_recon_full = reconstruct_frequencies_via_KMT(mat_data, struct_idx, N_eig)
                except Exception as e:
                    print(f"  WARNING: Reconstruction failed: {e}")
                    do_reconstruct = False

            # Plot IBZ contour wavevectors (for first structure only)
            if struct_idx == struct_idxs[0]:
                fig_contour = plt.figure(figsize=(8, 8))
                ax_contour = fig_contour.add_subplot(111)
                if use_interpolation:
                    ax_contour.plot(wavevectors_contour[:, 0], wavevectors_contour[:, 1], 'k.', markersize=2)
                else:
                    # Extract only points that lie on contour
                    contour_points, frequencies_contour_grid, contour_param_grid = \
                        extract_grid_points_on_contour(wavevectors, frequencies, contour_info, a)
                    ax_contour.plot(contour_points[:, 0], contour_points[:, 1], 'k.', markersize=2)
                ax_contour.set_aspect('equal')
                ax_contour.set_xlabel('Wavevector x component [1/m]')
                ax_contour.set_ylabel('Wavevector y component [1/m]')
                ax_contour.set_title('IBZ Contour Wavevectors')
                ax_contour.grid(True, alpha=0.3)
                if isExportPng:
                    png_path = output_dir / 'contour' / f'{struct_idx}.png'
                    png_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_contour.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                    print(f"    Saved: {png_path}")
                    plt.close(fig_contour)

            # Evaluate frequencies on contour
            if use_interpolation:
                print("  Interpolating frequencies to contour points...")
                grid_interp, wavevectors_grid = create_grid_interpolators(wavevectors, frequencies, [int(np.sqrt(N_wv)), int(np.sqrt(N_wv))])
                frequencies_contour = np.zeros((len(wavevectors_contour), frequencies.shape[1]))
                for eig_idx in range(frequencies.shape[1]):
                    points_yx = wavevectors_contour[:, [1, 0]]
                    frequencies_contour[:, eig_idx] = grid_interp[eig_idx](points_yx)
                contour_param = contour_info['wavevector_parameter']
                if do_reconstruct:
                    grid_interp_recon, _ = create_grid_interpolators(wavevectors, freqs_recon_full, [int(np.sqrt(N_wv)), int(np.sqrt(N_wv))])
                    frequencies_recon_contour = np.zeros_like(frequencies_contour)
                    for eig_idx in range(frequencies_contour.shape[1]):
                        points_yx = wavevectors_contour[:, [1, 0]]
                        frequencies_recon_contour[:, eig_idx] = grid_interp_recon[eig_idx](points_yx)
            else:
                print("  Using exact grid point frequencies on contour...")
                # Use already extracted from the plotting step above; recompute here for clarity
                _, frequencies_contour, contour_param = extract_grid_points_on_contour(wavevectors, frequencies, contour_info, a)
                if do_reconstruct:
                    _, frequencies_recon_contour, contour_param_recon = extract_grid_points_on_contour(wavevectors, freqs_recon_full, contour_info, a)
                    if len(contour_param_recon) == len(contour_param):
                        contour_param_recon_use = contour_param_recon
                    else:
                        contour_param_recon_use = contour_param

            # Plot dispersion (original)
            fig_disp = plt.figure(figsize=(10, 6))
            ax_disp = fig_disp.add_subplot(111)
            for band_idx in range(frequencies_contour.shape[1]):
                ax_disp.plot(contour_param, frequencies_contour[:, band_idx], linewidth=2)
            for i in range(contour_info['N_segment'] + 1):
                ax_disp.axvline(i, color='k', linestyle='--', alpha=0.3, linewidth=1)
            if 'vertex_labels' in contour_info and contour_info['vertex_labels']:
                vertex_positions = np.linspace(0, contour_info['N_segment'], len(contour_info['vertex_labels']))
                ax_disp.set_xticks(vertex_positions)
                ax_disp.set_xticklabels(contour_info['vertex_labels'])
            ax_disp.set_xlabel('Wavevector Contour Parameter')
            ax_disp.set_ylabel('Frequency [Hz]')
            ax_disp.set_title('Dispersion Relation (Original)')
            ax_disp.grid(True, alpha=0.3)
            if isExportPng:
                png_path = output_dir / 'dispersion' / f'{int(struct_idx)}.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig_disp.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                print(f"    Saved: {png_path}")
                plt.close(fig_disp)

            # Plot reconstructed and difference if available
            if do_reconstruct:
                # Reconstructed
                fig_recon = plt.figure(figsize=(10, 6))
                ax_recon = fig_recon.add_subplot(111)
                x_param_recon = contour_param
                if not use_interpolation:
                    try:
                        x_param_recon = contour_param_recon_use
                    except NameError:
                        x_param_recon = contour_param
                for band_idx in range(frequencies_recon_contour.shape[1]):
                    ax_recon.plot(x_param_recon, frequencies_recon_contour[:, band_idx], linewidth=2)
                for i in range(contour_info['N_segment'] + 1):
                    ax_recon.axvline(i, color='k', linestyle='--', alpha=0.3, linewidth=1)
                if 'vertex_labels' in contour_info and contour_info['vertex_labels']:
                    vertex_positions = np.linspace(0, contour_info['N_segment'], len(contour_info['vertex_labels']))
                    ax_recon.set_xticks(vertex_positions)
                    ax_recon.set_xticklabels(contour_info['vertex_labels'])
                ax_recon.set_xlabel('Wavevector Contour Parameter')
                ax_recon.set_ylabel('Frequency [Hz]')
                ax_recon.set_title('Dispersion Relation (Reconstructed)')
                ax_recon.grid(True, alpha=0.3)
                if isExportPng:
                    png_path = output_dir / 'dispersion' / f'{int(struct_idx)}_recon.png'
                    png_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_recon.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                    print(f"    Saved: {png_path}")
                    plt.close(fig_recon)

                # Difference
                fig_diff = plt.figure(figsize=(10, 6))
                ax_diff = fig_diff.add_subplot(111)
                x_param = contour_param
                y_diff = frequencies_contour - frequencies_recon_contour
                if len(x_param) != y_diff.shape[0]:
                    min_len = min(len(x_param), y_diff.shape[0])
                    x_param = x_param[:min_len]
                    y_diff = y_diff[:min_len, :]
                for band_idx in range(y_diff.shape[1]):
                    ax_diff.plot(x_param, y_diff[:, band_idx], linewidth=2)
                for i in range(contour_info['N_segment'] + 1):
                    ax_diff.axvline(i, color='k', linestyle='--', alpha=0.3, linewidth=1)
                ax_diff.axhline(0, color='r', linestyle='--', linewidth=1.5)
                ax_diff.set_xlabel('Wavevector Contour Parameter')
                ax_diff.set_ylabel('Frequency Difference [Hz]')
                max_abs_diff = float(np.max(np.abs(y_diff))) if y_diff.size else 0.0
                ax_diff.set_title(f'Difference (Original - Reconstructed)\nMax abs diff = {max_abs_diff:.3e} Hz')
                ax_diff.grid(True, alpha=0.3)
                if isExportPng:
                    png_path = output_dir / 'dispersion' / f'{int(struct_idx)}_diff.png'
                    png_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_diff.savefig(png_path, dpi=png_resolution, bbox_inches='tight')
                    print(f"    Saved: {png_path}")
                    plt.close(fig_diff)

        print(f"\n{'='*70}")
        print("Processing complete!")
        print(f"Output saved to: {output_dir}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()


