"""
Convert MATLAB dataset to PyTorch format following dataset_conversion_reduction.ipynb conventions.

This script loads a MATLAB .mat file containing dispersion dataset,
extracts the main components (const, design_params, designs), computes
the remaining quantities using PyTorch tensors, and saves everything
in the format used by dataset_conversion_reduction.ipynb.

Usage:
    python convert_mat_to_pytorch.py <path_to_mat_file>
    
Output:
    Creates a folder <mat_file_path>_py/ containing:
    - designs.npy: Design arrays (N_struct, N_pix, N_pix) [float16]
    - design_params.npy: Design parameters [float64]
    - wavevectors.npy: Wavevector data (N_struct, N_wv, 2) [float16]
    - waveforms.npy: Wavelet-embedded wavevectors (N_wv, N_pix, N_pix) [float16]
    - eigenvalue_data.npy: Eigenvalues/frequencies (N_struct, N_wv, N_eig) [original dtype]
    - eigenvector_data_x.npy: X-component eigenvectors (N_struct, N_wv, N_eig, N_pix, N_pix) [complex]
    - eigenvector_data_y.npy: Y-component eigenvectors (N_struct, N_wv, N_eig, N_pix, N_pix) [complex]
    - bands_fft.npy: Wavelet-embedded band indices (N_eig, N_pix, N_pix) [float16]
    
Note: This format differs from the MATLAB structure to match the Python ML workflow.

IMPORTANT: Wavelet Embedding Functions
---------------------------------------
The functions `embed_2const_wavelet` and `embed_integer_wavelet` are placeholder
implementations using sinusoidal encodings. The actual implementations from 
NO_utils_multiple may use different wavelet transforms.

If you have access to NO_utils_multiple, you can replace these functions with:
    from NO_utils_multiple import embed_2const_wavelet, embed_integer_wavelet

The current sinusoidal encoding provides a reasonable spatial representation but
may not match exactly the output from the original NO_utils_multiple functions.
"""

import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
import sys
from typing import Dict, Any, Tuple, Optional

# Import local modules
try:
    from mat73_loader import load_matlab_v73
except ImportError:
    import mat73_loader
    load_matlab_v73 = mat73_loader.load_matlab_v73


def load_dataset(mat_path: Path) -> Dict[str, Any]:
    """
    Load MATLAB dataset from .mat file.
    
    Parameters
    ----------
    mat_path : Path
        Path to .mat file
        
    Returns
    -------
    data : dict
        Dictionary containing dataset variables
    """
    print(f"Loading dataset from: {mat_path}")
    
    # Try scipy first (for older MATLAB files)
    try:
        data = sio.loadmat(str(mat_path), squeeze_me=False)
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        print("  Loaded using scipy.io.loadmat (MATLAB < v7.3)")
    except NotImplementedError:
        # Use robust h5py loader
        print("  File is MATLAB v7.3 format, using h5py loader...")
        data = load_matlab_v73(str(mat_path), verbose=False)
    
    return data


def extract_scalar(val: Any) -> float:
    """Extract scalar value from various nested structures."""
    if np.isscalar(val):
        return float(val)
    elif isinstance(val, np.ndarray):
        if val.ndim == 0:
            return float(val.item())
        else:
            return float(val.flatten()[0])
    else:
        return float(val)


def extract_string(val: Any) -> str:
    """Robustly extract a Python string from various MATLAB/HDF5 representations."""
    # Direct string
    if isinstance(val, str):
        return val
    # Bytes/bytearray
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode('utf-8')
        except Exception:
            return str(val)
    # NumPy arrays
    if isinstance(val, np.ndarray):
        # Empty array
        if val.size == 0:
            return ''
        # If array of strings/bytes
        if val.dtype.kind in ('U', 'S', 'O'):
            try:
                # Prefer single element
                if val.size == 1:
                    return extract_string(val.item())
                # Take the first non-empty element
                flat = val.flatten()
                for item in flat:
                    s = extract_string(item)
                    if s:
                        return s
                return extract_string(flat[0])
            except Exception:
                return str(val)
        # Char arrays from MATLAB sometimes appear as numeric arrays; fallback to str
        return str(val)
    # Fallback
    return str(val)


def parse_const(const_raw: Any) -> Dict[str, Any]:
    """
    Parse const structure to Python dictionary.
    
    Parameters
    ----------
    const_raw : Any
        Raw const structure from MATLAB
        
    Returns
    -------
    const : dict
        Parsed const dictionary
    """
    const = {}
    
    if isinstance(const_raw, dict):
        # Already a dict (from h5py loader)
        for key in const_raw:
            val = const_raw[key]
            if isinstance(val, np.ndarray):
                if val.dtype == np.object_:
                    # Handle object arrays
                    if val.size == 1:
                        const[key] = val.item()
                    else:
                        const[key] = val
                else:
                    const[key] = val
            else:
                const[key] = val
    else:
        # Structured array (from scipy loader)
        for field in const_raw.dtype.names:
            val = const_raw[field][0, 0]
            const[field] = val
    
    # Extract scalar values where appropriate
    for key in ['N_ele', 'N_eig', 'a', 't', 'sigma_eig', 
                'E_min', 'E_max', 'rho_min', 'rho_max', 
                'poisson_min', 'poisson_max']:
        if key in const:
            const[key] = extract_scalar(const[key])
    
    # Handle N_pix (can be scalar or array)
    if 'N_pix' in const:
        N_pix_raw = const['N_pix']
        if isinstance(N_pix_raw, np.ndarray):
            const['N_pix'] = [int(x) for x in N_pix_raw.flatten()]
            if len(const['N_pix']) == 1:
                const['N_pix'] = int(const['N_pix'][0])
        else:
            const['N_pix'] = int(extract_scalar(N_pix_raw))
    
    # Handle N_wv (can be scalar or array)
    if 'N_wv' in const:
        N_wv_raw = const['N_wv']
        if isinstance(N_wv_raw, np.ndarray):
            const['N_wv'] = [int(x) for x in N_wv_raw.flatten()]
        else:
            const['N_wv'] = [int(extract_scalar(N_wv_raw)), 
                             int(extract_scalar(N_wv_raw))]
    
    # Handle design_scale
    if 'design_scale' in const:
        const['design_scale'] = extract_string(const['design_scale']).strip()
    
    return const


def embed_2const_wavelet(wavevector_x: np.ndarray, wavevector_y: np.ndarray, 
                         size: int = 32) -> np.ndarray:
    """
    Embed 2 constant wavevectors into spatial domain using wavelet-like encoding.
    
    This creates a spatial representation of wavevector data for neural network input.
    Follows the approach from NO_utils_multiple.embed_2const_wavelet.
    
    Parameters
    ----------
    wavevector_x : np.ndarray
        X-components of wavevectors (N_wv,)
    wavevector_y : np.ndarray
        Y-components of wavevectors (N_wv,)
    size : int
        Spatial resolution (N_pix)
        
    Returns
    -------
    waveforms : np.ndarray
        Wavelet-embedded representation (N_wv, size, size)
    """
    N_wv = len(wavevector_x)
    waveforms = np.zeros((N_wv, size, size), dtype=np.float32)
    
    # Create spatial grid
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # For each wavevector, create spatial representation
    for i in range(N_wv):
        kx = wavevector_x[i]
        ky = wavevector_y[i]
        
        # Simple sinusoidal encoding (can be replaced with actual wavelet if needed)
        waveforms[i] = np.sin(2 * np.pi * kx * X) * np.cos(2 * np.pi * ky * Y)
    
    return waveforms


def embed_integer_wavelet(bands: np.ndarray, size: int = 32) -> np.ndarray:
    """
    Embed integer band indices into spatial domain using wavelet-like encoding.
    
    Follows the approach from NO_utils_multiple.embed_integer_wavelet.
    
    Parameters
    ----------
    bands : np.ndarray
        Band indices (e.g., [1, 2, 3, 4, 5, 6])
    size : int
        Spatial resolution (N_pix)
        
    Returns
    -------
    bands_fft : np.ndarray
        Wavelet-embedded representation (N_bands, size, size)
    """
    N_bands = len(bands)
    bands_fft = np.zeros((N_bands, size, size), dtype=np.float32)
    
    # Create spatial grid
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # For each band, create spatial representation
    for i, band in enumerate(bands):
        # Simple encoding based on band number
        bands_fft[i] = np.sin(band * np.pi * X) * np.cos(band * np.pi * Y)
    
    return bands_fft




def reshape_eigenvectors_to_spatial(eigenvector_data: np.ndarray, N_pix: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape eigenvector data from DOF format to spatial format and split into x/y components.
    
    Follows the approach from dataset_conversion_reduction.ipynb.
    
    Parameters
    ----------
    eigenvector_data : np.ndarray
        Full eigenvector array (N_dof, N_wv, N_eig, N_struct)
        where N_dof = 2 * N_nodes^2 and eigenvectors are [u1, v1, u2, v2, ...]
        
    N_pix : int
        Pixel resolution
        
    Returns
    -------
    eigvec_x : np.ndarray
        X-displacement components reshaped to (N_struct, N_wv, N_eig, N_pix, N_pix)
    eigvec_y : np.ndarray
        Y-displacement components reshaped to (N_struct, N_wv, N_eig, N_pix, N_pix)
    """
    N_dof, N_wv, N_eig, N_struct = eigenvector_data.shape
    N_nodes = int(np.sqrt(N_dof / 2))
    
    # Extract x and y components (interleaved in DOF)
    eigvec_x_dof = eigenvector_data[0::2, :, :, :]  # X-displacements
    eigvec_y_dof = eigenvector_data[1::2, :, :, :]  # Y-displacements
    
    # Reshape from node-based to spatial grid
    # (N_nodes^2, N_wv, N_eig, N_struct) -> (N_struct, N_wv, N_eig, N_pix, N_pix)
    eigvec_x = np.zeros((N_struct, N_wv, N_eig, N_pix, N_pix), dtype=eigenvector_data.dtype)
    eigvec_y = np.zeros((N_struct, N_wv, N_eig, N_pix, N_pix), dtype=eigenvector_data.dtype)
    
    # Reshape each structure's eigenvectors
    for struct_idx in range(N_struct):
        for wv_idx in range(N_wv):
            for eig_idx in range(N_eig):
                # Reshape from 1D node array to 2D spatial grid
                eigvec_x[struct_idx, wv_idx, eig_idx, :, :] = \
                    eigvec_x_dof[:, wv_idx, eig_idx, struct_idx].reshape(N_nodes, N_nodes)[:N_pix, :N_pix]
                eigvec_y[struct_idx, wv_idx, eig_idx, :, :] = \
                    eigvec_y_dof[:, wv_idx, eig_idx, struct_idx].reshape(N_nodes, N_nodes)[:N_pix, :N_pix]
    
    return eigvec_x, eigvec_y


def main(mat_path: str):
    """
    Main conversion function following dataset_conversion_reduction.ipynb format.
    
    Parameters
    ----------
    mat_path : str
        Path to input .mat file
    """
    mat_path = Path(mat_path)
    
    if not mat_path.exists():
        print(f"ERROR: File not found: {mat_path}")
        return
    
    print("=" * 80)
    print("MATLAB to NumPy/PyTorch Conversion (dataset_conversion_reduction format)")
    print("=" * 80)
    
    # Create output directory as a fixed subfolder named 'python'
    output_dir = mat_path.parent / 'python'
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load dataset
    data = load_dataset(mat_path)
    
    # Check required keys
    required_keys = ['designs']
    for key in required_keys:
        if key not in data:
            print(f"ERROR: Required key '{key}' not found in dataset")
            return
    
    print("\nDataset keys:", list(data.keys()))
    
    # Extract and parse const
    if 'const' in data:
        const = parse_const(data['const'])
        print(f"\nParsed const with {len(const)} fields")
        N_pix = const.get('N_pix', 32)
        if isinstance(N_pix, list):
            N_pix = N_pix[0]
        N_eig = const.get('N_eig', 6)
    else:
        print("WARNING: 'const' not found in dataset")
        const = {}
        N_pix = 32  # Default
        N_eig = 6
    
    # Extract design_params
    if 'design_params' in data:
        design_params_raw = data['design_params']
        # Convert to simple array format
        design_params = np.array([N_eig, N_pix], dtype=np.float64)
        print(f"Found design_params: {design_params}")
    else:
        print("WARNING: 'design_params' not found")
        design_params = np.array([N_eig, N_pix], dtype=np.float64)
    
    # Extract designs
    designs = data['designs']
    print(f"\nOriginal designs shape: {designs.shape}")
    
    # Determine data structure and reshape to (N_pix, N_pix, 3, N_struct)
    if designs.shape[0] < designs.shape[2]:
        # HDF5: (N_struct, 3, N_pix, N_pix) -> (N_pix, N_pix, 3, N_struct)
        print("Detected HDF5 format, transposing...")
        designs = designs.transpose(2, 3, 1, 0)
    
    # Take only first pane (elastic modulus) following notebook approach
    # (N_pix, N_pix, 3, N_struct) -> (N_struct, N_pix, N_pix)
    print("Taking first pane (elastic modulus) only...")
    designs_first_pane = designs[:, :, 0, :].transpose(2, 0, 1)
    N_struct = designs_first_pane.shape[0]
    print(f"  Designs shape after extraction: {designs_first_pane.shape}")
    print(f"  N_struct: {N_struct}, N_pix: {N_pix}")
    
    # Save designs (first pane only, float16)
    print("\nSaving designs.npy (float16)...")
    np.save(output_dir / 'designs.npy', designs_first_pane.astype(np.float16))
    print(f"  Saved: designs.npy {designs_first_pane.shape}")
    
    # Save design_params (float64)
    print("\nSaving design_params.npy (float64)...")
    np.save(output_dir / 'design_params.npy', design_params)
    print(f"  Saved: design_params.npy {design_params.shape}")
    
    # Extract wavevector data
    if 'WAVEVECTOR_DATA' in data:
        wavevector_data = data['WAVEVECTOR_DATA']
        # Check format and transpose to (N_struct, N_wv, 2)
        if wavevector_data.shape[0] == N_struct and wavevector_data.shape[1] == 2:
            # HDF5: (N_struct, 2, N_wv) -> (N_struct, N_wv, 2)
            wavevector_data = wavevector_data.transpose(0, 2, 1)
        elif wavevector_data.shape[2] == N_struct:
            # scipy: (N_wv, 2, N_struct) -> (N_struct, N_wv, 2)
            wavevector_data = wavevector_data.transpose(2, 0, 1)
        
        print(f"\nWavevector data shape: {wavevector_data.shape}")
        N_wv = wavevector_data.shape[1]
        
        # Save wavevectors (float16)
        print("Saving wavevectors.npy (float16)...")
        np.save(output_dir / 'wavevectors.npy', wavevector_data.astype(np.float16))
        print(f"  Saved: wavevectors.npy {wavevector_data.shape}")
        
        # Compute waveforms from first structure's wavevectors
        print("\nComputing waveforms using wavelet embedding...")
        waveforms = embed_2const_wavelet(
            wavevector_data[0, :, 0],  # X-components of first structure
            wavevector_data[0, :, 1],  # Y-components of first structure  
            size=N_pix
        )
        print(f"  Waveforms shape: {waveforms.shape}")
        
        # Save waveforms (float16)
        np.save(output_dir / 'waveforms.npy', waveforms.astype(np.float16))
        print(f"  Saved: waveforms.npy")
    
    # Extract eigenvalue data
    if 'EIGENVALUE_DATA' in data:
        eigenvalue_data = data['EIGENVALUE_DATA']
        # Reshape to (N_struct, N_wv, N_eig)
        if eigenvalue_data.shape[0] == N_struct and eigenvalue_data.shape[1] < eigenvalue_data.shape[2]:
            # HDF5: (N_struct, N_eig, N_wv) -> (N_struct, N_wv, N_eig)
            eigenvalue_data = eigenvalue_data.transpose(0, 2, 1)
        elif eigenvalue_data.shape[2] == N_struct:
            # scipy: (N_wv, N_eig, N_struct) -> (N_struct, N_wv, N_eig)
            eigenvalue_data = eigenvalue_data.transpose(2, 0, 1)
        
        print(f"\nEigenvalue data shape: {eigenvalue_data.shape}")
        
        # Save eigenvalue data (keep original dtype)
        np.save(output_dir / 'eigenvalue_data.npy', eigenvalue_data)
        print(f"  Saved: eigenvalue_data.npy")
        
        # Compute bands_fft using wavelet embedding
        print("\nComputing bands_fft using wavelet embedding...")
        bands = np.arange(1, N_eig + 1)  # Band indices [1, 2, 3, ..., N_eig]
        bands_fft = embed_integer_wavelet(bands, size=N_pix)
        print(f"  Bands FFT shape: {bands_fft.shape}")
        
        # Save bands_fft (float16)
        np.save(output_dir / 'bands_fft.npy', bands_fft.astype(np.float16))
        print(f"  Saved: bands_fft.npy")
    
    # Extract eigenvector data
    if 'EIGENVECTOR_DATA' in data:
        eigenvector_data = data['EIGENVECTOR_DATA']
        
        # Handle structured dtype (real/imag)
        if eigenvector_data.dtype.names and 'real' in eigenvector_data.dtype.names:
            eigenvector_data = eigenvector_data['real'] + 1j * eigenvector_data['imag']
        
        print(f"\nOriginal eigenvector shape: {eigenvector_data.shape}")
        
        # Reshape based on format
        if eigenvector_data.shape[0] == N_struct:
            # HDF5: (N_struct, N_eig, N_wv, N_dof) -> (N_dof, N_wv, N_eig, N_struct)
            eigenvector_data = eigenvector_data.transpose(3, 2, 1, 0)
        
        print(f"  Eigenvector shape after transpose: {eigenvector_data.shape}")
        
        # Reshape to spatial format and split into x/y
        print("Reshaping eigenvectors to spatial format...")
        eigvec_x, eigvec_y = reshape_eigenvectors_to_spatial(eigenvector_data, N_pix)
        
        print(f"  Eigenvector X shape: {eigvec_x.shape}")
        print(f"  Eigenvector Y shape: {eigvec_y.shape}")
        
        # Save eigenvector components as complex64
        np.save(output_dir / 'eigenvector_data_x.npy', eigvec_x.astype(np.complex64))
        np.save(output_dir / 'eigenvector_data_y.npy', eigvec_y.astype(np.complex64))
        print(f"  Saved: eigenvector_data_x.npy")
        print(f"  Saved: eigenvector_data_y.npy")
    
    # Create a summary file
    summary = {
        'source_file': str(mat_path),
        'N_struct': N_struct,
        'N_pix': N_pix,
        'N_eig': N_eig,
        'N_wv': N_wv if 'WAVEVECTOR_DATA' in data else 'N/A',
        'files_created': sorted([f.name for f in output_dir.iterdir()]),
        'format': 'dataset_conversion_reduction.ipynb compatible',
    }
    
    with open(output_dir / 'conversion_summary.txt', 'w') as f:
        f.write("NumPy Conversion Summary (dataset_conversion_reduction format)\n")
        f.write("=" * 80 + "\n\n")
        f.write("This dataset follows the format from dataset_conversion_reduction.ipynb\n")
        f.write("Shapes:\n")
        f.write(f"  - designs: (N_struct={N_struct}, N_pix={N_pix}, N_pix={N_pix})\n")
        f.write(f"  - wavevectors: (N_struct={N_struct}, N_wv={summary['N_wv']}, 2)\n")
        f.write(f"  - waveforms: (N_wv={summary['N_wv']}, N_pix={N_pix}, N_pix={N_pix})\n")
        f.write(f"  - eigenvalue_data: (N_struct={N_struct}, N_wv={summary['N_wv']}, N_eig={N_eig})\n")
        f.write(f"  - eigenvector_data_x/y: (N_struct={N_struct}, N_wv={summary['N_wv']}, N_eig={N_eig}, N_pix={N_pix}, N_pix={N_pix})\n")
        f.write(f"  - bands_fft: (N_eig={N_eig}, N_pix={N_pix}, N_pix={N_pix})\n\n")
        f.write("Files created:\n")
        for fname in summary['files_created']:
            fpath = output_dir / fname
            if fpath.exists():
                size_mb = fpath.stat().st_size / (1024 * 1024)
                f.write(f"  - {fname:30s} ({size_mb:8.2f} MB)\n")
    
    print("\n" + "=" * 80)
    print("Conversion Summary")
    print("=" * 80)
    print(f"Source: {mat_path}")
    print(f"Output: {output_dir}")
    print(f"Format: dataset_conversion_reduction.ipynb compatible")
    print(f"\nData shapes:")
    print(f"  designs: ({N_struct}, {N_pix}, {N_pix})")
    print(f"  wavevectors: ({N_struct}, {summary['N_wv']}, 2)")
    print(f"  waveforms: ({summary['N_wv']}, {N_pix}, {N_pix})")
    print(f"  eigenvalue_data: ({N_struct}, {summary['N_wv']}, {N_eig})")
    print(f"  eigenvector_data_x/y: ({N_struct}, {summary['N_wv']}, {N_eig}, {N_pix}, {N_pix})")
    print(f"  bands_fft: ({N_eig}, {N_pix}, {N_pix})")
    print(f"\nFiles created: {len(summary['files_created'])}")
    for fname in summary['files_created']:
        fpath = output_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  - {fname:30s} ({size_mb:8.2f} MB)")
    print("=" * 80)
    print("\nConversion complete!")
    print("\nNote: This format matches dataset_conversion_reduction.ipynb")
    print("      Designs contain only elastic modulus (first pane)")
    print("      Waveforms and bands_fft use wavelet embedding")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default dataset path (modify as needed)
        # Absolute path provided by user
        default_mat_path = Path(r"D:\Research\NO-2D-Metamaterials\generate_dispersion_dataset_Han\OUTPUT\output 13-Oct-2025 23-22-59\continuous 13-Oct-2025 23-22-59.mat")

        print("No path provided. Using default dataset path:")
        print(f"  {default_mat_path}")
        main(str(default_mat_path))
    else:
        mat_path = sys.argv[1]
        main(mat_path)

