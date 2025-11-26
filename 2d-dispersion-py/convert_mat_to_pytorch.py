"""
Convert MATLAB dataset to Reduced PyTorch Format (matching matlab_to_reduced_pt.ipynb)

This script converts MATLAB .mat files directly to reduced, wavelet-embedded, float16 PyTorch .pt format,
matching the exact functionality of matlab_to_reduced_pt.ipynb.

**Features:**
- Uses NO_utils.extract_data() for loading MATLAB files
- Uses NO_utils_multiple.embed_2const_wavelet() and embed_integer_wavelet() for wavelet embedding
- Applies dataset reduction by random sampling (wavevectors and bands)
- Converts to float16 precision (or float8_e4m3fn if requested)
- Saves as PyTorch .pt files

**Output:**
- displacements_dataset.pt: TensorDataset with eigenvector components (real/imag, x/y)
- reduced_indices.pt: List of (design_idx, wavevector_idx, band_idx) tuples
- geometries_full.pt: Design arrays (N_struct, N_pix, N_pix) [float16]
- waveforms_full.pt: Wavelet-embedded wavevectors (N_wv, N_pix, N_pix) [float16]
- wavevectors_full.pt: Wavevector data (N_struct, N_wv, 2) [float16]
- band_fft_full.pt: Wavelet-embedded band indices (N_bands, N_pix, N_pix) [float16]
- design_params_full.pt: Design parameters [float16]

Usage:
    python convert_mat_to_pytorch.py <path_to_single_mat_file> [--output OUTPUT] [--wvr WVR] [--br BR] [--do DO] [--e4m3]

Note:
    This script processes a SINGLE .mat file. If a folder contains multiple .mat files,
    you must specify the full path to the specific file you want to convert.

Example:
    python convert_mat_to_pytorch.py ../OUTPUT/test\ dataset/out_binarized_1.mat --output ../data_debug --wvr 0.66 --br 0.5
"""

import numpy as np
import torch
import tempfile
import shutil
import time
import random
from pathlib import Path
import sys
import argparse

# Add parent directory to path to import NO_utils modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import required modules
try:
    import NO_utils
    import NO_utils_multiple
except ImportError:
    print("ERROR: Could not import NO_utils or NO_utils_multiple")
    print("Make sure these modules are in the parent directory")
    sys.exit(1)


def convert_matlab_to_reduced_pt(mat_file_path, output_base_path, WVR=1.0, BR=1.0, DO=0, use_e4m3=False):
    """
    Convert a single MATLAB .mat file to reduced PyTorch format.
    
    This function matches the exact functionality from matlab_to_reduced_pt.ipynb.
    
    Parameters
    ----------
    mat_file_path : Path
        Path to the .mat file to convert
    output_base_path : Path
        Base output folder (subfolder will be created based on file name)
    WVR : float
        Wavevector Reduction Ratio (default: 1.0, i.e., no reduction)
    BR : float
        Band Reduction Ratio (default: 1.0, i.e., no reduction)
    DO : int
        Dataset Offset (default: 0)
    use_e4m3 : bool
        Use float8_e4m3fn format instead of float16 (default: False)
    
    Returns
    -------
    dict
        Information about the converted dataset with keys:
        - file_name: Name of the converted file
        - output_path: Path where files were saved
        - samples: Number of reduced samples
        - size_mb: Total size in MB
        - indices: List of reduced indices
    """
    file_name = mat_file_path.stem  # Name without .mat extension
    output_path = output_base_path / file_name
    
    print("\n" + "=" * 80)
    print(f"Processing: {mat_file_path.name}")
    print("=" * 80)
    
    # Step 1: Load MATLAB Dataset
    print("\nStep 1: Loading MATLAB Dataset")
    start_time = time.time()
    
    # Create a temporary directory with only this .mat file
    # (because NO_utils.extract_data expects a folder with a single .mat file)
    temp_dir = tempfile.mkdtemp(prefix=f"matlab_convert_{file_name}_")
    temp_mat_path = Path(temp_dir) / mat_file_path.name
    
    try:
        # Copy the .mat file to temporary directory
        shutil.copy2(mat_file_path, temp_mat_path)
        
        # Extract data from MATLAB file using NO_utils.extract_data()
        (designs, design_params, n_designs, n_panes, design_res,
         WAVEVECTOR_DATA, WAVEFORM_DATA, n_dim, n_wavevectors,
         EIGENVALUE_DATA, n_bands, EIGENVECTOR_DATA_x,
         EIGENVECTOR_DATA_y, const, N_struct,
         imag_tol, rng_seed_offset) = NO_utils.extract_data(temp_dir)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Take first pane of designs (elastic modulus only)
    designs = designs[:, 0, :, :]
    
    elapsed_time = time.time() - start_time
    print(f"Dataset loaded in {elapsed_time:.2f} seconds")
    print(f"  Original shapes: designs={designs.shape}, eigenvectors={EIGENVECTOR_DATA_x.shape}")
    
    # Step 2: Apply Wavelet Embedding
    print("\nStep 2: Applying Wavelet Embedding")
    waveforms = NO_utils_multiple.embed_2const_wavelet(
        WAVEVECTOR_DATA[0, :, 0],  # X-components of first structure
        WAVEVECTOR_DATA[0, :, 1],  # Y-components of first structure
        size=design_res
    )
    bands = np.arange(1, n_bands + 1)
    bands_fft = NO_utils_multiple.embed_integer_wavelet(bands, size=design_res)
    print(f"  Embedded shapes: waveforms={waveforms.shape}, bands_fft={bands_fft.shape}")
    
    # Step 3: Reduce Dataset
    print("\nStep 3: Reducing Dataset")
    if DO != 0:
        design_params[:, 0] += DO
    
    waveforms_reduced_amount = int(waveforms.shape[0] * WVR)
    bands_fft_reduced_amount = int(bands_fft.shape[0] * BR)
    samples_reduced_amount = int(designs.shape[0] * waveforms_reduced_amount * bands_fft_reduced_amount)
    
    print(f"  Reduction: {waveforms.shape[0]} waveforms -> {waveforms_reduced_amount}")
    print(f"             {bands_fft.shape[0]} bands -> {bands_fft_reduced_amount}")
    print(f"             Total samples: {samples_reduced_amount}")
    
    # Generate indices (random if reduction is applied, otherwise sequential)
    reduced_indices_reserved = [None] * samples_reduced_amount
    current_idx = 0
    
    if WVR < 1.0 or BR < 1.0:
        # Random sampling for reduced dataset
        for d_idx in range(designs.shape[0]):
            waveform_indices = np.random.choice(waveforms.shape[0], size=waveforms_reduced_amount, replace=False)
            for w_idx in waveform_indices:
                band_indices = np.random.choice(bands_fft.shape[0], size=bands_fft_reduced_amount, replace=False)
                for b_idx in band_indices:
                    reduced_indices_reserved[current_idx] = (d_idx, w_idx, b_idx)
                    current_idx += 1
    else:
        # No reduction: use all combinations sequentially
        for d_idx in range(designs.shape[0]):
            for w_idx in range(waveforms.shape[0]):
                for b_idx in range(bands_fft.shape[0]):
                    reduced_indices_reserved[current_idx] = (d_idx, w_idx, b_idx)
                    current_idx += 1
    
    # Extract reduced eigenvectors
    eigenvector_data_x_reduced = EIGENVECTOR_DATA_x[
        [idx[0] for idx in reduced_indices_reserved],
        [idx[1] for idx in reduced_indices_reserved],
        [idx[2] for idx in reduced_indices_reserved]
    ]
    eigenvector_data_y_reduced = EIGENVECTOR_DATA_y[
        [idx[0] for idx in reduced_indices_reserved],
        [idx[1] for idx in reduced_indices_reserved],
        [idx[2] for idx in reduced_indices_reserved]
    ]
    print(f"  Reduced shapes: {eigenvector_data_x_reduced.shape}")
    
    # Step 4: Convert to PyTorch Tensors
    print("\nStep 4: Converting to PyTorch Tensors")
    if use_e4m3:
        print("  Using float8_e4m3fn format")
        eigenvector_x_real_tensor = torch.from_numpy(eigenvector_data_x_reduced.real).to(torch.float8_e4m3fn)
        eigenvector_x_imag_tensor = torch.from_numpy(eigenvector_data_x_reduced.imag).to(torch.float8_e4m3fn)
        eigenvector_y_real_tensor = torch.from_numpy(eigenvector_data_y_reduced.real).to(torch.float8_e4m3fn)
        eigenvector_y_imag_tensor = torch.from_numpy(eigenvector_data_y_reduced.imag).to(torch.float8_e4m3fn)
    else:
        print("  Using float16 format")
        eigenvector_x_real_tensor = torch.from_numpy(eigenvector_data_x_reduced.real).to(torch.float16)
        eigenvector_x_imag_tensor = torch.from_numpy(eigenvector_data_x_reduced.imag).to(torch.float16)
        eigenvector_y_real_tensor = torch.from_numpy(eigenvector_data_y_reduced.real).to(torch.float16)
        eigenvector_y_imag_tensor = torch.from_numpy(eigenvector_data_y_reduced.imag).to(torch.float16)
    
    designs_tensor = torch.from_numpy(designs).to(torch.float16)
    waveforms_tensor = torch.from_numpy(waveforms).to(torch.float16)
    wavevectors_tensor = torch.from_numpy(WAVEVECTOR_DATA).to(torch.float16)
    bands_fft_tensor = torch.from_numpy(bands_fft).to(torch.float16)
    
    design_params_clean = design_params.copy()
    if np.any(np.isnan(design_params_clean)):
        design_params_clean = np.nan_to_num(design_params_clean, nan=0)
    design_params_tensor = torch.from_numpy(design_params_clean.astype(np.float16))
    
    displacements_dataset = torch.utils.data.TensorDataset(
        eigenvector_x_real_tensor,
        eigenvector_x_imag_tensor,
        eigenvector_y_real_tensor,
        eigenvector_y_imag_tensor
    )
    
    # Step 5: Save Dataset
    print("\nStep 5: Saving Dataset")
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(displacements_dataset, output_path / "displacements_dataset.pt")
    torch.save(reduced_indices_reserved, output_path / "reduced_indices.pt")
    torch.save(designs_tensor, output_path / "geometries_full.pt")
    torch.save(waveforms_tensor, output_path / "waveforms_full.pt")
    torch.save(wavevectors_tensor, output_path / "wavevectors_full.pt")
    torch.save(bands_fft_tensor, output_path / "band_fft_full.pt")
    torch.save(design_params_tensor, output_path / "design_params_full.pt")
    
    total_size = sum(f.stat().st_size for f in output_path.glob("*.pt")) / (1024 * 1024)
    
    print(f"  Saved to: {output_path}")
    print(f"  Total size: {total_size:.2f} MB")
    
    return {
        'file_name': file_name,
        'output_path': output_path,
        'samples': samples_reduced_amount,
        'size_mb': total_size,
        'indices': reduced_indices_reserved
    }


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Convert MATLAB dataset to Reduced PyTorch Format (matching matlab_to_reduced_pt.ipynb)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('mat_path', help='Path to a single input .mat file (not a folder)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output base folder (default: data/, creates data/[filename]/ subfolder)')
    parser.add_argument('--wvr', type=float, default=1.0,
                        help='Wavevector Reduction Ratio (default: 1.0, i.e., no reduction)')
    parser.add_argument('--br', type=float, default=1.0,
                        help='Band Reduction Ratio (default: 1.0, i.e., no reduction)')
    parser.add_argument('--do', type=int, default=0,
                        help='Dataset Offset (default: 0)')
    parser.add_argument('--e4m3', action='store_true',
                        help='Use float8_e4m3fn format instead of float16')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility when reduction is applied (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility (only needed if reduction is applied)
    if args.wvr < 1.0 or args.br < 1.0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed set to {args.seed} for reproducible random sampling")
    
    # Convert paths
    mat_file_path = Path(args.mat_path)
    if not mat_file_path.exists():
        print(f"ERROR: File not found: {mat_file_path}")
        sys.exit(1)
    
    if not mat_file_path.is_file():
        print(f"ERROR: Path is not a file: {mat_file_path}")
        print("Please specify the full path to a single .mat file, not a folder.")
        sys.exit(1)
    
    if not mat_file_path.suffix.lower() == '.mat':
        print(f"ERROR: File is not a .mat file: {mat_file_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        # Default: data\ (the function will append the filename)
        output_base_path = Path("data")
    else:
        output_base_path = Path(args.output)
    
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MATLAB to Reduced PyTorch Converter")
    print("=" * 80)
    print(f"Input file: {mat_file_path}")
    print(f"Output folder: {output_base_path}")
    print(f"Reduction parameters: WVR={args.wvr}, BR={args.br}, DO={args.do}")
    if args.wvr < 1.0 or args.br < 1.0:
        print(f"  (Reduction enabled - random seed: {args.seed})")
    else:
        print(f"  (No reduction - full dataset)")
    print(f"Precision: {'float8_e4m3fn' if args.e4m3 else 'float16'}")
    print("=" * 80)
    
    # Convert the file
    try:
        result = convert_matlab_to_reduced_pt(
            mat_file_path,
            output_base_path,
            WVR=args.wvr,
            BR=args.br,
            DO=args.do,
            use_e4m3=args.e4m3
        )
        
        print("\n" + "=" * 80)
        print("Conversion Complete!")
        print("=" * 80)
        print(f"Output: {result['output_path']}")
        print(f"Samples: {result['samples']}")
        print(f"Size: {result['size_mb']:.2f} MB")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
