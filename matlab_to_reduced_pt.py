#!/usr/bin/env python3
"""
MATLAB to Reduced PyTorch Dataset Converter

This script converts MATLAB `.mat` files directly to reduced, wavelet-embedded, float16 PyTorch `.pt` format.

Features:
- Processes multiple `.mat` files in the same folder
- Each file gets its own output folder matching the input name
- Applies wavelet embedding to wavevectors and bands
- Reduces dataset by random sampling (wavevectors and bands)
- Converts to float16 precision
- Saves as PyTorch `.pt` files

Output: Reduced datasets ready for neural operator training, one folder per input file.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time
import random
import tempfile
import shutil
import argparse
import sys

# Custom utilities
try:
    import NO_utils_multiple
    import NO_utils
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Please ensure NO_utils.py and NO_utils_multiple.py are in the same directory or PYTHONPATH")
    sys.exit(1)


def convert_matlab_to_reduced_pt(mat_file_path, output_base_path, WVR, BR, DO, use_e4m3, file_index=None, total_files=None):
    """
    Convert a single MATLAB .mat file to reduced PyTorch format.
    
    Parameters:
    -----------
    mat_file_path : Path
        Path to the .mat file to convert
    output_base_path : Path
        Base output folder (subfolder will be created based on file name)
    WVR : float
        Wavevector Reduction Ratio
    BR : float
        Band Reduction Ratio
    DO : int
        Dataset Offset
    use_e4m3 : bool
        Use float8_e4m3fn format
    file_index : int, optional
        Current file index (for progress display)
    total_files : int, optional
        Total number of files (for progress display)
    
    Returns:
    --------
    dict : Information about the converted dataset
    """
    file_name = mat_file_path.stem  # Name without .mat extension
    output_path = output_base_path / file_name
    
    progress_str = f"[{file_index}/{total_files}]" if file_index is not None else ""
    
    print("\n" + "=" * 80)
    print(f"{progress_str} Processing: {mat_file_path.name}")
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
        
        # Extract data from MATLAB file
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
        WAVEVECTOR_DATA[0, :, 0], 
        WAVEVECTOR_DATA[0, :, 1], 
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
    
    # Generate random indices
    reduced_indices_reserved = [None] * samples_reduced_amount
    current_idx = 0
    
    for d_idx in range(designs.shape[0]):
        waveform_indices = np.random.choice(waveforms.shape[0], size=waveforms_reduced_amount, replace=False)
        for w_idx in waveform_indices:
            band_indices = np.random.choice(bands_fft.shape[0], size=bands_fft_reduced_amount, replace=False)
            for b_idx in band_indices:
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
    print("\nStep 4: Converting to PyTorch Tensors (Float16)")
    if use_e4m3:
        eigenvector_x_real_tensor = torch.from_numpy(eigenvector_data_x_reduced.real).to(torch.float8_e4m3fn)
        eigenvector_x_imag_tensor = torch.from_numpy(eigenvector_data_x_reduced.imag).to(torch.float8_e4m3fn)
        eigenvector_y_real_tensor = torch.from_numpy(eigenvector_data_y_reduced.real).to(torch.float8_e4m3fn)
        eigenvector_y_imag_tensor = torch.from_numpy(eigenvector_data_y_reduced.imag).to(torch.float8_e4m3fn)
    else:
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
    """Main function to process MATLAB files."""
    parser = argparse.ArgumentParser(
        description='Convert MATLAB .mat files to reduced PyTorch format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all .mat files in a folder with default settings
  python matlab_to_reduced_pt.py --input_folder "path/to/matlab/data" --output_folder "path/to/output"
  
  # Process with custom reduction ratios
  python matlab_to_reduced_pt.py --input_folder "path/to/matlab/data" --output_folder "path/to/output" --wvr 0.2 --br 0.5
  
  # Process only specific files
  python matlab_to_reduced_pt.py --input_folder "path/to/matlab/data" --output_folder "path/to/output" --files "file1.mat" "file2.mat"
        """
    )
    
    parser.add_argument(
        '--input_folder', '-i',
        type=str,
        required=True,
        help='Path to MATLAB dataset folder (containing one or more .mat files)'
    )
    
    parser.add_argument(
        '--output_folder', '-o',
        type=str,
        required=True,
        help='Base output folder (subfolders will be created for each .mat file)'
    )
    
    parser.add_argument(
        '--wvr',
        type=float,
        default=1.0,
        help='Wavevector Reduction Ratio (0.0-1.0, default: 1.0 = keep all)'
    )
    
    parser.add_argument(
        '--br',
        type=float,
        default=1.0,
        help='Band Reduction Ratio (0.0-1.0, default: 1.0 = keep all)'
    )
    
    parser.add_argument(
        '--do',
        type=int,
        default=0,
        help='Dataset Offset (add to geometry indices when combining datasets, default: 0)'
    )
    
    parser.add_argument(
        '--use_e4m3',
        action='store_true',
        help='Use float8_e4m3fn instead of float16 precision'
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        default=None,
        help='Process only specific files (e.g., --files "file1.mat" "file2.mat"). If not specified, processes all .mat files.'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Random seed: {SEED}')
    
    current_dir = os.getcwd()
    print(f'Current directory: {current_dir}')
    
    # Convert to Path objects
    matlab_input_path = Path(args.input_folder)
    output_base_path = Path(args.output_folder)
    
    print(f"\nMATLAB input folder: {matlab_input_path}")
    print(f"Output base folder: {output_base_path}")
    print(f"Reduction parameters: WVR={args.wvr}, BR={args.br}, DO={args.do}")
    print(f"Use e4m3 format: {args.use_e4m3}")
    
    # Validate input
    if not matlab_input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {matlab_input_path}")
    
    # Find all .mat files
    all_mat_files = sorted(matlab_input_path.glob("*.mat"))
    
    if args.files is not None:
        # Filter to only specified files
        process_only_set = set(Path(f).name for f in args.files)
        mat_files = [f for f in all_mat_files if f.name in process_only_set]
        print(f"\nFiltering: Processing {len(mat_files)} of {len(all_mat_files)} .mat files")
    else:
        mat_files = all_mat_files
    
    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in {matlab_input_path}")
    
    print(f"\nFound {len(mat_files)} .mat file(s) to process:")
    for i, mat_file in enumerate(mat_files, 1):
        print(f"  [{i}] {mat_file.name}")
    
    # Create output base folder
    output_base_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput base folder ready: {output_base_path}")
    
    # Process all MATLAB files
    print("\n" + "=" * 80)
    print("Processing All MATLAB Files")
    print("=" * 80)
    
    converted_datasets = []
    errors = []
    
    for i, mat_file in enumerate(mat_files, 1):
        try:
            result = convert_matlab_to_reduced_pt(
                mat_file,
                output_base_path,
                WVR=args.wvr,
                BR=args.br,
                DO=args.do,
                use_e4m3=args.use_e4m3,
                file_index=i,
                total_files=len(mat_files)
            )
            converted_datasets.append(result)
            print(f"  [OK] Successfully converted {mat_file.name}")
        except Exception as e:
            error_msg = f"  [ERROR] Failed to convert {mat_file.name}: {e}"
            print(error_msg)
            errors.append((mat_file.name, str(e)))
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("Conversion Summary")
    print("=" * 80)
    print(f"Successfully converted: {len(converted_datasets)}/{len(mat_files)} files")
    
    if converted_datasets:
        print("\nConverted datasets:")
        total_size = 0
        for result in converted_datasets:
            print(f"  - {result['file_name']:30s} {result['samples']:8d} samples, {result['size_mb']:8.2f} MB")
            total_size += result['size_mb']
        print(f"\nTotal size: {total_size:.2f} MB")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for filename, error in errors:
            print(f"  - {filename}: {error}")
    
    print("=" * 80)
    
    return 0 if len(errors) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

