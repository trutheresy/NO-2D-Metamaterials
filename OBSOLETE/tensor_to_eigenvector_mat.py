"""
Script to convert a tensor of shape (4, n, 32, 32) to EIGENVECTOR_DATA.mat file.

The input tensor should have:
- Index 0: eigenvector_x_real
- Index 1: eigenvector_x_imag
- Index 2: eigenvector_y_real
- Index 3: eigenvector_y_imag

Uses the same mapping logic as reduced_pt_to_matlab.ipynb to reconstruct the full
EIGENVECTOR_DATA structure from reduced samples.
"""

import numpy as np
import torch
import h5py
import scipy.io as sio
from pathlib import Path
import argparse


def tensor_to_eigenvector_mat(
    tensor, 
    reduced_indices_path, 
    output_path,
    n_designs=None,
    n_wavevectors=None,
    n_bands=None,
    design_res=32
):
    """
    Convert a tensor of shape (4, n, design_res, design_res) to EIGENVECTOR_DATA.mat
    
    Parameters:
    -----------
    tensor : torch.Tensor or np.ndarray
        Tensor of shape (4, n, design_res, design_res) containing:
        [0]: eigenvector_x_real
        [1]: eigenvector_x_imag
        [2]: eigenvector_y_real
        [3]: eigenvector_y_imag
    reduced_indices_path : str or Path
        Path to reduced_indices.pt file
    output_path : str or Path
        Path to output EIGENVECTOR_DATA.mat file
    n_designs : int, optional
        Number of designs. If None, will be inferred from reduced_indices
    n_wavevectors : int, optional
        Number of wavevectors. If None, will be inferred from reduced_indices
    n_bands : int, optional
        Number of bands. If None, will be inferred from reduced_indices
    design_res : int
        Design resolution (default: 32)
    
    Returns:
    --------
    dict : Information about the conversion
    """
    print("=" * 80)
    print("Converting Tensor to EIGENVECTOR_DATA.mat")
    print("=" * 80)
    
    # Handle TensorDataset (from displacements_dataset.pt)
    if isinstance(tensor, torch.utils.data.TensorDataset):
        print("  Detected TensorDataset, extracting tensors...")
        if len(tensor.tensors) != 4:
            raise ValueError(f"Expected TensorDataset with 4 tensors, got {len(tensor.tensors)}")
        # Stack tensors into (4, n, 32, 32) format
        tensor = torch.stack(tensor.tensors, dim=0)
        print(f"  Stacked to shape: {tensor.shape}")
    
    # Convert tensor to numpy if needed
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    
    # Validate tensor shape
    if tensor.shape[0] != 4:
        raise ValueError(f"Expected tensor first dimension to be 4, got {tensor.shape[0]}")
    if len(tensor.shape) != 4:
        raise ValueError(f"Expected 4D tensor, got {len(tensor.shape)}D")
    if tensor.shape[2] != design_res or tensor.shape[3] != design_res:
        raise ValueError(f"Expected spatial dimensions ({design_res}, {design_res}), got ({tensor.shape[2]}, {tensor.shape[3]})")
    
    n_samples = tensor.shape[1]
    print(f"\nInput tensor shape: {tensor.shape}")
    print(f"  Number of samples: {n_samples}")
    
    # Extract components
    eigenvector_x_real = tensor[0, :, :, :]
    eigenvector_x_imag = tensor[1, :, :, :]
    eigenvector_y_real = tensor[2, :, :, :]
    eigenvector_y_imag = tensor[3, :, :, :]
    
    # Load reduced indices
    print(f"\nLoading reduced indices from: {reduced_indices_path}")
    reduced_indices = torch.load(reduced_indices_path, map_location='cpu')
    
    if len(reduced_indices) != n_samples:
        raise ValueError(
            f"Number of samples in tensor ({n_samples}) doesn't match "
            f"number of indices ({len(reduced_indices)})"
        )
    
    # Infer dimensions from reduced_indices if not provided
    if n_designs is None or n_wavevectors is None or n_bands is None:
        print("\nInferring dimensions from reduced_indices...")
        max_d_idx = max(int(idx[0]) if isinstance(idx[0], torch.Tensor) else int(idx[0]) 
                       for idx in reduced_indices) + 1
        max_w_idx = max(int(idx[1]) if isinstance(idx[1], torch.Tensor) else int(idx[1]) 
                       for idx in reduced_indices) + 1
        max_b_idx = max(int(idx[2]) if isinstance(idx[2], torch.Tensor) else int(idx[2]) 
                       for idx in reduced_indices) + 1
        
        if n_designs is None:
            n_designs = max_d_idx
        if n_wavevectors is None:
            n_wavevectors = max_w_idx
        if n_bands is None:
            n_bands = max_b_idx
        
        print(f"  Inferred: n_designs={n_designs}, n_wavevectors={n_wavevectors}, n_bands={n_bands}")
    
    print(f"\nDimensions:")
    print(f"  n_designs: {n_designs}")
    print(f"  n_wavevectors: {n_wavevectors}")
    print(f"  n_bands: {n_bands}")
    print(f"  design_res: {design_res}")
    
    # Step 1: Reconstruct Full EIGENVECTOR_DATA
    print("\nStep 1: Reconstructing Full EIGENVECTOR_DATA")
    
    # Initialize full eigenvector arrays (fill with zeros for missing entries)
    EIGENVECTOR_DATA_x_full = np.zeros((n_designs, n_wavevectors, n_bands, design_res, design_res), 
                                        dtype=np.complex64)
    EIGENVECTOR_DATA_y_full = np.zeros((n_designs, n_wavevectors, n_bands, design_res, design_res), 
                                        dtype=np.complex64)
    
    # Place reduced eigenvectors at correct indices
    for sample_idx, (d_idx, w_idx, b_idx) in enumerate(reduced_indices):
        # Convert indices to int if they're tensors
        d_idx = int(d_idx) if isinstance(d_idx, torch.Tensor) else int(d_idx)
        w_idx = int(w_idx) if isinstance(w_idx, torch.Tensor) else int(w_idx)
        b_idx = int(b_idx) if isinstance(b_idx, torch.Tensor) else int(b_idx)
        
        # Validate indices
        if d_idx >= n_designs or w_idx >= n_wavevectors or b_idx >= n_bands:
            print(f"  Warning: Sample {sample_idx} has out-of-range indices: ({d_idx}, {w_idx}, {b_idx})")
            continue
        
        # Reconstruct complex eigenvectors
        eigenvector_x = eigenvector_x_real[sample_idx] + 1j * eigenvector_x_imag[sample_idx]
        eigenvector_y = eigenvector_y_real[sample_idx] + 1j * eigenvector_y_imag[sample_idx]
        
        # Place in full array
        EIGENVECTOR_DATA_x_full[d_idx, w_idx, b_idx, :, :] = eigenvector_x
        EIGENVECTOR_DATA_y_full[d_idx, w_idx, b_idx, :, :] = eigenvector_y
    
    print(f"  Reconstructed EIGENVECTOR_DATA_x shape: {EIGENVECTOR_DATA_x_full.shape}")
    print(f"  Reconstructed EIGENVECTOR_DATA_y shape: {EIGENVECTOR_DATA_y_full.shape}")
    
    # Step 2: Combine x and y eigenvectors into single array
    print("\nStep 2: Combining x and y eigenvectors")
    
    # Reshape to (n_designs, n_wavevectors, n_bands, 2*design_res*design_res)
    EIGENVECTOR_DATA_x_flat = EIGENVECTOR_DATA_x_full.reshape(n_designs, n_wavevectors, n_bands, -1)
    EIGENVECTOR_DATA_y_flat = EIGENVECTOR_DATA_y_full.reshape(n_designs, n_wavevectors, n_bands, -1)
    
    # Interleave x and y components
    n_dof = 2 * design_res * design_res
    EIGENVECTOR_DATA_combined = np.zeros((n_designs, n_wavevectors, n_bands, n_dof), dtype=np.complex64)
    
    # Interleave manually
    EIGENVECTOR_DATA_combined[:, :, :, 0::2] = EIGENVECTOR_DATA_x_flat
    EIGENVECTOR_DATA_combined[:, :, :, 1::2] = EIGENVECTOR_DATA_y_flat
    
    # Transpose to match MATLAB format: (n_designs, n_eig, n_wv, n_dof)
    EIGENVECTOR_DATA = EIGENVECTOR_DATA_combined.transpose(0, 2, 1, 3)
    
    print(f"  Combined EIGENVECTOR_DATA shape: {EIGENVECTOR_DATA.shape}")
    
    # Step 3: Save as MATLAB v7.3 format
    print(f"\nStep 3: Saving to {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to float32 (single precision) to match original MATLAB format
    EIGENVECTOR_DATA_real = EIGENVECTOR_DATA.real.astype(np.float32)
    EIGENVECTOR_DATA_imag = EIGENVECTOR_DATA.imag.astype(np.float32)
    
    # Create structured array with compound dtype (matches MATLAB format)
    # This is how MATLAB v7.3 stores complex arrays
    structured_dtype = np.dtype([('real', np.float32), ('imag', np.float32)])
    EIGENVECTOR_DATA_structured = np.empty(EIGENVECTOR_DATA.shape, dtype=structured_dtype)
    EIGENVECTOR_DATA_structured['real'] = EIGENVECTOR_DATA_real
    EIGENVECTOR_DATA_structured['imag'] = EIGENVECTOR_DATA_imag
    
    # Save using h5py with proper MATLAB structure
    with h5py.File(output_path, 'w') as f:
        # Create dataset with structured dtype (compound datatype)
        dset = f.create_dataset(
            'EIGENVECTOR_DATA',
            data=EIGENVECTOR_DATA_structured,
            dtype=structured_dtype
        )
        # Add MATLAB_class attribute to indicate it's a single-precision complex array
        dset.attrs['MATLAB_class'] = np.bytes_(b'single')
    
    print("  Saved using h5py with structured array (MATLAB v7.3 format)")
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"  Saved to: {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    
    return {
        'output_path': output_path,
        'file_size_mb': file_size,
        'n_designs': n_designs,
        'n_wavevectors': n_wavevectors,
        'n_bands': n_bands,
        'EIGENVECTOR_DATA_shape': EIGENVECTOR_DATA.shape
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert tensor (4, n, 32, 32) to EIGENVECTOR_DATA.mat file'
    )
    parser.add_argument(
        'tensor_path',
        type=str,
        help='Path to input tensor file (.pt or .npy) with shape (4, n, 32, 32)'
    )
    parser.add_argument(
        'reduced_indices_path',
        type=str,
        help='Path to reduced_indices.pt file'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path to output EIGENVECTOR_DATA.mat file'
    )
    parser.add_argument(
        '--n-designs',
        type=int,
        default=None,
        help='Number of designs (will be inferred if not provided)'
    )
    parser.add_argument(
        '--n-wavevectors',
        type=int,
        default=None,
        help='Number of wavevectors (will be inferred if not provided)'
    )
    parser.add_argument(
        '--n-bands',
        type=int,
        default=None,
        help='Number of bands (will be inferred if not provided)'
    )
    parser.add_argument(
        '--design-res',
        type=int,
        default=32,
        help='Design resolution (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Load tensor
    tensor_path = Path(args.tensor_path)
    print(f"Loading tensor from: {tensor_path}")
    
    if tensor_path.suffix == '.pt':
        tensor = torch.load(tensor_path, map_location='cpu', weights_only=False)
    elif tensor_path.suffix == '.npy':
        tensor = np.load(tensor_path)
    else:
        raise ValueError(f"Unsupported file format: {tensor_path.suffix}. Use .pt or .npy")
    
    # Run conversion
    result = tensor_to_eigenvector_mat(
        tensor=tensor,
        reduced_indices_path=args.reduced_indices_path,
        output_path=args.output_path,
        n_designs=args.n_designs,
        n_wavevectors=args.n_wavevectors,
        n_bands=args.n_bands,
        design_res=args.design_res
    )
    
    print("\n" + "=" * 80)
    print("Conversion Summary")
    print("=" * 80)
    print(f"Output file: {result['output_path']}")
    print(f"File size: {result['file_size_mb']:.2f} MB")
    print(f"EIGENVECTOR_DATA shape: {result['EIGENVECTOR_DATA_shape']}")
    print(f"n_designs: {result['n_designs']}")
    print(f"n_wavevectors: {result['n_wavevectors']}")
    print(f"n_bands: {result['n_bands']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

