"""
Example: Using Converted Dispersion Dataset (dataset_conversion_reduction.ipynb format)

This script demonstrates how to load and use the NumPy-converted
dispersion dataset for neural network training with PyTorch.

Usage:
    python example_pytorch_dataset.py <path_to_converted_dataset>
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import numpy as np


class DispersionDataset(Dataset):
    """
    PyTorch Dataset for dispersion data following dataset_conversion_reduction.ipynb format.
    
    Loads NumPy arrays and converts to PyTorch tensors for neural network training.
    """
    
    def __init__(self, data_path, device='cpu'):
        """
        Initialize dataset.
        
        Parameters
        ----------
        data_path : str or Path
            Path to converted dataset folder (ending in _py)
        device : str
            Device to load tensors to ('cpu', 'cuda', etc.)
        """
        self.data_path = Path(data_path)
        self.device = device
        
        print(f"Loading dataset from: {self.data_path}")
        
        # Load core data (NumPy arrays)
        designs_np = np.load(self.data_path / 'designs.npy')
        eigenvalues_np = np.load(self.data_path / 'eigenvalue_data.npy')
        wavevectors_np = np.load(self.data_path / 'wavevectors.npy')
        waveforms_np = np.load(self.data_path / 'waveforms.npy')
        bands_fft_np = np.load(self.data_path / 'bands_fft.npy')
        
        # Convert to PyTorch tensors
        self.designs = torch.from_numpy(designs_np).to(device)
        self.eigenvalues = torch.from_numpy(eigenvalues_np).to(device)
        self.wavevectors = torch.from_numpy(wavevectors_np).to(device)
        self.waveforms = torch.from_numpy(waveforms_np).to(device)
        self.bands_fft = torch.from_numpy(bands_fft_np).to(device)
        
        # Load design params
        design_params_np = np.load(self.data_path / 'design_params.npy')
        self.design_params = torch.from_numpy(design_params_np)
        
        # Dataset dimensions (following dataset_conversion_reduction format)
        self.N_struct = self.designs.shape[0]  # First dimension
        self.N_pix = self.designs.shape[1]
        self.N_wv = self.eigenvalues.shape[1]
        self.N_eig = self.eigenvalues.shape[2]
        
        print(f"  N_struct: {self.N_struct}")
        print(f"  N_pix: {self.N_pix}")
        print(f"  N_wv: {self.N_wv}")
        print(f"  N_eig: {self.N_eig}")
        
        # Optionally load eigenvectors (if needed)
        self.has_eigenvectors = False
        if (self.data_path / 'eigenvector_data_x.npy').exists():
            eigvec_x_np = np.load(self.data_path / 'eigenvector_data_x.npy')
            eigvec_y_np = np.load(self.data_path / 'eigenvector_data_y.npy')
            self.eigvec_x = torch.from_numpy(eigvec_x_np).to(device)
            self.eigvec_y = torch.from_numpy(eigvec_y_np).to(device)
            self.has_eigenvectors = True
            print(f"  Eigenvectors loaded: {self.eigvec_x.shape}")
    
    def __len__(self):
        return self.N_struct
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns
        -------
        sample : dict
            Dictionary containing:
            - design: Material property design (N_pix, N_pix) - elastic modulus only
            - eigenvalues: Eigenfrequencies (N_wv, N_eig)
            - wavevectors: Wavevector grid (N_wv, 2)
            - eigvec_x, eigvec_y: (optional) Eigenvector components (N_wv, N_eig, N_pix, N_pix)
        """
        sample = {
            'design': self.designs[idx],  # (N_pix, N_pix)
            'eigenvalues': self.eigenvalues[idx],  # (N_wv, N_eig)
            'wavevectors': self.wavevectors[idx],  # (N_wv, 2)
            'index': idx
        }
        
        if self.has_eigenvectors:
            sample['eigvec_x'] = self.eigvec_x[idx]  # (N_wv, N_eig, N_pix, N_pix)
            sample['eigvec_y'] = self.eigvec_y[idx]  # (N_wv, N_eig, N_pix, N_pix)
        
        return sample


def example_neural_network_training(data_path):
    """
    Example: Training a simple neural network on dispersion data.
    
    This demonstrates loading the dataset and using it with PyTorch
    DataLoader for batched training.
    """
    print("\n" + "=" * 80)
    print("Example: Neural Network Training with Dispersion Dataset")
    print("=" * 80)
    
    # Create dataset
    dataset = DispersionDataset(data_path)
    
    # Create dataloader
    batch_size = 4
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to > 0 for parallel loading
    )
    
    print(f"\nDataLoader created with batch_size={batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Example: Iterate through batches
    print("\nExample batch iteration:")
    for batch_idx, batch in enumerate(dataloader):
        designs = batch['design']  # (batch, N_pix, N_pix) - elastic modulus only
        eigenvals = batch['eigenvalues']  # (batch, N_wv, N_eig)
        wavevecs = batch['wavevectors']  # (batch, N_wv, 2)
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Designs: {list(designs.shape)} dtype={designs.dtype}")
        print(f"  Eigenvalues: {list(eigenvals.shape)} dtype={eigenvals.dtype}")
        print(f"  Wavevectors: {list(wavevecs.shape)} dtype={wavevecs.dtype}")
        
        # Example: Check value ranges
        print(f"  Design range: [{designs.min():.3f}, {designs.max():.3f}]")
        print(f"  Eigenvalue range: [{eigenvals.min():.3e}, {eigenvals.max():.3e}]")
        
        # Only show first batch as example
        if batch_idx == 0:
            break
    
    # Example: Memory usage
    print("\n" + "=" * 80)
    print("Memory Usage Estimate:")
    print("=" * 80)
    
    def get_tensor_size_mb(tensor):
        return tensor.element_size() * tensor.nelement() / (1024 * 1024)
    
    print(f"  Designs: {get_tensor_size_mb(dataset.designs):.2f} MB")
    print(f"  Eigenvalues: {get_tensor_size_mb(dataset.eigenvalues):.2f} MB")
    print(f"  Wavevectors: {get_tensor_size_mb(dataset.wavevectors):.2f} MB")
    print(f"  Waveforms: {get_tensor_size_mb(dataset.waveforms):.2f} MB")
    print(f"  Bands FFT: {get_tensor_size_mb(dataset.bands_fft):.2f} MB")
    
    if dataset.has_eigenvectors:
        eigvec_size = get_tensor_size_mb(dataset.eigvec_x) + \
                      get_tensor_size_mb(dataset.eigvec_y)
        print(f"  Eigenvectors (x+y): {eigvec_size:.2f} MB")
    
    total_size = get_tensor_size_mb(dataset.designs) + \
                 get_tensor_size_mb(dataset.eigenvalues) + \
                 get_tensor_size_mb(dataset.wavevectors) + \
                 get_tensor_size_mb(dataset.waveforms) + \
                 get_tensor_size_mb(dataset.bands_fft)
    
    if dataset.has_eigenvectors:
        total_size += eigvec_size
    
    print(f"  Total (in memory): {total_size:.2f} MB")


def example_data_statistics(data_path):
    """
    Example: Computing statistics on the dataset.
    """
    print("\n" + "=" * 80)
    print("Example: Dataset Statistics")
    print("=" * 80)
    
    dataset = DispersionDataset(data_path)
    
    # Design statistics (elastic modulus only in this format)
    designs = dataset.designs
    print("\nDesign Statistics (Elastic Modulus):")
    print(f"  Mean: {designs.mean():.6f}")
    print(f"  Std:  {designs.std():.6f}")
    print(f"  Min:  {designs.min():.6f}")
    print(f"  Max:  {designs.max():.6f}")
    
    # Eigenvalue statistics
    eigenvals = dataset.eigenvalues
    print("\nEigenvalue (Frequency) Statistics:")
    print(f"  Mean: {eigenvals.mean():.3e}")
    print(f"  Std:  {eigenvals.std():.3e}")
    print(f"  Min:  {eigenvals.min():.3e}")
    print(f"  Max:  {eigenvals.max():.3e}")
    
    # Per-band statistics
    print("\nPer-Band Statistics:")
    for band_idx in range(min(dataset.N_eig, 6)):  # Show first 6 bands
        band_data = eigenvals[:, :, band_idx]  # All structures, all wavevectors, this band
        print(f"  Band {band_idx + 1}: mean={band_data.mean():.3e}, "
              f"std={band_data.std():.3e}")


def example_single_sample_access(data_path):
    """
    Example: Accessing individual samples.
    """
    print("\n" + "=" * 80)
    print("Example: Single Sample Access")
    print("=" * 80)
    
    dataset = DispersionDataset(data_path)
    
    # Get a single sample
    sample_idx = 0
    sample = dataset[sample_idx]
    
    print(f"\nSample {sample_idx}:")
    print(f"  Design shape: {sample['design'].shape}")
    print(f"  Eigenvalues shape: {sample['eigenvalues'].shape}")
    print(f"  Wavevectors shape: {sample['wavevectors'].shape}")
    
    # Access specific components
    design = sample['design']  # (N_pix, N_pix) - elastic modulus only
    eigenvals = sample['eigenvalues']  # (N_wv, N_eig)
    
    print(f"\nDesign properties (Elastic Modulus only):")
    print(f"  Range: [{design.min():.3f}, {design.max():.3f}]")
    print(f"  Mean: {design.mean():.3f}")
    
    print(f"\nFirst band eigenvalues:")
    print(f"  Min: {eigenvals[:, 0].min():.3e}")
    print(f"  Max: {eigenvals[:, 0].max():.3e}")
    print(f"  Mean: {eigenvals[:, 0].mean():.3e}")


def main(data_path):
    """Run all examples."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"ERROR: Dataset path not found: {data_path}")
        print("\nMake sure you've run the conversion script first:")
        print("  python convert_mat_to_pytorch.py <mat_file>")
        return
    
    # Run examples
    example_single_sample_access(data_path)
    example_data_statistics(data_path)
    example_neural_network_training(data_path)
    
    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_pytorch_dataset.py <path_to_pytorch_dataset>")
        print("\nExample:")
        print("  python example_pytorch_dataset.py ../data/continuous_dataset_py")
        sys.exit(1)
    
    data_path = sys.argv[1]
    main(data_path)

