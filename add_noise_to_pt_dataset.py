#!/usr/bin/env python3
"""
Add Gaussian noise to PyTorch .pt dataset files.

This script:
1. Loads all .pt files from a source directory
2. Adds Gaussian noise (~1% average error)
3. Saves to a new directory
4. Analyzes and reports errors between original and noisy datasets
"""

import numpy as np
import torch
from pathlib import Path
import sys
import json
from typing import Dict, Any

def add_gaussian_noise(data, noise_std=0.01):
    """
    Add Gaussian noise to data using multiplicative relative error model.
    
    For each element: noisy_value = original_value * (1 + ε)
    where ε ~ N(0, noise_std²) represents relative error.
    
    Parameters:
    -----------
    data : torch.Tensor or np.ndarray
        Input data (can be real or complex)
    noise_std : float
        Standard deviation of relative error Gaussian (default: 0.01 = 1%)
    
    Returns:
    --------
    noisy_data : torch.Tensor or np.ndarray
        Data with added noise
    """
    # Convert to numpy if torch tensor
    if isinstance(data, torch.Tensor):
        data_np = data.numpy()
        was_tensor = True
        dtype = data.dtype
    else:
        data_np = data
        was_tensor = False
        dtype = data.dtype
    
    # Generate relative error samples from N(0, noise_std²)
    # One sample per element - same for real and complex
    relative_error = np.random.normal(0, noise_std, data_np.shape).astype(data_np.real.dtype)
    
    # Apply multiplicative noise: noisy = original * (1 + relative_error)
    # This works for both real and complex arrays (element-wise multiplication)
    noisy_data_np = data_np * (1 + relative_error)
    
    # Convert back to torch tensor if original was tensor
    if was_tensor:
        noisy_data = torch.from_numpy(noisy_data_np).to(dtype)
    else:
        noisy_data = noisy_data_np.astype(dtype)
    
    return noisy_data


def analyze_errors(original, noisy):
    """
    Analyze errors between original and noisy data.
    
    Returns:
    --------
    dict : Error statistics
    """
    # Convert to numpy for analysis
    if isinstance(original, torch.Tensor):
        orig_np = original.numpy().astype(np.float32)  # Convert to float32 for analysis
    else:
        orig_np = np.asarray(original, dtype=np.float32)
    
    if isinstance(noisy, torch.Tensor):
        noisy_np = noisy.numpy().astype(np.float32)
    else:
        noisy_np = np.asarray(noisy, dtype=np.float32)
    
    # Compute absolute error
    abs_error = np.abs(noisy_np - orig_np)
    
    # Compute magnitude
    magnitude = np.abs(orig_np)
    
    # Mask for non-zero values (for relative error on non-zero elements)
    non_zero_mask = magnitude > 1e-10
    
    # Compute relative error only for non-zero elements
    if np.any(non_zero_mask):
        relative_error_nonzero = abs_error[non_zero_mask] / magnitude[non_zero_mask]
    else:
        relative_error_nonzero = np.array([])
    
    # Compute relative error for all elements (with epsilon for zeros)
    epsilon = 1e-10
    magnitude_with_epsilon = np.where(magnitude > epsilon, magnitude, np.mean(magnitude[magnitude > epsilon]) if np.any(magnitude > epsilon) else epsilon)
    relative_error_all = abs_error / magnitude_with_epsilon
    
    # Statistics
    stats = {
        'mean_absolute_error': float(np.mean(abs_error)),
        'max_absolute_error': float(np.max(abs_error)),
        'std_absolute_error': float(np.std(abs_error)),
        'mean_magnitude': float(np.mean(magnitude)),
        'num_elements': int(np.prod(orig_np.shape)),
        'num_zero_elements': int(np.sum(magnitude <= 1e-10)),
        'num_nonzero_elements': int(np.sum(non_zero_mask)),
        'shape': list(orig_np.shape),
        'dtype': str(orig_np.dtype),
        'is_complex': np.iscomplexobj(orig_np),
    }
    
    # Add relative error stats for non-zero elements (more meaningful)
    if len(relative_error_nonzero) > 0:
        stats['mean_relative_error_nonzero'] = float(np.mean(relative_error_nonzero))
        stats['max_relative_error_nonzero'] = float(np.max(relative_error_nonzero))
        stats['median_relative_error_nonzero'] = float(np.median(relative_error_nonzero))
        stats['std_relative_error_nonzero'] = float(np.std(relative_error_nonzero))
        stats['p95_relative_error_nonzero'] = float(np.percentile(relative_error_nonzero, 95))
        stats['p99_relative_error_nonzero'] = float(np.percentile(relative_error_nonzero, 99))
    
    # Add relative error stats for all elements (using epsilon)
    valid_rel_error = relative_error_all[np.isfinite(relative_error_all)]
    if len(valid_rel_error) > 0:
        stats['mean_relative_error_all'] = float(np.mean(valid_rel_error))
        stats['median_relative_error_all'] = float(np.median(valid_rel_error))
        stats['p95_relative_error_all'] = float(np.percentile(valid_rel_error, 95))
        stats['p99_relative_error_all'] = float(np.percentile(valid_rel_error, 99))
    
    return stats


def process_pt_file(input_path: Path, output_path: Path, noise_std=0.01, seed=None):
    """
    Process a single .pt file: add noise and save.
    
    Returns:
    --------
    dict : Processing info and error statistics
    """
    # Skip index files - they shouldn't have noise
    if 'indices' in input_path.name.lower():
        print(f"  Skipping (index file): {input_path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        original_data = torch.load(input_path, map_location='cpu', weights_only=False)
        torch.save(original_data, output_path)
        return {
            'file': input_path.name,
            'input_path': str(input_path),
            'output_path': str(output_path),
            'error_stats': {'note': 'Index file - copied without noise'}
        }
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Load original data
    print(f"  Loading: {input_path.name}")
    original_data = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Handle different data types
    if isinstance(original_data, torch.utils.data.TensorDataset):
        # TensorDataset: contains multiple tensors
        tensors = original_data.tensors
        noisy_tensors = tuple(add_gaussian_noise(t, noise_std) for t in tensors)
        noisy_data = torch.utils.data.TensorDataset(*noisy_tensors)
        
        # Analyze errors for each tensor
        error_stats = []
        for i, (orig_t, noisy_t) in enumerate(zip(tensors, noisy_tensors)):
            stats = analyze_errors(orig_t, noisy_t)
            stats['tensor_index'] = i
            error_stats.append(stats)
    elif isinstance(original_data, (torch.Tensor, np.ndarray)):
        # Single tensor/array
        noisy_data = add_gaussian_noise(original_data, noise_std)
        error_stats = [analyze_errors(original_data, noisy_data)]
    elif isinstance(original_data, (list, tuple)):
        # Check if list/tuple contains tensors/arrays or other data types
        if len(original_data) > 0 and isinstance(original_data[0], (torch.Tensor, np.ndarray)):
            # List/tuple of tensors
            noisy_data = type(original_data)(add_gaussian_noise(item, noise_std) for item in original_data)
            error_stats = [analyze_errors(orig_item, noisy_item) 
                          for orig_item, noisy_item in zip(original_data, noisy_data)]
        else:
            # List/tuple of non-tensor data (e.g., indices) - copy as is
            print(f"    Warning: List/tuple contains non-tensor data, copying without noise")
            noisy_data = original_data
            error_stats = {'note': 'Non-tensor list/tuple - copied without noise'}
    elif isinstance(original_data, dict):
        # Dictionary
        noisy_data = {key: add_gaussian_noise(val, noise_std) for key, val in original_data.items()}
        error_stats = {key: analyze_errors(original_data[key], noisy_data[key]) 
                      for key in original_data.keys()}
    else:
        # Unknown type - copy as is
        print(f"    Warning: Unknown data type {type(original_data)}, copying without noise")
        noisy_data = original_data
        error_stats = {'note': 'Data type not processed for noise'}
    
    # Save noisy data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(noisy_data, output_path)
    print(f"    Saved: {output_path.name}")
    
    return {
        'file': input_path.name,
        'input_path': str(input_path),
        'output_path': str(output_path),
        'error_stats': error_stats
    }


def main():
    # Paths
    source_dir = Path("D:/Research/NO-2D-Metamaterials/data/out_test_10_pt_noise/out_binarized_1")
    output_dir = Path("D:/Research/NO-2D-Metamaterials/data/out_test_10_pt_noise/out_binarized_1_noise")
    
    noise_std = 0.01  # Standard deviation of relative error Gaussian (1%)
    seed = 42  # For reproducibility
    
    print("=" * 80)
    print("Adding Gaussian Noise to PyTorch Dataset")
    print("=" * 80)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Relative error std: {noise_std * 100:.1f}% (Gaussian: N(0, {noise_std}²))")
    print(f"Noise model: noisy = original * (1 + ε) where ε ~ N(0, {noise_std}²)")
    print()
    
    # Find all .pt files
    pt_files = list(source_dir.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: No .pt files found in {source_dir}")
        return 1
    
    print(f"Found {len(pt_files)} .pt file(s):")
    for pt_file in sorted(pt_files):
        print(f"  - {pt_file.name}")
    print()
    
    # Process each file
    all_results = []
    for pt_file in sorted(pt_files):
        output_file = output_dir / pt_file.name
        result = process_pt_file(pt_file, output_file, noise_std=noise_std, seed=seed)
        all_results.append(result)
    
    # Print summary
    print()
    print("=" * 80)
    print("Error Analysis Summary")
    print("=" * 80)
    
    for result in all_results:
        print(f"\n{result['file']}:")
        error_stats = result['error_stats']
        
        if isinstance(error_stats, dict) and 'note' in error_stats:
            print(f"  {error_stats['note']}")
        elif isinstance(error_stats, list):
            for i, stats in enumerate(error_stats):
                print(f"  Tensor {i}:")
                print(f"    Shape: {stats['shape']}, Dtype: {stats['dtype']}")
                print(f"    Mean absolute error: {stats['mean_absolute_error']:.6e}")
                print(f"    Mean magnitude: {stats['mean_magnitude']:.6e}")
                print(f"    Non-zero elements: {stats['num_nonzero_elements']}/{stats['num_elements']} ({stats['num_nonzero_elements']/stats['num_elements']*100:.1f}%)")
                
                if 'mean_relative_error_nonzero' in stats:
                    print(f"    Relative error (non-zero elements only):")
                    print(f"      Mean: {stats['mean_relative_error_nonzero']*100:.4f}%")
                    print(f"      Median: {stats['median_relative_error_nonzero']*100:.4f}%")
                    print(f"      Max: {stats['max_relative_error_nonzero']*100:.4f}%")
                    print(f"      Std: {stats['std_relative_error_nonzero']*100:.4f}%")
                    print(f"      95th percentile: {stats['p95_relative_error_nonzero']*100:.4f}%")
                    print(f"      99th percentile: {stats['p99_relative_error_nonzero']*100:.4f}%")
                
                if 'mean_relative_error_all' in stats:
                    print(f"    Relative error (all elements, using epsilon):")
                    print(f"      Mean: {stats['mean_relative_error_all']*100:.4f}%")
                    print(f"      Median: {stats['median_relative_error_all']*100:.4f}%")
                
                if stats['is_complex']:
                    print(f"    Complex data: Yes")
        elif isinstance(error_stats, dict):
            for key, stats in error_stats.items():
                if isinstance(stats, dict) and 'mean_absolute_error' in stats:
                    print(f"  Key '{key}':")
                    print(f"    Shape: {stats['shape']}")
                    if 'mean_relative_error_nonzero' in stats:
                        print(f"    Mean relative error (non-zero): {stats['mean_relative_error_nonzero']*100:.4f}%")
                        print(f"    Median relative error (non-zero): {stats['median_relative_error_nonzero']*100:.4f}%")
    
    # Save detailed report to JSON
    report_path = output_dir / "noise_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed report saved to: {report_path}")
    
    print()
    print("=" * 80)
    print("Processing complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

