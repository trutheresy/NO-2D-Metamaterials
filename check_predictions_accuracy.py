#!/usr/bin/env python3
"""
Compute MSE and weighted average relative error between predictions and ground truth.
"""

import torch
import numpy as np
from pathlib import Path

def compute_metrics(predictions_file, ground_truth_file):
    """
    Compute MSE and weighted average relative error.
    
    Parameters
    ----------
    predictions_file : str or Path
        Path to predictions .pt file
    ground_truth_file : str or Path
        Path to ground truth .pt file
    
    Returns
    -------
    dict : Dictionary with metrics
    """
    print("=" * 80)
    print("Computing Prediction Accuracy Metrics")
    print("=" * 80)
    
    # Load data
    print(f"\n1. Loading data...")
    predictions = torch.load(str(predictions_file), map_location='cpu', weights_only=False)
    ground_truth = torch.load(str(ground_truth_file), map_location='cpu', weights_only=False)
    
    print(f"   Predictions: {predictions_file}")
    print(f"   Ground truth: {ground_truth_file}")
    
    # Check structure
    assert hasattr(predictions, 'tensors'), "Predictions must be TensorDataset"
    assert hasattr(ground_truth, 'tensors'), "Ground truth must be TensorDataset"
    assert len(predictions.tensors) == 4, "Expected 4 tensors (x_real, x_imag, y_real, y_imag)"
    assert len(ground_truth.tensors) == 4, "Expected 4 tensors (x_real, x_imag, y_real, y_imag)"
    
    # Extract tensors
    pred_x_real = predictions.tensors[0]
    pred_x_imag = predictions.tensors[1]
    pred_y_real = predictions.tensors[2]
    pred_y_imag = predictions.tensors[3]
    
    gt_x_real = ground_truth.tensors[0]
    gt_x_imag = ground_truth.tensors[1]
    gt_y_real = ground_truth.tensors[2]
    gt_y_imag = ground_truth.tensors[3]
    
    # Check shapes match
    assert pred_x_real.shape == gt_x_real.shape, f"Shape mismatch: {pred_x_real.shape} vs {gt_x_real.shape}"
    
    print(f"   Sample shape: {pred_x_real.shape}")
    print(f"   Number of samples: {pred_x_real.shape[0]}")
    
    # Convert to complex
    pred_x = (pred_x_real.float() + 1j * pred_x_imag.float())
    pred_y = (pred_y_real.float() + 1j * pred_y_imag.float())
    gt_x = (gt_x_real.float() + 1j * gt_x_imag.float())
    gt_y = (gt_y_real.float() + 1j * gt_y_imag.float())
    
    # Stack into single array: (n_samples, 2, H, W) or (n_samples, 2, H*W)
    # Flatten spatial dimensions if needed
    if pred_x.ndim > 2:
        # (n_samples, H, W) -> (n_samples, H*W)
        pred_x_flat = pred_x.reshape(pred_x.shape[0], -1)
        pred_y_flat = pred_y.reshape(pred_y.shape[0], -1)
        gt_x_flat = gt_x.reshape(gt_x.shape[0], -1)
        gt_y_flat = gt_y.reshape(gt_y.shape[0], -1)
    else:
        pred_x_flat = pred_x
        pred_y_flat = pred_y
        gt_x_flat = gt_x
        gt_y_flat = gt_y
    
    # Combine x and y
    pred_combined = torch.cat([pred_x_flat, pred_y_flat], dim=1)  # (n_samples, 2*n_pixels)
    gt_combined = torch.cat([gt_x_flat, gt_y_flat], dim=1)  # (n_samples, 2*n_pixels)
    
    # Convert to numpy for easier computation
    pred_np = pred_combined.numpy()
    gt_np = gt_combined.numpy()
    
    # Compute statistics on ground truth magnitude
    print(f"\n2. Ground truth statistics...")
    gt_magnitude = np.abs(gt_np)
    avg_pixel_value = np.mean(gt_magnitude)
    print(f"   Average pixel value (magnitude): {avg_pixel_value:.6e}")
    print(f"   Pixel value range: [{np.min(gt_magnitude):.6e}, {np.max(gt_magnitude):.6e}]")
    print(f"   Pixel value median: {np.median(gt_magnitude):.6e}")
    
    # Compute MSE
    print(f"\n3. Computing MSE...")
    mse = np.mean(np.abs(pred_np - gt_np)**2)
    print(f"   MSE: {mse:.6e}")
    
    # Compute per-sample MSE
    mse_per_sample = np.mean(np.abs(pred_np - gt_np)**2, axis=1)
    print(f"   MSE per sample: mean={np.mean(mse_per_sample):.6e}, std={np.std(mse_per_sample):.6e}")
    print(f"   MSE per sample: min={np.min(mse_per_sample):.6e}, max={np.max(mse_per_sample):.6e}")
    
    # Compute average absolute error
    print(f"\n4. Computing average absolute error...")
    abs_error = np.abs(pred_np - gt_np)
    avg_abs_error = np.mean(abs_error)
    print(f"   Average absolute error: {avg_abs_error:.6e}")
    print(f"   Average absolute error per sample: mean={np.mean(np.mean(abs_error, axis=1)):.6e}, std={np.std(np.mean(abs_error, axis=1)):.6e}")
    print(f"   Average absolute error per sample: min={np.min(np.mean(abs_error, axis=1)):.6e}, max={np.max(np.mean(abs_error, axis=1)):.6e}")
    
    # Compute weighted average relative error
    # Weight by magnitude of ground truth (pixel value)
    print(f"\n5. Computing weighted average relative error...")
    
    # Compute relative error: |pred - gt| / |gt|
    # Avoid division by zero
    epsilon = 1e-10
    relative_error = np.abs(pred_np - gt_np) / (gt_magnitude + epsilon)
    
    # Weight by ground truth magnitude
    weights = gt_magnitude
    weights_normalized = weights / (np.sum(weights) + epsilon)
    
    weighted_avg_relative_error = np.sum(relative_error * weights_normalized)
    
    print(f"   Weighted average relative error: {weighted_avg_relative_error:.6f} ({weighted_avg_relative_error*100:.4f}%)")
    
    # Also compute unweighted average relative error for comparison
    unweighted_avg_relative_error = np.mean(relative_error)
    print(f"   Unweighted average relative error: {unweighted_avg_relative_error:.6f} ({unweighted_avg_relative_error*100:.4f}%)")
    
    # Per-sample statistics
    per_sample_weighted_re = []
    for i in range(len(pred_np)):
        w = gt_magnitude[i]
        w_norm = w / (np.sum(w) + epsilon)
        re = np.abs(pred_np[i] - gt_np[i]) / (gt_magnitude[i] + epsilon)
        per_sample_weighted_re.append(np.sum(re * w_norm))
    
    per_sample_weighted_re = np.array(per_sample_weighted_re)
    print(f"   Weighted RE per sample: mean={np.mean(per_sample_weighted_re):.6f}, std={np.std(per_sample_weighted_re):.6f}")
    print(f"   Weighted RE per sample: min={np.min(per_sample_weighted_re):.6f}, max={np.max(per_sample_weighted_re):.6f}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Average pixel value (magnitude): {avg_pixel_value:.6e}")
    print(f"Average absolute error: {avg_abs_error:.6e}")
    print(f"MSE: {mse:.6e}")
    print(f"Weighted average relative error: {weighted_avg_relative_error:.6f} ({weighted_avg_relative_error*100:.4f}%)")
    print(f"Unweighted average relative error: {unweighted_avg_relative_error:.6f} ({unweighted_avg_relative_error*100:.4f}%)")
    print("=" * 80)
    
    return {
        'avg_pixel_value': avg_pixel_value,
        'avg_abs_error': avg_abs_error,
        'mse': mse,
        'weighted_avg_relative_error': weighted_avg_relative_error,
        'unweighted_avg_relative_error': unweighted_avg_relative_error,
        'mse_per_sample': mse_per_sample,
        'weighted_re_per_sample': per_sample_weighted_re
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute prediction accuracy metrics')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions .pt file')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth .pt file')
    args = parser.parse_args()
    
    compute_metrics(args.predictions, args.ground_truth)

