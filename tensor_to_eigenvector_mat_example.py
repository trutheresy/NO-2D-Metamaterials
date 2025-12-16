"""
Example usage of tensor_to_eigenvector_mat.py

This shows how to use the script programmatically (as a Python function)
rather than from the command line.
"""

import torch
import numpy as np
from pathlib import Path
from tensor_to_eigenvector_mat import tensor_to_eigenvector_mat

# Example: Convert a tensor to EIGENVECTOR_DATA.mat

# Create a dummy tensor of shape (4, n_samples, 32, 32)
# In practice, this would come from your model predictions
n_samples = 100  # Example: 100 samples
design_res = 32

# Create dummy tensor (replace with your actual predictions)
tensor = torch.randn(4, n_samples, design_res, design_res)

# Or load from file:
# tensor = torch.load("your_predictions.pt")

# Path to reduced_indices.pt (from the original dataset)
reduced_indices_path = r"D:\Research\NO-2D-Metamaterials\data\dispersion_binarized_1\reduced_indices.pt"

# Output path
output_path = r"D:\Research\NO-2D-Metamaterials\EIGENVECTOR_DATA.mat"

# Run conversion
result = tensor_to_eigenvector_mat(
    tensor=tensor,
    reduced_indices_path=reduced_indices_path,
    output_path=output_path,
    # Optional: specify dimensions if you know them
    # n_designs=200,
    # n_wavevectors=91,
    # n_bands=6,
    design_res=32
)

print("\nConversion completed!")
print(f"Output saved to: {result['output_path']}")

