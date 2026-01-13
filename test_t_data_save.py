"""Test script to verify T_data.pt was saved correctly."""
import torch
from pathlib import Path

data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
T_data = torch.load(data_dir / 'T_data.pt', map_location='cpu')

print(f'T_data type: {type(T_data)}')
print(f'T_data length: {len(T_data)}')
print(f'First structure T_data length: {len(T_data[0])}')
if hasattr(T_data[0][0], 'shape'):
    print(f'First T matrix shape: {T_data[0][0].shape}')
    print(f'First T matrix nnz: {T_data[0][0].nnz if hasattr(T_data[0][0], "nnz") else "N/A"}')
print('T_data.pt file loaded successfully!')

