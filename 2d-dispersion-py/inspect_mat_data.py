"""Inspect the saved .mat file to see what values are stored."""
import scipy.io as sio
import numpy as np
from pathlib import Path

test_plots_dir = Path('test_plots')
data_path = test_plots_dir / 'plot_dispersion_contour_data.mat'

if not data_path.exists():
    print(f"❌ File not found: {data_path}")
    exit(1)

data = sio.loadmat(str(data_path))

print("=" * 80)
print("INSPECTING SAVED .MAT FILE")
print("=" * 80)
print(f"\nFile: {data_path}")
print(f"\nKeys (excluding MATLAB metadata):")
for key in data.keys():
    if not key.startswith('__'):
        print(f"  - {key}: shape {data[key].shape}, dtype {data[key].dtype}")

print("\n" + "=" * 80)
print("X MATRIX (wavevector x-component)")
print("=" * 80)
print(f"Shape: {data['X'].shape}")
print(f"Range: [{np.min(data['X']):.6f}, {np.max(data['X']):.6f}]")
print(f"\nFirst 5x5:")
print(data['X'][:5, :5])
print(f"\nLast 5x5:")
print(data['X'][-5:, -5:])

print("\n" + "=" * 80)
print("Y MATRIX (wavevector y-component)")
print("=" * 80)
print(f"Shape: {data['Y'].shape}")
print(f"Range: [{np.min(data['Y']):.6f}, {np.max(data['Y']):.6f}]")
print(f"\nFirst 5x5:")
print(data['Y'][:5, :5])
print(f"\nLast 5x5:")
print(data['Y'][-5:, -5:])

print("\n" + "=" * 80)
print("Z MATRIX (frequency values)")
print("=" * 80)
print(f"Shape: {data['Z'].shape}")
print(f"Range: [{np.min(data['Z']):.2f}, {np.max(data['Z']):.2f}] Hz")
print(f"Mean: {np.mean(data['Z']):.2f} Hz")
print(f"Std: {np.std(data['Z']):.2f} Hz")
print(f"\nFirst 5x5:")
print(data['Z'][:5, :5])
print(f"\nLast 5x5:")
print(data['Z'][-5:, -5:])

print("\n" + "=" * 80)
print("WAVEVECTOR GRID")
print("=" * 80)
print(f"Shape: {data['wv_grid'].shape}")
print(f"First 10 wavevectors:")
print(data['wv_grid'][:10, :])
print(f"Last 10 wavevectors:")
print(data['wv_grid'][-10:, :])

