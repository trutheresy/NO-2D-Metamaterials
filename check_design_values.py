"""
Check design values to see if they're causing E overflow.
"""
import numpy as np
import torch
from pathlib import Path

python_data_dir = Path(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1')
struct_idx = 0

# Load design
if (python_data_dir / 'geometries_full.pt').exists():
    geometries = torch.load(python_data_dir / 'geometries_full.pt', map_location='cpu')
    if isinstance(geometries, torch.Tensor):
        geometries = geometries.numpy()
    design = geometries[struct_idx]
else:
    print("Error: Could not find design file")
    exit(1)

print("="*70)
print("CHECKING DESIGN VALUES")
print("="*70)

print(f"\nDesign shape: {design.shape}")
print(f"Design dtype: {design.dtype}")
print(f"Design min: {np.min(design):.6f}")
print(f"Design max: {np.max(design):.6f}")
print(f"Design mean: {np.mean(design):.6f}")

# Check if values are in [0, 1]
if np.any(design < 0) or np.any(design > 1):
    print(f"\n⚠️  Design values are OUTSIDE [0, 1] range!")
    print(f"   Values < 0: {np.sum(design < 0)}")
    print(f"   Values > 1: {np.sum(design > 1)}")
    print(f"   Min: {np.min(design):.6f}, Max: {np.max(design):.6f}")
    
    # Check what happens when computing E
    E_min = 20e6
    E_max = 200e9
    E = E_min + design * (E_max - E_min)
    print(f"\n   E values (without clamping):")
    print(f"   Min: {np.min(E):.6e}, Max: {np.max(E):.6e}")
    if np.any(np.isinf(E)) or np.any(np.isnan(E)):
        print(f"   ⚠️  E contains inf or nan values!")
else:
    print(f"\n✅ Design values are within [0, 1] range")

# Check if design needs normalization
if np.max(design) > 1.0 or np.min(design) < 0.0:
    design_normalized = np.clip(design, 0, 1)
    print(f"\n   After clamping to [0, 1]:")
    print(f"   Min: {np.min(design_normalized):.6f}, Max: {np.max(design_normalized):.6f}")
    
    E_normalized = E_min + design_normalized * (E_max - E_min)
    print(f"   E values (with clamping):")
    print(f"   Min: {np.min(E_normalized):.6e}, Max: {np.max(E_normalized):.6e}")

print("\n" + "="*70)

