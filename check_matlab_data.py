"""Check what data is in the MATLAB file."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from mat73_loader import load_matlab_v73

data = load_matlab_v73('data/out_test_10_matlab/out_binarized_1.mat', verbose=False)

print("Keys in loaded data:")
for k in sorted(data.keys()):
    val = data[k]
    if hasattr(val, 'shape'):
        print(f"  {k}: shape={val.shape}, dtype={val.dtype}")
    elif hasattr(val, '__len__'):
        print(f"  {k}: length={len(val)}, type={type(val)}")
    else:
        print(f"  {k}: type={type(val)}")

print("\nChecking for reconstruction data:")
for key in ['EIGENVECTOR_DATA', 'K_DATA', 'M_DATA', 'T_DATA']:
    if key in data:
        print(f"  ✓ {key} found")
        val = data[key]
        if hasattr(val, 'shape'):
            print(f"    shape: {val.shape}, dtype: {val.dtype}")
    else:
        print(f"  ✗ {key} NOT found")

