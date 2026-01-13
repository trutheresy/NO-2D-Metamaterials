"""Check symmetry type in the MATLAB dataset."""
import sys
from pathlib import Path
sys.path.insert(0, '2d-dispersion-py')
from mat73_loader import load_matlab_v73
import numpy as np

mat_file = '2D-dispersion-han/OUTPUT/out_test_10/out_binarized_1.mat'
data = load_matlab_v73(mat_file, verbose=False)

const = data.get('const', {})
print(f"const type: {type(const)}")

if isinstance(const, dict):
    sym_type = const.get('symmetry_type', None)
else:
    # Try to access as struct-like
    try:
        sym_type = const['symmetry_type']
    except (KeyError, TypeError):
        sym_type = None

print(f"\nSymmetry type raw: {sym_type}")
print(f"Type: {type(sym_type)}")

# Handle different formats
if sym_type is None:
    print("  ✗ symmetry_type not found")
elif isinstance(sym_type, str):
    print(f"  String: '{sym_type}'")
elif isinstance(sym_type, np.ndarray):
    print(f"  Array shape: {sym_type.shape}, dtype: {sym_type.dtype}")
    if sym_type.dtype.kind == 'U' or sym_type.dtype.kind == 'S':
        # String array
        try:
            sym_str = ''.join(sym_type.flatten())
            print(f"  Converted string: '{sym_str}'")
        except:
            print(f"  Array values: {sym_type}")
    elif sym_type.dtype == np.uint16:
        # Character codes
        try:
            sym_str = ''.join([chr(int(x)) for x in sym_type.flatten()])
            print(f"  Converted from uint16: '{sym_str}'")
        except:
            print(f"  Array values: {sym_type}")
    else:
        print(f"  Array values: {sym_type}")

