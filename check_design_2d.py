"""
Check design 2D comparison approach
"""

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))

from mat73_loader import load_matlab_v73
from get_design import get_design

# Load MATLAB data
matlab_data = load_matlab_v73('test_outputs_matlab/test1_designs.mat', verbose=False)
python_data = np.load('test_outputs_python/test1_designs.npz', allow_pickle=True)

designs = matlab_data.get('designs_matlab', {})
matlab_q1d = None
if isinstance(designs, np.ndarray) and designs.dtype.names:
    if 'quasi_1D' in designs.dtype.names:
        matlab_q1d = designs['quasi_1D'][0]
        print('MATLAB quasi-1D shape:', matlab_q1d.shape)
        
        # Handle dimension ordering
        if matlab_q1d.shape[0] == 3 and len(matlab_q1d.shape) == 3:
            matlab_q1d = np.transpose(matlab_q1d, (1, 2, 0))
            print('After transpose to (8,8,3):', matlab_q1d.shape)
        
        # Extract 2D slices
        for prop_idx in range(3):
            matlab_2d = matlab_q1d[:, :, prop_idx]
            print(f'\nProperty {prop_idx} (MATLAB):')
            print(matlab_2d)
            print('Shape:', matlab_2d.shape)

python_q1d = python_data['quasi-1D']
print(f'\nPython quasi-1D shape: {python_q1d.shape}')

# Extract 2D slices
for prop_idx in range(3):
    python_2d = python_q1d[:, :, prop_idx]
    print(f'\nProperty {prop_idx} (Python):')
    print(python_2d)
    print('Shape:', python_2d.shape)

# Compare directly
if matlab_q1d is not None:
    print('\n' + '='*60)
    print('Direct comparison (property 0):')
    matlab_2d_0 = matlab_q1d[:, :, 0] if len(matlab_q1d.shape) == 3 else matlab_q1d
    python_2d_0 = python_q1d[:, :, 0]
    print(f'MATLAB shape: {matlab_2d_0.shape}')
    print(f'Python shape: {python_2d_0.shape}')
    print(f'Direct match: {np.array_equal(matlab_2d_0, python_2d_0)}')
    print(f'Transpose match: {np.array_equal(matlab_2d_0.T, python_2d_0)}')
    print(f'Fliplr match: {np.array_equal(np.fliplr(matlab_2d_0), python_2d_0)}')
    print(f'Flipud match: {np.array_equal(np.flipud(matlab_2d_0), python_2d_0)}')

