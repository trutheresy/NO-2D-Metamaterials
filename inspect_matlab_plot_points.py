"""Inspect MATLAB saved plot points to understand contour structure."""
import h5py
import numpy as np

mat_file = '2D-dispersion-han/plots/out_binarized_1_mat/plot_points.mat'

with h5py.File(mat_file, 'r') as f:
    print('Top-level keys:')
    for k in f.keys():
        if not k.startswith('#'):
            print(f'  {k}')
    
    print('\n' + '='*70)
    if 'plot_points_data' in f:
        pp = f['plot_points_data']
        print('plot_points_data keys:')
        for k in sorted(pp.keys()):
            val = np.array(pp[k])
            print(f'  {k}: shape={val.shape}, dtype={val.dtype}')
        
        # Find wavevectors and contour_param
        wv_keys = [k for k in pp.keys() if 'wavevectors_contour' in k]
        param_keys = [k for k in pp.keys() if 'contour_param' in k]
        
        if wv_keys and param_keys:
            wv_key = wv_keys[0]
            param_key = param_keys[0]
            
            wv = np.array(pp[wv_key])
            param = np.array(pp[param_key])
            
            # Handle MATLAB's (N, 1) shape
            if param.ndim == 2 and param.shape[1] == 1:
                param = param.flatten()
            if wv.ndim == 2 and wv.shape[0] == 2 and wv.shape[1] > 2:
                wv = wv.T
            
            print(f'\n{wv_key}:')
            print(f'  Shape: {wv.shape}')
            print(f'  First 3 points:\n{wv[:3]}')
            print(f'  Last 3 points:\n{wv[-3:]}')
            
            print(f'\n{param_key}:')
            print(f'  Shape: {param.shape}')
            print(f'  Range: [{np.min(param):.4f}, {np.max(param):.4f}]')
            print(f'  First 10 values: {param[:10]}')
            print(f'  Last 10 values: {param[-10:]}')
            print(f'  Unique value count: {len(np.unique(param))}')
            
            # Check if it's evenly spaced
            if len(param) > 1:
                diffs = np.diff(param)
                print(f'  Step size (first): {diffs[0]:.6f}')
                print(f'  Step size (last): {diffs[-1]:.6f}')
                print(f'  Step size (mean): {np.mean(diffs):.6f}')
                print(f'  Step size (std): {np.std(diffs):.6f}')
                if np.std(diffs) < 1e-10:
                    print(f'  ✓ Evenly spaced')
                else:
                    print(f'  ⚠ Not evenly spaced (std={np.std(diffs):.6e})')

