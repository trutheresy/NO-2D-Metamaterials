"""
Compare plot points for Case 2 (with eigenfrequencies) between Python and MATLAB.

Dataset: 2D-dispersion-han/OUTPUT/out_test_10/out_binarized_1.mat
"""

import numpy as np
from pathlib import Path
import scipy.io as sio
import h5py
import glob
import os

def load_python_plot_points(npz_path):
    """Load plot points from Python .npz file."""
    data = np.load(npz_path)
    return dict(data)

def load_matlab_plot_points(mat_path):
    """Load plot points from MATLAB .mat file."""
    try:
        # Try scipy.io.loadmat first (for older MATLAB files)
        data = sio.loadmat(mat_path, squeeze_me=False)
        if 'plot_points_data' in data:
            return data['plot_points_data']
        else:
            return {k: v for k, v in data.items() if not k.startswith('__')}
    except (NotImplementedError, ValueError):
        # Try h5py for v7.3 files
        try:
            with h5py.File(mat_path, 'r') as f:
                if 'plot_points_data' in f:
                    plot_data = {}
                    ref = f['plot_points_data']
                    if isinstance(ref, h5py.Group):
                        for key in ref.keys():
                            val = np.array(ref[key])
                            # MATLAB stores arrays in column-major order
                            if val.ndim == 2:
                                if 'wavevectors_contour' in key and val.shape[0] == 2:
                                    val = val.T
                                elif 'frequencies_contour' in key and val.shape[0] < val.shape[1] and val.shape[0] <= 10:
                                    val = val.T
                            elif val.ndim == 2 and val.shape[1] == 1:
                                val = val.flatten()
                            plot_data[key] = val
                    return plot_data
                else:
                    return {k: np.array(f[k]) for k in f.keys() if not k.startswith('#')}
        except Exception as e:
            print(f"Error loading MATLAB file: {e}")
            return None

def compare_plot_points(py_data, ml_data, py_struct_idx, ml_struct_idx=None, tolerance=1e-6):
    """Compare plot points for a specific structure."""
    if ml_struct_idx is None:
        ml_struct_idx = py_struct_idx + 1  # MATLAB uses 1-based indexing
    
    print(f"\n{'='*70}")
    print(f"Comparing plot points: Python struct_{py_struct_idx} vs MATLAB struct_{ml_struct_idx}")
    print(f"{'='*70}")
    
    keys = {
        'wavevectors_contour': 'wavevectors_contour',
        'frequencies_contour': 'frequencies_contour',
        'contour_param': 'contour_param'
    }
    
    discrepancies = []
    
    for py_key, ml_key in keys.items():
        py_full_key = f'struct_{py_struct_idx}_{py_key}'
        ml_full_key = f'struct_{ml_struct_idx}_{py_key}'
        
        if py_full_key not in py_data:
            print(f"  ✗ Python key not found: {py_full_key}")
            discrepancies.append(f"Missing Python key: {py_key}")
            continue
        
        if ml_full_key not in ml_data:
            print(f"  ✗ MATLAB key not found: {ml_full_key}")
            discrepancies.append(f"Missing MATLAB key: {py_key}")
            continue
        
        py_val = py_data[py_full_key]
        ml_val = ml_data[ml_full_key]
        
        # Handle MATLAB's (N, 1) vs Python's (N,) shape
        if ml_val.ndim == 2 and ml_val.shape[1] == 1:
            ml_val = ml_val.flatten()
        
        # Check shapes
        if py_val.shape != ml_val.shape:
            print(f"  ⚠ Shape mismatch for {py_key}:")
            print(f"    Python: {py_val.shape}")
            print(f"    MATLAB: {ml_val.shape}")
            
            # Check if it's just a difference in interpolation modes
            py_use_interp = py_data.get(f'struct_{py_struct_idx}_use_interpolation', np.array([False]))[0]
            ml_use_interp = ml_data.get(f'struct_{ml_struct_idx}_use_interpolation', np.array([True]))[0]
            
            if py_use_interp != ml_use_interp:
                print(f"    NOTE: Different interpolation modes (Python: {py_use_interp}, MATLAB: {ml_use_interp})")
                print(f"    This is expected - comparing overlapping points only")
                # Compare subset
                min_len = min(len(py_val), len(ml_val))
                py_val_subset = py_val[:min_len]
                ml_val_subset = ml_val[:min_len]
                
                if np.allclose(py_val_subset, ml_val_subset, rtol=tolerance, atol=tolerance):
                    print(f"    ✓ First {min_len} points match")
                    continue
                else:
                    diff = np.abs(py_val_subset - ml_val_subset)
                    max_diff = np.max(diff)
                    print(f"    ✗ First {min_len} points differ (max diff: {max_diff:.6e})")
                    discrepancies.append(f"Shape/Value mismatch: {py_key} (different interpolation, overlapping points differ)")
                    continue
            else:
                discrepancies.append(f"Shape mismatch: {py_key}")
                continue
        
        # Compare values
        if np.allclose(py_val, ml_val, rtol=tolerance, atol=tolerance):
            print(f"  ✓ {py_key}: Values match (shape: {py_val.shape})")
        else:
            diff = np.abs(py_val - ml_val)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            rel_error = max_diff / (np.max(np.abs(ml_val)) + 1e-15)
            
            print(f"  ✗ {py_key}: Values differ")
            print(f"    Max absolute difference: {max_diff:.6e}")
            print(f"    Mean absolute difference: {mean_diff:.6e}")
            print(f"    Max relative error: {rel_error:.6e}")
            print(f"    Shape: {py_val.shape}")
            
            discrepancies.append({
                'key': py_key,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'rel_error': rel_error
            })
    
    return discrepancies

def main():
    """Main comparison function."""
    print("="*70)
    print("Case 2 Plot Points Comparison: Python vs MATLAB (WITH eigenfrequencies)")
    print("="*70)
    print("Dataset: 2D-dispersion-han/OUTPUT/out_test_10/out_binarized_1.mat")
    print("="*70)
    
    # Find most recent Python output
    py_patterns = [
        "dispersion_plots_*_mat/plot_points.npz",
    ]
    all_py = []
    for pattern in py_patterns:
        all_py.extend(glob.glob(pattern))
    
    if not all_py:
        print("\nERROR: No Python plot_points.npz files found!")
        print("  Run: python 2d-dispersion-py/plot_dispersion_with_eigenfrequencies.py \"2D-dispersion-han/OUTPUT/out_test_10/out_binarized_1.mat\" -n 1 --save-plot-points")
        return
    
    # Sort by modification time, most recent first
    all_py.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    case2_py = Path(all_py[0])
    print(f"\nPython output: {case2_py}")
    
    # Find MATLAB output
    case2_ml = Path("2D-dispersion-han/plots/out_binarized_1_mat/plot_points.mat")
    case2_ml_alt = Path("plots/out_binarized_1_mat/plot_points.mat")
    
    case2_ml_actual = case2_ml if case2_ml.exists() else (case2_ml_alt if case2_ml_alt.exists() else None)
    
    if case2_ml_actual is None:
        print(f"\nERROR: MATLAB plot_points.mat not found!")
        print(f"  Expected locations:")
        print(f"    - {case2_ml}")
        print(f"    - {case2_ml_alt}")
        print(f"\n  Run MATLAB script:")
        print(f"    cd('2D-dispersion-han')")
        print(f"    data_fn = 'D:\\Research\\NO-2D-Metamaterials\\2D-dispersion-han\\OUTPUT\\out_test_10\\out_binarized_1.mat';")
        print(f"    plot_dispersion")
        return
    
    print(f"MATLAB output: {case2_ml_actual}")
    
    # Load data
    py_data = load_python_plot_points(case2_py)
    ml_data = load_matlab_plot_points(case2_ml_actual)
    
    if ml_data is None:
        print("\nERROR: Could not load MATLAB data")
        return
    
    # Find structure indices
    py_struct_indices = set()
    for key in py_data.keys():
        if key.startswith('struct_') and '_wavevectors_contour' in key:
            idx = int(key.split('_')[1])
            py_struct_indices.add(idx)
    
    ml_struct_indices = set()
    for key in ml_data.keys():
        if key.startswith('struct_') and '_wavevectors_contour' in key:
            idx = int(key.split('_')[1])
            ml_struct_indices.add(idx)
    
    print(f"\nPython structure indices: {sorted(py_struct_indices)}")
    print(f"MATLAB structure indices: {sorted(ml_struct_indices)}")
    
    all_discrepancies = {}
    
    # Compare matching structures
    for py_idx in sorted(py_struct_indices):
        ml_idx = py_idx + 1  # MATLAB uses 1-based indexing
        
        if ml_idx in ml_struct_indices:
            print(f"\n  Comparing: Python struct_{py_idx} <-> MATLAB struct_{ml_idx}")
            disc = compare_plot_points(py_data, ml_data, py_idx, ml_idx)
            if disc:
                all_discrepancies[f'Case2_struct_{py_idx}'] = disc
        else:
            print(f"\n  WARNING: No MATLAB equivalent for Python struct_{py_idx}")
            all_discrepancies[f'Case2_struct_{py_idx}'] = ['No MATLAB equivalent']
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if all_discrepancies:
        print(f"Found discrepancies in {len(all_discrepancies)} structure(s):")
        for key, disc in all_discrepancies.items():
            print(f"  {key}: {len(disc)} discrepancy(ies)")
    else:
        print("✓ No discrepancies found! Python and MATLAB plot points match.")
    
    print("="*70)

if __name__ == "__main__":
    main()

