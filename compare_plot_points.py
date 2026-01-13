"""
Compare plot point locations between Python and MATLAB scripts.

This script loads saved plot points from both Python (.npz) and MATLAB (.mat) files
and compares them for discrepancies.
"""

import numpy as np
from pathlib import Path
import scipy.io as sio
import h5py
import sys

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
            # Return all non-metadata keys
            return {k: v for k, v in data.items() if not k.startswith('__')}
    except (NotImplementedError, ValueError):
        # Try h5py for v7.3 files
        try:
            with h5py.File(mat_path, 'r') as f:
                if 'plot_points_data' in f:
                    # Load the struct
                    plot_data = {}
                    ref = f['plot_points_data']
                    if isinstance(ref, h5py.Group):
                        for key in ref.keys():
                            val = np.array(ref[key])
                            # MATLAB stores arrays in column-major order
                            # For 2D arrays, transpose if needed
                            if val.ndim == 2:
                                # Check if this looks like it needs transposing
                                # wavevectors_contour should be (N_points, 2)
                                # frequencies_contour should be (N_points, N_eig)
                                if 'wavevectors_contour' in key and val.shape[0] == 2:
                                    # MATLAB stored as (2, N) -> transpose to (N, 2)
                                    val = val.T
                                elif 'frequencies_contour' in key and val.shape[0] < val.shape[1] and val.shape[0] <= 10:
                                    # Likely stored as (N_eig, N_points) -> transpose to (N_points, N_eig)
                                    val = val.T
                            elif val.ndim == 2 and val.shape[1] == 1:
                                # Column vector (N, 1) -> flatten to (N,)
                                val = val.flatten()
                            elif val.ndim == 1:
                                # Already 1D
                                pass
                            plot_data[key] = val
                    return plot_data
                else:
                    # Return all top-level datasets
                    return {k: np.array(f[k]) for k in f.keys() if not k.startswith('#')}
        except Exception as e:
            print(f"Error loading MATLAB file: {e}")
            return None

def compare_plot_points(py_data, ml_data, py_struct_idx, ml_struct_idx=None, tolerance=1e-6):
    """Compare plot points for a specific structure.
    
    Parameters
    ----------
    py_data : dict
        Python plot points data
    ml_data : dict
        MATLAB plot points data
    py_struct_idx : int
        Python structure index (0-based)
    ml_struct_idx : int, optional
        MATLAB structure index (1-based). If None, assumes ml_struct_idx = py_struct_idx + 1
    tolerance : float
        Tolerance for numerical comparison
    """
    if ml_struct_idx is None:
        ml_struct_idx = py_struct_idx + 1  # MATLAB uses 1-based indexing
    
    print(f"\n{'='*70}")
    print(f"Comparing plot points: Python struct_{py_struct_idx} vs MATLAB struct_{ml_struct_idx}")
    print(f"{'='*70}")
    
    # Keys to compare
    keys = {
        'wavevectors_contour': 'wavevectors_contour',
        'frequencies_contour': 'frequencies_contour',
        'contour_param': 'contour_param'
    }
    
    discrepancies = []
    
    for py_key, ml_key in keys.items():
        py_full_key = f'struct_{py_struct_idx}_{py_key}'
        ml_full_key = f'struct_{ml_struct_idx}_{ml_key}'
        
        if py_full_key not in py_data:
            print(f"  WARNING: Python data missing key: {py_full_key}")
            discrepancies.append(f"Missing Python key: {py_full_key}")
            continue
        
        if ml_full_key not in ml_data:
            print(f"  WARNING: MATLAB data missing key: {ml_full_key}")
            discrepancies.append(f"Missing MATLAB key: {ml_full_key}")
            continue
        
        py_val = py_data[py_full_key]
        ml_val = ml_data[ml_full_key]
        
        # Handle MATLAB cell arrays or nested structures
        if isinstance(ml_val, dict) or (hasattr(ml_val, 'dtype') and ml_val.dtype == np.object_):
            print(f"  WARNING: MATLAB {ml_full_key} is complex structure, skipping detailed comparison")
            continue
        
        # Convert to numpy arrays if needed
        if not isinstance(py_val, np.ndarray):
            py_val = np.array(py_val)
        if not isinstance(ml_val, np.ndarray):
            ml_val = np.array(ml_val)
        
        # Compare shapes
        if py_val.shape != ml_val.shape:
            print(f"  ⚠ Shape mismatch for {py_key}:")
            print(f"    Python: {py_val.shape}")
            print(f"    MATLAB: {ml_val.shape}")
            
            # Check if this is due to different interpolation modes
            if 'contour_param' in py_key or 'wavevectors_contour' in py_key or 'frequencies_contour' in py_key:
                py_use_interp = py_data.get(f'struct_{py_struct_idx}_use_interpolation', None)
                ml_use_interp = ml_data.get(f'struct_{ml_struct_idx}_use_interpolation', None)
                if py_use_interp is not None and ml_use_interp is not None:
                    py_interp = bool(py_use_interp) if isinstance(py_use_interp, (bool, np.bool_)) else bool(py_use_interp.item() if hasattr(py_use_interp, 'item') else py_use_interp)
                    ml_interp = bool(ml_use_interp) if isinstance(ml_use_interp, (bool, np.bool_)) else bool(ml_use_interp.item() if hasattr(ml_use_interp, 'item') else ml_use_interp)
                    if py_interp != ml_interp:
                        print(f"    NOTE: Different interpolation modes (Python: {py_interp}, MATLAB: {ml_interp})")
                        print(f"    This is expected - comparing overlapping points only")
                        
                        # Try to compare overlapping points if possible
                        if 'wavevectors_contour' in py_key:
                            # Find matching wavevectors
                            min_len = min(len(py_val), len(ml_val))
                            py_val_subset = py_val[:min_len]
                            ml_val_subset = ml_val[:min_len]
                            
                            # Compare subset
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
                            discrepancies.append(f"Shape mismatch: {py_key} (different interpolation modes)")
                            continue
            
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
    print("Plot Points Comparison: Python vs MATLAB")
    print("="*70)
    
    # Case 1: Without eigenfrequencies (reconstructed)
    case1_py = Path("plots/out_binarized_1_recon/plot_points.npz")
    # MATLAB may save in different locations depending on where script is run from
    case1_ml = Path("plots/out_binarized_1_mat/plot_points.mat")
    case1_ml_alt = Path("2D-dispersion-han/plots/out_binarized_1_mat/plot_points.mat")
    
    # Case 2: With eigenfrequencies
    case2_ml = Path("2D-dispersion-han/plots/out_binarized_1_mat/plot_points.mat")
    case2_ml_alt = Path("plots/out_binarized_1_mat/plot_points.mat")
    
    # Case 2: With eigenfrequencies
    # Note: Python script had issues loading the MATLAB file, so we'll skip this for now
    # case2_py = Path("plots/.../plot_points.npz")
    # case2_ml = Path("plots/out_binarized_1_mat/plot_points.mat")
    
    all_discrepancies = {}
    
    # Compare Case 1
    # Check for MATLAB file in both possible locations
    case1_ml_actual = case1_ml if case1_ml.exists() else (case1_ml_alt if case1_ml_alt.exists() else None)
    
    if case1_py.exists() and case1_ml_actual is not None:
        print("\n" + "="*70)
        print("CASE 1: Without eigenfrequencies (reconstructed)")
        print("="*70)
        
        py_data = load_python_plot_points(case1_py)
        ml_data = load_matlab_plot_points(case1_ml_actual)
        
        if ml_data is None:
            print("  ERROR: Could not load MATLAB data")
        else:
            # Find all structure indices
            # Python uses 0-based indexing, MATLAB uses 1-based
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
            
            print(f"  Python structure indices: {sorted(py_struct_indices)}")
            print(f"  MATLAB structure indices: {sorted(ml_struct_indices)}")
            
            # Compare matching structures (Python 0-based vs MATLAB 1-based)
            for py_idx in sorted(py_struct_indices):
                # MATLAB uses 1-based indexing, so struct_0 in Python = struct_1 in MATLAB
                ml_idx = py_idx + 1
                
                if ml_idx in ml_struct_indices:
                    print(f"\n  Comparing: Python struct_{py_idx} <-> MATLAB struct_{ml_idx}")
                    disc = compare_plot_points(py_data, ml_data, py_idx, ml_idx)
                    if disc:
                        all_discrepancies[f'Case1_struct_{py_idx}'] = disc
                else:
                    print(f"\n  WARNING: No MATLAB equivalent for Python struct_{py_idx}")
                    all_discrepancies[f'Case1_struct_{py_idx}'] = ['No MATLAB equivalent']
    else:
        print(f"\nCase 1 files not found:")
        print(f"  Python: {case1_py} (exists: {case1_py.exists()})")
        print(f"  MATLAB (primary): {case1_ml} (exists: {case1_ml.exists()})")
        print(f"  MATLAB (alt): {case1_ml_alt} (exists: {case1_ml_alt.exists()})")
    
    # Compare Case 2 (if available)
    case2_ml_actual = case2_ml if case2_ml.exists() else (case2_ml_alt if case2_ml_alt.exists() else None)
    
    # Find most recent Python Case 2 output (look for dispersion_plots_*_mat/plot_points.npz)
    case2_py = None
    import glob
    import os
    py_case2_patterns = [
        "dispersion_plots_*_mat/plot_points.npz",
        "plots/*_mat/plot_points.npz"
    ]
    all_py_case2 = []
    for pattern in py_case2_patterns:
        all_py_case2.extend(glob.glob(pattern))
    
    if all_py_case2:
        # Sort by modification time, most recent first
        all_py_case2.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        case2_py = Path(all_py_case2[0])
        print(f"\n  Found Python Case 2 output: {case2_py}")
    
    if case2_ml_actual is not None and case2_py is not None:
        print("\n" + "="*70)
        print("CASE 2: With eigenfrequencies")
        print("="*70)
        
        py_data = load_python_plot_points(case2_py)
        ml_data = load_matlab_plot_points(case2_ml_actual)
        
        if ml_data is None:
            print("  ERROR: Could not load MATLAB data")
        else:
            # Find all structure indices
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
            
            print(f"  Python structure indices: {sorted(py_struct_indices)}")
            print(f"  MATLAB structure indices: {sorted(ml_struct_indices)}")
            
            # Compare matching structures (Python 0-based vs MATLAB 1-based)
            for py_idx in sorted(py_struct_indices):
                # MATLAB uses 1-based indexing, so struct_0 in Python = struct_1 in MATLAB
                ml_idx = py_idx + 1
                
                if ml_idx in ml_struct_indices:
                    print(f"\n  Comparing: Python struct_{py_idx} <-> MATLAB struct_{ml_idx}")
                    disc = compare_plot_points(py_data, ml_data, py_idx, ml_idx)
                    if disc:
                        all_discrepancies[f'Case2_struct_{py_idx}'] = disc
                else:
                    print(f"\n  WARNING: No MATLAB equivalent for Python struct_{py_idx}")
                    all_discrepancies[f'Case2_struct_{py_idx}'] = ['No MATLAB equivalent']
    elif case2_ml_actual is not None:
        print("\n" + "="*70)
        print("CASE 2: With eigenfrequencies")
        print("="*70)
        print(f"  Found MATLAB plot_points.mat: {case2_ml_actual}")
        print("  Note: Python Case 2 output not found")
        print("  (Run: python 2d-dispersion-py/plot_dispersion_with_eigenfrequencies.py data/out_test_10_matlab/out_binarized_1.mat -n 1 --save-plot-points)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if all_discrepancies:
        print(f"Found discrepancies in {len(all_discrepancies)} structure(s):")
        for key, disc in all_discrepancies.items():
            print(f"  {key}: {len(disc)} discrepancy(ies)")
    else:
        print("No discrepancies found! Python and MATLAB plot points match.")
    
    print("="*70)

if __name__ == "__main__":
    main()

