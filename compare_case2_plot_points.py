"""Compare Case 2 (with eigenfrequencies) plot points between Python and MATLAB."""

import numpy as np
import h5py
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from mat73_loader import load_matlab_v73

def load_matlab_plot_points(mat_path):
    """Load plot points from MATLAB .mat file."""
    data = load_matlab_v73(mat_path, verbose=False)
    if 'plot_points_data' not in data:
        return None
    
    plot_data = {}
    pp_data = data['plot_points_data']
    
    # Handle different MATLAB struct formats
    if isinstance(pp_data, dict):
        for key, val in pp_data.items():
            if isinstance(val, np.ndarray):
                # Handle transposed arrays
                if val.ndim == 2:
                    if 'wavevectors_contour' in key and val.shape[0] == 2:
                        val = val.T
                    elif 'frequencies_contour' in key and val.shape[0] < val.shape[1] and val.shape[0] <= 10:
                        val = val.T
                elif val.ndim == 2 and val.shape[1] == 1:
                    val = val.flatten()
                plot_data[key] = val
            else:
                plot_data[key] = val
    else:
        # Try to extract from structured array
        if hasattr(pp_data, 'dtype') and pp_data.dtype.names:
            for field_name in pp_data.dtype.names:
                val = pp_data[field_name]
                if isinstance(val, np.ndarray):
                    if val.ndim == 2 and val.shape[0] == 2 and 'wavevectors' in field_name:
                        val = val.T
                    elif val.ndim == 2 and val.shape[1] == 1:
                        val = val.flatten()
                plot_data[field_name] = val
    
    return plot_data

def load_python_plot_points(npz_path):
    """Load plot points from Python .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return dict(data)

def compare_case2(py_file, ml_file, struct_idx_py=0, struct_idx_ml=1, tolerance=1e-6):
    """Compare Case 2 plot points for a specific structure."""
    print("="*70)
    print(f"Case 2 Comparison: Python struct_{struct_idx_py} vs MATLAB struct_{struct_idx_ml}")
    print("="*70)
    
    # Load data
    py_data = load_python_plot_points(py_file)
    ml_data = load_matlab_plot_points(ml_file)
    
    if ml_data is None:
        print("ERROR: Could not load MATLAB data")
        return
    
    # Keys to compare
    keys = ['wavevectors_contour', 'frequencies_contour', 'contour_param', 'use_interpolation']
    
    discrepancies = []
    
    for key in keys:
        py_full_key = f'struct_{struct_idx_py}_{key}'
        ml_full_key = f'struct_{struct_idx_ml}_{key}'
        
        if py_full_key not in py_data:
            print(f"  ✗ Python missing: {py_full_key}")
            discrepancies.append(f"Missing Python key: {py_full_key}")
            continue
        
        if ml_full_key not in ml_data:
            print(f"  ✗ MATLAB missing: {ml_full_key}")
            discrepancies.append(f"Missing MATLAB key: {ml_full_key}")
            continue
        
        py_val = py_data[py_full_key]
        ml_val = ml_data[ml_full_key]
        
        # Handle use_interpolation flag
        if key == 'use_interpolation':
            py_interp = bool(py_val) if isinstance(py_val, (bool, np.bool_)) else bool(py_val.item() if hasattr(py_val, 'item') else py_val)
            ml_interp = bool(ml_val) if isinstance(ml_val, (bool, np.bool_)) else bool(ml_val.item() if hasattr(ml_val, 'item') else ml_val)
            print(f"\n  {key}:")
            print(f"    Python: {py_interp}")
            print(f"    MATLAB: {ml_interp}")
            if py_interp == ml_interp:
                print(f"    ✓ Match")
            else:
                print(f"    ✗ Mismatch")
                discrepancies.append(f"{key}: Python={py_interp}, MATLAB={ml_interp}")
            continue
        
        # Convert to numpy arrays
        if not isinstance(py_val, np.ndarray):
            py_val = np.array(py_val)
        if not isinstance(ml_val, np.ndarray):
            ml_val = np.array(ml_val)
        
        # Compare shapes
        print(f"\n  {key}:")
        print(f"    Python shape: {py_val.shape}")
        print(f"    MATLAB shape: {ml_val.shape}")
        
        if py_val.shape != ml_val.shape:
            print(f"    ✗ Shape mismatch")
            discrepancies.append(f"{key}: shape mismatch")
            
            # Try to compare overlapping portion
            if py_val.ndim == ml_val.ndim:
                min_len = min(py_val.shape[0], ml_val.shape[0])
                if min_len > 0:
                    print(f"    Comparing first {min_len} points...")
                    py_subset = py_val[:min_len]
                    ml_subset = ml_val[:min_len]
                    if np.allclose(py_subset, ml_subset, rtol=tolerance, atol=tolerance):
                        print(f"    ✓ First {min_len} points match")
                    else:
                        diff = np.abs(py_subset - ml_subset)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        rel_error = max_diff / (np.max(np.abs(ml_subset)) + 1e-15)
                        print(f"    ✗ First {min_len} points differ")
                        print(f"      Max abs diff: {max_diff:.6e}")
                        print(f"      Mean abs diff: {mean_diff:.6e}")
                        print(f"      Max rel error: {rel_error:.6e}")
                        discrepancies.append(f"{key}: value mismatch in overlapping region")
            continue
        
        # Compare values
        if np.allclose(py_val, ml_val, rtol=tolerance, atol=tolerance):
            print(f"    ✓ Values match")
        else:
            diff = np.abs(py_val - ml_val)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            rel_error = max_diff / (np.max(np.abs(ml_val)) + 1e-15)
            
            print(f"    ✗ Values differ")
            print(f"      Max abs diff: {max_diff:.6e}")
            print(f"      Mean abs diff: {mean_diff:.6e}")
            print(f"      Max rel error: {rel_error:.6e}")
            
            # Show some sample differences
            if py_val.size > 0:
                idx_max = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"      Max diff at index {idx_max}:")
                print(f"        Python: {py_val[idx_max]:.6e}")
                print(f"        MATLAB: {ml_val[idx_max]:.6e}")
            
            discrepancies.append({
                'key': key,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'rel_error': rel_error
            })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if discrepancies:
        print(f"Found {len(discrepancies)} discrepancy(ies)")
        for disc in discrepancies:
            if isinstance(disc, dict):
                print(f"  {disc['key']}: max_diff={disc['max_diff']:.6e}, rel_error={disc['rel_error']:.6e}")
            else:
                print(f"  {disc}")
    else:
        print("✓ All values match within tolerance!")
    print("="*70)
    
    return discrepancies

if __name__ == "__main__":
    # Find latest Python output
    py_files = list(Path('.').glob('dispersion_plots_*_mat/plot_points.npz'))
    if not py_files:
        print("ERROR: No Python plot_points.npz files found")
        sys.exit(1)
    
    py_file = sorted(py_files, key=lambda p: p.stat().st_mtime)[-1]  # Most recent
    ml_file = Path('2D-dispersion-han/plots/out_binarized_1_mat/plot_points.mat')
    
    print(f"Python file: {py_file}")
    print(f"MATLAB file: {ml_file}")
    print()
    
    if not py_file.exists():
        print(f"ERROR: Python file not found: {py_file}")
        sys.exit(1)
    
    if not ml_file.exists():
        print(f"ERROR: MATLAB file not found: {ml_file}")
        sys.exit(1)
    
    compare_case2(py_file, ml_file)

