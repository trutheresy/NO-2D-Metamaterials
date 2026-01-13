"""Detailed comparison of Python and MATLAB plot points."""
import numpy as np
import h5py
from pathlib import Path
import sys
sys.path.insert(0, '2d-dispersion-py')
from mat73_loader import load_matlab_v73

# Find most recent Python output
py_files = sorted(Path('.').glob('dispersion_plots_*_mat/plot_points.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
if not py_files:
    raise FileNotFoundError("No Python plot_points.npz files found")
py_file = py_files[0]
print(f"Using Python file: {py_file}")
py_data = np.load(py_file)

# Load MATLAB data
ml_file = Path('2D-dispersion-han/plots/out_binarized_1_mat/plot_points.mat')

print("="*70)
print("DETAILED COMPARISON: Python vs MATLAB Plot Points")
print("="*70)

# Load MATLAB data
with h5py.File(ml_file, 'r') as f:
    ml_pp = f['plot_points_data']
    
    # Get struct_0 (Python) vs struct_1 (MATLAB)
    py_wv = py_data['struct_0_wavevectors_contour']
    py_freq = py_data['struct_0_frequencies_contour']
    py_param = py_data['struct_0_contour_param']
    
    ml_wv_raw = np.array(ml_pp['struct_1_wavevectors_contour'])
    ml_freq_raw = np.array(ml_pp['struct_1_frequencies_contour'])
    ml_param_raw = np.array(ml_pp['struct_1_contour_param'])
    
    # Transpose MATLAB arrays (stored as (2, 55) and (6, 55))
    ml_wv = ml_wv_raw.T if ml_wv_raw.shape[0] == 2 else ml_wv_raw
    ml_freq = ml_freq_raw.T if ml_freq_raw.shape[0] == 6 else ml_freq_raw
    ml_param = ml_param_raw.flatten() if ml_param_raw.ndim == 2 else ml_param_raw
    
    print("\n1. WAVEVECTORS_CONTOUR")
    print("-" * 70)
    print(f"   Python shape: {py_wv.shape}")
    print(f"   MATLAB shape: {ml_wv.shape}")
    if py_wv.shape == ml_wv.shape:
        diff_wv = np.abs(py_wv - ml_wv)
        max_diff = np.max(diff_wv)
        mean_diff = np.mean(diff_wv)
        if np.allclose(py_wv, ml_wv, rtol=1e-10, atol=1e-10):
            print(f"   ✓ MATCH EXACTLY")
            print(f"   Max absolute difference: {max_diff:.2e}")
            print(f"   Mean absolute difference: {mean_diff:.2e}")
        else:
            print(f"   ✗ DO NOT MATCH")
            print(f"   Max absolute difference: {max_diff:.2e}")
            print(f"   Mean absolute difference: {mean_diff:.2e}")
    else:
        print(f"   ✗ SHAPE MISMATCH")
    
    print("\n2. FREQUENCIES_CONTOUR")
    print("-" * 70)
    print(f"   Python shape: {py_freq.shape}")
    print(f"   MATLAB shape: {ml_freq.shape}")
    print(f"   Python NaN count: {np.isnan(py_freq).sum()}/{py_freq.size}")
    print(f"   MATLAB NaN count: {np.isnan(ml_freq).sum()}/{ml_freq.size}")
    
    if py_freq.shape == ml_freq.shape:
        if np.isnan(ml_freq).all():
            print(f"   ✗ MATLAB VALUES ARE ALL NaN")
            print(f"   This indicates MATLAB script needs to be re-run with the fix")
        elif np.isnan(py_freq).all():
            print(f"   ✗ PYTHON VALUES ARE ALL NaN")
        else:
            # Compare non-NaN values
            valid_mask = ~(np.isnan(py_freq) | np.isnan(ml_freq))
            if valid_mask.sum() > 0:
                py_valid = py_freq[valid_mask]
                ml_valid = ml_freq[valid_mask]
                diff_freq = np.abs(py_valid - ml_valid)
                max_diff = np.max(diff_freq)
                mean_diff = np.mean(diff_freq)
                rel_error = max_diff / (np.max(np.abs(ml_valid)) + 1e-15)
                
                print(f"   Valid values: {valid_mask.sum()}/{py_freq.size}")
                print(f"   Max absolute difference: {max_diff:.6e}")
                print(f"   Mean absolute difference: {mean_diff:.6e}")
                print(f"   Max relative error: {rel_error:.6e}")
                
                if np.allclose(py_valid, ml_valid, rtol=1e-6, atol=1e-6):
                    print(f"   ✓ MATCH (within tolerance)")
                else:
                    print(f"   ✗ DO NOT MATCH")
            else:
                print(f"   ✗ NO VALID VALUES TO COMPARE")
    else:
        print(f"   ✗ SHAPE MISMATCH")
    
    print("\n3. CONTOUR_PARAM")
    print("-" * 70)
    print(f"   Python shape: {py_param.shape}")
    print(f"   MATLAB shape: {ml_param.shape}")
    if py_param.shape == ml_param.shape:
        diff_param = np.abs(py_param - ml_param)
        max_diff = np.max(diff_param)
        mean_diff = np.mean(diff_param)
        if np.allclose(py_param, ml_param, rtol=1e-10, atol=1e-10):
            print(f"   ✓ MATCH EXACTLY")
            print(f"   Max absolute difference: {max_diff:.2e}")
            print(f"   Mean absolute difference: {mean_diff:.2e}")
        else:
            print(f"   ✗ DO NOT MATCH")
            print(f"   Max absolute difference: {max_diff:.2e}")
            print(f"   Mean absolute difference: {mean_diff:.2e}")
    else:
        print(f"   ✗ SHAPE MISMATCH")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    matches = []
    discrepancies = []
    
    if py_wv.shape == ml_wv.shape and np.allclose(py_wv, ml_wv, rtol=1e-10, atol=1e-10):
        matches.append("wavevectors_contour")
    else:
        discrepancies.append("wavevectors_contour")
    
    if not np.isnan(ml_freq).all() and py_freq.shape == ml_freq.shape:
        valid_mask = ~(np.isnan(py_freq) | np.isnan(ml_freq))
        if valid_mask.sum() > 0:
            if np.allclose(py_freq[valid_mask], ml_freq[valid_mask], rtol=1e-6, atol=1e-6):
                matches.append("frequencies_contour")
            else:
                discrepancies.append("frequencies_contour")
        else:
            discrepancies.append("frequencies_contour (no valid values)")
    elif np.isnan(ml_freq).all():
        discrepancies.append("frequencies_contour (MATLAB all NaN - needs re-run)")
    else:
        discrepancies.append("frequencies_contour")
    
    if py_param.shape == ml_param.shape and np.allclose(py_param, ml_param, rtol=1e-10, atol=1e-10):
        matches.append("contour_param")
    else:
        discrepancies.append("contour_param")
    
    print(f"\n✓ MATCHING: {len(matches)}/{3}")
    for m in matches:
        print(f"  - {m}")
    
    print(f"\n✗ DISCREPANCIES: {len(discrepancies)}/{3}")
    for d in discrepancies:
        print(f"  - {d}")
    
    print("="*70)

