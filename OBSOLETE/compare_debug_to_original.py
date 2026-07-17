"""
Compare debug reconstruction output to original MATLAB data.
Compares K, M matrices and eigenvalue_data.
"""

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from pathlib import Path
import sys

# Import the safe loader function for original file
sys.path.insert(0, '.')
from compare_matlab_files import load_mat_data_safe

# Import mat73_loader for better MATLAB v7.3 support
sys.path.insert(0, '2d-dispersion-py')
try:
    from mat73_loader import load_matlab_v73
    HAS_MAT73_LOADER = True
except ImportError:
    HAS_MAT73_LOADER = False
    print("Warning: mat73_loader not available, using scipy.io.loadmat")

# Paths
original_mat_path = Path(r"D:\Research\NO-2D-Metamaterials\data\out_test_10_mat_original\out_binarized_1.mat")
debug_data_path = Path(r"D:\Research\NO-2D-Metamaterials\2D-dispersion-han\debug\reconstruction_debug_data.mat")

print("=" * 80)
print("COMPARING DEBUG RECONSTRUCTION TO ORIGINAL MATLAB DATA")
print("=" * 80)
print(f"\nOriginal file: {original_mat_path}")
print(f"Debug data file: {debug_data_path}")

# Load original MATLAB file
print("\nLoading original MATLAB file...")
if not original_mat_path.exists():
    print(f"  ERROR: Original file not found: {original_mat_path}")
    sys.exit(1)

try:
    if HAS_MAT73_LOADER:
        orig_data = load_matlab_v73(str(original_mat_path), verbose=False)
        print("  Successfully loaded original file using mat73_loader")
    else:
        orig_data = load_mat_data_safe(original_mat_path)
        print("  Successfully loaded original file using load_mat_data_safe")
except Exception as e:
    print(f"  ERROR loading original file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load debug data
print("\nLoading debug reconstruction data...")
if not debug_data_path.exists():
    print(f"  ERROR: Debug file not found: {debug_data_path}")
    print("  Make sure the MATLAB script has completed running.")
    sys.exit(1)

try:
    debug_data = sio.loadmat(str(debug_data_path))
    # Remove MATLAB metadata
    debug_data = {k: v for k, v in debug_data.items() if not k.startswith('__')}
    print("  Successfully loaded debug file")
    print(f"  Keys: {list(debug_data.keys())[:20]}...")  # First 20 keys
except Exception as e:
    print(f"  ERROR loading debug file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare K and M matrices
print("\n" + "=" * 80)
print("COMPARING K AND M MATRICES")
print("=" * 80)

struct_idx = 1  # Compare first structure (MATLAB uses 1-indexed)

# Get K and M from original file
K_orig = None
M_orig = None

if 'K_DATA' in orig_data and 'M_DATA' in orig_data:
    K_data_orig = orig_data['K_DATA']
    M_data_orig = orig_data['M_DATA']
    
    print(f"\nOriginal K_DATA type: {type(K_data_orig)}")
    print(f"Original M_DATA type: {type(M_data_orig)}")
    
    # Handle different formats
    if isinstance(K_data_orig, (list, np.ndarray)):
        if isinstance(K_data_orig, np.ndarray) and K_data_orig.dtype == object:
            # Cell array format
            if K_data_orig.size > struct_idx - 1:
                K_orig = K_data_orig.flat[struct_idx - 1]
                M_orig = M_data_orig.flat[struct_idx - 1]
                print(f"  Extracted K and M for structure {struct_idx} (cell array format)")
        elif isinstance(K_data_orig, list):
            if len(K_data_orig) > struct_idx - 1:
                K_orig = K_data_orig[struct_idx - 1]
                M_orig = M_data_orig[struct_idx - 1]
                print(f"  Extracted K and M for structure {struct_idx} (list format)")
    
    if K_orig is not None:
        if not sp.issparse(K_orig):
            K_orig = sp.csr_matrix(K_orig)
        if not sp.issparse(M_orig):
            M_orig = sp.csr_matrix(M_orig)
        print(f"  K_orig: shape={K_orig.shape}, nnz={K_orig.nnz}")
        print(f"  M_orig: shape={M_orig.shape}, nnz={M_orig.nnz}")
    else:
        print("  Could not extract K and M from original file")
else:
    print("  K_DATA or M_DATA not found in original file")

# Get K and M from debug data
K_debug = None
M_debug = None

if 'debug_K_M_data' in debug_data:
    km_data = debug_data['debug_K_M_data']
    print(f"\nDebug K_M_data type: {type(km_data)}")
    
    # Extract field names from structured array
    if isinstance(km_data, np.ndarray) and km_data.dtype.names:
        field_names = km_data.dtype.names
        print(f"  Field names: {field_names[:10]}...")  # First 10
        
        K_key = f'struct_{struct_idx}_K'
        M_key = f'struct_{struct_idx}_M'
        
        if K_key in field_names and M_key in field_names:
            K_debug = km_data[K_key][0, 0]
            M_debug = km_data[M_key][0, 0]
            
            print(f"  Extracted K and M from debug data for structure {struct_idx}")
            
            # Convert to sparse if needed
            if not sp.issparse(K_debug):
                K_debug = sp.csr_matrix(K_debug)
            if not sp.issparse(M_debug):
                M_debug = sp.csr_matrix(M_debug)
            
            print(f"  K_debug: shape={K_debug.shape}, nnz={K_debug.nnz}")
            print(f"  M_debug: shape={M_debug.shape}, nnz={M_debug.nnz}")
        else:
            print(f"  Keys {K_key} or {M_key} not found in debug data")
            print(f"  Available keys: {[k for k in field_names if 'struct_' in k][:10]}")
    else:
        print(f"  Unexpected debug_K_M_data format: {type(km_data)}")
else:
    print("  debug_K_M_data not found in debug file")

# Compare if we have both
if K_orig is not None and K_debug is not None and M_orig is not None and M_debug is not None:
    print(f"\n  Comparing K and M matrices...")
    
    # Compare shapes
    if K_debug.shape == K_orig.shape and M_debug.shape == M_orig.shape:
        # Compare values
        K_diff = K_debug - K_orig
        M_diff = M_debug - M_orig
        
        K_max_diff = np.max(np.abs(K_diff.data)) if K_diff.nnz > 0 else 0
        M_max_diff = np.max(np.abs(M_diff.data)) if M_diff.nnz > 0 else 0
        
        # Also compute relative differences
        K_orig_max = np.max(np.abs(K_orig.data)) if K_orig.nnz > 0 else 1
        M_orig_max = np.max(np.abs(M_orig.data)) if M_orig.nnz > 0 else 1
        
        K_rel_diff = K_max_diff / K_orig_max if K_orig_max > 0 else 0
        M_rel_diff = M_max_diff / M_orig_max if M_orig_max > 0 else 0
        
        print(f"\n  Comparison results:")
        print(f"    K max difference: {K_max_diff:.6e} (relative: {K_rel_diff:.6e})")
        print(f"    M max difference: {M_max_diff:.6e} (relative: {M_rel_diff:.6e})")
        
        if K_max_diff < 1e-10 and M_max_diff < 1e-10:
            print(f"    ✅ K and M matrices match exactly!")
        elif K_max_diff < 1e-6 and M_max_diff < 1e-6:
            print(f"    ✅ K and M matrices match closely (numerical precision)")
        else:
            print(f"    ⚠️  K and M matrices have differences")
    else:
        print(f"    ❌ Shape mismatch!")
        print(f"    K: {K_debug.shape} vs {K_orig.shape}")
        print(f"    M: {M_debug.shape} vs {M_orig.shape}")

# Compare eigenvalues
print("\n" + "=" * 80)
print("COMPARING EIGENVALUE DATA")
print("=" * 80)

eigenvalue_orig = None
if 'EIGENVALUE_DATA' in orig_data:
    eigenvalue_orig = orig_data['EIGENVALUE_DATA']
    print(f"\nOriginal EIGENVALUE_DATA shape: {eigenvalue_orig.shape}")
else:
    print("\nEIGENVALUE_DATA not found in original file")

eigenvalue_debug = None
if 'EIGENVALUE_DATA' in debug_data:
    eigenvalue_debug = debug_data['EIGENVALUE_DATA']
    print(f"Debug EIGENVALUE_DATA shape: {eigenvalue_debug.shape}")
else:
    print("\nEIGENVALUE_DATA not found in debug file")

if eigenvalue_orig is not None and eigenvalue_debug is not None:
    print(f"\nComparing eigenvalue data...")
    
    # Handle shape differences (MATLAB may transpose)
    orig_shape = eigenvalue_orig.shape
    debug_shape = eigenvalue_debug.shape
    
    print(f"  Original shape: {orig_shape}")
    print(f"  Debug shape: {debug_shape}")
    
    # Try to align shapes - debug should be (N_wv, N_eig, N_struct)
    if len(orig_shape) == 3 and len(debug_shape) == 3:
        # Original might be (N_struct, N_eig, N_wv) or (N_wv, N_eig, N_struct)
        if orig_shape[0] == debug_shape[2] and orig_shape[2] == debug_shape[0]:
            # Likely transposed: (N_struct, N_eig, N_wv) -> (N_wv, N_eig, N_struct)
            eigenvalue_orig_aligned = eigenvalue_orig.transpose(2, 1, 0)
            print(f"  Transposed original to: {eigenvalue_orig_aligned.shape}")
        elif orig_shape == debug_shape:
            eigenvalue_orig_aligned = eigenvalue_orig
        else:
            print(f"  ⚠️  Cannot automatically align shapes")
            eigenvalue_orig_aligned = None
        
        if eigenvalue_orig_aligned is not None and eigenvalue_orig_aligned.shape == debug_shape:
            # Compare
            diff = np.abs(eigenvalue_debug - eigenvalue_orig_aligned)
            valid_mask = ~(np.isnan(eigenvalue_debug) | np.isnan(eigenvalue_orig_aligned) |
                          np.isinf(eigenvalue_debug) | np.isinf(eigenvalue_orig_aligned))
            
            if np.any(valid_mask):
                diff_valid = diff[valid_mask]
                orig_valid = eigenvalue_orig_aligned[valid_mask]
                rel_diff = diff_valid / (np.abs(orig_valid) + 1e-20)
                
                print(f"\n  Comparison statistics:")
                print(f"    Valid entries: {np.sum(valid_mask)} / {eigenvalue_debug.size}")
                print(f"    Max abs diff: {np.max(diff_valid):.6e} Hz")
                print(f"    Mean abs diff: {np.mean(diff_valid):.6e} Hz")
                print(f"    Max rel diff: {np.max(rel_diff):.6e} ({np.max(rel_diff)*100:.2f}%)")
                print(f"    Mean rel diff: {np.mean(rel_diff):.6e} ({np.mean(rel_diff)*100:.2f}%)")
                
                if np.max(rel_diff) < 1e-6:
                    print(f"    ✅ Eigenvalues match very well!")
                elif np.max(rel_diff) < 0.01:
                    print(f"    ✅ Eigenvalues match well (within 1%)")
                elif np.max(rel_diff) < 0.05:
                    print(f"    ⚠️  Eigenvalues have small differences (within 5%)")
                else:
                    print(f"    ❌ Eigenvalues have significant differences")
            else:
                print(f"    ❌ No valid entries for comparison")
        else:
            print(f"    ⚠️  Shape mismatch after alignment")
    else:
        print(f"    ⚠️  Unexpected shape dimensions")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

