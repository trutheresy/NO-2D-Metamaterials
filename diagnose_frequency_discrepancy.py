"""
Diagnostic script to identify sources of eigenfrequency discrepancies.

This script compares intermediate values in the eigenfrequency reconstruction
process between original and reconstructed datasets.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys

# Import loaders
sys.path.insert(0, '.')
from compare_matlab_files import load_mat_data_safe

sys.path.insert(0, '2d-dispersion-py')
from mat73_loader import load_matlab_v73

# Paths
path_original = Path(r"D:\Research\NO-2D-Metamaterials\data\out_test_10_mat_original\out_binarized_1_mat")
path_reconstructed = Path(r"D:\Research\NO-2D-Metamaterials\data\out_test_10_mat_reconstructed\out_binarized_1_mat")

print("=" * 80)
print("DIAGNOSING EIGENFREQUENCY RECONSTRUCTION DISCREPANCIES")
print("=" * 80)

# Load K, M, T matrices and eigenvectors
print("\n1. Loading K, M, T matrices and eigenvectors...")
kmt_orig = path_original / "computed_K_M_T_matrices.mat"
kmt_recon = path_reconstructed / "computed_K_M_T_matrices.mat"

data_orig_kmt = load_matlab_v73(str(kmt_orig), verbose=False)
data_recon_kmt = load_matlab_v73(str(kmt_recon), verbose=False)

# Load full datasets for eigenvectors
mat_orig = Path(r"D:\Research\NO-2D-Metamaterials\data\out_test_10_mat_original\out_binarized_1.mat")
mat_recon = Path(r"D:\Research\NO-2D-Metamaterials\data\out_test_10_mat_reconstructed\out_binarized_1.mat")

data_orig = load_matlab_v73(str(mat_orig), verbose=False)
data_recon = load_matlab_v73(str(mat_recon), verbose=False)

struct_idx = 0  # First structure
wv_idx = 0  # First wavevector
band_idx = 0  # First band

print(f"\n2. Comparing intermediate values for structure {struct_idx+1}, wavevector {wv_idx+1}, band {band_idx+1}...")

# Extract K, M matrices
if 'debug_K_M_data' in data_orig_kmt:
    K_orig = data_orig_kmt['debug_K_M_data'][f'struct_{struct_idx+1}_K']
    M_orig = data_orig_kmt['debug_K_M_data'][f'struct_{struct_idx+1}_M']
else:
    print("ERROR: Could not find K, M in original file")
    sys.exit(1)

if 'debug_K_M_data' in data_recon_kmt:
    K_recon = data_recon_kmt['debug_K_M_data'][f'struct_{struct_idx+1}_K']
    M_recon = data_recon_kmt['debug_K_M_data'][f'struct_{struct_idx+1}_M']
else:
    print("ERROR: Could not find K, M in reconstructed file")
    sys.exit(1)

# Convert to sparse if needed
if not sp.issparse(K_orig):
    K_orig = sp.csr_matrix(K_orig)
if not sp.issparse(M_orig):
    M_orig = sp.csr_matrix(M_orig)
if not sp.issparse(K_recon):
    K_recon = sp.csr_matrix(K_recon)
if not sp.issparse(M_recon):
    M_recon = sp.csr_matrix(M_recon)

print(f"   K shapes: {K_orig.shape} vs {K_recon.shape}")
print(f"   M shapes: {M_orig.shape} vs {M_recon.shape}")

# Extract T matrices - need to compute from wavevectors
sys.path.insert(0, '2d-dispersion-py')
from system_matrices import get_transformation_matrix

# Get wavevectors and const
if 'const' in data_orig:
    const_orig = data_orig['const']
    if isinstance(const_orig, dict):
        wavevectors_orig = const_orig.get('wavevectors', None)
    else:
        wavevectors_orig = getattr(const_orig, 'wavevectors', None)
else:
    wavevectors_orig = None

if 'const' in data_recon:
    const_recon = data_recon['const']
    if isinstance(const_recon, dict):
        wavevectors_recon = const_recon.get('wavevectors', None)
    else:
        wavevectors_recon = getattr(const_recon, 'wavevectors', None)
else:
    wavevectors_recon = None

if wavevectors_orig is None or wavevectors_recon is None:
    print("ERROR: Could not find wavevectors")
    sys.exit(1)

# Extract single wavevector (2D array)
wavevectors_orig = np.array(wavevectors_orig)
wavevectors_recon = np.array(wavevectors_recon)

if wavevectors_orig.ndim == 1:
    # Single wavevector case - need to check if it's actually a 2D array
    if len(wavevectors_orig) == 2:
        wv_orig = wavevectors_orig
    else:
        print(f"ERROR: Unexpected wavevector shape: {wavevectors_orig.shape}")
        sys.exit(1)
else:
    wv_orig = wavevectors_orig[wv_idx, :] if wavevectors_orig.shape[0] > wv_idx else wavevectors_orig[0, :]

if wavevectors_recon.ndim == 1:
    if len(wavevectors_recon) == 2:
        wv_recon = wavevectors_recon
    else:
        print(f"ERROR: Unexpected wavevector shape: {wavevectors_recon.shape}")
        sys.exit(1)
else:
    wv_recon = wavevectors_recon[wv_idx, :] if wavevectors_recon.shape[0] > wv_idx else wavevectors_recon[0, :]

wv_orig = np.array(wv_orig).flatten()[:2]  # Ensure 2D
wv_recon = np.array(wv_recon).flatten()[:2]  # Ensure 2D

print(f"   Wavevector {wv_idx}: {wv_orig} vs {wv_recon}")

# Convert const to dict format
def extract_const_value(const, key, as_int=False):
    """Extract a scalar value from const structure."""
    if isinstance(const, dict):
        val = const.get(key, None)
    else:
        val = getattr(const, key, None)
    
    if val is None:
        return None
    
    val = np.array(val)
    if val.ndim == 0:
        result = float(val)
    elif val.size == 1:
        result = float(val.flat[0])
    else:
        result = float(val[0, 0]) if val.ndim == 2 else float(val[0])
    
    return int(result) if as_int else result

const_orig_dict = {
    'N_pix': extract_const_value(const_orig, 'N_pix', as_int=True),
    'N_ele': extract_const_value(const_orig, 'N_ele', as_int=True),
    'a': extract_const_value(const_orig, 'a')
}

const_recon_dict = {
    'N_pix': extract_const_value(const_recon, 'N_pix', as_int=True),
    'N_ele': extract_const_value(const_recon, 'N_ele', as_int=True),
    'a': extract_const_value(const_recon, 'a')
}

print(f"   Const: N_pix={const_orig_dict['N_pix']} vs {const_recon_dict['N_pix']}")
print(f"   Const: N_ele={const_orig_dict['N_ele']} vs {const_recon_dict['N_ele']}")
print(f"   Const: a={const_orig_dict['a']} vs {const_recon_dict['a']}")

try:
    T_orig = get_transformation_matrix(wv_orig.astype(np.float32), const_orig_dict)
    T_recon = get_transformation_matrix(wv_recon.astype(np.float32), const_recon_dict)
except Exception as e:
    print(f"ERROR: Could not compute T matrices: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if T_orig is None or T_recon is None:
    print("ERROR: T matrix computation returned None")
    sys.exit(1)

print(f"   T shapes: {T_orig.shape} vs {T_recon.shape}")

# Extract eigenvectors
eigvec_orig = None
eigvec_recon = None

if 'EIGENVECTOR_DATA' in data_orig:
    eigvec_data_orig = data_orig['EIGENVECTOR_DATA']
    print(f"   EIGENVECTOR_DATA shape (orig): {eigvec_data_orig.shape}")
    # HDF5 order: (struct, band, wv, dof) based on dimension ordering guide
    if eigvec_data_orig.ndim == 4:
        # Use (struct, band, wv, dof) indexing
        eigvec_orig = eigvec_data_orig[struct_idx, band_idx, wv_idx, :]
        print(f"   Using indexing: [{struct_idx}, {band_idx}, {wv_idx}, :] -> shape {eigvec_orig.shape}")
    else:
        print(f"   ERROR: Unexpected EIGENVECTOR_DATA ndim: {eigvec_data_orig.ndim}")
        sys.exit(1)

if 'EIGENVECTOR_DATA' in data_recon:
    eigvec_data_recon = data_recon['EIGENVECTOR_DATA']
    print(f"   EIGENVECTOR_DATA shape (recon): {eigvec_data_recon.shape}")
    if eigvec_data_recon.ndim == 4:
        # Use (struct, band, wv, dof) indexing
        eigvec_recon = eigvec_data_recon[struct_idx, band_idx, wv_idx, :]
        print(f"   Using indexing: [{struct_idx}, {band_idx}, {wv_idx}, :] -> shape {eigvec_recon.shape}")
    else:
        print(f"   ERROR: Unexpected EIGENVECTOR_DATA ndim: {eigvec_data_recon.ndim}")
        sys.exit(1)

if eigvec_orig is None or eigvec_recon is None:
    print("ERROR: Could not extract eigenvectors")
    sys.exit(1)

# Handle structured dtype
if eigvec_orig.dtype.names and 'real' in eigvec_orig.dtype.names:
    eigvec_orig = eigvec_orig['real'] + 1j * eigvec_orig['imag']
if eigvec_recon.dtype.names and 'real' in eigvec_recon.dtype.names:
    eigvec_recon = eigvec_recon['real'] + 1j * eigvec_recon['imag']

eigvec_orig = np.array(eigvec_orig, dtype=np.complex128)
eigvec_recon = np.array(eigvec_recon, dtype=np.complex128)

print(f"   Eigenvector shapes: {eigvec_orig.shape} vs {eigvec_recon.shape}")

# Step-by-step comparison
print("\n" + "=" * 80)
print("STEP-BY-STEP COMPARISON")
print("=" * 80)

# Step 1: Compare T matrices
print("\nStep 1: T Matrix Comparison")
T_orig_dense = T_orig.toarray() if sp.issparse(T_orig) else T_orig
T_recon_dense = T_recon.toarray() if sp.issparse(T_recon) else T_recon
T_diff = np.abs(T_orig_dense - T_recon_dense)
T_max_diff = np.max(T_diff)
T_rel_diff = T_max_diff / (np.max(np.abs(T_orig_dense)) + 1e-20)
print(f"   Max absolute difference: {T_max_diff:.6e}")
print(f"   Max relative difference: {T_rel_diff:.6e}")
print(f"   T_orig dtype: {T_orig.dtype}, T_recon dtype: {T_recon.dtype}")

# Step 2: Compare reduced matrices Kr
print("\nStep 2: Reduced Stiffness Matrix (Kr) Comparison")
Kr_orig = T_orig.conj().T @ K_orig @ T_orig
Kr_recon = T_recon.conj().T @ K_recon @ T_recon
Kr_orig_dense = Kr_orig.toarray() if sp.issparse(Kr_orig) else Kr_orig
Kr_recon_dense = Kr_recon.toarray() if sp.issparse(Kr_recon) else Kr_recon
Kr_diff = np.abs(Kr_orig_dense - Kr_recon_dense)
Kr_max_diff = np.max(Kr_diff)
Kr_rel_diff = Kr_max_diff / (np.max(np.abs(Kr_orig_dense)) + 1e-20)
print(f"   Kr shapes: {Kr_orig.shape} vs {Kr_recon.shape}")
print(f"   Max absolute difference: {Kr_max_diff:.6e}")
print(f"   Max relative difference: {Kr_rel_diff:.6e}")

# Step 3: Compare reduced matrices Mr
print("\nStep 3: Reduced Mass Matrix (Mr) Comparison")
Mr_orig = T_orig.conj().T @ M_orig @ T_orig
Mr_recon = T_recon.conj().T @ M_recon @ T_recon
Mr_orig_dense = Mr_orig.toarray() if sp.issparse(Mr_orig) else Mr_orig
Mr_recon_dense = Mr_recon.toarray() if sp.issparse(Mr_recon) else Mr_recon
Mr_diff = np.abs(Mr_orig_dense - Mr_recon_dense)
Mr_max_diff = np.max(Mr_diff)
Mr_rel_diff = Mr_max_diff / (np.max(np.abs(Mr_orig_dense)) + 1e-20)
print(f"   Mr shapes: {Mr_orig.shape} vs {Mr_recon.shape}")
print(f"   Max absolute difference: {Mr_max_diff:.6e}")
print(f"   Max relative difference: {Mr_rel_diff:.6e}")

# Step 4: Compare eigenvectors
print("\nStep 4: Eigenvector Comparison")
eigvec_diff = np.abs(eigvec_orig - eigvec_recon)
eigvec_max_diff = np.max(eigvec_diff)
eigvec_rel_diff = eigvec_max_diff / (np.max(np.abs(eigvec_orig)) + 1e-20)
print(f"   Max absolute difference: {eigvec_max_diff:.6e}")
print(f"   Max relative difference: {eigvec_rel_diff:.6e}")
print(f"   Eigenvector norms: ||v_orig||={np.linalg.norm(eigvec_orig):.6e}, ||v_recon||={np.linalg.norm(eigvec_recon):.6e}")

# Check if eigenvectors match reduced space dimension
print(f"   Eigenvector length: {len(eigvec_orig)} vs {len(eigvec_recon)}")
print(f"   Kr/Mr dimension: {Kr_orig.shape[0]} vs {Kr_recon.shape[0]}")
if len(eigvec_orig) != Kr_orig.shape[0]:
    print(f"   ⚠️  WARNING: Eigenvector dimension ({len(eigvec_orig)}) != Kr dimension ({Kr_orig.shape[0]})")
if len(eigvec_recon) != Kr_recon.shape[0]:
    print(f"   ⚠️  WARNING: Eigenvector dimension ({len(eigvec_recon)}) != Kr dimension ({Kr_recon.shape[0]})")

# Step 5: Compare Kr*eigvec
print("\nStep 5: Kr * eigvec Comparison")
Kr_eigvec_orig = Kr_orig @ eigvec_orig
Kr_eigvec_recon = Kr_recon @ eigvec_recon
if sp.issparse(Kr_eigvec_orig):
    Kr_eigvec_orig = Kr_eigvec_orig.toarray().flatten()
if sp.issparse(Kr_eigvec_recon):
    Kr_eigvec_recon = Kr_eigvec_recon.toarray().flatten()

Kr_eigvec_diff = np.abs(Kr_eigvec_orig - Kr_eigvec_recon)
Kr_eigvec_max_diff = np.max(Kr_eigvec_diff)
Kr_eigvec_rel_diff = Kr_eigvec_max_diff / (np.max(np.abs(Kr_eigvec_orig)) + 1e-20)
print(f"   Max absolute difference: {Kr_eigvec_max_diff:.6e}")
print(f"   Max relative difference: {Kr_eigvec_rel_diff:.6e}")
print(f"   Norms: ||Kr*v_orig||={np.linalg.norm(Kr_eigvec_orig):.6e}, ||Kr*v_recon||={np.linalg.norm(Kr_eigvec_recon):.6e}")

# Step 6: Compare Mr*eigvec
print("\nStep 6: Mr * eigvec Comparison")
Mr_eigvec_orig = Mr_orig @ eigvec_orig
Mr_eigvec_recon = Mr_recon @ eigvec_recon
if sp.issparse(Mr_eigvec_orig):
    Mr_eigvec_orig = Mr_eigvec_orig.toarray().flatten()
if sp.issparse(Mr_eigvec_recon):
    Mr_eigvec_recon = Mr_eigvec_recon.toarray().flatten()

Mr_eigvec_diff = np.abs(Mr_eigvec_orig - Mr_eigvec_recon)
Mr_eigvec_max_diff = np.max(Mr_eigvec_diff)
Mr_eigvec_rel_diff = Mr_eigvec_max_diff / (np.max(np.abs(Mr_eigvec_orig)) + 1e-20)
print(f"   Max absolute difference: {Mr_eigvec_max_diff:.6e}")
print(f"   Max relative difference: {Mr_eigvec_rel_diff:.6e}")
print(f"   Norms: ||Mr*v_orig||={np.linalg.norm(Mr_eigvec_orig):.6e}, ||Mr*v_recon||={np.linalg.norm(Mr_eigvec_recon):.6e}")

# Step 7: Compare eigenvalues
print("\nStep 7: Eigenvalue Comparison")
eigval_orig = np.linalg.norm(Kr_eigvec_orig) / np.linalg.norm(Mr_eigvec_orig)
eigval_recon = np.linalg.norm(Kr_eigvec_recon) / np.linalg.norm(Mr_eigvec_recon)
eigval_diff = np.abs(eigval_orig - eigval_recon)
eigval_rel_diff = eigval_diff / (np.abs(eigval_orig) + 1e-20)
print(f"   Eigenvalues: {eigval_orig:.6e} vs {eigval_recon:.6e}")
print(f"   Absolute difference: {eigval_diff:.6e}")
print(f"   Relative difference: {eigval_rel_diff:.6e}")

# Step 8: Compare frequencies
print("\nStep 8: Frequency Comparison")
freq_orig = np.sqrt(np.real(eigval_orig)) / (2 * np.pi)
freq_recon = np.sqrt(np.real(eigval_recon)) / (2 * np.pi)
freq_diff = np.abs(freq_orig - freq_recon)
freq_rel_diff = freq_diff / (np.abs(freq_orig) + 1e-20)
print(f"   Frequencies: {freq_orig:.6e} vs {freq_recon:.6e}")
print(f"   Absolute difference: {freq_diff:.6e}")
print(f"   Relative difference: {freq_rel_diff:.6e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: ERROR PROPAGATION")
print("=" * 80)
print(f"T matrix relative error:     {T_rel_diff*100:.4f}%")
print(f"Kr matrix relative error:     {Kr_rel_diff*100:.4f}%")
print(f"Mr matrix relative error:     {Mr_rel_diff*100:.4f}%")
print(f"Eigenvector relative error:   {eigvec_rel_diff*100:.4f}%")
print(f"Kr*eigvec relative error:     {Kr_eigvec_rel_diff*100:.4f}%")
print(f"Mr*eigvec relative error:     {Mr_eigvec_rel_diff*100:.4f}%")
print(f"Eigenvalue relative error:    {eigval_rel_diff*100:.4f}%")
print(f"Frequency relative error:    {freq_rel_diff*100:.4f}%")
print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

