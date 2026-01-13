#!/usr/bin/env python3
"""
Test H6: Eigenvalue Computation Formula

Tests if the eigenvalue computation formula is correct:
eigval = norm(Kr*eigvec) / norm(Mr*eigvec)

This assumes: Kr*eigvec = eigval*Mr*eigvec
Taking norms: ||Kr*eigvec|| = |eigval| * ||Mr*eigvec||
Therefore: eigval = ||Kr*eigvec|| / ||Mr*eigvec||

But this only works if the eigenvector is exact. If the eigenvector
is approximate (from float16 conversion), this relationship may not hold.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys
import h5py

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
from system_matrices import get_transformation_matrix

def test_eigenvalue_formula():
    """Test eigenvalue computation formula."""
    print("=" * 80)
    print("TEST H6: Eigenvalue Computation Formula")
    print("=" * 80)
    
    # Load data
    mat_file = Path("data/out_test_10_mat_original/out_binarized_1.mat")
    if not mat_file.exists():
        print(f"   File not found: {mat_file}")
        return False
    
    print(f"\n1. Loading data...")
    with h5py.File(str(mat_file), 'r') as f:
        eigvec_data = np.array(f['EIGENVECTOR_DATA'])
        designs = np.array(f['designs'])
        wavevectors = np.array(f['WAVEVECTOR_DATA'])
        const_dict = {key: np.array(f['const'][key]) for key in f['const']}
    
    # Convert eigenvector
    if eigvec_data.dtype.names and 'real' in eigvec_data.dtype.names:
        eigvec_data = eigvec_data['real'] + 1j * eigvec_data['imag']
    
    struct_idx = 0
    band_idx = 0
    wv_idx = 0
    
    # Extract data
    design = designs[struct_idx, :, :, :].transpose(1, 2, 0)  # (H, W, 3)
    wv = wavevectors[struct_idx, :, wv_idx].astype(np.float32)  # (2,)
    eigvec = eigvec_data[struct_idx, band_idx, wv_idx, :].astype(np.complex128)  # (n_dof,)
    
    print(f"   Design shape: {design.shape}")
    print(f"   Wavevector: {wv}")
    print(f"   Eigenvector shape: {eigvec.shape}, dtype: {eigvec.dtype}")
    
    # Setup const
    const = {
        'N_pix': int(np.array(const_dict['N_pix']).item()),
        'N_ele': int(np.array(const_dict['N_ele']).item()),
        'a': float(np.array(const_dict['a']).item()),
        'E_min': float(np.array(const_dict['E_min']).item()),
        'E_max': float(np.array(const_dict['E_max']).item()),
        'rho_min': float(np.array(const_dict['rho_min']).item()),
        'rho_max': float(np.array(const_dict['rho_max']).item()),
        'poisson_min': float(np.array(const_dict.get('nu_min', const_dict.get('poisson_min', 0.3))).item()),
        'poisson_max': float(np.array(const_dict.get('nu_max', const_dict.get('poisson_max', 0.3))).item()),
        't': float(np.array(const_dict.get('t', 1.0)).item()) if 't' in const_dict else 1.0,
        'design_scale': 'linear',
        'isUseImprovement': True,
        'isUseSecondImprovement': False,
        'design': design
    }
    
    print(f"\n2. Computing K, M, T matrices...")
    K, M = get_system_matrices_VEC(const)
    T = get_transformation_matrix(wv, const)
    
    print(f"   K: shape={K.shape}, dtype={K.dtype}")
    print(f"   M: shape={M.shape}, dtype={M.dtype}")
    print(f"   T: shape={T.shape}, dtype={T.dtype}")
    
    # Convert to sparse
    T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.complex64) if np.iscomplexobj(T) else T.astype(np.float32))
    K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
    M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
    
    print(f"\n3. Computing reduced matrices...")
    Kr = T_sparse.conj().T @ K_sparse @ T_sparse
    Mr = T_sparse.conj().T @ M_sparse @ T_sparse
    
    print(f"   Kr: shape={Kr.shape}, dtype={Kr.dtype}")
    print(f"   Mr: shape={Mr.shape}, dtype={Mr.dtype}")
    
    print(f"\n4. Testing eigenvalue computation formula...")
    
    # Method 1: Current formula (norm ratio)
    Kr_eigvec = Kr @ eigvec
    Mr_eigvec = Mr @ eigvec
    
    if sp.issparse(Kr_eigvec):
        Kr_eigvec = Kr_eigvec.toarray().flatten()
    if sp.issparse(Mr_eigvec):
        Mr_eigvec = Mr_eigvec.toarray().flatten()
    
    eigval_norm = np.linalg.norm(Kr_eigvec) / np.linalg.norm(Mr_eigvec)
    
    print(f"   Method 1 (norm ratio): eigval = {eigval_norm:.6e}")
    
    # Method 2: Direct ratio (if eigenvector is exact)
    # For exact eigenvector: Kr*eigvec = eigval*Mr*eigvec
    # So: eigval = (Kr*eigvec) / (Mr*eigvec) element-wise
    # But this only works if Mr*eigvec != 0 everywhere
    ratio = Kr_eigvec / (Mr_eigvec + 1e-15)  # Avoid division by zero
    eigval_direct = np.mean(ratio)  # Average ratio
    
    print(f"   Method 2 (direct ratio): eigval = {eigval_direct:.6e}")
    
    # Method 3: Rayleigh quotient (more standard)
    # eigval = (eigvec^H * Kr * eigvec) / (eigvec^H * Mr * eigvec)
    numerator = np.dot(eigvec.conj(), Kr @ eigvec)
    denominator = np.dot(eigvec.conj(), Mr @ eigvec)
    eigval_rayleigh = numerator / denominator
    
    print(f"   Method 3 (Rayleigh quotient): eigval = {eigval_rayleigh:.6e}")
    
    # Check consistency
    print(f"\n5. Checking formula consistency...")
    diff_norm_rayleigh = np.abs(eigval_norm - eigval_rayleigh) / max(np.abs(eigval_norm), np.abs(eigval_rayleigh))
    print(f"   Difference (norm vs Rayleigh): {diff_norm_rayleigh:.6e}")
    
    if diff_norm_rayleigh > 1e-3:
        print(f"   WARNING: Large difference between norm and Rayleigh quotient!")
        print(f"   This suggests the eigenvector may not be exact.")
    else:
        print(f"   OK: Norm and Rayleigh quotient are consistent")
    
    # Check if eigenvector satisfies the eigenvalue equation
    print(f"\n6. Checking if eigenvector satisfies eigenvalue equation...")
    # Kr*eigvec should equal eigval*Mr*eigvec
    lhs = Kr @ eigvec
    rhs = eigval_rayleigh * (Mr @ eigvec)
    
    if sp.issparse(lhs):
        lhs = lhs.toarray().flatten()
    if sp.issparse(rhs):
        rhs = rhs.toarray().flatten()
    
    residual = np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs)
    print(f"   Residual (||Kr*eigvec - eigval*Mr*eigvec|| / ||Kr*eigvec||): {residual:.6e}")
    
    if residual > 1e-3:
        print(f"   WARNING: Large residual! Eigenvector does not satisfy eigenvalue equation well.")
        print(f"   This could cause errors in the norm-based formula.")
        return False
    else:
        print(f"   OK: Eigenvector satisfies eigenvalue equation well")
    
    # Check real part extraction
    print(f"\n7. Checking real part extraction...")
    print(f"   eigval_rayleigh: {eigval_rayleigh:.6e}")
    print(f"   real(eigval_rayleigh): {np.real(eigval_rayleigh):.6e}")
    print(f"   imag(eigval_rayleigh): {np.imag(eigval_rayleigh):.6e}")
    
    if np.abs(np.imag(eigval_rayleigh)) > 1e-6:
        print(f"   WARNING: Eigenvalue has significant imaginary part!")
        print(f"   Discarding it with np.real() could cause errors.")
    else:
        print(f"   OK: Eigenvalue is essentially real")
    
    return True

if __name__ == "__main__":
    result = test_eigenvalue_formula()
    
    print(f"\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    
    if result:
        print("H6: Need to check if eigenvector satisfies eigenvalue equation.")
        print("    If residual is large, norm-based formula may be inaccurate.")
    else:
        print("H6 CONFIRMED: Eigenvalue computation formula issue found!")

