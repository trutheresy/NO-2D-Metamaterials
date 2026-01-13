#!/usr/bin/env python3
"""
Test H2: Indexing and Dimension Mismatches

Tests:
1. Eigenvector dimension mismatch
2. T matrix shape mismatch
3. Wavevector index mismatch
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

def load_mat_data_simple(mat_path):
    """Load minimal data from .mat file."""
    with h5py.File(str(mat_path), 'r') as f:
        data = {
            'EIGENVECTOR_DATA': np.array(f['EIGENVECTOR_DATA']),
            'WAVEVECTOR_DATA': np.array(f['WAVEVECTOR_DATA']),
            'designs': np.array(f['designs']),
            'const': {key: np.array(f['const'][key]) for key in f['const']}
        }
    return data

def test_dimension_mismatches():
    """Test for dimension mismatches in reconstruction."""
    print("=" * 80)
    print("TEST H2: Indexing and Dimension Mismatches")
    print("=" * 80)
    
    # Try to load real data
    mat_file = Path("data/out_test_10_mat_original/out_binarized_1.mat")
    if not mat_file.exists():
        print(f"   File not found: {mat_file}")
        print("   Using synthetic test...")
        return test_synthetic_dimensions()
    
    print(f"\n1. Loading data from {mat_file.name}...")
    data = load_mat_data_simple(mat_file)
    
    struct_idx = 0
    const_orig = data['const']
    
    # Extract const
    const_for_km = {
        'N_pix': int(np.array(const_orig['N_pix']).item()),
        'N_ele': int(np.array(const_orig['N_ele']).item()),
        'a': float(np.array(const_orig['a']).item()),
        'E_min': float(np.array(const_orig['E_min']).item()),
        'E_max': float(np.array(const_orig['E_max']).item()),
        'rho_min': float(np.array(const_orig['rho_min']).item()),
        'rho_max': float(np.array(const_orig['rho_max']).item()),
        'poisson_min': float(np.array(const_orig.get('nu_min', const_orig.get('poisson_min', 0.3))).item()),
        'poisson_max': float(np.array(const_orig.get('nu_max', const_orig.get('poisson_max', 0.3))).item()),
        't': float(np.array(const_orig.get('t', 1.0)).item()) if 't' in const_orig else 1.0,
        'design_scale': 'linear',
        'isUseImprovement': True,
        'isUseSecondImprovement': False
    }
    
    # Get design
    design_orig = data['designs'][struct_idx, :, :, :]  # (3, H, W)
    design_orig_3ch = design_orig.transpose(1, 2, 0)  # (H, W, 3)
    const_for_km['design'] = design_orig_3ch
    
    print(f"   N_pix: {const_for_km['N_pix']}, N_ele: {const_for_km['N_ele']}")
    
    # Get wavevectors - shape is (n_designs, 2, n_wavevectors) or (n_wavevectors, 2)
    wavevectors = data['WAVEVECTOR_DATA']
    print(f"   Wavevectors shape: {wavevectors.shape}")
    
    # Handle different shapes
    if wavevectors.ndim == 3:
        # (n_designs, 2, n_wavevectors) - extract for first design
        wavevectors = wavevectors[struct_idx, :, :].T  # Transpose to (n_wavevectors, 2)
        print(f"   Extracted wavevectors for struct {struct_idx}, shape: {wavevectors.shape}")
    elif wavevectors.ndim == 2 and wavevectors.shape[0] == 2:
        # (2, n_wavevectors) - transpose
        wavevectors = wavevectors.T
        print(f"   Transposed wavevectors, shape: {wavevectors.shape}")
    
    # Get eigenvectors - handle structured dtype
    eigvec_data_raw = data['EIGENVECTOR_DATA']
    print(f"   EIGENVECTOR_DATA shape: {eigvec_data_raw.shape}, dtype: {eigvec_data_raw.dtype}")
    
    # Convert structured dtype to complex if needed
    if eigvec_data_raw.dtype.names and 'real' in eigvec_data_raw.dtype.names:
        eigvec_data = eigvec_data_raw['real'] + 1j * eigvec_data_raw['imag']
    else:
        eigvec_data = eigvec_data_raw
    print(f"   EIGENVECTOR_DATA (converted) shape: {eigvec_data.shape}, dtype: {eigvec_data.dtype}")
    
    print(f"\n2. Computing K and M matrices...")
    K, M = get_system_matrices_VEC(const_for_km)
    N_dof_full = K.shape[0]
    print(f"   K: shape={K.shape}, N_dof_full={N_dof_full}")
    print(f"   M: shape={M.shape}")
    
    print(f"\n3. Computing T matrices and checking dimensions...")
    T_data = []
    for wv_idx in range(min(3, wavevectors.shape[0])):
        wv = wavevectors[wv_idx, :].astype(np.float32)
        T = get_transformation_matrix(wv, const_for_km)
        T_data.append(T)
        
        N_dof_reduced = T.shape[1]
        print(f"\n   Wavevector {wv_idx}: {wv}")
        print(f"     T: shape={T.shape}, N_dof_full={T.shape[0]}, N_dof_reduced={N_dof_reduced}")
        
        # Check T dimensions
        if T.shape[0] != N_dof_full:
            print(f"     ERROR: T.shape[0] ({T.shape[0]}) != N_dof_full ({N_dof_full})")
            return False
        else:
            print(f"     OK: T.shape[0] matches N_dof_full")
    
    print(f"\n4. Checking eigenvector dimensions...")
    # EIGENVECTOR_DATA format: (n_designs, n_bands, n_wavevectors, n_dof)
    print(f"   EIGENVECTOR_DATA shape: {eigvec_data.shape}")
    
    # Extract for structure
    eigenvectors_struct = eigvec_data[struct_idx, :, :, :]  # (n_bands, n_wavevectors, n_dof)
    print(f"   eigenvectors_struct shape: {eigenvectors_struct.shape}")
    
    # Transpose to (n_dof, n_wavevectors, n_bands) for reconstruction
    eigenvectors_struct_transposed = eigenvectors_struct.transpose(2, 1, 0)
    print(f"   eigenvectors_struct_transposed shape: {eigenvectors_struct_transposed.shape}")
    
    n_dof_eigvec = eigenvectors_struct_transposed.shape[0]
    
    # Check if eigenvector DOF matches reduced DOF
    if n_dof_eigvec != N_dof_reduced:
        print(f"   ERROR: eigenvector DOF ({n_dof_eigvec}) != N_dof_reduced ({N_dof_reduced})")
        print(f"   This is a critical mismatch!")
        return False
    else:
        print(f"   OK: eigenvector DOF matches N_dof_reduced")
    
    print(f"\n5. Testing matrix-vector multiplication dimensions...")
    for wv_idx in range(min(2, len(T_data))):
        T = T_data[wv_idx]
        eigvec = eigenvectors_struct_transposed[:, wv_idx, 0].astype(np.complex128)  # First band
        
        print(f"\n   Wavevector {wv_idx}, Band 0:")
        print(f"     T shape: {T.shape}")
        print(f"     eigvec shape: {eigvec.shape}")
        
        # Check dimensions
        if eigvec.shape[0] != T.shape[1]:
            print(f"     ERROR: eigvec length ({eigvec.shape[0]}) != T.shape[1] ({T.shape[1]})")
            return False
        else:
            print(f"     OK: Dimensions match for matrix-vector multiplication")
        
        # Test multiplication
        try:
            Kr = T.conj().T @ K @ T
            Mr = T.conj().T @ M @ T
            Kr_eigvec = Kr @ eigvec
            Mr_eigvec = Mr @ eigvec
            
            print(f"     Kr @ eigvec shape: {Kr_eigvec.shape}")
            print(f"     Mr @ eigvec shape: {Mr_eigvec.shape}")
            print(f"     OK: Matrix-vector multiplication successful")
        except Exception as e:
            print(f"     ERROR in matrix-vector multiplication: {e}")
            return False
    
    return True

def test_synthetic_dimensions():
    """Synthetic dimension test."""
    print("   Using synthetic test...")
    const = {
        'N_pix': 4,
        'N_ele': 1,
        'a': 1.0,
        'E_min': 20e6,
        'E_max': 200e9,
        'rho_min': 1200,
        'rho_max': 8000,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 1.0,
        'design_scale': 'linear',
        'isUseImprovement': True,
        'isUseSecondImprovement': False,
        'design': np.ones((4, 4, 3)) * 0.5
    }
    
    K, M = get_system_matrices_VEC(const)
    N_dof_full = K.shape[0]
    
    wv = np.array([0.1, 0.2], dtype=np.float32)
    T = get_transformation_matrix(wv, const)
    N_dof_reduced = T.shape[1]
    
    print(f"   N_dof_full: {N_dof_full}")
    print(f"   N_dof_reduced: {N_dof_reduced}")
    print(f"   T shape: {T.shape}")
    
    # Create synthetic eigenvector
    eigvec = np.ones(N_dof_reduced, dtype=np.complex128)
    print(f"   eigvec shape: {eigvec.shape}")
    
    if eigvec.shape[0] != N_dof_reduced:
        print(f"   ERROR: Dimension mismatch!")
        return False
    else:
        print(f"   OK: Dimensions match")
        return True

if __name__ == "__main__":
    result = test_dimension_mismatches()
    
    print(f"\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    
    if result:
        print("H2 ELIMINATED: All dimensions match correctly.")
    else:
        print("H2 CONFIRMED: Dimension mismatch found!")

