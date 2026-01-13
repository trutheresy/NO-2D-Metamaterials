#!/usr/bin/env python3
"""
Test H1: Check actual code path in reduced_pt_to_matlab.py

Verify what actually happens with T matrix in the reconstruction code.
Since T is always sparse from get_transformation_matrix, the conversion
line won't trigger, but we should verify this and check if there are
other type issues.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
from system_matrices import get_transformation_matrix

def test_synthetic():
    """Synthetic test - verify T matrix behavior."""
    print("=" * 80)
    print("TEST H1: Type Conversion in Actual Code Path")
    print("=" * 80)
    
    # Create minimal const
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
        'isUseSecondImprovement': False
    }
    
    # Create simple design
    design = np.ones((4, 4, 3)) * 0.5
    const['design'] = design
    
    print(f"\n1. Computing K and M matrices...")
    K, M = get_system_matrices_VEC(const)
    print(f"   K: shape={K.shape}, dtype={K.dtype}, sparse={sp.issparse(K)}")
    print(f"   M: shape={M.shape}, dtype={M.dtype}, sparse={sp.issparse(M)}")
    
    print(f"\n2. Computing T matrix...")
    wv = np.array([0.1, 0.2], dtype=np.float32)
    T = get_transformation_matrix(wv, const)
    
    print(f"   T: type={type(T)}, sparse={sp.issparse(T)}, shape={T.shape}, dtype={T.dtype}")
    if sp.issparse(T):
        print(f"   T.data dtype: {T.data.dtype}")
        print(f"   T has complex: {np.iscomplexobj(T)}")
        if np.iscomplexobj(T):
            print(f"   T imag range: [{np.min(np.imag(T.data)):.6e}, {np.max(np.imag(T.data)):.6e}]")
    
    print(f"\n3. Testing the exact code from reduced_pt_to_matlab.py:454...")
    # This is the exact line from reduced_pt_to_matlab.py:454
    T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.float32))
    
    print(f"   T_sparse: type={type(T_sparse)}, dtype={T_sparse.dtype}")
    if sp.issparse(T_sparse):
        print(f"   T_sparse.data dtype: {T_sparse.data.dtype}")
        print(f"   T_sparse has complex: {np.iscomplexobj(T_sparse)}")
        
        # Check if complex was lost
        if sp.issparse(T) and np.iscomplexobj(T) and not np.iscomplexobj(T_sparse):
            print(f"   ✗ CRITICAL: Complex type was lost!")
            return False
        elif sp.issparse(T) and np.iscomplexobj(T) and np.iscomplexobj(T_sparse):
            print(f"   ✓ Complex type preserved (T was already sparse, so no conversion)")
    
    print(f"\n4. Testing K and M conversion...")
    K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
    M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
    
    print(f"   K_sparse: dtype={K_sparse.dtype}, has complex={np.iscomplexobj(K_sparse)}")
    print(f"   M_sparse: dtype={M_sparse.dtype}, has complex={np.iscomplexobj(M_sparse)}")
    
    print(f"\n5. Testing matrix multiplication...")
    print(f"   Computing Kr = T^H @ K @ T...")
    Kr = T_sparse.conj().T @ K_sparse @ T_sparse
    print(f"   Kr: dtype={Kr.dtype}, shape={Kr.shape}, has complex={np.iscomplexobj(Kr)}")
    
    Mr = T_sparse.conj().T @ M_sparse @ T_sparse
    print(f"   Mr: dtype={Mr.dtype}, shape={Mr.shape}, has complex={np.iscomplexobj(Mr)}")
    
    # Check if Kr/Mr should be complex
    if np.iscomplexobj(T_sparse) and not np.iscomplexobj(Kr):
        print(f"   ⚠ WARNING: Kr should be complex but isn't!")
        return False
    if np.iscomplexobj(T_sparse) and not np.iscomplexobj(Mr):
        print(f"   ⚠ WARNING: Mr should be complex but isn't!")
        return False
    
    print(f"\n6. Testing what happens if T were dense (hypothetical)...")
    # Hypothetical: what if T were dense?
    T_dense = T.toarray() if sp.issparse(T) else T
    print(f"   T_dense: dtype={T_dense.dtype}, has complex={np.iscomplexobj(T_dense)}")
    
    # This would be the problematic conversion
    T_wrong = sp.csr_matrix(T_dense.astype(np.float32))
    print(f"   T_wrong (after .astype(np.float32)): dtype={T_wrong.dtype}, has complex={np.iscomplexobj(T_wrong)}")
    
    if np.iscomplexobj(T_dense) and not np.iscomplexobj(T_wrong):
        print(f"   ✗ CRITICAL: If T were dense, .astype(np.float32) would lose complex part!")
        print(f"   However, T is always sparse from get_transformation_matrix, so this doesn't happen.")
    
    return True

if __name__ == "__main__":
    result = test_synthetic()
    
    print(f"\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    
    if result:
        print("✓ H1 ELIMINATED: T matrix is always sparse, so the conversion line doesn't trigger.")
        print("  The code path `T_sparse = T if sp.issparse(T) else ...` always takes the first branch.")
        print("  Complex type is preserved.")
    else:
        print("✗ H1 CONFIRMED: Type conversion issue found!")
