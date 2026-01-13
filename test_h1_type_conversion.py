#!/usr/bin/env python3
"""
Test H1: Type Conversion Issues

Tests if T matrix complex64 → float32 conversion is happening incorrectly.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices import get_transformation_matrix

def test_t_matrix_dtype():
    """Test T matrix dtype and conversion behavior."""
    print("=" * 80)
    print("TEST H1: Type Conversion Issues")
    print("=" * 80)
    
    # Create minimal const
    const = {
        'N_pix': 4,  # Small for efficiency
        'N_ele': 1,
        'a': 1.0
    }
    
    # Create test wavevector
    wavevector = np.array([0.1, 0.2], dtype=np.float32)
    
    print(f"\n1. Computing T matrix...")
    T = get_transformation_matrix(wavevector, const)
    
    print(f"   T matrix:")
    print(f"     Type: {type(T)}")
    print(f"     Sparse: {sp.issparse(T)}")
    print(f"     Shape: {T.shape}")
    print(f"     Dtype: {T.dtype}")
    
    # Check if T is complex
    if sp.issparse(T):
        sample_values = T.data[:10] if len(T.data) > 10 else T.data
        print(f"     Sample values: {sample_values[:5]}")
        print(f"     Has complex values: {np.any(np.iscomplex(T.data))}")
        print(f"     Real part range: [{np.min(np.real(T.data)):.6e}, {np.max(np.real(T.data)):.6e}]")
        print(f"     Imag part range: [{np.min(np.imag(T.data)):.6e}, {np.max(np.imag(T.data)):.6e}]")
    
    print(f"\n2. Testing conversion in reduced_pt_to_matlab.py style...")
    # This is the problematic line from reduced_pt_to_matlab.py:454
    T_sparse_wrong = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.float32))
    
    print(f"   T_sparse_wrong:")
    print(f"     Type: {type(T_sparse_wrong)}")
    print(f"     Dtype: {T_sparse_wrong.dtype}")
    if sp.issparse(T_sparse_wrong):
        print(f"     Has complex values: {np.any(np.iscomplex(T_sparse_wrong.data))}")
        if np.any(np.iscomplex(T_sparse_wrong.data)):
            print(f"     ✓ T_sparse_wrong is complex (correct)")
        else:
            print(f"     ✗ T_sparse_wrong lost complex part (WRONG!)")
            print(f"     This is the bug!")
    
    print(f"\n3. Testing correct conversion (preserve complex)...")
    if sp.issparse(T):
        T_sparse_correct = T  # Already sparse
    else:
        # Preserve complex type
        if np.iscomplexobj(T):
            T_sparse_correct = sp.csr_matrix(T.astype(np.complex64))
        else:
            T_sparse_correct = sp.csr_matrix(T.astype(np.float32))
    
    print(f"   T_sparse_correct:")
    print(f"     Type: {type(T_sparse_correct)}")
    print(f"     Dtype: {T_sparse_correct.dtype}")
    if sp.issparse(T_sparse_correct):
        print(f"     Has complex values: {np.any(np.iscomplex(T_sparse_correct.data))}")
    
    print(f"\n4. Testing what happens with .astype(np.float32) on complex matrix...")
    if sp.issparse(T) and np.iscomplexobj(T):
        T_dense = T.toarray()
        T_wrong_dtype = T_dense.astype(np.float32)
        print(f"   Original T (dense) dtype: {T_dense.dtype}")
        print(f"   After .astype(np.float32): {T_wrong_dtype.dtype}")
        print(f"   Original has complex: {np.any(np.iscomplex(T_dense))}")
        print(f"   After conversion has complex: {np.any(np.iscomplex(T_wrong_dtype))}")
        print(f"   Data loss: {np.any(np.abs(np.imag(T_dense)) > 1e-10)}")
        if np.any(np.abs(np.imag(T_dense)) > 1e-10):
            print(f"   ✗ CRITICAL: Imaginary parts will be lost!")
            max_imag = np.max(np.abs(np.imag(T_dense)))
            print(f"   Max imaginary part: {max_imag:.6e}")
    
    print(f"\n5. Testing matrix multiplication with wrong type...")
    # Create small test matrices
    K_test = sp.csr_matrix(np.eye(10, dtype=np.float32))
    M_test = sp.csr_matrix(np.eye(10, dtype=np.float32))
    
    if sp.issparse(T) and T.shape[0] == 10:
        print(f"   Using actual T matrix")
        T_test = T
    else:
        # Create small test T
        T_test = sp.csr_matrix((np.ones(5, dtype=np.complex64), 
                               (np.arange(5), np.arange(5))), 
                              shape=(10, 5))
        print(f"   Created test T matrix: shape={T_test.shape}, dtype={T_test.dtype}")
    
    # Test with correct complex T
    Kr_correct = T_test.conj().T @ K_test @ T_test
    print(f"   Kr_correct: dtype={Kr_correct.dtype}, shape={Kr_correct.shape}")
    
    # Test with wrong real T (if conversion happened)
    if not np.iscomplexobj(T_test):
        T_test_wrong = sp.csr_matrix(T_test.toarray().astype(np.float32))
        Kr_wrong = T_test_wrong.conj().T @ K_test @ T_test_wrong
        print(f"   Kr_wrong: dtype={Kr_wrong.dtype}, shape={Kr_wrong.shape}")
        print(f"   Difference: {np.max(np.abs(Kr_correct.toarray() - Kr_wrong.toarray())):.6e}")
    
    return T, T_sparse_wrong, T_sparse_correct

if __name__ == "__main__":
    T, T_wrong, T_correct = test_t_matrix_dtype()
    
    print(f"\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    
    if sp.issparse(T) and np.iscomplexobj(T):
        if not np.iscomplexobj(T_wrong):
            print("✗ BUG FOUND: T matrix is complex64 but gets converted to float32!")
            print("  This causes loss of imaginary parts in T matrix.")
            print("  Location: reduced_pt_to_matlab.py:454")
            print("  Fix: Preserve complex type when converting T to sparse")
        else:
            print("✓ T matrix type is preserved correctly")
    else:
        print("? T matrix is not complex - need to check if this is expected")

