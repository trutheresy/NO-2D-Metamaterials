"""
Unit tests for system matrix functions.

Tests system matrix assembly and transformation matrix functions
to ensure equivalence with MATLAB versions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from scipy.sparse import issparse, csr_matrix

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from get_system_matrices import get_system_matrices
from get_transformation_matrix import get_transformation_matrix
from system_matrices_vec import get_system_matrices_VEC, get_system_matrices_VEC_simplified
from get_design import get_design


def create_test_const():
    """Create a test constants structure."""
    design = get_design('homogeneous', 5)
    
    const = {
        'N_pix': 5,
        'N_ele': 4,
        'a': 1.0,
        'design': design,
        'design_scale': 'linear',
        'E_min': 2e9,
        'E_max': 200e9,
        'rho_min': 1000,
        'rho_max': 8000,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 0.01
    }
    return const


class TestSystemMatrices:
    """Test get_system_matrices function."""
    
    def test_system_matrices_basic(self):
        """Test basic system matrix assembly."""
        const = create_test_const()
        
        K, M = get_system_matrices(const)
        
        # Check that matrices are sparse
        assert issparse(K), "K should be sparse matrix"
        assert issparse(M), "M should be sparse matrix"
        
        # Check dimensions
        N_dof = 2 * (const['N_ele'] * const['N_pix'] + 1)**2
        assert K.shape == (N_dof, N_dof), \
            f"K shape should be ({N_dof}, {N_dof}), got {K.shape}"
        assert M.shape == (N_dof, N_dof), \
            f"M shape should be ({N_dof}, {N_dof}), got {M.shape}"
        
        # Check symmetry
        K_dense = K.toarray()
        M_dense = M.toarray()
        assert np.allclose(K_dense, K_dense.T, rtol=1e-5), "K should be symmetric"
        assert np.allclose(M_dense, M_dense.T, rtol=1e-5), "M should be symmetric"
        
        # Check positive definiteness
        K_eigenvals = np.linalg.eigvals(K_dense)
        M_eigenvals = np.linalg.eigvals(M_dense)
        assert np.all(K_eigenvals > 0), "K should be positive definite"
        assert np.all(M_eigenvals > 0), "M should be positive definite"
    
    def test_system_matrices_sparsity(self):
        """Test that matrices have reasonable sparsity."""
        const = create_test_const()
        
        K, M = get_system_matrices(const)
        
        # Check sparsity (should be mostly zeros)
        K_density = K.nnz / (K.shape[0] * K.shape[1])
        M_density = M.nnz / (M.shape[0] * M.shape[1])
        
        # For FEM matrices, density should be low (< 1%)
        assert K_density < 0.1, f"K should be sparse, density is {K_density}"
        assert M_density < 0.1, f"M should be sparse, density is {M_density}"


class TestTransformationMatrix:
    """Test get_transformation_matrix function."""
    
    def test_transformation_matrix_basic(self):
        """Test basic transformation matrix calculation."""
        const = create_test_const()
        wavevector = np.array([0.5, 0.3])
        
        T = get_transformation_matrix(wavevector, const)
        
        # Check that T is sparse
        assert issparse(T), "T should be sparse matrix"
        
        # Check dimensions
        N_node = const['N_ele'] * const['N_pix'] + 1
        N_dof_full = 2 * N_node * N_node
        N_dof_reduced = 2 * (N_node - 1) * (N_node - 1)
        
        assert T.shape == (N_dof_full, N_dof_reduced), \
            f"T shape should be ({N_dof_full}, {N_dof_reduced}), got {T.shape}"
        
        # Check that T is complex (due to phase factors)
        T_dense = T.toarray()
        assert np.iscomplexobj(T_dense), "T should be complex"
    
    def test_transformation_matrix_gamma_point(self):
        """Test transformation matrix at Gamma point (k=0)."""
        const = create_test_const()
        wavevector = np.array([0.0, 0.0])
        
        T = get_transformation_matrix(wavevector, const)
        T_dense = T.toarray()
        
        # At Gamma point, phase factors should be 1, so T should be mostly real
        # (some numerical error in imaginary part is OK)
        assert np.max(np.abs(np.imag(T_dense))) < 1e-10, \
            "At Gamma point, T should be real"
    
    def test_transformation_matrix_phase_factors(self):
        """Test that phase factors are correct."""
        const = create_test_const()
        # Use a wavevector that definitely creates phase factors (non-zero in both directions)
        wavevector = np.array([0.5 * np.pi / const['a'], 0.3 * np.pi / const['a']])
        
        T = get_transformation_matrix(wavevector, const)
        T_dense = T.toarray()
        
        # Check that phase factors are applied (non-zero imaginary parts)
        # Use a smaller threshold to account for float32 precision
        max_imag = np.max(np.abs(np.imag(T_dense)))
        assert max_imag > 1e-7, \
            f"Phase factors should create complex values, but max imaginary part is {max_imag}"


class TestSystemMatricesVEC:
    """Test vectorized system matrix functions."""
    
    def test_system_matrices_vec_basic(self):
        """Test basic vectorized system matrix assembly."""
        const = create_test_const()
        
        K, M = get_system_matrices_VEC(const)
        
        # Check that matrices are sparse
        assert issparse(K), "K should be sparse matrix"
        assert issparse(M), "M should be sparse matrix"
        
        # Check dimensions
        N_dof = 2 * (const['N_ele'] * const['N_pix'] + 1)**2
        assert K.shape == (N_dof, N_dof)
        assert M.shape == (N_dof, N_dof)
        
        # Check symmetry
        K_dense = K.toarray()
        M_dense = M.toarray()
        assert np.allclose(K_dense, K_dense.T, rtol=1e-5)
        assert np.allclose(M_dense, M_dense.T, rtol=1e-5)
    
    def test_system_matrices_vec_vs_standard(self):
        """Test that VEC version produces same results as standard version."""
        const = create_test_const()
        
        K_std, M_std = get_system_matrices(const)
        K_vec, M_vec = get_system_matrices_VEC(const)
        
        # Convert to dense for comparison
        K_std_dense = K_std.toarray()
        K_vec_dense = K_vec.toarray()
        M_std_dense = M_std.toarray()
        M_vec_dense = M_vec.toarray()
        
        # Should be approximately equal (within numerical precision)
        assert np.allclose(K_std_dense, K_vec_dense, rtol=1e-4), \
            "VEC and standard K matrices should match"
        assert np.allclose(M_std_dense, M_vec_dense, rtol=1e-4), \
            "VEC and standard M matrices should match"
    
    def test_system_matrices_vec_simplified(self):
        """Test simplified vectorized system matrix assembly."""
        const = create_test_const()
        
        K, M = get_system_matrices_VEC_simplified(const)
        
        # Check that matrices are sparse
        assert issparse(K), "K should be sparse matrix"
        assert issparse(M), "M should be sparse matrix"
        
        # Check dimensions
        N_dof = 2 * (const['N_ele'] * const['N_pix'] + 1)**2
        assert K.shape == (N_dof, N_dof)
        assert M.shape == (N_dof, N_dof)
        
        # Check symmetry
        K_dense = K.toarray()
        M_dense = M.toarray()
        assert np.allclose(K_dense, K_dense.T, rtol=1e-5)
        assert np.allclose(M_dense, M_dense.T, rtol=1e-5)


class TestReducedMatrices:
    """Test reduced matrix creation (Kr, Mr)."""
    
    def test_reduced_matrices_creation(self):
        """Test creation of reduced matrices Kr = T'*K*T, Mr = T'*M*T."""
        const = create_test_const()
        wavevector = np.array([0.5, 0.3])
        
        # Get full matrices
        K, M = get_system_matrices(const)
        T = get_transformation_matrix(wavevector, const)
        
        # Create reduced matrices
        Kr = T.conj().T @ K @ T
        Mr = T.conj().T @ M @ T
        
        # Check dimensions
        N_dof_reduced = 2 * (const['N_ele'] * const['N_pix'])**2
        assert Kr.shape == (N_dof_reduced, N_dof_reduced)
        assert Mr.shape == (N_dof_reduced, N_dof_reduced)
        
        # Check symmetry (Hermitian for complex matrices)
        # Note: Due to float32/complex64 precision and sparse matrix operations,
        # the reduced matrices may not be exactly Hermitian, but should be approximately so
        Kr_dense = Kr.toarray()
        Mr_dense = Mr.toarray()
        
        # Check relative error rather than absolute
        Kr_max = np.max(np.abs(Kr_dense))
        Kr_hermitian_diff = np.max(np.abs(Kr_dense - Kr_dense.conj().T))
        Kr_relative_error = Kr_hermitian_diff / Kr_max if Kr_max > 0 else 0
        # Allow up to 1e-5 relative error for Hermitian property
        assert Kr_relative_error < 1e-5, \
            f"Kr should be approximately Hermitian, but relative error is {Kr_relative_error:.2e}"
        
        Mr_max = np.max(np.abs(Mr_dense))
        Mr_hermitian_diff = np.max(np.abs(Mr_dense - Mr_dense.conj().T))
        Mr_relative_error = Mr_hermitian_diff / Mr_max if Mr_max > 0 else 0
        assert Mr_relative_error < 1e-5, \
            f"Mr should be approximately Hermitian, but relative error is {Mr_relative_error:.2e}"
        
        # Check positive definiteness
        Kr_eigenvals = np.linalg.eigvals(Kr_dense)
        Mr_eigenvals = np.linalg.eigvals(Mr_dense)
        assert np.all(np.real(Kr_eigenvals) > 0), "Kr should be positive definite"
        assert np.all(np.real(Mr_eigenvals) > 0), "Mr should be positive definite"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



