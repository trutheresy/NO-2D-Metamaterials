"""
Unit tests for utility functions.

Tests utility functions to ensure equivalence with MATLAB versions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
import scipy.sparse as sp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from get_mask import get_mask
from get_mesh import get_mesh
from get_global_idxs import get_global_idxs
from utils_han import make_chunks, init_storage
from cellofsparse_to_full import cellofsparse_to_full
from symmetry import apply_p4mm_symmetry


class TestGetMask:
    """Test get_mask function."""
    
    def test_get_mask_c1m1(self):
        """Test mask generation for c1m1 symmetry."""
        symmetry_type = 'c1m1'
        N_wv = [5, 5]
        
        mask = get_mask(symmetry_type, N_wv)
        
        # Check that mask is boolean
        assert mask.dtype == bool or mask.dtype == np.bool_
        
        # Check shape - mask is created by stacking flipped and original, so:
        # Initial shape: (N_wv[0], (N_wv[1] + 1) // 2)
        # After flipud and vstack: (2*N_wv[0] - 1, (N_wv[1] + 1) // 2)
        assert mask.shape[0] == 2 * N_wv[0] - 1
        assert mask.shape[1] == (N_wv[1] + 1) // 2
    
    def test_get_mask_other_symmetry(self):
        """Test mask generation for other symmetry types."""
        # For non-c1m1, should return None
        mask = get_mask('p4mm', [5, 5])
        assert mask is None


class TestGetMesh:
    """Test get_mesh function."""
    
    def test_get_mesh_basic(self):
        """Test basic mesh generation."""
        const = {
            'N_pix': 5,
            'N_ele': 4,
            'a': 1.0
        }
        
        mesh = get_mesh(const)
        
        # Check structure
        assert 'dim' in mesh
        assert 'node_coords' in mesh
        
        # Check dimensions
        assert mesh['dim'] == 2
        
        # Check node coordinates
        assert len(mesh['node_coords']) == 2  # x and y coordinates
        
        # Check coordinate arrays
        N_node = const['N_ele'] * const['N_pix'] + 1
        assert mesh['node_coords'][0].shape == (N_node, N_node)
        assert mesh['node_coords'][1].shape == (N_node, N_node)
        
        # Check coordinate ranges
        assert np.all(mesh['node_coords'][0] >= 0)
        assert np.all(mesh['node_coords'][0] <= const['a'])
        assert np.all(mesh['node_coords'][1] >= 0)
        assert np.all(mesh['node_coords'][1] <= const['a'])


class TestGetGlobalIdxs:
    """Test get_global_idxs function."""
    
    def test_get_global_idxs_basic(self):
        """Test basic global index calculation."""
        const = {
            'N_pix': 5,
            'N_ele': 4,
            'a': 1.0
        }
        
        ele_idx_x = 1
        ele_idx_y = 1
        
        global_idxs = get_global_idxs(ele_idx_x, ele_idx_y, const)
        
        # Should return 8 indices (4 nodes * 2 DOF per node)
        assert len(global_idxs) == 8
        
        # All indices should be non-negative integers (0-based indexing in Python)
        assert np.all(global_idxs >= 0)
        assert np.all(global_idxs == np.floor(global_idxs))
    
    def test_get_global_idxs_different_elements(self):
        """Test global indices for different elements."""
        const = {
            'N_pix': 5,
            'N_ele': 4,
            'a': 1.0
        }
        
        # Get indices for different elements
        idxs1 = get_global_idxs(1, 1, const)
        idxs2 = get_global_idxs(2, 1, const)
        idxs3 = get_global_idxs(1, 2, const)
        
        # Should all be different
        assert not np.array_equal(idxs1, idxs2)
        assert not np.array_equal(idxs1, idxs3)
        assert not np.array_equal(idxs2, idxs3)


class TestMakeChunks:
    """Test make_chunks function."""
    
    def test_make_chunks_basic(self):
        """Test basic chunk creation."""
        N = 10
        M = 3
        
        ranges = make_chunks(N, M)
        
        # Check shape
        assert ranges.shape[1] == 2  # [start, end] pairs
        
        # Check that chunks cover 1 to N
        all_values = []
        for start, end in ranges:
            all_values.extend(range(start, end + 1))
        
        assert set(all_values) == set(range(1, N + 1))
        
        # Check chunk sizes
        for start, end in ranges:
            chunk_size = end - start + 1
            assert chunk_size <= M
    
    def test_make_chunks_edge_cases(self):
        """Test edge cases for make_chunks."""
        # N < M
        ranges = make_chunks(2, 5)
        assert len(ranges) == 1
        assert ranges[0, 0] == 1
        assert ranges[0, 1] == 2
        
        # N == M
        ranges = make_chunks(5, 5)
        assert len(ranges) == 1
        assert ranges[0, 0] == 1
        assert ranges[0, 1] == 5


class TestInitStorage:
    """Test init_storage function."""
    
    def test_init_storage_basic(self):
        """Test basic storage initialization."""
        const = {
            'N_pix': 5,
            'N_ele': 4,
            'a': 1.0,
            'N_eig': 10,
            'wavevectors': np.random.rand(20, 2)
        }
        
        N_struct_batch = 5
        
        result = init_storage(const, N_struct_batch)
        
        # Unpack results
        (designs, WAVEVECTOR_DATA, EIGENVALUE_DATA, N_dof, DESIGN_NUMBERS,
         EIGENVECTOR_DATA, ELASTIC_MODULUS_DATA, DENSITY_DATA, POISSON_DATA,
         K_DATA, M_DATA, T_DATA) = result
        
        # Check shapes
        assert designs.shape == (5, 5, 3, N_struct_batch)
        assert WAVEVECTOR_DATA.shape == (20, 2, N_struct_batch)
        assert EIGENVALUE_DATA.shape == (20, 10, N_struct_batch)
        assert len(DESIGN_NUMBERS) == N_struct_batch
        assert EIGENVECTOR_DATA.shape == (N_dof, 20, 10, N_struct_batch)
        assert ELASTIC_MODULUS_DATA.shape == (5, 5, N_struct_batch)
        assert DENSITY_DATA.shape == (5, 5, N_struct_batch)
        assert POISSON_DATA.shape == (5, 5, N_struct_batch)
        assert len(K_DATA) == N_struct_batch
        assert len(M_DATA) == N_struct_batch


class TestCellOfSparseToFull:
    """Test cellofsparse_to_full function."""
    
    def test_cellofsparse_to_full_basic(self):
        """Test conversion of sparse matrices to full array."""
        # Create list of sparse matrices
        N = 5
        matrix_size = 10
        system_matrix_data = []
        
        for i in range(N):
            # Create random sparse matrix
            matrix = sp.random(matrix_size, matrix_size, density=0.1, format='csr')
            system_matrix_data.append(matrix)
        
        # Convert to full
        out = cellofsparse_to_full(system_matrix_data)
        
        # Check shape
        assert out.shape == (N, matrix_size, matrix_size)
        
        # Check that conversion is correct
        for i in range(N):
            expected = system_matrix_data[i].toarray()
            assert np.allclose(out[i, :, :], expected, rtol=1e-6)
    
    def test_cellofsparse_to_full_dtype(self):
        """Test that output is float32."""
        matrix_size = 5
        system_matrix_data = [sp.eye(matrix_size, format='csr')]
        
        out = cellofsparse_to_full(system_matrix_data)
        
        assert out.dtype == np.float32


class TestApplyP4mmSymmetry:
    """Test apply_p4mm_symmetry function."""
    
    def test_apply_p4mm_symmetry_basic(self):
        """Test basic p4mm symmetry application."""
        # Create asymmetric array
        A = np.random.rand(5, 5)
        
        A_sym = apply_p4mm_symmetry(A)
        
        # Check shape
        assert A_sym.shape == A.shape
        
        # Check symmetry properties
        # Should be symmetric about center
        assert np.allclose(A_sym, A_sym.T, rtol=1e-5), "Should be symmetric"
        assert np.allclose(A_sym, np.fliplr(A_sym), rtol=1e-5), "Should be left-right symmetric"
        assert np.allclose(A_sym, np.flipud(A_sym), rtol=1e-5), "Should be up-down symmetric"
    
    def test_apply_p4mm_symmetry_preserves_range(self):
        """Test that symmetry preserves data range."""
        A = np.random.rand(5, 5) * 10 + 5  # Range [5, 15]
        
        A_sym = apply_p4mm_symmetry(A)
        
        # Range should be approximately preserved
        assert np.min(A_sym) >= np.min(A) - 1e-5
        assert np.max(A_sym) <= np.max(A) + 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



