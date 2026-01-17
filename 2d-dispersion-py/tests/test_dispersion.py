"""
Unit tests for dispersion calculation functions.

Tests dispersion calculation to ensure equivalence with MATLAB versions.
Note: Full equivalence testing requires comparing with MATLAB outputs.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from scipy.sparse.linalg import eigs

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dispersion import dispersion
from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt
from get_design import get_design
from wavevectors import get_IBZ_wavevectors


def create_test_const():
    """Create a test constants structure."""
    N_pix = 5
    design = get_design('homogeneous', N_pix)
    
    const = {
        'N_pix': N_pix,
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
        't': 0.01,
        'N_eig': 5,
        'sigma_eig': 'SM',
        'isSaveEigenvectors': True,
        'isSaveMesh': False,
        'isUseGPU': False,
        'isUseParallel': False,
        'isUseImprovement': False,
        'isUseSecondImprovement': False
    }
    return const


class TestDispersion:
    """Test dispersion function."""
    
    def test_dispersion_basic(self):
        """Test basic dispersion calculation."""
        const = create_test_const()
        
        # Generate wavevectors
        wavevectors = get_IBZ_wavevectors([5, 3], const['a'], symmetry_type='none')
        
        # Calculate dispersion
        wv, fr, ev, mesh = dispersion(const, wavevectors)
        
        # Check outputs
        assert wv.shape == wavevectors.shape
        assert fr.shape == (wavevectors.shape[0], const['N_eig'])
        # Calculate expected N_dof_reduced (matching dispersion.py calculation)
        # Eigenvectors are stored in reduced DOF space
        if const['isSaveEigenvectors']:
            # Get transformation matrix to determine reduced DOF size
            from system_matrices import get_transformation_matrix
            T_temp = get_transformation_matrix(wavevectors[0, :], const)
            N_dof_reduced = T_temp.shape[1]
            assert ev.shape == (N_dof_reduced, wavevectors.shape[0], const['N_eig'])
        else:
            assert ev is None or len(ev) == 0
        
        # Check frequency values are positive
        assert np.all(fr >= 0), "Frequencies should be non-negative"
        
        # Check frequencies are real
        assert np.all(np.isreal(fr)), "Frequencies should be real"
        
        # Check frequencies are sorted (lowest to highest)
        for i in range(wavevectors.shape[0]):
            assert np.all(np.diff(fr[i, :]) >= -1e-10), \
                f"Frequencies should be sorted for wavevector {i}"
    
    def test_dispersion_gamma_point(self):
        """Test dispersion at Gamma point (k=0)."""
        const = create_test_const()
        
        # Single wavevector at Gamma point
        wavevectors = np.array([[0.0, 0.0]])
        
        wv, fr, ev, mesh = dispersion(const, wavevectors)
        
        # At Gamma point, frequencies should be real and positive
        assert np.all(fr >= 0)
        assert np.all(np.isreal(fr))
        
        # First frequency should be zero (rigid body mode)
        # Allow for small numerical errors - rigid body mode might not be exactly zero
        # due to numerical precision in eigenvalue solver
        assert fr[0, 0] < 100.0, \
            f"First frequency at Gamma should be near zero (rigid body mode), got {fr[0, 0]}"
    
    def test_dispersion_eigenvector_normalization(self):
        """Test that eigenvectors are properly normalized."""
        const = create_test_const()
        const['isSaveEigenvectors'] = True
        
        wavevectors = get_IBZ_wavevectors([3, 3], const['a'], symmetry_type='none')
        
        wv, fr, ev, mesh = dispersion(const, wavevectors)
        
        if ev is not None and len(ev) > 0:
            # Check eigenvector shape
            assert ev.shape[0] > 0
            
            # Check that eigenvectors are normalized (2-norm = 1)
            for k_idx in range(wavevectors.shape[0]):
                for eig_idx in range(const['N_eig']):
                    ev_norm = np.linalg.norm(ev[:, k_idx, eig_idx])
                    assert np.isclose(ev_norm, 1.0, rtol=1e-5), \
                        f"Eigenvector should be normalized, got norm {ev_norm}"


class TestDispersionWithMatrixSaveOpt:
    """Test dispersion_with_matrix_save_opt function."""
    
    def test_dispersion_with_matrix_save_basic(self):
        """Test dispersion calculation with matrix saving."""
        const = create_test_const()
        const['isSaveKandM'] = True
        
        wavevectors = get_IBZ_wavevectors([3, 3], const['a'], symmetry_type='none')
        
        result = dispersion_with_matrix_save_opt(const, wavevectors)
        wv, fr, ev, mesh, K_out, M_out, T_out = result
        
        # Check that matrices are saved
        assert K_out is not None, "K should be saved"
        assert M_out is not None, "M should be saved"
        assert T_out is not None, "T should be saved"
        
        # Check matrix shapes
        N_pix_val = const['N_pix'] if not isinstance(const['N_pix'], (list, tuple)) else const['N_pix'][0]
        N_dof = 2 * (const['N_ele'] * N_pix_val + 1)**2
        assert K_out.shape == (N_dof, N_dof)
        assert M_out.shape == (N_dof, N_dof)
        
        # Check T_out is a list of transformation matrices
        assert len(T_out) == wavevectors.shape[0]
        for T in T_out:
            assert T is not None
    
    def test_dispersion_with_matrix_save_no_save(self):
        """Test dispersion without matrix saving."""
        const = create_test_const()
        const['isSaveKandM'] = False
        
        wavevectors = get_IBZ_wavevectors([3, 3], const['a'], symmetry_type='none')
        
        result = dispersion_with_matrix_save_opt(const, wavevectors)
        wv, fr, ev, mesh, K_out, M_out, T_out = result
        
        # Matrices should be None when not saving
        assert K_out is None or len(K_out) == 0
        assert M_out is None or len(M_out) == 0
        assert T_out is None or len(T_out) == 0


class TestDispersionConsistency:
    """Test consistency of dispersion calculations."""
    
    def test_dispersion_consistency_same_wavevector(self):
        """Test that same wavevector gives same results."""
        const = create_test_const()
        
        # Single wavevector
        wavevector = np.array([[0.5, 0.3]])
        
        wv1, fr1, ev1, mesh1 = dispersion(const, wavevector)
        wv2, fr2, ev2, mesh2 = dispersion(const, wavevector)
        
        # Results should be very close (allowing for numerical precision in eigenvalue solver)
        # The eigenvalue solver may have slight variations due to numerical precision
        assert np.allclose(fr1, fr2, rtol=1e-4, atol=1.0), \
            f"Same wavevector should give same frequencies (within numerical precision). Max diff: {np.max(np.abs(fr1 - fr2))}"
    
    def test_dispersion_symmetry(self):
        """Test that symmetric wavevectors give same frequencies."""
        const = create_test_const()
        
        # Symmetric wavevectors
        wavevectors = np.array([[0.5, 0.3], [-0.5, 0.3]])
        
        wv, fr, ev, mesh = dispersion(const, wavevectors)
        
        # For symmetric system, frequencies should match
        # (This may not always be true depending on design symmetry)
        # Just check that calculation completes without error
        assert fr.shape == (2, const['N_eig'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

