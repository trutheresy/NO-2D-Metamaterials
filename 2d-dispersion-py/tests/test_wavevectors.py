"""
Unit tests for wavevector functions.

Tests wavevector generation functions to ensure equivalence
with MATLAB versions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavevectors import get_IBZ_wavevectors, get_IBZ_contour_wavevectors


class TestIBZWavevectors:
    """Test get_IBZ_wavevectors function."""
    
    def test_ibz_wavevectors_none_symmetry(self):
        """Test wavevector generation with no symmetry."""
        N_wv = [10, 5]
        a = 1.0
        
        wavevectors = get_IBZ_wavevectors(N_wv, a, symmetry_type='none')
        
        # Check shape
        assert wavevectors.shape[1] == 2, "Wavevectors should be 2D"
        
        # Check range: x from -pi/a to pi/a, y from 0 to pi/a
        assert np.all(wavevectors[:, 0] >= -np.pi/a - 1e-10)
        assert np.all(wavevectors[:, 0] <= np.pi/a + 1e-10)
        assert np.all(wavevectors[:, 1] >= -1e-10)
        assert np.all(wavevectors[:, 1] <= np.pi/a + 1e-10)
        
        # Check number of wavevectors
        expected_count = N_wv[0] * N_wv[1]
        assert wavevectors.shape[0] == expected_count, \
            f"Expected {expected_count} wavevectors, got {wavevectors.shape[0]}"
    
    def test_ibz_wavevectors_p4mm_symmetry(self):
        """Test wavevector generation with p4mm symmetry."""
        N_wv = [10, 10]
        a = 1.0
        
        wavevectors = get_IBZ_wavevectors(N_wv, a, symmetry_type='p4mm')
        
        # Check shape
        assert wavevectors.shape[1] == 2
        
        # For p4mm, wavevectors should satisfy: x >= 0, y >= 0, y <= x
        assert np.all(wavevectors[:, 0] >= -1e-10), "x should be >= 0"
        assert np.all(wavevectors[:, 1] >= -1e-10), "y should be >= 0"
        assert np.all(wavevectors[:, 1] - wavevectors[:, 0] <= 1e-10), \
            "y should be <= x"
        
        # Should have fewer wavevectors than full grid
        full_count = N_wv[0] * N_wv[1]
        assert wavevectors.shape[0] < full_count, \
            "p4mm symmetry should reduce number of wavevectors"
    
    def test_ibz_wavevectors_omit_symmetry(self):
        """Test wavevector generation with omit symmetry (full square)."""
        N_wv = [10, 10]
        a = 1.0
        
        wavevectors = get_IBZ_wavevectors(N_wv, a, symmetry_type='omit')
        
        # Check shape
        assert wavevectors.shape[1] == 2
        
        # For omit, should cover full square: -pi/a to pi/a in both directions
        assert np.all(wavevectors[:, 0] >= -np.pi/a - 1e-10)
        assert np.all(wavevectors[:, 0] <= np.pi/a + 1e-10)
        assert np.all(wavevectors[:, 1] >= -np.pi/a - 1e-10)
        assert np.all(wavevectors[:, 1] <= np.pi/a + 1e-10)
        
        # Should have all wavevectors
        expected_count = N_wv[0] * N_wv[1]
        assert wavevectors.shape[0] == expected_count
    
    def test_ibz_wavevectors_tesselations(self):
        """Test wavevector generation with tesselations."""
        N_wv = [5, 5]
        a = 1.0
        N_tesselations = 2
        
        wavevectors = get_IBZ_wavevectors(N_wv, a, symmetry_type='none', 
                                         N_tesselations=N_tesselations)
        
        # With tesselations, wavevectors should be scaled
        wavevectors_single = get_IBZ_wavevectors(N_wv, a, symmetry_type='none', 
                                                N_tesselations=1)
        
        # Check that tesselated wavevectors are scaled
        assert np.allclose(wavevectors, wavevectors_single * N_tesselations, rtol=1e-10)


class TestIBZContourWavevectors:
    """Test get_IBZ_contour_wavevectors function."""
    
    def test_ibz_contour_wavevectors_p4mm(self):
        """Test contour wavevector generation for p4mm symmetry."""
        N_k = 10
        a = 1.0
        
        wavevectors, contour_info = get_IBZ_contour_wavevectors(N_k, a, 
                                                               symmetry_type='p4mm')
        
        # Check shape
        assert wavevectors.shape[1] == 2
        
        # Check contour info
        assert 'N_segment' in contour_info
        assert 'vertex_labels' in contour_info
        assert 'vertices' in contour_info
        assert 'wavevector_parameter' in contour_info
        
        # For p4mm, should have 3 segments (Gamma->X->M->Gamma)
        assert contour_info['N_segment'] == 3
        
        # Check that wavevectors follow the contour
        # Should start and end at Gamma (0, 0)
        assert np.allclose(wavevectors[0], [0, 0], atol=1e-10)
        assert np.allclose(wavevectors[-1], [0, 0], atol=1e-10)
    
    def test_ibz_contour_wavevectors_none(self):
        """Test contour wavevector generation with no symmetry."""
        N_k = 10
        a = 1.0
        
        wavevectors, contour_info = get_IBZ_contour_wavevectors(N_k, a, 
                                                               symmetry_type='none')
        
        # Check shape
        assert wavevectors.shape[1] == 2
        
        # Should start and end at Gamma
        assert np.allclose(wavevectors[0], [0, 0], atol=1e-10)
        assert np.allclose(wavevectors[-1], [0, 0], atol=1e-10)
        
        # Should have more segments than p4mm
        assert contour_info['N_segment'] > 3
    
    def test_ibz_contour_wavevectors_vertex_labels(self):
        """Test that vertex labels are correct."""
        N_k = 10
        a = 1.0
        
        wavevectors, contour_info = get_IBZ_contour_wavevectors(N_k, a, 
                                                               symmetry_type='p4mm')
        
        # Check vertex labels
        labels = contour_info['vertex_labels']
        assert len(labels) == contour_info['N_segment'] + 1
        
        # First and last should be Gamma
        assert 'Gamma' in labels[0] or r'$\Gamma$' in labels[0]
        assert 'Gamma' in labels[-1] or r'$\Gamma$' in labels[-1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



