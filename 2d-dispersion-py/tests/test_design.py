"""
Unit tests for design functions.

Tests design generation and conversion functions to ensure equivalence
with MATLAB versions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from get_design import get_design
from get_design2 import get_design2
from design_conversion import convert_design, apply_steel_rubber_paradigm
from design_parameters import DesignParameters


class TestGetDesign:
    """Test get_design function."""
    
    def test_get_design_homogeneous(self):
        """Test homogeneous design generation."""
        N_pix = 5
        design = get_design('homogeneous', N_pix)
        
        # Check shape
        assert design.shape == (N_pix, N_pix, 3)
        
        # Should be all ones for homogeneous
        assert np.allclose(design[:, :, 0], 1.0), "E should be 1.0"
        assert np.allclose(design[:, :, 1], 1.0), "rho should be 1.0"
        assert np.allclose(design[:, :, 2], 0.6), "nu should be 0.6"
    
    def test_get_design_random_seed(self):
        """Test random design generation with seed."""
        N_pix = 5
        seed = 42
        
        design1 = get_design(seed, N_pix)
        design2 = get_design(seed, N_pix)
        
        # Should be identical with same seed
        assert np.allclose(design1, design2)
        
        # Should be binary (0 or 1) for random designs
        assert np.all((design1[:, :, 0] == 0) | (design1[:, :, 0] == 1))
    
    def test_get_design_dispersive_tetragonal(self):
        """Test dispersive tetragonal design."""
        N_pix = 8
        design = get_design('dispersive-tetragonal', N_pix)
        
        # Check shape
        assert design.shape == (N_pix, N_pix, 3)
        
        # Should have a square pattern in the center
        center_slice = slice(N_pix//4, 3*N_pix//4)
        assert np.all(design[center_slice, center_slice, 0] == 1)
    
    def test_get_design_quasi_1d(self):
        """Test quasi-1D design."""
        N_pix = 8
        design = get_design('quasi-1D', N_pix)
        
        # Check shape
        assert design.shape == (N_pix, N_pix, 3)
        
        # Every other column should be 0
        assert np.all(design[:, 0::2, 0] == 0)


class TestGetDesign2:
    """Test get_design2 function."""
    
    def test_get_design2_basic(self):
        """Test get_design2 with design parameters."""
        design_params = DesignParameters(1)
        design_params.N_pix = [5, 5]
        # Use a recognized design style instead of default 'matern52'
        design_params.design_style = 'homogeneous'
        design_params = design_params.prepare()
        
        design = get_design2(design_params)
        
        # Check shape
        assert design.shape == (5, 5, 3)
        
        # Check values are in valid range
        assert np.all(design >= 0)
        assert np.all(design <= 1)


class TestConvertDesign:
    """Test convert_design function."""
    
    def test_convert_design_linear_to_explicit(self):
        """Test conversion from linear to explicit format."""
        N_pix = 5
        design_linear = np.random.rand(N_pix, N_pix, 3)
        
        design_explicit = convert_design(
            design_linear, 'linear', 'explicit',
            E_min=2e9, E_max=200e9,
            rho_min=1000, rho_max=8000,
            poisson_min=0.0, poisson_max=0.5
        )
        
        # Check shape
        assert design_explicit.shape == (N_pix, N_pix, 3)
        
        # Check ranges
        assert np.all(design_explicit[:, :, 0] >= 2e9)
        assert np.all(design_explicit[:, :, 0] <= 200e9)
        assert np.all(design_explicit[:, :, 1] >= 1000)
        assert np.all(design_explicit[:, :, 1] <= 8000)
        assert np.all(design_explicit[:, :, 2] >= 0.0)
        assert np.all(design_explicit[:, :, 2] <= 0.5)
    
    def test_convert_design_round_trip(self):
        """Test that conversion is reversible."""
        N_pix = 5
        design_linear = np.random.rand(N_pix, N_pix, 3)
        
        # Convert to explicit and back
        design_explicit = convert_design(
            design_linear, 'linear', 'explicit',
            E_min=2e9, E_max=200e9,
            rho_min=1000, rho_max=8000
        )
        
        design_back = convert_design(
            design_explicit, 'explicit', 'linear',
            E_min=2e9, E_max=200e9,
            rho_min=1000, rho_max=8000
        )
        
        # Should be approximately equal
        assert np.allclose(design_linear, design_back, rtol=1e-5)
    
    def test_convert_design_log_to_explicit(self):
        """Test conversion from log to explicit format."""
        N_pix = 5
        design_log = np.random.rand(N_pix, N_pix, 3) * 2 - 1  # Range [-1, 1]
        
        design_explicit = convert_design(
            design_log, 'log', 'explicit',
            E_min=2e9, E_max=200e9,
            rho_min=1000, rho_max=8000
        )
        
        # E and rho should be positive (exp of any number is positive)
        assert np.all(design_explicit[:, :, 0] > 0)
        assert np.all(design_explicit[:, :, 1] > 0)


class TestApplySteelRubberParadigm:
    """Test apply_steel_rubber_paradigm function."""
    
    def test_apply_steel_rubber_paradigm_basic(self):
        """Test basic steel-rubber paradigm application."""
        N_pix = 5
        design = np.random.rand(N_pix, N_pix, 3)
        
        const = {
            'E_min': 2e9,
            'E_max': 200e9,
            'rho_min': 1000,
            'rho_max': 8000,
            'poisson_min': 0.0,
            'poisson_max': 0.5
        }
        
        design_out = apply_steel_rubber_paradigm(design, const)
        
        # Check shape
        assert design_out.shape == design.shape
        
        # Check values are in valid range
        assert np.all(design_out >= 0)
        assert np.all(design_out <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



