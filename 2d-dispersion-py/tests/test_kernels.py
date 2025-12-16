"""
Unit tests for kernel functions.

Tests Gaussian process kernel functions to ensure equivalence
with MATLAB versions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels import (matern52_kernel, periodic_kernel, periodic_kernel_not_squared,
                    kernel_prop, matern52_prop)


class TestMatern52Kernel:
    """Test matern52_kernel function."""
    
    def test_matern52_kernel_basic(self):
        """Test basic Matern 5/2 kernel calculation."""
        n_points = 5
        # Use same points for both to ensure symmetry
        points = np.random.rand(n_points, 2)
        sigma_f = 1.0
        sigma_l = 0.5
        
        C = matern52_kernel(points, points, sigma_f, sigma_l)
        
        # Check shape
        assert C.shape == (n_points, n_points)
        
        # Check symmetry (only symmetric when points_i == points_j)
        assert np.allclose(C, C.T, rtol=1e-10), "Covariance matrix should be symmetric"
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(C)
        assert np.all(eigenvals > 0), "Covariance matrix should be positive definite"
        
        # Check diagonal elements
        assert np.allclose(np.diag(C), sigma_f**2), \
            "Diagonal should be sigma_f^2"
    
    def test_matern52_kernel_different_sizes(self):
        """Test kernel with different input sizes."""
        points_i = np.random.rand(3, 2)
        points_j = np.random.rand(5, 2)
        sigma_f = 1.0
        sigma_l = 0.5
        
        C = matern52_kernel(points_i, points_j, sigma_f, sigma_l)
        
        # Check shape
        assert C.shape == (3, 5)
    
    def test_matern52_kernel_scale_parameters(self):
        """Test that scale parameters work correctly."""
        points = np.random.rand(5, 2)
        sigma_l = 0.5
        
        # Test with different sigma_f
        C1 = matern52_kernel(points, points, sigma_f=1.0, sigma_l=sigma_l)
        C2 = matern52_kernel(points, points, sigma_f=2.0, sigma_l=sigma_l)
        
        # Should scale with sigma_f^2
        assert np.allclose(C2, C1 * 4, rtol=1e-10)


class TestPeriodicKernel:
    """Test periodic_kernel function."""
    
    def test_periodic_kernel_basic(self):
        """Test basic periodic kernel calculation."""
        n_points = 5
        # Use same points for both to ensure symmetry
        points = np.random.rand(n_points, 2)
        sigma_f = 1.0
        sigma_l = 0.5
        period = [1.0, 1.0]
        
        C = periodic_kernel(points, points, sigma_f, sigma_l, period)
        
        # Check shape
        assert C.shape == (n_points, n_points)
        
        # Check symmetry (only symmetric when points_i == points_j)
        assert np.allclose(C, C.T, rtol=1e-10)
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(C)
        assert np.all(eigenvals > 0)
        
        # Check diagonal
        assert np.allclose(np.diag(C), sigma_f**2)
    
    def test_periodic_kernel_periodicity(self):
        """Test that kernel is periodic."""
        points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        sigma_f = 1.0
        sigma_l = 0.5
        period = [1.0, 1.0]
        
        C = periodic_kernel(points, points, sigma_f, sigma_l, period)
        
        # Points at 0 and 1 should have same covariance (periodic)
        assert np.isclose(C[0, 2], C[0, 0], rtol=1e-5)


class TestPeriodicKernelNotSquared:
    """Test periodic_kernel_not_squared function."""
    
    def test_periodic_kernel_not_squared_basic(self):
        """Test basic periodic kernel (not squared) calculation."""
        n_points = 5
        # Use same points for both to ensure symmetry
        points = np.random.rand(n_points, 2)
        sigma_f = 1.0
        sigma_l = 0.5
        period = [1.0, 1.0]
        
        C = periodic_kernel_not_squared(points, points, sigma_f, sigma_l, period)
        
        # Check shape
        assert C.shape == (n_points, n_points)
        
        # Check symmetry (only symmetric when points_i == points_j)
        assert np.allclose(C, C.T, rtol=1e-10)
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(C)
        assert np.all(eigenvals > 0)


class TestKernelProp:
    """Test kernel_prop function."""
    
    def test_kernel_prop_matern52(self):
        """Test property generation with Matern52 kernel."""
        kernel = 'matern52'
        N_pix = 5
        design_options = {
            'sigma_f': 1.0,
            'sigma_l': 0.5
        }
        
        prop = kernel_prop(kernel, N_pix, design_options)
        
        # Check shape
        assert prop.shape == (N_pix, N_pix)
        
        # Check range (should be thresholded to [0, 1])
        assert np.all(prop >= 0)
        assert np.all(prop <= 1)
    
    def test_kernel_prop_periodic(self):
        """Test property generation with periodic kernel."""
        kernel = 'periodic'
        N_pix = 5
        design_options = {
            'sigma_f': 1.0,
            'sigma_l': 0.5
        }
        
        prop = kernel_prop(kernel, N_pix, design_options)
        
        # Check shape
        assert prop.shape == (N_pix, N_pix)
        
        # Check range
        assert np.all(prop >= 0)
        assert np.all(prop <= 1)


class TestMatern52Prop:
    """Test matern52_prop function."""
    
    def test_matern52_prop_basic(self):
        """Test basic Matern52 property generation."""
        kernel = 'matern52'
        N_pix = [5, 5]
        design_options = {
            'sigma_f': 1.0,
            'sigma_l': 0.5
        }
        
        prop = matern52_prop(kernel, N_pix, design_options)
        
        # Check shape
        assert prop.shape == tuple(N_pix)
        
        # Check range
        assert np.all(prop >= 0)
        assert np.all(prop <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



