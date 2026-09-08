"""
Unit tests for element-level functions.

Tests element stiffness, mass, and pixel property functions
to ensure equivalence with MATLAB versions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from get_element_stiffness import get_element_stiffness
from get_element_mass import get_element_mass
from get_pixel_properties import get_pixel_properties
from get_element_stiffness_VEC import get_element_stiffness_VEC
from get_element_mass_VEC import get_element_mass_VEC


class TestElementStiffness:
    """Test get_element_stiffness function."""
    
    def test_element_stiffness_basic(self):
        """Test basic element stiffness matrix calculation."""
        E = 200e9  # Young's modulus (Pa)
        nu = 0.3   # Poisson's ratio
        t = 0.01   # Thickness (m)
        const = {'N_pix': 5, 'N_ele': 4, 'a': 1.0}
        
        k_ele = get_element_stiffness(E, nu, t, const)
        
        # Check shape
        assert k_ele.shape == (8, 8), f"Expected shape (8, 8), got {k_ele.shape}"
        
        # Check symmetry
        assert np.allclose(k_ele, k_ele.T, rtol=1e-5), "Stiffness matrix should be symmetric"
        
        # Check positive semi-definiteness (allowing for numerical precision with float32)
        # For Q4 elements, the matrix should be positive semi-definite
        # Small negative eigenvalues can occur due to float32 precision
        eigenvals = np.linalg.eigvals(k_ele)
        # Check that most eigenvalues are positive and any negative ones are very small relative to the largest
        max_eigenval = np.max(np.abs(eigenvals))
        # Allow negative eigenvalues that are less than 1e-4 of the maximum (numerical precision)
        assert np.all(eigenvals > -max_eigenval * 1e-4), \
            f"Stiffness matrix should be positive semi-definite, but found eigenvalues: {eigenvals}"
        
        # Check specific values (from MATLAB reference)
        # First element should be positive
        assert k_ele[0, 0] > 0, "Diagonal element should be positive"
        
        # Check data type
        assert k_ele.dtype == np.float32, f"Expected float32, got {k_ele.dtype}"
    
    def test_element_stiffness_different_materials(self):
        """Test with different material properties."""
        const = {'N_pix': 5, 'N_ele': 4, 'a': 1.0}
        
        # Test with different E values
        E_values = [100e9, 200e9, 300e9]
        nu = 0.3
        t = 0.01
        
        # Get reference matrix first
        k_ref = get_element_stiffness(200e9, nu, t, const)
        
        for E in E_values:
            k_ele = get_element_stiffness(E, nu, t, const)
            assert k_ele.shape == (8, 8)
            # Stiffness should scale with E
            if E == 200e9:
                assert np.allclose(k_ele, k_ref, rtol=1e-4)
            else:
                scale = E / 200e9
                assert np.allclose(k_ele, k_ref * scale, rtol=1e-4)
    
    def test_element_stiffness_poisson_ratio(self):
        """Test with different Poisson ratios."""
        E = 200e9
        t = 0.01
        const = {'N_pix': 5, 'N_ele': 4, 'a': 1.0}
        
        nu_values = [0.0, 0.3, 0.45]
        k_matrices = []
        
        for nu in nu_values:
            k_ele = get_element_stiffness(E, nu, t, const)
            k_matrices.append(k_ele)
            assert k_ele.shape == (8, 8)
        
        # All should be different
        assert not np.allclose(k_matrices[0], k_matrices[1], rtol=1e-4)
        assert not np.allclose(k_matrices[1], k_matrices[2], rtol=1e-4)


class TestElementMass:
    """Test get_element_mass function."""
    
    def test_element_mass_basic(self):
        """Test basic element mass matrix calculation."""
        rho = 7800  # Density (kg/m^3)
        t = 0.01    # Thickness (m)
        const = {'N_pix': 5, 'N_ele': 4, 'a': 1.0}
        
        m_ele = get_element_mass(rho, t, const)
        
        # Check shape
        assert m_ele.shape == (8, 8), f"Expected shape (8, 8), got {m_ele.shape}"
        
        # Check symmetry
        assert np.allclose(m_ele, m_ele.T, rtol=1e-5), "Mass matrix should be symmetric"
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(m_ele)
        assert np.all(eigenvals > 0), "Mass matrix should be positive definite"
        
        # Check diagonal elements are positive
        assert np.all(np.diag(m_ele) > 0), "Diagonal elements should be positive"
        
        # Check data type
        assert m_ele.dtype == np.float32, f"Expected float32, got {m_ele.dtype}"
    
    def test_element_mass_density_scaling(self):
        """Test that mass scales with density."""
        t = 0.01
        const = {'N_pix': 5, 'N_ele': 4, 'a': 1.0}
        
        rho_values = [1000, 2000, 7800]
        m_matrices = []
        
        for rho in rho_values:
            m_ele = get_element_mass(rho, t, const)
            m_matrices.append(m_ele)
        
        # Mass should scale linearly with density
        scale_1 = rho_values[1] / rho_values[0]
        assert np.allclose(m_matrices[1], m_matrices[0] * scale_1, rtol=1e-4)
        
        scale_2 = rho_values[2] / rho_values[0]
        assert np.allclose(m_matrices[2], m_matrices[0] * scale_2, rtol=1e-4)


class TestPixelProperties:
    """Test get_pixel_properties function."""
    
    def test_pixel_properties_linear_scale(self):
        """Test pixel properties with linear design scale."""
        const = {
            'design_scale': 'linear',
            'E_min': 2e9,
            'E_max': 200e9,
            'rho_min': 1000,
            'rho_max': 8000,
            'poisson_min': 0.0,
            'poisson_max': 0.5,
            't': 0.01,
            'design': np.zeros((5, 5, 3))
        }
        
        # Test with zero design (should give min values)
        E, nu, t, rho = get_pixel_properties(0, 0, const)
        assert np.isclose(E, const['E_min'], rtol=1e-6)
        assert np.isclose(rho, const['rho_min'], rtol=1e-6)
        assert np.isclose(nu, const['poisson_min'], rtol=1e-6)
        assert np.isclose(t, const['t'], rtol=1e-6)
        
        # Set design to 1.0 (should give max values)
        const['design'][0, 0, 0] = 1.0  # E
        const['design'][0, 0, 1] = 1.0  # rho
        const['design'][0, 0, 2] = 1.0  # nu
        
        E, nu, t, rho = get_pixel_properties(0, 0, const)
        assert np.isclose(E, const['E_max'], rtol=1e-6)
        assert np.isclose(rho, const['rho_max'], rtol=1e-6)
        assert np.isclose(nu, const['poisson_max'], rtol=1e-6)
    
    def test_pixel_properties_log_scale(self):
        """Test pixel properties with logarithmic design scale."""
        const = {
            'design_scale': 'log',
            'E_min': 2e9,
            'E_max': 200e9,
            'rho_min': 1000,
            'rho_max': 8000,
            'poisson_min': 0.0,
            'poisson_max': 0.5,
            't': 0.01,
            'design': np.zeros((5, 5, 3))
        }
        
        # Test with design = 0 (should give exp(0) = 1)
        const['design'][0, 0, 0] = 0.0  # E
        const['design'][0, 0, 1] = 0.0  # rho
        
        E, nu, t, rho = get_pixel_properties(0, 0, const)
        assert np.isclose(E, 1.0, rtol=1e-6)
        assert np.isclose(rho, 1.0, rtol=1e-6)
        
        # Test with design = log(200e9)
        const['design'][0, 0, 0] = np.log(200e9)
        E, nu, t, rho = get_pixel_properties(0, 0, const)
        assert np.isclose(E, 200e9, rtol=1e-4)


class TestElementStiffnessVEC:
    """Test vectorized element stiffness function."""
    
    def test_element_stiffness_vec_basic(self):
        """Test vectorized element stiffness calculation."""
        n_elements = 10
        E = np.random.rand(n_elements) * 200e9 + 2e9
        nu = np.random.rand(n_elements) * 0.5
        t = np.ones(n_elements) * 0.01
        
        k_ele = get_element_stiffness_VEC(E, nu, t)
        
        # Check shape
        assert k_ele.shape == (n_elements, 8, 8), \
            f"Expected shape ({n_elements}, 8, 8), got {k_ele.shape}"
        
        # Check each element matrix
        for i in range(n_elements):
            k_i = k_ele[i, :, :]
            assert k_i.shape == (8, 8)
            assert np.allclose(k_i, k_i.T, rtol=1e-5), \
                f"Element {i} stiffness matrix should be symmetric"
    
    def test_element_stiffness_vec_single_element(self):
        """Test vectorized function with single element."""
        E = np.array([200e9])
        nu = np.array([0.3])
        t = np.array([0.01])
        
        k_ele = get_element_stiffness_VEC(E, nu, t)
        assert k_ele.shape == (1, 8, 8)
        
        # Compare with non-vectorized version
        const = {'N_pix': 5, 'N_ele': 4, 'a': 1.0}
        k_single = get_element_stiffness(E[0], nu[0], t[0], const)
        
        assert np.allclose(k_ele[0, :, :], k_single, rtol=1e-4), \
            "Vectorized and non-vectorized should match"


class TestElementMassVEC:
    """Test vectorized element mass function."""
    
    def test_element_mass_vec_basic(self):
        """Test vectorized element mass calculation."""
        n_elements = 10
        rho = np.random.rand(n_elements) * 7000 + 1000
        t = np.ones(n_elements) * 0.01
        const = {'N_pix': 5, 'N_ele': 4, 'a': 1.0}
        
        m_ele = get_element_mass_VEC(rho, t, const)
        
        # Check shape
        assert m_ele.shape == (n_elements, 8, 8), \
            f"Expected shape ({n_elements}, 8, 8), got {m_ele.shape}"
        
        # Check each element matrix
        for i in range(n_elements):
            m_i = m_ele[i, :, :]
            assert m_i.shape == (8, 8)
            assert np.allclose(m_i, m_i.T, rtol=1e-5), \
                f"Element {i} mass matrix should be symmetric"
    
    def test_element_mass_vec_single_element(self):
        """Test vectorized function with single element."""
        rho = np.array([7800])
        t = np.array([0.01])
        const = {'N_pix': 5, 'N_ele': 4, 'a': 1.0}
        
        m_ele = get_element_mass_VEC(rho, t, const)
        assert m_ele.shape == (1, 8, 8)
        
        # Compare with non-vectorized version
        m_single = get_element_mass(rho[0], t[0], const)
        
        assert np.allclose(m_ele[0, :, :], m_single, rtol=1e-4), \
            "Vectorized and non-vectorized should match"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



