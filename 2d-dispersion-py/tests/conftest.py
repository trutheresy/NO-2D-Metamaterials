"""
Pytest configuration file.

This file provides shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_constants():
    """Create a standard test constants structure."""
    from get_design import get_design
    
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


@pytest.fixture
def tolerance():
    """Standard tolerance values for tests."""
    return {
        'rtol': 1e-5,  # Relative tolerance
        'atol': 1e-6,  # Absolute tolerance
        'sparse_rtol': 1e-4,  # For sparse matrix comparisons
    }



