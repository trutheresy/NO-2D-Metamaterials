"""
2D Dispersion Analysis Package

A Python translation of the MATLAB 2D dispersion analysis code for metamaterials.
This package provides functionality for calculating dispersion relations, 
group velocities, and design sensitivities for 2D periodic structures.
"""

__version__ = "1.0.0"
__author__ = "Translated from MATLAB by Alex"

# Import main functions
from .dispersion import dispersion, dispersion2
from .dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt
from .design_parameters import DesignParameters
from .get_design import get_design
from .get_design2 import get_design2
from .get_prop import get_prop
from .system_matrices import get_system_matrices, get_transformation_matrix
from .system_matrices_vec import get_system_matrices_VEC, get_system_matrices_VEC_simplified
from .elements import get_element_stiffness, get_element_mass, get_pixel_properties
from .elements_vec import get_element_stiffness_VEC, get_element_mass_VEC
from .wavevectors import get_IBZ_wavevectors, get_IBZ_contour_wavevectors
from .plotting import (plot_dispersion, plot_design, plot_dispersion_surface, 
                      plot_dispersion_contour, plot_mode, visualize_designs)
from .kernels import matern52_kernel, periodic_kernel, generate_correlated_design, kernel_prop
from .symmetry import apply_p4mm_symmetry, apply_rotational_symmetry, check_symmetry
from .design_conversion import convert_design, design_to_explicit, explicit_to_design
from .utils import validate_constants, check_contour_analysis, compute_band_gap, linspaceNDim

__all__ = [
    # Core dispersion functions
    'dispersion',
    'dispersion2',
    'dispersion_with_matrix_save_opt',
    
    # Design and parameters
    'DesignParameters',
    'get_design',
    'get_design2',
    'get_prop',
    'convert_design',
    'design_to_explicit',
    'explicit_to_design',
    
    # System assembly
    'get_system_matrices',
    'get_system_matrices_VEC',
    'get_system_matrices_VEC_simplified',
    'get_transformation_matrix',
    'get_element_stiffness',
    'get_element_stiffness_VEC',
    'get_element_mass',
    'get_element_mass_VEC',
    'get_pixel_properties',
    
    # Wavevectors
    'get_IBZ_wavevectors',
    'get_IBZ_contour_wavevectors',
    
    # Visualization
    'plot_dispersion',
    'plot_design',
    'plot_dispersion_surface',
    'plot_dispersion_contour',
    'plot_mode',
    'visualize_designs',
    
    # Kernels and GP
    'matern52_kernel',
    'periodic_kernel',
    'kernel_prop',
    'generate_correlated_design',
    
    # Symmetry
    'apply_p4mm_symmetry',
    'apply_rotational_symmetry',
    'check_symmetry',
    
    # Utilities
    'validate_constants',
    'check_contour_analysis',
    'compute_band_gap',
    'linspaceNDim'
]
