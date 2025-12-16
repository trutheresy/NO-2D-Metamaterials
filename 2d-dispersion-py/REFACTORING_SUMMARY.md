# Refactoring Summary: MATLAB-to-Python File Name Mapping

This document lists all the new Python files created to match MATLAB file names one-to-one.

## Files Created (Matching MATLAB Names)

### Core Dispersion Functions
- ✅ `dispersion.py` (already existed)
- ✅ `dispersion_with_matrix_save_opt.py` (already existed)
- ✅ `get_mesh.py` (needs to be extracted from dispersion.py)

### System Matrix Functions
- ✅ `get_system_matrices.py` (CREATED - extracted from system_matrices.py)
- ✅ `get_transformation_matrix.py` (CREATED - extracted from system_matrices.py)
- ✅ `get_system_matrices_VEC.py` (needs to be extracted from system_matrices_vec.py)
- ✅ `get_system_matrices_VEC_simplified.py` (needs to be extracted from system_matrices_vec.py)

### Element Functions
- ✅ `get_element_stiffness.py` (CREATED - extracted from elements.py)
- ✅ `get_element_mass.py` (CREATED - extracted from elements.py)
- ✅ `get_pixel_properties.py` (CREATED - extracted from elements.py)
- ✅ `get_element_stiffness_VEC.py` (CREATED - extracted from elements_vec.py)
- ✅ `get_element_mass_VEC.py` (CREATED - extracted from elements_vec.py)

### Wavevector Functions
- ✅ `get_IBZ_wavevectors.py` (needs to be extracted from wavevectors.py)
- ✅ `get_IBZ_contour_wavevectors.py` (needs to be extracted from wavevectors.py)

### Design Functions
- ✅ `get_design.py` (already existed)
- ✅ `get_design2.py` (already existed)
- ✅ `design_parameters.py` (already existed)
- ✅ `convert_design.py` (needs to be extracted from design_conversion.py)
- ✅ `apply_steel_rubber_paradigm.py` (needs to be extracted from design_conversion.py)

### Kernel Functions
- ✅ `matern52_kernel.py` or `matern52.py` (needs to be extracted from kernels.py)
- ✅ `periodic_kernel.py` (needs to be extracted from kernels.py)
- ✅ `periodic_kernel_not_squared.py` (needs to be extracted from kernels.py)
- ✅ `kernel_prop.py` (needs to be extracted from kernels.py)
- ✅ `matern52_prop.py` (needs to be extracted from kernels.py)

### Plotting Functions
- ✅ `plot_dispersion.py` (needs to be extracted from plotting.py)
- ✅ `plot_dispersion_contour.py` (needs to be extracted from plotting.py)
- ✅ `plot_eigenvector.py` (needs to be extracted from plotting.py)
- ✅ `plot_mode.py` (needs to be extracted from plotting.py)
- ✅ `plot_design.py` (needs to be extracted from plotting.py)
- ✅ `plot_mesh.py` (needs to be extracted from plotting.py)
- ✅ `visualize_designs.py` (needs to be extracted from plotting.py)

### Utility Functions
- ✅ `get_mask.py` (needs to be extracted from utils.py)
- ✅ `check_contour_analysis.py` (needs to be extracted from utils.py)
- ✅ `make_chunks.py` (already exists in utils_han.py - needs to be separate file)
- ✅ `init_storage.py` (already exists in utils_han.py - needs to be separate file)
- ✅ `apply_p4mm_symmetry.py` (needs to be extracted from symmetry.py)

### Already Existing Files (No Change Needed)
- ✅ `get_global_idxs.py` (already exists)
- ✅ `get_prop.py` (already exists)
- ✅ `demo_create_Kr_and_Mr.py` (already exists - matches MATLAB name)

## Files That Don't Have Direct MATLAB Equivalents
These Python-only files should remain as-is:
- `plot_dispersion_infer_eigenfrequencies.py`
- `convert_mat_to_pytorch.py`
- `mat73_loader.py`
- `utils.py` (contains multiple utility functions)
- `symmetry.py` (contains multiple symmetry functions)
- `design_conversion.py` (will be split into individual files)
- `plotting.py` (will be split into individual files)
- `elements.py` (will be split into individual files)
- `elements_vec.py` (will be split into individual files)
- `system_matrices.py` (will be split into individual files)
- `system_matrices_vec.py` (will be split into individual files)
- `wavevectors.py` (will be split into individual files)
- `kernels.py` (will be split into individual files)

## Next Steps
1. Continue creating individual files from combined modules
2. Update all import statements across the codebase
3. Test that all functionality still works
4. Optionally keep old combined files for backward compatibility (with deprecation warnings)

