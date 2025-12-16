# New Python Files Created (Matching MATLAB Names)

This document lists all the new Python files created to match MATLAB file names one-to-one.

## ✅ Files Created

### System Matrix Functions
1. **`get_system_matrices.py`** - Extracted from `system_matrices.py`
   - Contains: `get_system_matrices()` function

2. **`get_transformation_matrix.py`** - Extracted from `system_matrices.py`
   - Contains: `get_transformation_matrix()` function

3. **`get_system_matrices_VEC.py`** - Needs to be extracted from `system_matrices_vec.py`
   - Contains: `get_system_matrices_VEC()` function

4. **`get_system_matrices_VEC_simplified.py`** - Needs to be extracted from `system_matrices_vec.py`
   - Contains: `get_system_matrices_VEC_simplified()` function

### Element Functions
5. **`get_element_stiffness.py`** - Extracted from `elements.py`
   - Contains: `get_element_stiffness()` function

6. **`get_element_mass.py`** - Extracted from `elements.py`
   - Contains: `get_element_mass()` function

7. **`get_pixel_properties.py`** - Extracted from `elements.py`
   - Contains: `get_pixel_properties()` function

8. **`get_element_stiffness_VEC.py`** - Extracted from `elements_vec.py`
   - Contains: `get_element_stiffness_VEC()` function

9. **`get_element_mass_VEC.py`** - Extracted from `elements_vec.py`
   - Contains: `get_element_mass_VEC()` function

## 📋 Files Still To Be Created

### Wavevector Functions
- `get_IBZ_wavevectors.py` - Extract from `wavevectors.py`
- `get_IBZ_contour_wavevectors.py` - Extract from `wavevectors.py`

### Kernel Functions
- `matern52_kernel.py` or `matern52.py` - Extract from `kernels.py`
- `periodic_kernel.py` - Extract from `kernels.py`
- `periodic_kernel_not_squared.py` - Extract from `kernels.py`
- `kernel_prop.py` - Extract from `kernels.py`
- `matern52_prop.py` - Extract from `kernels.py`

### Design Functions
- `convert_design.py` - Extract from `design_conversion.py`
- `apply_steel_rubber_paradigm.py` - Extract from `design_conversion.py`

### Plotting Functions
- `plot_dispersion.py` - Extract from `plotting.py`
- `plot_dispersion_contour.py` - Extract from `plotting.py`
- `plot_eigenvector.py` - Extract from `plotting.py`
- `plot_mode.py` - Extract from `plotting.py`
- `plot_design.py` - Extract from `plotting.py`
- `plot_mesh.py` - Extract from `plotting.py`
- `visualize_designs.py` - Extract from `plotting.py`

### Utility Functions
- `get_mesh.py` - Extract from `dispersion.py`
- `get_mask.py` - Extract from `utils.py`
- `check_contour_analysis.py` - Extract from `utils.py`
- `make_chunks.py` - Extract from `utils_han.py` (or keep as separate file)
- `init_storage.py` - Extract from `utils_han.py` (or keep as separate file)
- `apply_p4mm_symmetry.py` - Extract from `symmetry.py`

## ✅ Files Already Matching MATLAB Names (No Change Needed)
- `dispersion.py`
- `dispersion_with_matrix_save_opt.py`
- `get_design.py`
- `get_design2.py`
- `design_parameters.py`
- `get_global_idxs.py`
- `get_prop.py`
- `demo_create_Kr_and_Mr.py`

## Notes
- All new files follow Python naming conventions (snake_case) matching MATLAB camelCase
- Each file contains a single function matching the MATLAB equivalent
- Import statements will need to be updated across the codebase
- Original combined files can be kept for backward compatibility (with deprecation warnings)

