# Complete MATLAB to Python File Mapping

This document provides a complete one-to-one mapping of all MATLAB files in `2D-dispersion-han` to Python files in `2d-dispersion-py`.

## ✅ Complete Matches (Direct File-to-File)

| MATLAB File | Python File | Status |
|------------|-------------|--------|
| `apply_p4mm_symmetry.m` | `symmetry.py` → `apply_p4mm_symmetry()` | ✅ Function exists |
| `apply_steel_rubber_paradigm.m` | `design_conversion.py` → `apply_steel_rubber_paradigm()` | ✅ Function exists |
| `cellofsparse_to_full.m` | `cellofsparse_to_full.py` | ✅ **CREATED** |
| `check_contour_analysis.m` | `check_contour_analysis.py` | ✅ **CREATED** |
| `convert_design.m` | `design_conversion.py` → `convert_design()` | ✅ Function exists |
| `demo_create_Kr_and_Mr.m` | `demo_create_Kr_and_Mr.py` | ✅ Direct match |
| `design_parameters.m` | `design_parameters.py` | ✅ Direct match |
| `dispersion.m` | `dispersion.py` | ✅ Direct match |
| `dispersion_with_matrix_save_opt.m` | `dispersion_with_matrix_save_opt.py` | ✅ Direct match |
| `get_design.m` | `get_design.py` | ✅ Direct match |
| `get_design2.m` | `get_design2.py` | ✅ Direct match |
| `get_element_mass.m` | `get_element_mass.py` | ✅ Direct match |
| `get_element_mass_VEC.m` | `get_element_mass_VEC.py` | ✅ Direct match |
| `get_element_stiffness.m` | `get_element_stiffness.py` | ✅ Direct match |
| `get_element_stiffness_VEC.m` | `get_element_stiffness_VEC.py` | ✅ Direct match |
| `get_global_idxs.m` | `get_global_idxs.py` | ✅ Direct match |
| `get_mask.m` | `get_mask.py` | ✅ **CREATED** |
| `get_mesh.m` | `get_mesh.py` | ✅ **CREATED** |
| `get_pixel_properties.m` | `get_pixel_properties.py` | ✅ Direct match |
| `get_prop.m` | `get_prop.py` | ✅ Direct match |
| `get_system_matrices.m` | `get_system_matrices.py` | ✅ Direct match |
| `get_transformation_matrix.m` | `get_transformation_matrix.py` | ✅ Direct match |
| `init_storage.m` | `utils_han.py` → `init_storage()` | ✅ Function exists |
| `make_chunks.m` | `utils_han.py` → `make_chunks()` | ✅ Function exists |

## ⚠️ Files Needing Extraction from Combined Modules

These MATLAB files have Python equivalents but they're currently in combined modules. They need to be extracted to individual files:

| MATLAB File | Current Python Location | Action Needed |
|------------|------------------------|---------------|
| `get_IBZ_wavevectors.m` | `wavevectors.py` → `get_IBZ_wavevectors()` | Extract to `get_IBZ_wavevectors.py` |
| `get_IBZ_contour_wavevectors.m` | `wavevectors.py` → `get_IBZ_contour_wavevectors()` | Extract to `get_IBZ_contour_wavevectors.py` |
| `get_system_matrices_VEC.m` | `system_matrices_vec.py` → `get_system_matrices_VEC()` | Extract to `get_system_matrices_VEC.py` |
| `get_system_matrices_VEC_simplified.m` | `system_matrices_vec.py` → `get_system_matrices_VEC_simplified()` | Extract to `get_system_matrices_VEC_simplified.py` |
| `matern52_kernel.m` / `matern52.m` | `kernels.py` → `matern52_kernel()` | Extract to `matern52_kernel.py` |
| `matern52_prop.m` | `kernels.py` → `matern52_prop()` | Extract to `matern52_prop.py` |
| `periodic_kernel.m` | `kernels.py` → `periodic_kernel()` | Extract to `periodic_kernel.py` |
| `periodic_kernel_not_squared.m` | `kernels.py` → `periodic_kernel_not_squared()` | Extract to `periodic_kernel_not_squared.py` |
| `kernel_prop.m` | `kernels.py` → `kernel_prop()` | Extract to `kernel_prop.py` |
| `plot_dispersion.m` | `plotting.py` → `plot_dispersion()` | Extract to `plot_dispersion.py` |
| `plot_dispersion_contour.m` | `plotting.py` → `plot_dispersion_contour()` | Extract to `plot_dispersion_contour.py` |
| `plot_design.m` | `plotting.py` → `plot_design()` | Extract to `plot_design.py` |
| `plot_mode.m` | `plotting.py` → `plot_mode()` | Extract to `plot_mode.py` |
| `visualize_designs.m` | `plotting.py` → `visualize_designs()` | Extract to `visualize_designs.py` |

## ❌ Missing Files (Need to be Created)

These MATLAB files don't have Python equivalents yet:

| MATLAB File | Type | Notes |
|------------|------|-------|
| `ex_dispersion_batch_save.m` | Script | Batch processing script - needs Python equivalent |
| `generate_dispersion_dataset_Han.m` | Script | Dataset generation script - needs Python equivalent |
| `gridMesh.m` | Class | MATLAB class - may not need direct equivalent |
| `plot_dispersion_only.m` | Script | Plotting script - needs Python equivalent |
| `plot_eigenvector.m` | Script | Plotting script - needs Python equivalent |
| `plot_mesh.m` | Function | Plotting function - needs Python equivalent |
| `plot_mode_ui.m` | Function | MATLAB UI function - may not need equivalent (MATLAB-specific) |
| `plot_node_labels.m` | Function | Plotting utility - needs Python equivalent |
| `plot_wavevectors.m` | Function | Plotting function - needs Python equivalent |
| `run_plot_binarized.m` | Script | Script - needs Python equivalent |
| `run_plot_continuous.m` | Script | Script - needs Python equivalent |
| `test_batch_output.m` | Script | Test script - needs Python equivalent |
| `visualize_mesh_ordering.m` | Script | Visualization script - needs Python equivalent |

## Summary

- **Total MATLAB files**: 52 (excluding .asv backup files)
- **✅ Complete matches**: 25 files (including newly created)
- **⚠️ Functions in combined modules**: 15 files (need extraction)
- **❌ Missing files**: 13 files (need creation)

## Next Steps

1. **Extract functions** from combined modules to create individual files
2. **Create missing files** for scripts and functions that don't have Python equivalents
3. **Update import statements** across the codebase after extraction
4. **Test** that all functionality works correctly

