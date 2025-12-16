# MATLAB to Python Script Mapping

This document maps MATLAB scripts from `2D-dispersion-han` to their equivalent Python scripts in `2d-dispersion-py`.

## Core Dispersion Calculation

| MATLAB Script | Python Script | Notes |
|--------------|---------------|-------|
| `dispersion.m` | `dispersion.py` | Main dispersion calculation function |
| `dispersion_with_matrix_save_opt.m` | `dispersion_with_matrix_save_opt.py` | Dispersion with option to save K, M, T matrices |

## System Matrix Assembly

| MATLAB Script | Python Script | Notes |
|--------------|---------------|-------|
| `get_system_matrices.m` | `system_matrices.py` → `get_system_matrices()` | Assembles global K and M matrices |
| `get_system_matrices_VEC.m` | `system_matrices_vec.py` → `get_system_matrices_VEC()` | Vectorized version for performance |
| `get_system_matrices_VEC_simplified.m` | `system_matrices_vec.py` → `get_system_matrices_VEC_simplified()` | Simplified vectorized version |
| `get_transformation_matrix.m` | `system_matrices.py` → `get_transformation_matrix()` | Creates transformation matrix T for Bloch conditions |

## Element-Level Calculations

| MATLAB Script | Python Script | Notes |
|--------------|---------------|-------|
| `get_element_stiffness.m` | `elements.py` → `get_element_stiffness()` | Element stiffness matrix |
| `get_element_mass.m` | `elements.py` → `get_element_mass()` | Element mass matrix |
| `get_element_stiffness_VEC.m` | `elements_vec.py` → `get_element_stiffness_VEC()` | Vectorized element stiffness |
| `get_element_mass_VEC.m` | `elements_vec.py` → `get_element_mass_VEC()` | Vectorized element mass |
| `get_pixel_properties.m` | `elements.py` → `get_pixel_properties()` | Material properties from design |
| `get_global_idxs.m` | `get_global_idxs.py` → `get_global_idxs()` | Global DOF indices for elements |

## Wavevector Generation

| MATLAB Script | Python Script | Notes |
|--------------|---------------|-------|
| `get_IBZ_wavevectors.m` | `wavevectors.py` → `get_IBZ_wavevectors()` | Generate wavevectors in irreducible Brillouin zone |
| `get_IBZ_contour_wavevectors.m` | `wavevectors.py` → `get_IBZ_contour_wavevectors()` | Generate wavevectors along IBZ contour |

## Design Generation

| MATLAB Script | Python Script | Notes |
|--------------|---------------|-------|
| `get_design.m` | `get_design.py` → `get_design()` | Generate predefined design patterns |
| `get_design2.m` | `get_design2.py` → `get_design2()` | Generate design from design_parameters object |
| `design_parameters.m` | `design_parameters.py` → `DesignParameters` class | Design parameter class |
| `convert_design.m` | `design_conversion.py` → `convert_design()` | Convert design between formats |
| `apply_steel_rubber_paradigm.m` | `design_conversion.py` → `apply_steel_rubber_paradigm()` | Apply steel-rubber material paradigm |

## Kernel Functions (for Correlated Designs)

| MATLAB Script | Python Script | Notes |
|--------------|---------------|-------|
| `matern52.m` / `matern52_kernel.m` | `kernels.py` → `matern52_kernel()` | Matern 5/2 kernel |
| `periodic_kernel.m` | `kernels.py` → `periodic_kernel()` | Periodic kernel |
| `periodic_kernel_not_squared.m` | `kernels.py` → `periodic_kernel_not_squared()` | Periodic kernel variant |
| `kernel_prop.m` | `kernels.py` → `kernel_prop()` | Generate properties using kernel |
| `matern52_prop.m` | `kernels.py` → `matern52_prop()` | Generate properties using Matern52 kernel |

## Plotting and Visualization

| MATLAB Script | Python Script | Notes |
|--------------|---------------|-------|
| `plot_dispersion.m` | `plotting.py` → `plot_dispersion()` | Plot dispersion curves |
| `plot_dispersion_only.m` | `plot_dispersion_mat.py`, `plot_dispersion_pt.py`, `plot_dispersion_np.py` | Multiple Python versions for different data formats |
| `plot_dispersion_contour.m` | `plotting.py` → `plot_dispersion_contour()` | Plot dispersion contour |
| `plot_eigenvector.m` | `plotting.py` → `plot_mode()` | Plot eigenvector/mode shape |
| `plot_mode.m` | `plotting.py` → `plot_mode()` | Plot mode shape |
| `plot_design.m` | `plotting.py` → `plot_design()` | Visualize design pattern |
| `plot_mesh.m` | `plotting.py` → (part of `plot_mode()`) | Plot mesh |
| `visualize_designs.m` | `plotting.py` → `visualize_designs()` | Visualize multiple designs |
| `plot_wavevectors.m` | (functionality in `wavevectors.py` or plotting utilities) | Plot wavevector paths |

## Utility Functions

| MATLAB Script | Python Script | Notes |
|--------------|---------------|-------|
| `get_mesh.m` | `dispersion.py` → `get_mesh()` | Generate mesh information |
| `get_mask.m` | `utils.py` → `get_mask()` | Get mask for symmetry |
| `get_prop.m` | `get_prop.py` → `get_prop()` | Extract property from design_parameters |
| `make_chunks.m` | `utils_han.py` → `make_chunks()` | Create chunk ranges |
| `init_storage.m` | `utils_han.py` → `init_storage()` | Initialize storage arrays |
| `check_contour_analysis.m` | `utils.py` → `check_contour_analysis()` | Validate contour analysis |
| `apply_p4mm_symmetry.m` | `symmetry.py` → `apply_p4mm_symmetry()` | Apply p4mm symmetry |

## Demo and Example Scripts

| MATLAB Script | Python Script | Notes |
|--------------|---------------|-------|
| `demo_create_Kr_and_Mr.m` | `demo_create_Kr_and_Mr.py` | Demo: create reduced matrices Kr and Mr |
| `ex_dispersion_batch_save.m` | `example_pytorch_dataset.py`, `examples.py` | Batch processing examples |
| `generate_dispersion_dataset_Han.m` | `example_pytorch_dataset.py` | Dataset generation examples |

## Additional Python-Only Files

These Python files don't have direct MATLAB equivalents but provide additional functionality:

- `plot_dispersion_infer_eigenfrequencies.py` - Infer frequencies from eigenvectors
- `convert_mat_to_pytorch.py` - Convert MATLAB data to PyTorch format
- `mat73_loader.py` - Load MATLAB v7.3 HDF5 files
- `utils.py` - Additional utility functions
- `symmetry.py` - Extended symmetry operations
- `design_conversion.py` - Extended design conversion utilities

## Notes

1. **Function Naming**: Python functions use snake_case while MATLAB uses camelCase, but the functionality is equivalent.

2. **Vectorized Versions**: The Python codebase includes both standard and vectorized (`_VEC`) versions, matching the MATLAB structure.

3. **Multiple Implementations**: Some plotting functions have multiple Python implementations (`plot_dispersion_mat.py`, `plot_dispersion_pt.py`, `plot_dispersion_np.py`) for different data formats (MATLAB .mat, PyTorch, NumPy).

4. **Class-based Design**: `design_parameters.m` is a MATLAB class, while `design_parameters.py` is a Python class with similar functionality.

5. **Sparse Matrices**: Both MATLAB and Python use sparse matrix representations (MATLAB's `sparse`, Python's `scipy.sparse`).

