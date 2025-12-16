# Test Equivalence Documentation

This document tracks all unit tests performed to ensure functional equivalence between Python files in `2d-dispersion-py` and MATLAB files in `2D-dispersion-han`.

## Test Status Legend

- ✅ **PASSED** - Test passed, functionality verified
- ⚠️ **PARTIAL** - Test passed but with minor differences (documented)
- ❌ **FAILED** - Test failed, needs investigation
- 🔄 **PENDING** - Test not yet implemented
- 📊 **VISUAL** - Requires visual inspection (plotting functions)

## Test Categories

### 1. Core Dispersion Functions
### 2. System Matrix Functions
### 3. Element Functions
### 4. Wavevector Functions
### 5. Design Functions
### 6. Kernel Functions
### 7. Plotting Functions (Visual Verification Required)
### 8. Utility Functions

---

## Test Results

### Core Dispersion Functions

| MATLAB File | Python File | Test Status | Test File | Notes |
|------------|-------------|-------------|-----------|-------|
| `dispersion.m` | `dispersion.py` | ✅ CREATED | `test_dispersion.py` | Tests created, ready to run |
| `dispersion_with_matrix_save_opt.m` | `dispersion_with_matrix_save_opt.py` | ✅ CREATED | `test_dispersion.py` | Tests created, ready to run |

### System Matrix Functions

| MATLAB File | Python File | Test Status | Test File | Notes |
|------------|-------------|-------------|-----------|-------|
| `get_system_matrices.m` | `get_system_matrices.py` | ✅ CREATED | `test_system_matrices.py` | Tests created, ready to run |
| `get_transformation_matrix.m` | `get_transformation_matrix.py` | ✅ CREATED | `test_system_matrices.py` | Tests created, ready to run |
| `get_system_matrices_VEC.m` | `system_matrices_vec.py` | ✅ CREATED | `test_system_matrices.py` | Tests created, ready to run |
| `get_system_matrices_VEC_simplified.m` | `system_matrices_vec.py` | ✅ CREATED | `test_system_matrices.py` | Tests created, ready to run |

### Element Functions

| MATLAB File | Python File | Test Status | Test File | Notes |
|------------|-------------|-------------|-----------|-------|
| `get_element_stiffness.m` | `get_element_stiffness.py` | ✅ CREATED | `test_elements.py` | Tests created, ready to run |
| `get_element_mass.m` | `get_element_mass.py` | ✅ CREATED | `test_elements.py` | Tests created, ready to run |
| `get_pixel_properties.m` | `get_pixel_properties.py` | ✅ CREATED | `test_elements.py` | Tests created, ready to run |
| `get_element_stiffness_VEC.m` | `get_element_stiffness_VEC.py` | ✅ CREATED | `test_elements.py` | Tests created, ready to run |
| `get_element_mass_VEC.m` | `get_element_mass_VEC.py` | ✅ CREATED | `test_elements.py` | Tests created, ready to run |

### Wavevector Functions

| MATLAB File | Python File | Test Status | Test File | Notes |
|------------|-------------|-------------|-----------|-------|
| `get_IBZ_wavevectors.m` | `wavevectors.py` | ✅ CREATED | `test_wavevectors.py` | Tests created, ready to run |
| `get_IBZ_contour_wavevectors.m` | `wavevectors.py` | ✅ CREATED | `test_wavevectors.py` | Tests created, ready to run |

### Design Functions

| MATLAB File | Python File | Test Status | Test File | Notes |
|------------|-------------|-------------|-----------|-------|
| `get_design.m` | `get_design.py` | ✅ CREATED | `test_design.py` | Tests created, ready to run |
| `get_design2.m` | `get_design2.py` | ✅ CREATED | `test_design.py` | Tests created, ready to run |
| `convert_design.m` | `design_conversion.py` | ✅ CREATED | `test_design.py` | Tests created, ready to run |
| `apply_steel_rubber_paradigm.m` | `design_conversion.py` | ✅ CREATED | `test_design.py` | Tests created, ready to run |

### Kernel Functions

| MATLAB File | Python File | Test Status | Test File | Notes |
|------------|-------------|-------------|-----------|-------|
| `matern52_kernel.m` | `kernels.py` | ✅ CREATED | `test_kernels.py` | Tests created, ready to run |
| `matern52_prop.m` | `kernels.py` | ✅ CREATED | `test_kernels.py` | Tests created, ready to run |
| `periodic_kernel.m` | `kernels.py` | ✅ CREATED | `test_kernels.py` | Tests created, ready to run |
| `periodic_kernel_not_squared.m` | `kernels.py` | ✅ CREATED | `test_kernels.py` | Tests created, ready to run |
| `kernel_prop.m` | `kernels.py` | ✅ CREATED | `test_kernels.py` | Tests created, ready to run |

### Utility Functions

| MATLAB File | Python File | Test Status | Test File | Notes |
|------------|-------------|-------------|-----------|-------|
| `get_mask.m` | `get_mask.py` | ✅ CREATED | `test_utils.py` | Tests created, ready to run |
| `get_mesh.m` | `get_mesh.py` | ✅ CREATED | `test_utils.py` | Tests created, ready to run |
| `get_global_idxs.m` | `get_global_idxs.py` | ✅ CREATED | `test_utils.py` | Tests created, ready to run |
| `make_chunks.m` | `utils_han.py` | ✅ CREATED | `test_utils.py` | Tests created, ready to run |
| `init_storage.m` | `utils_han.py` | ✅ CREATED | `test_utils.py` | Tests created, ready to run |
| `cellofsparse_to_full.m` | `cellofsparse_to_full.py` | ✅ CREATED | `test_utils.py` | Tests created, ready to run |
| `check_contour_analysis.m` | `check_contour_analysis.py` | ✅ CREATED | `test_utils.py` | Tests created, ready to run |
| `apply_p4mm_symmetry.m` | `symmetry.py` | ✅ CREATED | `test_utils.py` | Tests created, ready to run |

### Plotting Functions (Visual Verification Required)

| MATLAB File | Python File | Test Status | Test File | Visual Check | Notes |
|------------|-------------|-------------|-----------|--------------|-------|
| `plot_dispersion.m` | `plotting.py` | 📊 CREATED | `test_plotting.py` | ⏳ **PENDING USER VERIFICATION** | Plot saved to `test_plots/plot_dispersion_basic.png` |
| `plot_dispersion_contour.m` | `plotting.py` | 📊 CREATED | `test_plotting.py` | ⏳ **PENDING USER VERIFICATION** | Plot saved to `test_plots/plot_dispersion_contour.png` |
| `plot_design.m` | `plotting.py` | 📊 CREATED | `test_plotting.py` | ⏳ **PENDING USER VERIFICATION** | Plots saved to `test_plots/plot_design_*.png` |
| `plot_mode.m` | `plotting.py` | 📊 CREATED | `test_plotting.py` | ⏳ **PENDING USER VERIFICATION** | Plot saved to `test_plots/plot_mode_basic.png` |
| `plot_eigenvector.m` | (functionality in test) | 📊 CREATED | `test_plotting.py` | ⏳ **PENDING USER VERIFICATION** | Plot saved to `test_plots/plot_eigenvector_components.png` |
| `plot_mesh.m` | (partial implementation) | 📊 CREATED | `test_plotting.py` | ⏳ **PENDING USER VERIFICATION** | Plot saved to `test_plots/plot_mesh_basic.png` |
| `plot_wavevectors.m` | (functionality in test) | 📊 CREATED | `test_plotting.py` | ⏳ **PENDING USER VERIFICATION** | Plot saved to `test_plots/plot_wavevectors_basic.png` |
| `plot_node_labels.m` | (missing) | ❌ NOT IMPLEMENTED | - | - | Needs implementation |
| `visualize_designs.m` | `plotting.py` | 📊 CREATED | `test_plotting.py` | ⏳ **PENDING USER VERIFICATION** | Plot saved to `test_plots/visualize_designs_multiple.png` |

## Test Methodology

### Numerical Functions
1. Generate test inputs (same for MATLAB and Python)
2. Run MATLAB function and save outputs
3. Run Python function with same inputs
4. Compare outputs with tolerance checks:
   - Absolute tolerance: 1e-6 for float32, 1e-12 for float64
   - Relative tolerance: 1e-5
5. Check matrix sparsity patterns match
6. Verify matrix dimensions match

### Plotting Functions
1. Generate test data
2. Create plots in both MATLAB and Python
3. Save plots to files
4. **USER VERIFICATION REQUIRED**: Compare plots visually
5. Check that plot data (axes limits, colors, etc.) match

## Test Data

Test data is stored in `test_data/` directory:
- `test_constants.mat` - Standard constants structure
- `test_designs.mat` - Sample design arrays
- `test_wavevectors.mat` - Sample wavevector arrays
- `test_results/` - Saved MATLAB outputs for comparison

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_elements.py

# Run with verbose output
pytest tests/ -v

# Run plotting tests (generates plots for visual inspection)
pytest tests/test_plotting.py --plots
```

## Known Differences

(To be filled in as tests are run)

1. **Indexing**: MATLAB uses 1-based indexing, Python uses 0-based
2. **Array ordering**: MATLAB is column-major, Python is row-major (handled in code)
3. **Complex numbers**: Both use same representation
4. **Sparse matrices**: Both use CSR format (converted appropriately)

