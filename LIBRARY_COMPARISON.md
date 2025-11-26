# Library Comparison: 2D-dispersion-Han vs 2D-dispersion-Alex (Both MATLAB)

## Executive Summary

This document provides a comprehensive comparison between two MATLAB implementations of 2D dispersion analysis for metamaterials:

- **2D-dispersion-han**: Active MATLAB library by Han (located in `2D-dispersion-han/`)
- **2D-dispersion-alex**: Retired MATLAB library by Alex (located in `retired/2D-dispersion_alex/`)

Both libraries implement the same core functionality for calculating dispersion relations in 2D periodic metamaterials, but with some key differences in features, organization, and capabilities.

---

## 1. Library Status and Location

| Aspect | 2D-dispersion-Han | 2D-dispersion-Alex |
|--------|-------------------|-------------------|
| **Status** | ✅ Active | ⚠️ Retired |
| **Location** | `2D-dispersion-han/` | `retired/2D-dispersion_alex/` |
| **Language** | MATLAB | MATLAB |
| **File Count** | ~38 functions | ~40 functions + scripts |

---

## 2. File Structure Comparison

### 2D-dispersion-Han (Active)
```
2D-dispersion-han/
├── Core Functions (~38 .m files)
│   ├── dispersion.m
│   ├── dispersion_with_matrix_save_opt.m
│   ├── get_design.m, get_design2.m
│   ├── get_system_matrices.m
│   ├── get_system_matrices_VEC.m
│   ├── get_system_matrices_VEC_simplified.m ⭐
│   └── ... (element, wavevector, plotting functions)
├── Scripts
│   ├── generate_dispersion_dataset_Han.m
│   ├── ex_dispersion_batch_save.m
│   └── init_storage.m
└── OUTPUT/ (generated)
```

**Characteristics:**
- Clean, focused file structure
- Core functionality only
- Includes `get_system_matrices_VEC_simplified.m` optimization
- Supports `eigenvector_dtype` for memory optimization

### 2D-dispersion-Alex (Retired)
```
retired/2D-dispersion_alex/
├── Core Functions (~40 .m files)
│   ├── dispersion.m
│   ├── dispersion2.m ⭐ (Enhanced with group velocity)
│   ├── dispersion_with_matrix_save_opt.m
│   ├── get_design.m, get_design2.m
│   ├── get_system_matrices.m
│   ├── get_system_matrices_VEC.m
│   ├── get_element_stiffness_sensitivity.m ⭐
│   ├── get_element_mass_sensitivity.m ⭐
│   ├── get_duddesign.m ⭐
│   └── ... (more functions)
├── Scripts (Many more)
│   ├── generate_dispersion_dataset_Han.m
│   ├── dispersion_dataset_script.m
│   ├── dispersion2_dataset_script.m ⭐
│   ├── dispersion2_script.m ⭐
│   ├── dispersion_contour_vs_full_IBZ_script.m
│   ├── dispersion_dataset_visualizer.m
│   ├── dispersion_surface_script.m
│   └── ... (many more scripts)
├── Legacy Tools
│   ├── legacy/ (legacy plotting functions)
│   ├── legacy_tools_for_duke_project/
│   └── legacy_validation_files/ ⭐ (11 validation scripts)
├── Utilities
│   ├── convert_mat_precision.m ⭐
│   ├── slicing_data.m ⭐
│   └── linspaceNDim.m
└── png/ (generated plots)
```

**Characteristics:**
- More extensive file structure
- Additional advanced features (dispersion2, sensitivity analysis)
- Many example/validation scripts
- Legacy code preserved
- More experimental/development code

---

## 3. Core Functionality Comparison

### ✅ Common Core Functions (Both Libraries)

Both libraries share these core functions with identical or very similar implementations:

| Function | Purpose | Status |
|----------|---------|--------|
| `dispersion.m` | Basic dispersion calculation | ✅ Both |
| `dispersion_with_matrix_save_opt.m` | Dispersion with matrix saving | ✅ Both |
| `get_design.m`, `get_design2.m` | Design generation | ✅ Both |
| `get_system_matrices.m` | System matrix assembly | ✅ Both |
| `get_system_matrices_VEC.m` | Vectorized matrix assembly | ✅ Both |
| `get_transformation_matrix.m` | Periodic BC transformation | ✅ Both |
| `get_element_stiffness.m`, `get_element_mass.m` | Element matrices | ✅ Both |
| `get_IBZ_wavevectors.m` | Wavevector generation | ✅ Both |
| `apply_p4mm_symmetry.m` | Symmetry operations | ✅ Both |
| `plot_dispersion.m`, `plot_design.m` | Visualization | ✅ Both |

### ⭐ Unique to 2D-dispersion-Han

| Feature | Description |
|---------|-------------|
| `get_system_matrices_VEC_simplified.m` | Further optimized vectorized assembly |
| `eigenvector_dtype` support | Can save eigenvectors as 'single' or 'double' to save memory |
| `init_storage.m` | Batch storage initialization utility |
| `ex_dispersion_batch_save.m` | Batch processing example |

**Key Enhancement:**
- **Memory optimization**: Han's version supports `const.eigenvector_dtype = 'single'` to reduce memory usage by 50% when saving eigenvectors

### ⭐ Unique to 2D-dispersion-Alex

| Feature | Description |
|---------|-------------|
| `dispersion2.m` | **Enhanced dispersion with group velocity and sensitivity analysis** |
| `get_element_stiffness_sensitivity.m` | Element stiffness design sensitivity |
| `get_element_mass_sensitivity.m` | Element mass design sensitivity |
| `get_duddesign.m` | Eigenvector design sensitivity |
| `convert_mat_precision.m` | Convert .mat file precision |
| `slicing_data.m` | Data slicing utilities |
| `linspaceNDim.m` | N-dimensional linspace |
| Many validation scripts | 11 validation/verification scripts |

**Key Enhancement:**
- **Advanced sensitivity analysis**: Alex's version includes `dispersion2()` which computes:
  - Group velocities (`cg`)
  - Frequency design sensitivities (`dfrddesign`)
  - Group velocity design sensitivities (`dcgddesign`)

---

## 4. Detailed Function Differences

### 4.1 `dispersion.m` Comparison

**2D-dispersion-Han:**
```matlab
function [wv,fr,ev,mesh] = dispersion(const,wavevectors)
    % Supports isUseSecondImprovement flag
    if const.isUseSecondImprovement
        [K,M] = get_system_matrices_VEC_simplified(const);
    elseif const.isUseImprovement
        [K,M] = get_system_matrices_VEC(const);
    else
        [K,M] = get_system_matrices(const);
    end
    % Returns mesh if isSaveMesh is true
```

**2D-dispersion-Alex:**
```matlab
function [wv,fr,ev] = dispersion(const,wavevectors)
    % Only supports isUseImprovement flag
    if const.isUseImprovement
        [K,M] = get_system_matrices_VEC(const);
    else
        [K,M] = get_system_matrices(const);
    end
    % Does not return mesh
```

**Differences:**
- ✅ Han: Supports `isUseSecondImprovement` → `get_system_matrices_VEC_simplified`
- ✅ Han: Returns `mesh` output
- ❌ Alex: Simpler, no second improvement option

### 4.2 `dispersion_with_matrix_save_opt.m` Comparison

**2D-dispersion-Han:**
```matlab
if const.isSaveEigenvectors
    ev = zeros(N_dof,size(wavevectors,2),const.N_eig,const.eigenvector_dtype);
    % Supports eigenvector_dtype ('single' or 'double')
```

**2D-dispersion-Alex:**
```matlab
if const.isSaveEigenvectors
    ev = zeros(N_dof,size(wavevectors,2),const.N_eig);
    % Always uses default (double) precision
```

**Differences:**
- ✅ Han: Memory optimization via `eigenvector_dtype`
- ❌ Alex: Always uses double precision (more memory)

### 4.3 `dispersion2.m` - Only in Alex's Library

**Unique Advanced Feature:**
```matlab
function [wv,fr,ev,cg,dfrddesign,dcgddesign] = dispersion2(const,wavevectors)
    % Computes:
    % - cg: Group velocities (N_wv x 2 x N_eig)
    % - dfrddesign: Frequency design sensitivities
    % - dcgddesign: Group velocity design sensitivities
```

**Capabilities:**
- ✅ Group velocity calculation
- ✅ Frequency sensitivity to design parameters
- ✅ Group velocity sensitivity to design parameters
- ✅ Requires `get_element_stiffness_sensitivity.m` and `get_element_mass_sensitivity.m`

**Not available in Han's library.**

---

## 5. Script and Utility Comparison

### 2D-dispersion-Han Scripts

| Script | Purpose |
|--------|---------|
| `generate_dispersion_dataset_Han.m` | Main dataset generation |
| `ex_dispersion_batch_save.m` | Batch processing example |
| `init_storage.m` | Storage initialization |

**Characteristics:**
- Focused, production-oriented
- Clean, well-organized
- Supports `eigenvector_dtype` for memory efficiency

### 2D-dispersion-Alex Scripts

| Script | Purpose |
|--------|---------|
| `generate_dispersion_dataset_Han.m` | Main dataset generation |
| `dispersion_dataset_script.m` | Dataset generation script |
| `dispersion2_dataset_script.m` | **Dataset with sensitivity analysis** |
| `dispersion2_script.m` | **Group velocity example** |
| `dispersion_contour_vs_full_IBZ_script.m` | IBZ comparison |
| `dispersion_dataset_visualizer.m` | Visualization tool |
| `dispersion_surface_script.m` | 3D surface plotting |
| `pixelated_design_script.m` | Design generation examples |

**Characteristics:**
- More extensive examples
- Includes sensitivity analysis examples
- More experimental/development scripts
- Legacy code preserved

---

## 6. Advanced Features Comparison

### Sensitivity Analysis

| Feature | 2D-dispersion-Han | 2D-dispersion-Alex |
|--------|-------------------|-------------------|
| **Group Velocity** | ❌ Not available | ✅ `dispersion2()` computes `cg` |
| **Frequency Sensitivity** | ❌ Not available | ✅ `dispersion2()` computes `dfrddesign` |
| **Group Velocity Sensitivity** | ❌ Not available | ✅ `dispersion2()` computes `dcgddesign` |
| **Element Sensitivities** | ❌ Not available | ✅ `get_element_stiffness_sensitivity.m`<br>✅ `get_element_mass_sensitivity.m` |
| **Eigenvector Sensitivity** | ❌ Not available | ✅ `get_duddesign.m` |

**Winner:** ⭐ **2D-dispersion-Alex** for sensitivity analysis capabilities

### Memory Optimization

| Feature | 2D-dispersion-Han | 2D-dispersion-Alex |
|--------|-------------------|-------------------|
| **Eigenvector Precision** | ✅ `eigenvector_dtype = 'single'` or `'double'` | ❌ Always `double` |
| **Memory Savings** | ✅ 50% reduction with `'single'` | ❌ No option |
| **Matrix Optimization** | ✅ `get_system_matrices_VEC_simplified` | ❌ Not available |

**Winner:** ⭐ **2D-dispersion-Han** for memory optimization

### Performance Optimizations

| Feature | 2D-dispersion-Han | 2D-dispersion-Alex |
|--------|-------------------|-------------------|
| **VEC (Vectorized)** | ✅ `get_system_matrices_VEC` | ✅ `get_system_matrices_VEC` |
| **VEC Simplified** | ✅ `get_system_matrices_VEC_simplified` | ❌ Not available |
| **Parallel Processing** | ✅ `parfor` support | ✅ `parfor` support |

**Winner:** ⭐ **2D-dispersion-Han** for additional optimization level

---

## 7. Code Quality and Organization

### 2D-dispersion-Han

**Strengths:**
- ✅ Clean, focused codebase
- ✅ Production-ready
- ✅ Memory-efficient options
- ✅ Additional optimization level (`VEC_simplified`)
- ✅ Well-organized, minimal legacy code

**Limitations:**
- ❌ No sensitivity analysis
- ❌ No group velocity calculations
- ❌ Fewer example scripts

### 2D-dispersion-Alex

**Strengths:**
- ✅ Advanced sensitivity analysis
- ✅ Group velocity calculations
- ✅ Extensive validation scripts (11 files)
- ✅ More example scripts
- ✅ Utility functions (`convert_mat_precision.m`, `slicing_data.m`)

**Limitations:**
- ⚠️ Retired status (may not be maintained)
- ⚠️ More experimental code
- ⚠️ Legacy code included
- ❌ No memory optimization options
- ❌ No `VEC_simplified` optimization

---

## 8. Usage Examples

### Basic Dispersion (Both Libraries)

**2D-dispersion-Han:**
```matlab
const.isUseSecondImprovement = false; % or true for optimization
const.eigenvector_dtype = 'single'; % Memory optimization
[wv, fr, ev, mesh] = dispersion(const, wavevectors);
```

**2D-dispersion-Alex:**
```matlab
[wv, fr, ev] = dispersion(const, wavevectors);
```

### Advanced Features (Alex Only)

**Group Velocity and Sensitivity:**
```matlab
const.isComputeGroupVelocity = true;
const.isComputeFrequencyDesignSensitivity = true;
const.isComputeGroupVelocityDesignSensitivity = true;

[wv, fr, ev, cg, dfrddesign, dcgddesign] = dispersion2(const, wavevectors);
```

**Not available in Han's library.**

---

## 9. Dataset Generation Comparison

### 2D-dispersion-Han: `generate_dispersion_dataset_Han.m`

**Features:**
- ✅ Supports `eigenvector_dtype = 'single'` or `'double'`
- ✅ Uses `get_system_matrices_VEC_simplified` option
- ✅ Memory-efficient for large datasets
- ✅ Default: `binarize = true`
- ✅ Default: `N_struct = 10`
- ✅ Kernel: `'periodic - not squared'`

**Key Line:**
```matlab
eigenvector_dtype = 'double'; % Can be 'single' for 50% memory savings
const.eigenvector_dtype = eigenvector_dtype;
ev = zeros(..., const.eigenvector_dtype);
```

### 2D-dispersion-Alex: `generate_dispersion_dataset_Han.m`

**Features:**
- ❌ Always uses `double` precision
- ❌ No `VEC_simplified` option
- ✅ Default: `binarize = false` (continuous designs)
- ✅ Default: `N_struct = 5`
- ✅ Kernel: `'periodic'` (different from Han)

**Key Differences:**
- Different default material parameters (`E_min = 200e6` vs `2e6`)
- Different kernel options
- No memory optimization

---

## 10. Summary Table

| Feature | 2D-dispersion-Han | 2D-dispersion-Alex | Winner |
|---------|-------------------|-------------------|--------|
| **Status** | ✅ Active | ⚠️ Retired | **Han** |
| **Core Dispersion** | ✅ Complete | ✅ Complete | **Tie** |
| **Memory Optimization** | ✅ `eigenvector_dtype` | ❌ None | **Han** |
| **Performance** | ✅ `VEC_simplified` | ❌ Not available | **Han** |
| **Group Velocity** | ❌ Not available | ✅ `dispersion2()` | **Alex** |
| **Sensitivity Analysis** | ❌ Not available | ✅ Complete | **Alex** |
| **Example Scripts** | ⚠️ Few | ✅ Many | **Alex** |
| **Validation Tools** | ❌ None | ✅ 11 scripts | **Alex** |
| **Code Organization** | ✅ Clean | ⚠️ Legacy code | **Han** |
| **Production Ready** | ✅ Yes | ⚠️ Retired | **Han** |

---

## 11. Recommendations

### Use 2D-dispersion-Han if:
- ✅ You need **active, maintained code**
- ✅ You're generating **large datasets** (memory optimization)
- ✅ You need **maximum performance** (`VEC_simplified`)
- ✅ You want **clean, production code**
- ✅ You don't need sensitivity analysis

### Use 2D-dispersion-Alex if:
- ✅ You need **group velocity calculations**
- ✅ You need **design sensitivity analysis**
- ✅ You want **extensive examples and validation scripts**
- ✅ You're doing **research on sensitivities**
- ⚠️ You're okay with **retired/unmaintained code**

### Best Approach:
1. **Use Han's library** as the primary codebase (active, optimized)
2. **Reference Alex's library** for:
   - Sensitivity analysis implementation (`dispersion2.m`)
   - Validation approaches (legacy_validation_files)
   - Additional examples
3. **Consider porting** Alex's `dispersion2()` features to Han's library if needed

---

## 12. Key Differences Summary

### What Han's Library Has That Alex's Doesn't:
1. ✅ `get_system_matrices_VEC_simplified.m` - Additional optimization
2. ✅ `eigenvector_dtype` support - Memory optimization
3. ✅ Active maintenance status
4. ✅ Cleaner codebase

### What Alex's Library Has That Han's Doesn't:
1. ✅ `dispersion2.m` - Group velocity and sensitivity analysis
2. ✅ `get_element_stiffness_sensitivity.m` - Element sensitivity
3. ✅ `get_element_mass_sensitivity.m` - Element sensitivity
4. ✅ `get_duddesign.m` - Eigenvector sensitivity
5. ✅ Extensive validation scripts (11 files)
6. ✅ More example scripts
7. ✅ Utility functions (`convert_mat_precision.m`, `slicing_data.m`)

---

## 13. Migration Notes

If migrating from Alex's to Han's library:

**Compatible:**
- ✅ All core `dispersion()` calls work the same
- ✅ `dispersion_with_matrix_save_opt()` is compatible
- ✅ All design generation functions
- ✅ All plotting functions

**Not Compatible:**
- ❌ `dispersion2()` - Not available in Han's library
- ❌ Sensitivity analysis functions - Not available
- ⚠️ `eigenvector_dtype` - New feature in Han's (optional)

**Action Items:**
1. Replace `dispersion2()` calls with `dispersion()` (lose sensitivity features)
2. Add `const.eigenvector_dtype = 'single'` for memory savings
3. Consider `const.isUseSecondImprovement = true` for performance

---

## 14. Conclusion

Both libraries are **excellent MATLAB implementations** of 2D dispersion analysis:

- **2D-dispersion-Han** is the **active, optimized version** with memory efficiency and performance improvements
- **2D-dispersion-Alex** is the **feature-rich version** with advanced sensitivity analysis capabilities

**Recommendation:** Use **Han's library** as the primary codebase for production work, and reference **Alex's library** for sensitivity analysis features and validation approaches.

---

**Document Version:** 1.0  
**Last Updated:** Based on codebase analysis  
**Comparison Date:** Current
