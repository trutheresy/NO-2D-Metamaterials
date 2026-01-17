# Float64 Migration Summary

## Overview
All K, M, and T matrix computation functions have been updated to use float64 precision to match MATLAB's double precision. Duplicate implementations have been removed.

## Changes Made

### 1. Updated Functions to Float64

#### `system_matrices_vec.py`
- ✅ `get_system_matrices_VEC()` - Already updated to float64 (verified against MATLAB)
- ✅ `get_system_matrices_VEC_simplified()` - Already updated to float64

#### `system_matrices.py`
- ✅ `get_system_matrices()` - Updated to float64
- ✅ `get_transformation_matrix()` - Updated to float64/complex128

### 2. Deleted Duplicate Files
- ❌ `get_transformation_matrix.py` - Deleted (duplicate of `system_matrices.py::get_transformation_matrix`)
- ❌ `get_system_matrices.py` - Deleted (functionality merged into `system_matrices.py`)

### 3. Updated All Scripts
All scripts now import from the verified float64 versions:
- `from system_matrices_vec import get_system_matrices_VEC`
- `from system_matrices import get_transformation_matrix`
- `from system_matrices import get_system_matrices` (for sensitivity computation)

## Unique Functionality Preserved

### ⚠️ Important: Sensitivity Computation
The function `get_system_matrices()` in `system_matrices.py` has **unique functionality** that is NOT available in `get_system_matrices_VEC()`:

**`return_sensitivities=True` parameter:**
- Returns design sensitivities: `dKddesign` and `dMddesign`
- Used in `dispersion.py` for optimization/gradient computation
- **This functionality is preserved** and updated to float64

**Usage:**
```python
K, M, dKddesign, dMddesign = get_system_matrices(const, return_sensitivities=True)
```

### ⚠️ Important: Transformation Matrix Derivatives
The function `get_transformation_matrix()` in `system_matrices.py` has:

**`return_derivatives=True` parameter:**
- Returns derivatives with respect to wavevector: `dTdwavevector`
- Used in `dispersion.py` for optimization
- **This functionality is preserved** and updated to complex128

**Usage:**
```python
T, dTdwavevector = get_transformation_matrix(wavevector, const, return_derivatives=True)
```

## Recommended Usage

### For Reconstruction (Primary Use Case)
```python
from system_matrices_vec import get_system_matrices_VEC
from system_matrices import get_transformation_matrix

# Use vectorized version (faster, verified float64)
K, M = get_system_matrices_VEC(const)
T = get_transformation_matrix(wavevector, const)
```

### For Optimization (Requires Sensitivities)
```python
from system_matrices import get_system_matrices, get_transformation_matrix

# Use non-vectorized version for sensitivities
K, M, dKddesign, dMddesign = get_system_matrices(const, return_sensitivities=True)
T, dTdwavevector = get_transformation_matrix(wavevector, const, return_derivatives=True)
```

## Verification Status

✅ **Verified against MATLAB:**
- `get_system_matrices_VEC()` - Frobenius relative error: ~10⁻¹⁵% (machine precision)
- `get_transformation_matrix()` - Perfect match (same implementation)

⚠️ **Not yet verified (but updated to float64):**
- `get_system_matrices()` with `return_sensitivities=True` - Updated to float64 but not verified against MATLAB

## Files Modified

### Core Functions
- `2d-dispersion-py/system_matrices_vec.py` - Already float64
- `2d-dispersion-py/system_matrices.py` - Updated to float64

### Scripts Updated
- `2d-dispersion-py/tests/test_system_matrices.py`
- `2d-dispersion-py/tests/test_plotting.py`
- `2d-dispersion-py/tests/test_dispersion.py`
- `2d-dispersion-py/check_phase_alignment.py`
- `2d-dispersion-py/check_eigenvector_components.py`
- `2d-dispersion-py/check_imag_u.py`
- `2d-dispersion-py/check_matrix_properties.py`
- `2d-dispersion-py/investigate_eigenvector_difference.py`
- `2d-dispersion-py/investigate_system_matrices.py`
- `2d-dispersion-py/test_eigs_alternatives.py`

### Files Deleted
- `2d-dispersion-py/get_transformation_matrix.py`
- `2d-dispersion-py/get_system_matrices.py`

## Summary

✅ All K, M, T matrix computations now use float64 precision
✅ Duplicate implementations removed
✅ Unique functionality (sensitivities, derivatives) preserved
✅ All scripts updated to use verified versions
⚠️ Sensitivity computation updated to float64 but not yet verified against MATLAB

