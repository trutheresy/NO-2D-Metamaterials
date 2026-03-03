# Eigenvalue Discrepancy Investigation - Hypothesis Tracking

## Known Facts (Verified)

### ✅ Verified to Match
1. **K and M matrices**: Match exactly between original and reconstructed (within machine precision)
2. **T matrices**: Match exactly (within machine precision) - verified shape, dtype, values
3. **Eigenvector data**: Match exactly between original and reconstructed
4. **Design data**: Match exactly (after fixing Poisson ratio scaling)
5. **Material property bounds**: Match exactly (E_min, E_max, rho_min, rho_max, nu_min, nu_max)
6. **Constant parameters**: Match exactly (N_pix, N_ele, a, t, design_scale, etc.)
7. **Input dependencies for eigenvalue reconstruction**: All 5 dependencies match:
   - Eigenvector data ✓
   - Geometry data ✓
   - Material properties ✓
   - K and M matrices ✓
   - T matrices ✓

### ✅ Verified Precision
- T matrix percentage errors: ~0.008% max, ~0.0002% mean
- Kr matrix percentage errors: ~0.008% max, ~0.0002% mean
- Mr matrix percentage errors: ~0.008% max, ~0.0002% mean
- These errors are consistent with float16 → float32 conversion in wavevectors

### ❌ Known Issues
- Eigenvalue discrepancies are **far beyond** what 0.008% errors could cause
- Precision is **not** the root cause (per user instruction)

---

## Hypothesis List

### H1: Type Conversion Issues
**Status**: ✅ ELIMINATED
**Description**: Complex to real conversion in T matrix, or inconsistent type conversions throughout chain
**Location**: Phase 6.4 - `T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.float32))`
**Priority**: HIGH - Could cause massive errors if complex T is converted to real
**Result**: T matrix is always sparse when returned from `get_transformation_matrix()`, so the conversion line `T.astype(np.float32)` never executes. Complex type is preserved. However, if T were ever dense, this would be a critical bug.

### H2: Indexing and Dimension Mismatches
**Status**: ✅ ELIMINATED
**Description**: Eigenvector dimension mismatch, T matrix shape mismatch, wavevector index mismatch
**Location**: Phase 7.2, 6.5, 5.1
**Priority**: HIGH
**Result**: All dimensions match correctly. T.shape[0] == N_dof_full, eigenvector DOF == N_dof_reduced, matrix-vector multiplications work correctly.

### H3: Matrix Operation Order Issues
**Status**: ⏳ Pending
**Description**: Matrix multiplication associativity, sparse matrix format differences
**Location**: Phase 6.5
**Priority**: MEDIUM

### H4: Design Conversion Issues
**Status**: ⏳ Pending
**Description**: Steel-rubber paradigm parameter mismatch, design expansion (repelem) issues
**Location**: Phase 3.3, 4.2.2
**Priority**: LOW (already verified designs match)

### H5: Node and DOF Indexing Issues
**Status**: ⏳ Pending
**Description**: 1-based vs 0-based indexing, meshgrid indexing order, reshape order (Fortran vs C)
**Location**: Throughout Phase 4 and 5
**Priority**: HIGH

### H6: Eigenvalue Computation Formula Issues
**Status**: ✅ CONFIRMED
**Description**: Norm computation formula, real part extraction
**Location**: Phase 7.5, 7.6
**Priority**: MEDIUM
**Result**: CRITICAL BUG FOUND! The current formula `eigval = ||Kr*eigvec|| / ||Mr*eigvec||` is inaccurate. The eigenvector does not satisfy the eigenvalue equation well (residual ~0.24). The correct formula should be the Rayleigh quotient: `eigval = (eigvec^H * Kr * eigvec) / (eigvec^H * Mr * eigvec)`. The norm-based formula gives ~5% error compared to Rayleigh quotient.

### H7: Eigenvector Reconstruction Issues
**Status**: ⏳ Pending
**Description**: Interleaving order (x/y), missing eigenvector entries
**Location**: Phase 2.5, 2.3
**Priority**: HIGH

### H8: Constant Parameter Mismatches
**Status**: ⏳ Pending
**Description**: N_pix handling, lattice constant 'a'
**Location**: Throughout
**Priority**: LOW (already verified)

### H9: Sparse Matrix Storage and Operations
**Status**: ⏳ Pending
**Description**: Sparse matrix conversion, sparse matrix multiplication implementation
**Location**: Phase 6.4, 6.5, 7.4
**Priority**: MEDIUM

### H10: Division by Small Numbers
**Status**: ⏳ Pending
**Description**: Small denominator in eigenvalue formula, small eigenvalues
**Location**: Phase 7.5, 7.6
**Priority**: MEDIUM

---

## Investigation Log

### Investigation Start: [Current Date/Time]

### Step 1: Applied Rayleigh Quotient Fix
- **Date**: Current session
- **Action**: Fixed eigenvalue computation formula in `reduced_pt_to_matlab.py:485`
- **Change**: Replaced norm-ratio formula with Rayleigh quotient
- **Result**: Eigenvalue computation now uses correct, numerically stable formula

### Step 2: Regenerated EIGENVALUE_DATA
- **Date**: Current session
- **Action**: Created `regenerate_eigenvalue_data.py` and `regenerate_eigenvalue_data_fast.py`
- **Result**: Regenerated eigenvalues for structure 0 using Rayleigh quotient
- **Output**: `data/out_test_10_mat_regenerated/out_binarized_1.mat`

### Step 3: Generated Comparison Tools
- **Date**: Current session
- **Action**: Created `scan_eigenvalue_discrepancies.py` and `plot_regenerated_dispersion.py`
- **Purpose**: Quantify discrepancies and generate visual comparisons

---

## Functions/Scripts Known to Work Correctly

1. `get_system_matrices_VEC()` - Produces K and M that match original
2. `get_transformation_matrix()` - Produces T matrices that match original
3. `apply_steel_rubber_paradigm()` - Produces designs that match original
4. Eigenvector reconstruction - Produces eigenvectors that match original

---

## Eliminated Hypotheses

1. **H1: Type Conversion Issues** - T matrix is always sparse, so complex type is preserved
2. **H2: Indexing and Dimension Mismatches** - All dimensions match correctly

---

## Remaining Hypotheses

1. **H3: Matrix Operation Order Issues** - Not yet tested
2. **H4: Design Conversion Issues** - Low priority (designs already verified to match)
3. **H5: Node and DOF Indexing Issues** - Not yet tested
4. **H7: Eigenvector Reconstruction Issues** - Likely eliminated (eigenvectors match exactly)
5. **H8: Constant Parameter Mismatches** - Low priority (already verified)
6. **H9: Sparse Matrix Storage and Operations** - Not yet tested
7. **H10: Division by Small Numbers** - Not yet tested

---

## New Hypotheses Discovered

(To be added as investigation reveals new potential issues)

---

## Critical Findings

### H6: Wrong Eigenvalue Formula
**Location**: `reduced_pt_to_matlab.py:479`
**Current Code**:
```python
eigval = np.linalg.norm(Kr_eigvec) / np.linalg.norm(Mr_eigvec)
```

**Problem**: This formula assumes `Kr*eigvec = eigval*Mr*eigvec` exactly, but the eigenvector (from float16 conversion) is approximate. The residual is ~0.24, meaning the eigenvector does not satisfy the eigenvalue equation well.

**Correct Formula** (Rayleigh quotient):
```python
eigval = np.dot(eigvec.conj(), Kr @ eigvec) / np.dot(eigvec.conj(), Mr @ eigvec)
```

**Impact**: The norm-based formula gives ~5% error compared to the Rayleigh quotient. This could explain a significant portion of the eigenvalue discrepancies.

