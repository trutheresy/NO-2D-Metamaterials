# Eigenfrequency Reconstruction: Complete Step-by-Step Analysis

## Overview
This document details all steps involved in reconstructing eigenfrequencies from K, M, T matrices and eigenvectors, and explains why discrepancies can occur even when K & M matrices match closely.

## Complete Reconstruction Process

### Step 1: Load Required Data
**Inputs:**
- `K`: Stiffness matrix (sparse, shape: [N_dof, N_dof])
- `M`: Mass matrix (sparse, shape: [N_dof, N_dof])
- `T_data`: List/array of transformation matrices (one per wavevector)
- `EIGENVECTOR_DATA`: Eigenvectors (shape: [N_dof, N_wv, N_bands, N_struct] or similar)
- `wavevectors`: Wavevector coordinates (shape: [N_wv, 2])

**Data Types:**
- K, M: Typically float32 or float64, sparse matrices
- T: Complex sparse matrices (complex128)
- Eigenvectors: Complex arrays (complex128)

---

### Step 2: Loop Over Wavevectors
For each wavevector `wv_idx` from 0 to `N_wv-1`:

#### 2.1: Extract T Matrix
- Get transformation matrix `T` for current wavevector
- T matrix shape: [N_dof, N_reduced] where N_reduced < N_dof
- T is computed from wavevector using Bloch boundary conditions
- **Critical**: T matrix depends on wavevector coordinates

#### 2.2: Matrix Type Conversion
```python
T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.float32))
K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
```
- Converts to sparse format if needed
- **Note**: May cast to float32 (precision loss!)

---

### Step 3: Transform to Reduced Space
**Computation:**
```python
Kr = T_sparse.conj().T @ K_sparse @ T_sparse
Mr = T_sparse.conj().T @ M_sparse @ T_sparse
```

**Mathematical Operation:**
- `Kr = T^H * K * T` (Hermitian transpose)
- `Mr = T^H * M * T`
- Reduces dimension from N_dof × N_dof to N_reduced × N_reduced

**Critical Points:**
1. **Matrix multiplication order matters**
2. **Complex conjugation** (T.conj().T) is essential
3. **Sparse matrix operations** may have numerical differences
4. **Precision**: float32 vs float64 can accumulate errors

---

### Step 4: Loop Over Eigenvalue Bands
For each band `band_idx` from 0 to `N_eig-1`:

#### 4.1: Extract Eigenvector
```python
eigvec = eigenvectors[:, wv_idx, band_idx].astype(np.complex128)
```
- Extract eigenvector for current wavevector and band
- Cast to complex128 for computation
- **Critical**: Eigenvector must match reduced space dimension!

#### 4.2: Matrix-Vector Products
```python
Kr_eigvec = Kr @ eigvec
Mr_eigvec = Mr @ eigvec
```
- Compute `Kr * eigvec` and `Mr * eigvec`
- Results are vectors in reduced space

#### 4.3: Convert to Dense (if sparse)
```python
if sp.issparse(Kr_eigvec):
    Kr_eigvec = Kr_eigvec.toarray().flatten()
if sp.issparse(Mr_eigvec):
    Mr_eigvec = Mr_eigvec.toarray().flatten()
```
- Convert sparse results to dense arrays
- **Note**: `.toarray()` may introduce small numerical differences

#### 4.4: Compute Eigenvalue
```python
eigval = np.linalg.norm(Kr_eigvec) / np.linalg.norm(Mr_eigvec)
```
**Mathematical Formula:**
- `eigval = ||Kr * eigvec|| / ||Mr * eigvec||`
- This is derived from: `Kr * eigvec = eigval * Mr * eigvec`
- Taking norms: `||Kr * eigvec|| = |eigval| * ||Mr * eigvec||`
- Therefore: `eigval = ||Kr * eigvec|| / ||Mr * eigvec||`

**Critical Points:**
1. **Norm computation**: Uses L2 norm (Euclidean)
2. **Division**: Can amplify small differences
3. **Assumes eigenvector is exact**: If eigenvector is approximate, error propagates

#### 4.5: Convert to Frequency
```python
frequencies_recon[wv_idx, band_idx] = np.sqrt(np.real(eigval)) / (2 * np.pi)
```
- Take real part (discard imaginary component)
- Square root to get angular frequency
- Divide by 2π to get frequency in Hz

---

## Why Frequencies Can Differ Even When K & M Match

### 1. **T Matrix Differences**
**Impact: HIGH**
- T matrices are computed from wavevectors
- If wavevectors differ slightly → T matrices differ
- T appears **twice** in transformation: `T^H * K * T`
- Small differences in T can be **amplified** by matrix multiplication

**Example:**
- If T has 0.1% error, then `T^H * K * T` can have ~0.2% error (compounded)
- This error propagates through eigenvalue computation

---

### 2. **Eigenvector Differences**
**Impact: VERY HIGH**
- Eigenvectors are used directly in reconstruction
- If eigenvectors differ → reconstructed eigenvalues differ
- **No error correction**: Formula assumes exact eigenvectors

**Sources of eigenvector differences:**
- Different eigenvalue solvers (MATLAB vs Python)
- Different numerical precision
- Different convergence tolerances
- Rounding errors in storage/loading

---

### 3. **Numerical Precision Accumulation**
**Impact: MEDIUM-HIGH**

**Precision Loss Chain:**
1. K, M stored as float32 → 0.04% relative error
2. T matrix computation → additional error
3. Matrix multiplication `T^H * K * T` → error compounds
4. Matrix-vector products → more operations
5. Norm computation → another operation
6. Division → can amplify errors
7. Square root → final operation

**Error Amplification:**
- Division: `||Kr_eigvec|| / ||Mr_eigvec||`
- If denominator is small → large relative error
- If numerator and denominator both have errors → compounded

---

### 4. **Sparse Matrix Operations**
**Impact: MEDIUM**

**Differences in:**
- Sparse matrix format (CSR vs CSC)
- Sparse matrix multiplication algorithms
- Rounding in sparse operations
- Fill-in during matrix products

**Example:**
- `T^H * K * T` involves two sparse matrix multiplications
- Each multiplication can introduce small numerical differences
- Differences accumulate

---

### 5. **Eigenvector Normalization**
**Impact: MEDIUM**

**Issue:**
- Eigenvectors may be normalized differently
- Formula `||Kr*eigvec|| / ||Mr*eigvec||` assumes specific normalization
- If eigenvectors are scaled differently → different eigenvalues

**Check:**
- Are eigenvectors unit-normalized with respect to M?
- MATLAB: `eigvec' * M * eigvec = 1`?
- Python: Same normalization?

---

### 6. **Complex Number Handling**
**Impact: LOW-MEDIUM**

**Differences in:**
- Complex conjugation: `T.conj().T` vs `T'` in MATLAB
- Handling of imaginary parts
- Taking real part: `np.real(eigval)` vs MATLAB's handling

---

### 7. **Order of Operations**
**Impact: LOW**

**Potential differences:**
- Matrix multiplication associativity (floating point)
- Order of norm computation
- Order of type conversions

---

## Specific Issues in Current Comparison

### Observed: 39% Relative Error in Frequencies

**Possible Causes:**

1. **Eigenvector Mismatch** (Most Likely)
   - Original dataset: Eigenvectors from MATLAB eigenvalue solver
   - Reconstructed dataset: Eigenvectors from Python eigenvalue solver
   - Different solvers → different eigenvectors → different reconstructed eigenvalues

2. **T Matrix Computation Differences**
   - Original: T matrices computed in MATLAB
   - Reconstructed: T matrices computed in Python
   - Even if wavevectors match, T computation may differ

3. **Precision Chain**
   - Original: float64 throughout
   - Reconstructed: May use float32 in some steps
   - Accumulated precision loss

4. **Eigenvector Storage/Conversion**
   - Original: Stored as MATLAB complex arrays
   - Reconstructed: Converted through PyTorch (float16) → back to float32
   - Precision loss in conversion chain

---

## Recommendations for Investigation

### Priority 1: Check Eigenvectors
```python
# Compare eigenvectors between original and reconstructed
eigvec_orig = original['EIGENVECTOR_DATA'][:, 0, 0, 0]
eigvec_recon = reconstructed['EIGENVECTOR_DATA'][:, 0, 0, 0]
diff = np.abs(eigvec_orig - eigvec_recon)
print(f"Max eigenvector difference: {np.max(diff)}")
print(f"Relative difference: {np.max(diff) / np.max(np.abs(eigvec_orig))}")
```

### Priority 2: Check T Matrices
```python
# Compare T matrices for first wavevector
T_orig = original['T_DATA'][0]
T_recon = reconstructed['T_DATA'][0]
diff_T = np.abs(T_orig - T_recon)
print(f"Max T matrix difference: {np.max(diff_T)}")
```

### Priority 3: Check Reduced Matrices
```python
# Compare Kr and Mr for first wavevector
Kr_orig = T_orig.conj().T @ K_orig @ T_orig
Kr_recon = T_recon.conj().T @ K_recon @ T_recon
diff_Kr = np.abs(Kr_orig - Kr_recon)
print(f"Max Kr difference: {np.max(diff_Kr)}")
```

### Priority 4: Check Intermediate Computations
```python
# Compare Kr*eigvec and Mr*eigvec
Kr_eigvec_orig = Kr_orig @ eigvec_orig
Kr_eigvec_recon = Kr_recon @ eigvec_recon
diff_Kr_eigvec = np.abs(Kr_eigvec_orig - Kr_eigvec_recon)
print(f"Max Kr*eigvec difference: {np.max(diff_Kr_eigvec)}")
```

---

## Mathematical Sensitivity Analysis

### Error Propagation Formula

If we have:
- `K_error = ε_K` (relative error in K)
- `M_error = ε_M` (relative error in M)
- `T_error = ε_T` (relative error in T)
- `eigvec_error = ε_v` (relative error in eigenvector)

Then:
- `Kr_error ≈ 2*ε_T + ε_K` (T appears twice)
- `Mr_error ≈ 2*ε_T + ε_M`
- `Kr_eigvec_error ≈ Kr_error + ε_v`
- `Mr_eigvec_error ≈ Mr_error + ε_v`
- `eigval_error ≈ Kr_eigvec_error + Mr_eigvec_error` (division amplifies)
- `frequency_error ≈ 0.5 * eigval_error` (square root reduces)

**Example:**
- If K has 0.04% error, T has 0.1% error, eigenvector has 1% error:
- Kr_error ≈ 0.24%
- Kr_eigvec_error ≈ 1.24%
- eigval_error ≈ 2.48% (amplified by division)
- frequency_error ≈ 1.24%

**But if eigenvector has 10% error:**
- frequency_error ≈ 12.4%

This explains why small K/M differences (0.04%) can lead to large frequency differences (39%) if eigenvectors differ significantly!

---

## Conclusion

The large frequency differences (39%) despite small K/M differences (0.04%) are most likely due to:

1. **Eigenvector differences** between original and reconstructed datasets
2. **T matrix computation differences** (if computed differently)
3. **Error amplification** through the reconstruction formula (division operation)
4. **Precision loss** in the conversion chain (float64 → float16 → float32)

The reconstruction process is **highly sensitive** to eigenvector accuracy because eigenvectors are used directly in the formula without any error correction mechanism.

