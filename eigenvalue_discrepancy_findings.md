# Eigenvalue Discrepancy Investigation Results

## Summary

After confirming that eigenvectors and K/M matrices match, we investigated two potential sources of eigenvalue discrepancies:

1. **T Matrix (Transformation Matrix) Differences** 
2. **Sparse Matrix Operations**

## Key Finding: Wavevector Differences

### Investigation 1: T Matrix Differences

**CRITICAL FINDING**: Wavevectors differ between original and reconstructed datasets!

- **Max wavevector difference**: 9.676536e-04
- **Mean wavevector difference**: 4.721063e-04

**Impact on T Matrices**:
- T matrices computed from different wavevectors show differences
- Max absolute difference in T matrices: 8.064817e-05
- Mean absolute difference: 1.193303e-09
- Max relative difference: 8.064817e-05

**Example**:
- Original wavevector: [0.000000, 0.261799]
- Reconstructed wavevector: [0.000000, 0.261719]
- Difference: ~0.00008 in the y-component

**Why This Matters**:
The T matrix is computed from wavevector coordinates using Bloch boundary conditions. Even small differences in wavevector values will cause:
1. Different T matrices
2. Different Kr = T^H * K * T matrices
3. Different Mr = T^H * M * T matrices
4. Different eigenvalue reconstructions

### Investigation 3: Sparse Matrix Operations

(Investigation in progress - need to fix design shape issue)

## Root Cause Analysis

The eigenvalue discrepancies are likely caused by:

1. **Wavevector differences** (PRIMARY SUSPECT)
   - Wavevectors are stored/loaded with different precision
   - Or wavevectors are computed differently in the conversion process
   - This propagates through T matrix computation to eigenvalue reconstruction

2. **T Matrix computation differences** (SECONDARY)
   - Even with same wavevectors, T matrix computation might differ
   - But this is less likely given the wavevector differences found

## Root Cause Identified

**Wavevector Precision Loss in Conversion Chain**:

1. **Original .mat file**: Wavevectors stored as `float64` (double precision)
2. **matlab_to_reduced_pt.py** (line 170): Wavevectors converted to `float16`:
   ```python
   wavevectors_tensor = torch.from_numpy(WAVEVECTOR_DATA).to(torch.float16)
   ```
3. **reduced_pt_to_matlab.py** (line 256): Wavevectors loaded back as `float16`:
   ```python
   wavevectors = torch.load(pt_input_path / "wavevectors_full.pt", ...)
   wavevectors_np = wavevectors.numpy()  # Still float16!
   ```
4. **Final .mat file** (line 605): Cast to `float64`, but precision already lost:
   ```python
   'WAVEVECTOR_DATA': WAVEVECTOR_DATA.astype(np.float64)
   ```

**Impact**:
- `float16` has ~3-4 decimal digits of precision
- Wavevector differences of ~0.00008 (8e-5) are within float16 precision limits
- This precision loss propagates through:
  1. T matrix computation (depends on wavevector values)
  2. Kr = T^H * K * T computation
  3. Mr = T^H * M * T computation  
  4. Eigenvalue reconstruction

## Solution

**Fix**: Store wavevectors in `float32` instead of `float16` in `matlab_to_reduced_pt.py`

`float32` provides ~7 decimal digits of precision, which should be sufficient to preserve wavevector values and prevent T matrix differences.

## Next Steps

1. **Fix wavevector precision**: Change `torch.float16` to `torch.float32` in `matlab_to_reduced_pt.py`
2. **Regenerate .pt files**: Re-run conversion with float32 wavevectors
3. **Regenerate .mat files**: Re-run reconstruction
4. **Verify**: Check that wavevectors match and eigenvalue discrepancies are resolved

