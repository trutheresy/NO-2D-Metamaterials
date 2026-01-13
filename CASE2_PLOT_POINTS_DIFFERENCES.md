# Case 2 (With Eigenfrequencies) Plot Points Comparison

## Summary

Comparison between Python and MATLAB saved plot points for Case 2 (with eigenfrequencies).

## Files Compared

- **Python**: `dispersion_plots_20260112_015717_mat/plot_points.npz` (struct_0)
- **MATLAB**: `2D-dispersion-han/plots/out_binarized_1_mat/plot_points.mat` (struct_1)

## Key Differences Found

### 1. Number of Contour Points

- **Python**: 28 points
- **MATLAB**: 55 points
- **Difference**: Python generates ~51% fewer points

**Root Cause**: 
- Both use `N_k = 10` (points per segment)
- For p4mm symmetry (3 segments: Gamma→X→M→Gamma):
  - Python: 10 + 9 + 9 = 28 points (removes first point of each new segment)
  - MATLAB: Should be 10 + 9 + 9 = 28 points based on code, but generates 55 points
  - **Investigation needed**: MATLAB's `get_IBZ_contour_wavevectors` may have different behavior than expected

### 2. Wavevectors Contour

- **Python shape**: (28, 2)
- **MATLAB shape**: (55, 2)
- **First 28 points**: ✓ Match (within tolerance)

**Conclusion**: The first 28 wavevectors match, but MATLAB has additional points.

### 3. Frequencies Contour

- **Python shape**: (28, 6)
- **MATLAB shape**: (55, 6)
- **First 28 points**: ✗ Differ (contains NaN values)

**Issue**: Python's scattered interpolation produces NaN values, likely because:
- Some contour points fall outside the convex hull of the input wavevectors
- The interpolation domain doesn't cover all contour points

### 4. Contour Parameter

- **Python shape**: (28,)
- **MATLAB shape**: (55, 1) (flattened to (55,) after transpose fix)
- **Difference**: Different number of points (28 vs 55)

### 5. Use Interpolation Flag

- **Python**: `True` ✓
- **MATLAB**: `True` ✓
- **Status**: Match

## Detailed Analysis

### Wavevector Generation Discrepancy

The discrepancy in the number of points (28 vs 55) suggests that MATLAB's `get_IBZ_contour_wavevectors` may:
1. Use a different algorithm than Python's implementation
2. Have additional logic not captured in the Python port
3. Generate points differently for the p4mm symmetry case

**Next Steps**:
1. Debug MATLAB's `get_IBZ_contour_wavevectors` to understand why it generates 55 points
2. Check if MATLAB uses a different N_k value or has additional segments
3. Verify the contour generation logic matches between implementations

### Frequency Interpolation Issues

Python's scattered interpolation (`griddata`) produces NaN values because:
- The contour points extend beyond the input wavevector domain
- `griddata` with `method='linear'` returns NaN for points outside the convex hull
- MATLAB's `scatteredInterpolant` with `extrap_method='linear'` may handle extrapolation differently

**Solution Options**:
1. Use `fill_value` parameter in `griddata` to extrapolate instead of returning NaN
2. Filter out NaN values before saving (current approach)
3. Use a different interpolation method that handles extrapolation better
4. Match MATLAB's extrapolation behavior exactly

## Recommendations

1. **Fix contour point generation**: Investigate why MATLAB generates 55 points vs Python's 28
2. **Fix frequency interpolation**: Handle extrapolation to match MATLAB's behavior
3. **Re-run comparison**: After fixes, regenerate both Python and MATLAB plot points and compare again

## Current Status

- ✅ Both scripts save plot points successfully
- ✅ Interpolation flag matches
- ⚠️ Different number of contour points (28 vs 55)
- ⚠️ Frequency values differ (NaN in Python, valid in MATLAB)
- ⚠️ Wavevectors match for overlapping points, but MATLAB has more points

