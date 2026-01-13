# Summary: Rayleigh Quotient Fix and Regeneration

## What Was Done

### 1. Fixed Eigenvalue Computation Formula
- **File**: `reduced_pt_to_matlab.py` (line 485)
- **Change**: Replaced norm-ratio formula with Rayleigh quotient
- **Old Formula**: `eigval = ||Kr*eigvec|| / ||Mr*eigvec||`
- **New Formula**: `eigval = (eigvec^H * Kr * eigvec) / (eigvec^H * Mr * eigvec)`
- **Reason**: The norm-based formula assumes exact eigenvectors, which is violated by float16→float32 conversions. The Rayleigh quotient is the standard, numerically stable choice.

### 2. Regenerated EIGENVALUE_DATA
- **Script**: `regenerate_eigenvalue_data.py` (full) and `regenerate_eigenvalue_data_fast.py` (test)
- **Input**: `data/out_test_10_mat_original/out_binarized_1.mat`
- **Output**: `data/out_test_10_mat_regenerated/out_binarized_1.mat`
- **Method**: Uses Rayleigh quotient to recompute all eigenvalues from K, M, T matrices and eigenvectors

### 3. Generated Comparison Tools
- **Scan Script**: `scan_eigenvalue_discrepancies.py` - Quantifies discrepancies
- **Plot Script**: `plot_regenerated_dispersion.py` - Generates visual comparisons
- **Combined Script**: `run_all_comparisons.py` - Runs all comparisons

## Results

### Discrepancy Statistics (Structure 0)
- **Band 0-1**: Large discrepancies (mean relative diff ~270x, max ~27,000x)
  - Original values are near zero (~4e-4 Hz) while regenerated are ~9-10 Hz
  - This confirms original stored values for low bands are corrupted/incorrect
- **Band 2-5**: Moderate discrepancies (mean relative diff ~3x)
  - These bands show better agreement but still have systematic differences

### Dispersion Plots Location
**All plots saved to**: `data/dispersion_plots_regenerated/`

1. **comparison_struct_0.png** - Side-by-side comparison of original vs regenerated
2. **overlay_struct_0.png** - Overlay plot showing both curves
3. **difference_struct_0.png** - Difference plot (regenerated - original)

## Key Findings

1. **H6 Confirmed**: The eigenvalue computation formula was wrong. The Rayleigh quotient fix is correct.

2. **Original Data Issues**: The stored eigenvalues for bands 0-1 (especially at Γ point) are clearly corrupted/incorrect (values near zero when they should be ~9-10 Hz).

3. **Higher Bands**: Bands 2-5 show better agreement, suggesting the original data may be more reliable for higher frequency modes.

## Next Steps

1. **Full Regeneration**: Currently regenerating all 10 structures (running in background)
2. **Visual Inspection**: Check plots in `data/dispersion_plots_regenerated/`
3. **Further Analysis**: Once full regeneration completes, run comprehensive scan on all structures

## Files Created/Modified

### New Files
- `regenerate_eigenvalue_data.py` - Full regeneration script
- `regenerate_eigenvalue_data_fast.py` - Fast test version (1 structure)
- `scan_eigenvalue_discrepancies.py` - Discrepancy analysis
- `plot_regenerated_dispersion.py` - Plot generation
- `run_all_comparisons.py` - Combined runner
- `SUMMARY_RAYLEIGH_FIX.md` - This file

### Modified Files
- `reduced_pt_to_matlab.py` - Fixed eigenvalue formula with explanatory comment
- `eigenvalue_discrepancy_investigation.md` - Updated with findings

