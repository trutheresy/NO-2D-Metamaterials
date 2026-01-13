# Dispersion Plots Locations

## Overview
Dispersion plots have been generated comparing original vs regenerated eigenvalues for all structures.

## Plot Locations

### Main Comparison Plots
**Directory**: `data/dispersion_plots_all_structures/`

**For each structure (0-9), three plot types are generated:**

1. **comparison_struct_N.png** - Side-by-side comparison
   - Left panel: Original eigenvalues
   - Right panel: Regenerated eigenvalues (using Rayleigh quotient)
   - Shows dispersion curves along IBZ contour

2. **overlay_struct_N.png** - Overlay plot
   - Dashed lines: Original eigenvalues
   - Solid lines: Regenerated eigenvalues
   - Same colors for same bands
   - Easy to see differences visually

3. **difference_struct_N.png** - Difference plot
   - Shows: Regenerated - Original
   - Zero line shown as dashed black line
   - Positive values indicate regenerated > original
   - Negative values indicate regenerated < original

**Total**: 10 structures × 3 plots = **30 plots**

### File Naming Convention
- Structure 0: `comparison_struct_0.png`, `overlay_struct_0.png`, `difference_struct_0.png`
- Structure 1: `comparison_struct_1.png`, `overlay_struct_1.png`, `difference_struct_1.png`
- ... and so on for structures 0-9

## Additional Analysis Plots

### Discrepancy Summary
**Location**: `data/discrepancy_analysis/discrepancy_summary.png`

This plot shows:
- Distribution of relative differences (log scale)
- Distribution of absolute differences (log scale)
- Mean relative difference per band
- Scatter plot: Original vs Regenerated eigenvalues

## Key Observations from Plots

### Expected Patterns

1. **Bands 0-1 (Low Frequency Modes)**:
   - Large differences expected, especially at Γ point (wavevector 0)
   - Original values are corrupted (near zero)
   - Regenerated values are correct (~9-10 Hz)

2. **Bands 2-5 (Higher Frequency Modes)**:
   - Small differences expected (<1% relative error)
   - Good agreement between original and regenerated
   - Differences mainly due to numerical precision

3. **Wavevector Dependence**:
   - Largest discrepancies at Γ point (wavevector 0)
   - Better agreement at higher wavevectors
   - Systematic differences decrease with increasing wavevector magnitude

## How to Interpret the Plots

### Comparison Plots
- **Use**: Quick visual check of overall agreement
- **Look for**: Similar curve shapes and magnitudes
- **Red flags**: Completely different curve shapes or orders of magnitude differences

### Overlay Plots
- **Use**: Detailed comparison of specific bands
- **Look for**: Overlap between dashed (original) and solid (regenerated) lines
- **Red flags**: Large gaps between corresponding curves

### Difference Plots
- **Use**: Quantify exact differences
- **Look for**: 
  - Values near zero (good agreement)
  - Systematic offsets (indicates systematic error)
  - Large spikes (indicates corrupted data points)
- **Red flags**: Differences >10 Hz for higher bands, or >1 Hz for well-behaved regions

## Quick Access

To view all plots:
```bash
# Windows
explorer data\dispersion_plots_all_structures

# Or navigate to:
D:\Research\NO-2D-Metamaterials\data\dispersion_plots_all_structures
```

## Summary Statistics

From the discrepancy scan:
- **Bands 0-1**: Mean relative diff ~260x (original data corrupted)
- **Bands 2-5**: Mean relative diff ~0.06-0.15% (excellent agreement)
- **Overall**: Only 2.22% of values have >1 Hz difference (mostly corrupted low bands)

## Next Steps

1. **Visual Inspection**: Review all 30 plots to identify any patterns or anomalies
2. **Focus Areas**: Pay special attention to:
   - Structure 0 (already analyzed in detail)
   - Structures with unusual geometries
   - Regions with large differences in difference plots
3. **Further Analysis**: If needed, investigate specific structures or wavevectors with large discrepancies

