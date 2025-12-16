# Testing Guide: Verifying Python-MATLAB Equivalence

This guide explains how to run the comprehensive test suite to verify functional equivalence between Python and MATLAB implementations.

## Quick Start

### 1. Run All Numerical Tests
```bash
cd 2d-dispersion-py
pytest tests/ -v
```

### 2. Generate Plots for Visual Verification
```bash
python tests/run_plotting_tests.py
```

### 3. Review Test Results
Check `TEST_EQUIVALENCE.md` for detailed test status.

## Test Structure

### Numerical Function Tests

These tests verify mathematical correctness and can be run automatically:

1. **test_elements.py** - Element stiffness/mass matrices
2. **test_system_matrices.py** - System matrix assembly
3. **test_wavevectors.py** - Wavevector generation
4. **test_utils.py** - Utility functions
5. **test_design.py** - Design generation/conversion
6. **test_kernels.py** - Kernel functions
7. **test_dispersion.py** - Dispersion calculations

### Visual Verification Tests

These tests generate plots that **must be visually compared** with MATLAB outputs:

- **test_plotting.py** - Generates 8+ plots for visual inspection

## Running Tests

### Run All Tests
```bash
# Run all numerical tests
pytest tests/ -v

# Or use the test runner
python tests/run_tests.py
```

### Run Specific Test Category
```bash
pytest tests/test_elements.py -v
pytest tests/test_system_matrices.py -v
# etc.
```

### Generate Plots for Visual Check
```bash
python tests/run_plotting_tests.py
```

Plots will be saved to `test_plots/` directory.

## Visual Verification Process

### Step 1: Generate Plots
Run the plotting test script to generate all plots:
```bash
python tests/run_plotting_tests.py
```

### Step 2: Compare with MATLAB
For each generated plot, compare with the corresponding MATLAB output:

| Python Plot | MATLAB File | What to Check |
|------------|-------------|---------------|
| `plot_design_homogeneous.png` | `plot_design.m` | 3 subplots showing uniform material properties |
| `plot_design_random.png` | `plot_design.m` | 3 subplots showing binary pattern |
| `plot_dispersion_basic.png` | `plot_dispersion.m` | Line plot with dispersion curves, segment markers |
| `plot_dispersion_contour.png` | `plot_dispersion_contour.m` | Contour plot in k-space |
| `plot_mode_basic.png` | `plot_mode.m` | Quiver plot showing displacement field |
| `plot_eigenvector_components.png` | `plot_eigenvector.m` | 4 subplots: real(u), imag(u), real(v), imag(v) |
| `plot_mesh_basic.png` | `plot_mesh.m` | Mesh visualization with nodes |
| `plot_wavevectors_basic.png` | `plot_wavevectors.m` | Scatter plot with numbered points |
| `visualize_designs_multiple.png` | `visualize_designs.m` | Grid of design visualizations |

### Step 3: Update Test Status
After visual verification, update `TEST_EQUIVALENCE.md`:
- Mark as ✅ **VERIFIED** if plots match
- Mark as ⚠️ **MINOR DIFFERENCES** if close but not identical (document differences)
- Mark as ❌ **DOES NOT MATCH** if significant differences (document issues)

## Test Coverage Summary

### Functions with Automated Tests: 40+
- Element functions: 5 functions
- System matrices: 4 functions
- Wavevectors: 2 functions
- Utilities: 8 functions
- Design: 4 functions
- Kernels: 5 functions
- Dispersion: 2 functions

### Functions Requiring Visual Verification: 8+
- Plotting functions that generate visual outputs

## Expected Test Results

### Numerical Tests
- ✅ Should pass with tolerance checks (rtol=1e-5, atol=1e-6)
- ✅ Matrix properties verified (symmetry, positive definiteness)
- ✅ Output shapes and types validated
- ✅ Edge cases tested

### Visual Tests
- 📊 Plots generated successfully
- 📊 **USER VERIFICATION REQUIRED** for each plot
- 📊 Compare: colors, scales, labels, layout, data representation

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the `2d-dispersion-py` directory:
```bash
cd 2d-dispersion-py
python -m pytest tests/ -v
```

### Missing Dependencies
Install required packages:
```bash
pip install -r requirements.txt
pip install pytest scipy matplotlib numpy
```

### Plot Generation Issues
- Ensure `test_plots/` directory exists (created automatically)
- Check matplotlib backend if plots don't display
- Verify all plotting dependencies are installed

## Next Steps After Testing

1. **Review all test results** - Check for any failures
2. **Verify all plots** - Compare with MATLAB outputs
3. **Update TEST_EQUIVALENCE.md** - Mark verified tests
4. **Fix any issues** - Address failures or mismatches
5. **Document differences** - Note any intentional or acceptable differences

## Test Maintenance

- Add new tests when new functions are added
- Update tests when functions are modified
- Re-run visual verification when plotting code changes
- Keep TEST_EQUIVALENCE.md up to date

