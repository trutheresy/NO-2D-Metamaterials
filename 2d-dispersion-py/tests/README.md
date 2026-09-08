# Test Suite for 2d-dispersion-py

This directory contains comprehensive unit tests to verify functional equivalence between Python and MATLAB implementations.

## Test Structure

- `test_elements.py` - Element-level function tests
- `test_system_matrices.py` - System matrix assembly tests
- `test_wavevectors.py` - Wavevector generation tests
- `test_utils.py` - Utility function tests
- `test_design.py` - Design generation and conversion tests
- `test_kernels.py` - Kernel function tests
- `test_dispersion.py` - Dispersion calculation tests
- `test_plotting.py` - Plotting function tests (requires visual verification)
- `conftest.py` - Shared pytest fixtures
- `run_tests.py` - Test runner script

## Running Tests

### Run All Tests
```bash
cd 2d-dispersion-py
python tests/run_tests.py
```

### Run Specific Test File
```bash
pytest tests/test_elements.py -v
```

### Run Plotting Tests (Generates Visual Outputs)
```bash
python tests/test_plotting.py
```

Plots will be saved to `test_plots/` directory for visual comparison with MATLAB outputs.

## Test Coverage

See `TEST_EQUIVALENCE.md` in the parent directory for detailed test status and results.

## Visual Verification

Plotting tests generate PNG files that must be visually compared with MATLAB outputs:
- `plot_design_*.png` - Design visualizations
- `plot_dispersion_*.png` - Dispersion curves
- `plot_mode_*.png` - Mode shapes
- `plot_eigenvector_*.png` - Eigenvector components
- `plot_mesh_*.png` - Mesh visualizations
- `plot_wavevectors_*.png` - Wavevector distributions



