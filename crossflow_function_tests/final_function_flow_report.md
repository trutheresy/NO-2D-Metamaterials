# Function-Level Cross-Flow Report

- Output folder: `D:\Research\NO-2D-Metamaterials\crossflow_function_tests`
- Fixture MD5: `9a94ea115aa21b94c6d305dab04ba5d3`
- Constants matched: computational grid and material constants use the same values in MATLAB/Python fixed scripts.

## Function checks

- `get_IBZ_wavevectors`: set-equality with MATLAB is `True` (order differs).
- `design_to_explicit`: E/rho/nu checks are allclose.
- `dispersion_with_matrix_save_opt` (Python): produced finite `wv/fr/ev`, plus `K/M/T`.
- MATLAB fixed flow: produced expected keys and finite principal tensors.

## Comparison summary

- designs raw allclose: `True`
- wavevectors raw allclose: `False`
- eigenvalues aligned allclose: `False`
- eigenvectors aligned allclose: `False`
- eigenvectors phase-aligned max_rel_norm_diff: `1.9999821267514442`
- eigenvectors phase-aligned min_correlation: `0.01639519849637284`

## Artifacts

- `wavevector_parity.json`
- `design_to_explicit_checks.json`
- `python_dispersion_function_checks.json`
- `matlab_dispersion_function_checks.json`
- `quantity_comparison.json`
- `worst_case_examples.json`
- plots in `D:\Research\NO-2D-Metamaterials\crossflow_function_tests\plots`
