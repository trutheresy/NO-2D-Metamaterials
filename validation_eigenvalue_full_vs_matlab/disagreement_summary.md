# MATLAB vs Python disagreement summary

## What was executed

1. Baseline generation and comparison:
   - `generate_dispersion_dataset_Han_Alex.py`
   - `pt_tensor_to_mat.py`
   - `generate_dispersion_from_prescribed_geometries.m`
   - `compare_eigenvalue_full_pt_vs_matlab.py`
2. Mapping parity A/B:
   - MATLAB `apply_steel_rubber_paradigm.m` output dumps
   - Python `design_conversion.apply_steel_rubber_paradigm` output dumps
3. Wavevector/remap validation:
   - Updated `compare_eigenvalue_full_pt_vs_matlab.py` to exact-key-first remap + diagnostics
4. Solver ablations:
   - `run_solver_ablations.py` over 6 runs:
     - baseline
     - `PARITY_USE_COMPLEX128=1`
     - `PARITY_FORCE_SPARSE_EIGS=1`
     - `PARITY_DISABLE_NEG_CLAMP=1`
     - `PARITY_FR_FLOAT64=1`
     - parity bundle (all flags)

## Artifacts produced

- `validation_eigenvalue_full_vs_matlab/baseline_report.json`
- `validation_eigenvalue_full_vs_matlab/baseline_report_pkl_float64.json`
- `validation_eigenvalue_full_vs_matlab/mapping_variant_report.json`
- `validation_eigenvalue_full_vs_matlab/wv_exact_match_report.json`
- `validation_eigenvalue_full_vs_matlab/solver_ablation_summary.json`
- `validation_eigenvalue_full_vs_matlab/solver_toggle_*.json`

## Key measured results

- Baseline (PT->MATLAB comparison): mean abs = `79.9010`, max abs = `312.4818`
- Baseline (PKL float64->MATLAB): mean abs = `79.9055`, max abs = `312.3500`
  - Conclusion: float16 export is not the dominant source.

### Mapping A/B

From `mapping_variant_report.json`:
- Python current mapping vs MATLAB: max abs `5.005e-4`, mean abs `8.342e-5`
- If Python uses `E_polymer=100e6` (MATLAB value), difference collapses to machine precision (`~1e-16`).
- Interpolation boundary policy is not active for in-range `[0,1]` geometry values.

### Wavevector remap diagnostics

From `wv_exact_match_report.json` (all structures):
- `fallback_count = 324 / 325` (exact key match almost always fails)
- `unique_mapped_indices = 325`, no duplicate assignments
- nearest distance: max `1.368e-3`, mean `7.432e-4`
- Conclusion: ordering/quantization mismatch exists, but remap is still one-to-one and not the main cause of large residual errors.

### Solver ablation impact

From `solver_ablation_summary.json`:
- `use_complex128`: mean abs improves slightly (`79.9010` -> `79.8780`)
- `force_sparse_eigs`: no change
- `disable_neg_clamp`: no change
- `fr_float64`: no change
- parity bundle: `NaN` frequencies appear (expected when disabling clamp with square-root of negative values)

## Where disagreements are (ranked)

1. **Design/material mapping constant mismatch (confirmed):**
   - MATLAB: `2D-dispersion-mat/apply_steel_rubber_paradigm.m` uses `E_polymer = 100e6`
   - Python: `2d-dispersion-py/design_conversion.py` uses `E_polymer = 200e6`
   - Measured effect: mapping-level error `~5e-4` in normalized design pane 1.

2. **Wavevector representation mismatch (confirmed):**
   - Source locations:
     - `2d-dispersion-py/wavevectors.py`
     - `2D-dispersion-mat/get_IBZ_wavevectors.m`
     - plus PT float16 storage in `generate_dispersion_dataset_Han_Alex.py`
   - Evidence: exact-key remap fallback required for `324/325` points, though mapping remains unique.

3. **Residual eigenvalue disagreement concentrated in specific structures (confirmed):**
   - From `baseline_report_pkl_float64.json`:
     - structure 1 mean abs `183.45`, max abs `281.14`
     - structure 3 mean abs `205.33`, max abs `312.35`
     - other structures are much lower (means `~2.9` to `~4.2`)
   - This pattern indicates disagreement is not uniform numeric noise; it is localized to particular designs/bands.

4. **Solver precision/algorithm settings are not dominant under tested toggles (confirmed):**
   - Complex128 only improves mean error by `~0.023`.
   - Sparse/full eig toggle, clamp toggle, and fr dtype toggle did not materially move the baseline metric.

## Final conclusion

The largest remaining MATLAB vs Python disagreement in the current workflow is **not** driven by PT float16 export and **not** primarily by the tested solver toggles. The strongest confirmed disagreements are:
- a mapping constant mismatch in steel-rubber conversion (`E_polymer`),
- wavevector representation mismatch requiring nearest remap,
- and structure-specific residual spectrum disagreement that remains after those controls, concentrated in structures 1 and 3.

The next highest-value follow-up is to enforce exact MATLAB mapping constants in Python and compare per-wavevector reduced matrices/eigenpairs for structures 1 and 3 to localize the first divergence point within the solver pipeline.
