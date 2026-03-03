# Master Deviation Audit: Python vs MATLAB Dataset Pipeline

This document consolidates intermediate and saved-output parity checks for the fixed-geometry debug workflow and includes previously measured intermediate-step relative errors.

## 1) Saved data from `generate_dispersion_dataset_Han_Alex.py`

When `is_save_output=True`, the script writes:

- `OUTPUT/output_<timestamp>/<script_name>.py` (script snapshot)
- `OUTPUT/output_<timestamp>/<design_type>_<timestamp>.pkl` (raw Python dataset object)
- `OUTPUT/output_<timestamp>/<design_type>_<timestamp>_pt/displacements_dataset.pt`
- `OUTPUT/output_<timestamp>/<design_type>_<timestamp>_pt/reduced_indices.pt`
- `OUTPUT/output_<timestamp>/<design_type>_<timestamp>_pt/geometries_full.pt`
- `OUTPUT/output_<timestamp>/<design_type>_<timestamp>_pt/waveforms_full.pt`
- `OUTPUT/output_<timestamp>/<design_type>_<timestamp>_pt/wavevectors_full.pt`
- `OUTPUT/output_<timestamp>/<design_type>_<timestamp>_pt/band_fft_full.pt`
- `OUTPUT/output_<timestamp>/<design_type>_<timestamp>_pt/design_params_full.pt`
- `OUTPUT/output_<timestamp>/<design_type>_<timestamp>_pt/eigenvalue_data_full.pt`

Additional feature/demo outputs from the same script:

- `quick_checks/design_patterns_<timestamp>.png`
- `quick_checks/wavevector_patterns_<timestamp>.png`
- `quick_checks/matlab_equivalent_design_<timestamp>.png`

## 2) Call paths in `generate_dispersion_dataset_Han_Alex.py`

Core generation path:

1. `DesignParameters.prepare()` -> `get_design2()`  
2. `convert_design()` -> `apply_steel_rubber_paradigm()`  
3. `dispersion_with_matrix_save_opt(const, const["wavevectors"])` -> returns `wv, fr, ev, K, M, T`  
4. `design_to_explicit()` -> constitutive panes (`E`, `rho`, `nu`)  
5. `_build_pt_dataset_outputs(...)` builds `.pt` payloads  
6. `torch.save(...)` writes `.pt` artifacts

Embedding path inside `_build_pt_dataset_outputs(...)`:

- `NO_utils_multiple.embed_2const_wavelet(...)` -> `waveforms_full.pt`
- `NO_utils_multiple.embed_integer_wavelet(...)` -> `band_fft_full.pt`

Displacement dataset path:

- `EIGENVECTOR_DATA` (complex) -> split x/y real-imag channels -> `TensorDataset` -> `displacements_dataset.pt`

## 3) Intermediate parity (already measured, imported)

Source report: `comparisons/intermediate_comparison_debug.json` (float16 Python), `comparisons/intermediate_comparison_debug_float64.json` (float64 Python)

- `design_input`: 0.0% mean relative error
- `design_mapped`: ~`3.67e-15%`
- `wavevectors_raw`: ~`99.84%` raw-order mismatch
- `frequencies_aligned`:  
  - float16: `0.0150%`  
  - float64: `0.00153%`
- `K_triplet`:  
  - float16: `0.0178%`  
  - float64: `9.24e-15%`
- `M_triplet`:  
  - float16: `0.00862%`  
  - float64: `3.16e-15%`
- `T_diag`: 0.0%

Wavevector remap diagnostics (float64):

- `fallback_fraction = 0.0`
- `identity_fraction = 0.04` (same wavevector set, different index traversal/order)

## 4) Saved-output parity for full `.pt` bundle

Source report: `comparisons/full_saved_outputs_comparison_debug.json` (float16 Python)  
Reference precision run: `comparisons/full_saved_outputs_comparison_debug_float64.json`

### 4.1 Artifact-level comparisons

- `geometries_full` vs MATLAB `designs(:,:,1,:)`:
  - float16: `0.01825%`
- `wavevectors_full_raw` vs MATLAB `WAVEVECTOR_DATA`:
  - float16: `99.84%` (raw index-order mismatch)
- `wavevectors_full_aligned`:
  - float16: `0.03006%`
- `eigenvalue_data_full_aligned` vs MATLAB `EIGENVALUE_DATA`:
  - float16: `0.02404%`
  - float64: `0.02148%`
- `waveforms_full_raw` vs waveform embedding from MATLAB wavevectors:
  - float16: `145.80%` (raw index-order mismatch)
- `waveforms_full_aligned`:
  - float16: `1.30477%`
  - float64: `0.01759%`
- `band_fft_full_from_n_eig`:
  - float16: `0.01729%`
  - float64: `0.01729%`
- `design_params_full_design_number`:
  - `0.0%`
- `reduced_indices`:
  - exact count and content match for expected Python layout (`1950/1950`)

### 4.2 `displacements_dataset.pt` vs MATLAB `EIGENVECTOR_DATA`

Comparison uses:

- wavevector index alignment via remap
- per-(wavevector,band) complex global phase alignment

float16 mean relative errors:

- `displacements_x_real`: `0.7672%`
- `displacements_x_imag`: `0.2522%`
- `displacements_y_real`: `0.7751%`
- `displacements_y_imag`: `0.2422%`

float64 mean relative errors:

- `displacements_x_real`: `0.8373%`
- `displacements_x_imag`: `0.4220%`
- `displacements_y_real`: `0.8710%`
- `displacements_y_imag`: `0.4090%`

## 5) Exact non-precision deviation alerts

Based on intermediate and full-output comparisons:

1. **Wavevector indexing/traversal order mismatch** between MATLAB and Python layouts.
   - Evidence: large raw-order error (`~99.84%`) with low aligned errors (`~0.03%`).
   - Present in both float16 and float64 runs.

No other tested artifact shows a persistent >1% deviation after proper alignment and float64 precision control.

## 6) Debug artifact locations used for this audit

- Intermediate report (float16): `comparisons/intermediate_comparison_debug.json`
- Intermediate report (float64): `comparisons/intermediate_comparison_debug_float64.json`
- Full output report (float16): `comparisons/full_saved_outputs_comparison_debug.json`
- Full output report (float64): `comparisons/full_saved_outputs_comparison_debug_float64.json`
