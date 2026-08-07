# Data Generation

How metamaterial dispersion datasets are generated, from the batch driver to the
tensor bundles consumed by training.

**Main entry point:** [`run_generate_dispersion_batched.py`](run_generate_dispersion_batched.py).

**Related:** [ML_TRAINING.md](ML_TRAINING.md) · [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md) · [RULES.md](RULES.md)

---

## 1) Generation chain

1. `run_generate_dispersion_batched.py` — batch orchestrator (manifest, logs, inline encodings).
2. `generate_dispersion_dataset_Han_Alex.py` — per-batch generator subprocess → raw `*_pt` bundle.
3. `2d-dispersion-py/` — FEM assembly + eigensolve (called by the generator).
4. Inline encoders in the driver (`write_eigenfrequency_uniform_full` / `_fft_full`) →
   `eigenfrequency_uniform_full.pt`, `eigenfrequency_fft_full.pt`
   (bulk backfill: `encode_eigenfrequency_uniform_full.py`, `encode_eigenfrequency_fft_full.py`).
5. `build_inputs_outputs_from_reduced_indices.py` → `inputs.pt` + `outputs_w_*.pt`.
6. `rename_and_downselect_indices.py` → `indices_full.pt` + downselected `reduced_indices.pt`.
7. `plot_dataset_histograms.py` → `hist_*.png`.

| Stage | Script(s) | Produces |
|-------|-----------|----------|
| Orchestration | `run_generate_dispersion_batched.py` | batch loop, manifest, logs, inline encodings |
| Raw generation | `generate_dispersion_dataset_Han_Alex.py` | raw `*_pt` bundle per batch |
| FEM/solver core | `2d-dispersion-py/` | matrices, eigenpairs |
| Eigenfrequency encoding | driver inline / `encode_eigenfrequency_*.py` | `eigenfrequency_{uniform,fft}_full.pt` |
| Training-tensor assembly | `build_inputs_outputs_from_reduced_indices.py` | `inputs.pt`, `outputs_w_*.pt` |
| Index downselection | `rename_and_downselect_indices.py` | `indices_full.pt`, `reduced_indices.pt` |
| Histograms | `plot_dataset_histograms.py` | `hist_*.png` |

---

## 2) Main script: `run_generate_dispersion_batched.py`

Launches the heavy generator as subprocesses with deterministic seed offsets, then
post-processes each batch into encoded eigenfrequency tensors.

### Key CLI arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `--total-samples` | 24000 | Total structures (must be divisible by `--batch-size`). |
| `--batch-size` | 1000 | Structures per batch. |
| `--start-seed-offset` | 0 | Seed offset for the first training batch. |
| `--run-validation` | off | Validation batch after all training batches succeed. |
| `--validation-size` | 1000 | Validation structures. |
| `--validation-seed-offset` | 24000 | Disjoint seed offset for validation. |
| `--binarize` | off | Binary (0/1) designs instead of continuous. |
| `--parallel-workers` | 16 | Workers forwarded to the generator. |
| `--skip-uniform-encoding` / `--skip-fft-encoding` | off | Disable the respective inline encoder. |
| `--uniform-patch-size`, `--fft-wavelet-size` | 32 | Patch side length for encodings. |

### Flow

1. Validate `total_samples % batch_size == 0`; create `OUTPUT/batched_generation_<timestamp>/` and an in-memory `manifest`.
2. For each batch `i`: `seed_offset = start_seed_offset + i * batch_size`.
3. Launch `generate_dispersion_dataset_Han_Alex.py` as a subprocess (`--skip-demo`, optional `--binarize`), logging to `logs/train_batch_<i>.log`; parse stdout for `SUCCESS: PyTorch dataset bundle saved to: ...`.
4. On success, run inline encoders on that bundle's `eigenvalue_data_full.pt`:
   - `write_eigenfrequency_uniform_full` → `eigenfrequency_uniform_full.pt` (+ histogram).
   - `write_eigenfrequency_fft_full` → `eigenfrequency_fft_full.pt` (+ histogram + decode spot-check).
5. Append per-batch status to the manifest; **stop at the first non-zero exit code**.
6. Optionally run the validation batch (only if all training batches succeeded).
7. Write `manifest.json` and a summary.

### Inline encoders

- `write_eigenfrequency_uniform_full` clamps non-positive eigenvalues to float16 `1e-6`, then `NO_utilities.encode_eigenfrequency_uniform_torch` → uniform 32×32 patches of `ln(s)/100`.
- `write_eigenfrequency_fft_full` encodes **unique** eigenvalues via `NO_utilities.embed_eigenfrequency_wavelet`, scatters back to full shape, and spot-checks with `extract_eigenfrequency_from_wavelet`.

---

## 3) Per-batch generator: `generate_dispersion_dataset_Han_Alex.py`

Generates `--n-struct` designs, solves dispersion, and writes the raw `*_pt` bundle.

- Prepends `2d-dispersion-py/` to `sys.path` and imports FEM/design utilities.
- Fixed contract: `N_ele=1`, `N_pix=32`, `N_wv=[25,13]` (→ **325** wavevectors), `N_eig=6` bands, vectorized assembly (`isUseImprovement=True`), `isSaveEigenvectors=True`.
- Per structure: design synthesis (`get_design2` → `get_prop` → `kernel_prop`, `p4mm` symmetry) → material mapping (`apply_steel_polymer_paradigm`, optional `--binarize`) → `dispersion_with_matrix_save_opt` (assemble `K`,`M`; per-wavevector reduced eigensolve; `f = sqrt(max(real(λ),0))/(2π)`).
- A transformation-matrix cache may be written/reused as `precomputed_T_matrices.pkl` for a fixed wavevector grid (regenerate if missing).
- Determinism: `design_number = struct_idx + rng_seed_offset`.

### Core solver modules (`2d-dispersion-py/`)

`dispersion_with_matrix_save_opt.py`, `system_matrices_vec.py`, `system_matrices.py`,
`elements_vec.py`, `design_parameters.py`, `get_design2.py`, `get_prop.py`, `kernels.py`,
`symmetry.py`, `wavevectors.py`, `design_conversion.py`, `utils.py`, plus `NO_utilities.py`
for wavelet embeddings when building the `*_pt` bundle.

---

## 4) Outputs and shapes

Let `N_struct` = designs, `N_wv = 325`, `N_band = 6`, `N_pix = 32`.
Full row count `n = N_struct × N_wv × N_band` before downselection.

### 4.1 Raw `*_pt` bundle (generator)

Location: `OUTPUT/output_<timestamp>/<continuous|binarized>_<timestamp>_pt/`

| File | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `geometries_full.pt` | `(N_struct, 32, 32)` | float16 | Geometry image (input ch0). |
| `waveforms_full.pt` | `(N_wv, 32, 32)` | float16 | Wavevector wavelet embedding (input ch1). |
| `band_fft_full.pt` | `(N_band, 32, 32)` | float16 | Band wavelet embedding (input ch2). |
| `wavevectors_full.pt` | `(N_struct, N_wv, 2)` | float16 | Raw `(kx, ky)`. |
| `eigenvalue_data_full.pt` | `(N_struct, N_wv, N_band)` | float16 | Eigenfrequencies. |
| `displacements_dataset.pt` | 4 × `(n, 32, 32)` | float16 | x_real, x_imag, y_real, y_imag. |
| `reduced_indices.pt` | list of `(design, wv, band)`, length `n` | int | Sample map. |
| `design_params_full.pt` | `(N_struct, 1)` | — | Design handles. |

### 4.2 Encoded eigenfrequency tensors

| File | Shape | Dtype | Producer |
|------|-------|-------|----------|
| `eigenfrequency_uniform_full.pt` | `(N_struct, N_wv, N_band, 32, 32)` | float16 | uniform encoder |
| `eigenfrequency_fft_full.pt` | `(N_struct, N_wv, N_band, 32, 32)` | float16 | wavelet / FFT encoder |

### 4.3 Training tensors (`build_inputs_outputs_from_reduced_indices.py`)

| File | Shape | Dtype | Channels |
|------|-------|-------|----------|
| `inputs.pt` | `(n, 3, 32, 32)` | float16 | geo, waveform, band |
| `outputs_w_uniform.pt` / `outputs_w_fft.pt` | `(n, 5, 32, 32)` | float16 | ch0 eigenfreq patch, ch1–4 displacements |

`--eigen-ch0-encoding {uniform,fft}` selects which eigenfrequency tensor fills channel 0.

### 4.4 Index downselection (`rename_and_downselect_indices.py`)

Renames original `reduced_indices.pt` → `indices_full.pt`, then writes a downselected
`reduced_indices.pt` keeping `max(1, N_wv // 5)` wavevectors per `(design, band)`
(seed `20260309`).

### 4.5 Batch-run artifacts (driver)

Location: `OUTPUT/batched_generation_<timestamp>/`

| File | Contents |
|------|----------|
| `manifest.json` | Run params + per-batch status; optional `validation_batch`. |
| `logs/train_batch_<NNN>.log`, `logs/validation_batch.log` | Subprocess stdout/stderr. |

---

## 5) Notes

- Generator writes under `OUTPUT/`; training reads `DATASETS/` (`c_train_*`, `b_train_*`, `c_test`, `b_test`). Organizing/rename between the two is a separate step.
- `train_from_disk.py` does **not** require `outputs_w_*.pt`: it reads `inputs.pt`, displacements from `outputs.pt`, and builds ch0 from `eigenfrequency_*_full.pt` via `reduced_indices`. See [ML_TRAINING.md](ML_TRAINING.md).
- `.mat` export is disabled; the pipeline is `.pt`-first.
- Do not run the batched generator (16 workers by default) concurrently with active GPU training — see [RULES.md](RULES.md).
