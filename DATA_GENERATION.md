# Data Generation

End-to-end production of metamaterial dispersion datasets used for neural-operator
training. The pipeline is Python-only: designs are synthesized, Bloch FEA is solved
in `2d-dispersion-py/`, and results are stored as float16 PyTorch tensors.

**Entry point:** [`run_generate_dispersion_batched.py`](run_generate_dispersion_batched.py)

**Related:** [ML_TRAINING.md](ML_TRAINING.md) · [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md)

---

## 1) Pipeline overview

| Stage | Script / package | Output |
|-------|------------------|--------|
| Batch orchestration | `run_generate_dispersion_batched.py` | Per-batch runs, logs, manifest |
| Design + FEA solve | `generate_dispersion_dataset.py` → `2d-dispersion-py/` | Raw `*_pt` bundle |
| Eigenfrequency channel encodings | driver inline (or `encode_eigenfrequency_*.py`) | `eigenfrequency_{uniform,fft}_full.pt` |
| Training tensors | `build_inputs_outputs_from_reduced_indices.py` | `inputs.pt`, optional `outputs_w_*.pt` |
| Index downselection | `rename_and_downselect_indices.py` | `indices_full.pt`, downselected `reduced_indices.pt` |
| Optional QA histograms | `DIAGNOSTICS/plot_dataset_histograms.py` (local-only) | `hist_*.png` next to tensors |

Typical layout after organizing runs into the training tree:

- Train shards: `DATASETS/c_train_*`, `DATASETS/b_train_*` (continuous / binary)
- Test shards: `DATASETS/c_test`, `DATASETS/b_test`

Raw generator writes under `OUTPUT/`; copy or rename into `DATASETS/` before training.

---

## 2) Problem contract (fixed by the generator)

| Quantity | Value |
|----------|-------|
| Pixel grid | `32 × 32` |
| Elements per pixel | `1` |
| IBZ wavevector grid | `25 × 13` → **325** wavevectors |
| Eigenbands per wavevector | **6** |
| Symmetry | `p4mm` (eight-fold) |
| Geometry classes | continuous in `[0, 1]`, or binary via `--binarize` |
| Storage dtype | float16 for bulky tensors |

Materials are mapped through `apply_steel_polymer_paradigm` in `2d-dispersion-py`
(normalized design → physical modulus / density / Poisson ratio).

---

## 3) Batch driver: `run_generate_dispersion_batched.py`

Launches `generate_dispersion_dataset.py` as subprocesses with deterministic
seed offsets, then writes eigenfrequency encoding tensors for each successful batch.

### Useful CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--total-samples` | `24000` | Designs to generate (must divide evenly by `--batch-size`) |
| `--batch-size` | `1000` | Designs per subprocess |
| `--start-seed-offset` | `0` | Seed base for the first training batch |
| `--binarize` | off | Threshold designs to `{0,1}` |
| `--parallel-workers` | `16` | Workers forwarded to the generator |
| `--run-validation` | off | Extra held-out batch after all train batches succeed |
| `--validation-size` | `1000` | Designs in the validation batch |
| `--validation-seed-offset` | `24000` | Disjoint seed base for validation |
| `--skip-uniform-encoding` / `--skip-fft-encoding` | off | Skip either eigenfrequency encoding |

### Flow

1. Create `OUTPUT/batched_generation_<timestamp>/` and an in-memory manifest.
2. For batch `i`, set `seed_offset = start_seed_offset + i * batch_size`.
3. Run the generator (`--skip-demo`, optional `--binarize`); log to
   `logs/train_batch_<i>.log`. Parse
   `SUCCESS: PyTorch dataset bundle saved to: ...`.
4. On success, encode `eigenvalue_data_full.pt`:
   - uniform → `eigenfrequency_uniform_full.pt`
   - wavelet/FFT → `eigenfrequency_fft_full.pt` (with decode spot-check)
5. Stop on the first non-zero subprocess exit code.
6. Optionally run the validation batch; write `manifest.json`.

**Note:** Do not run this driver (CPU-heavy, many workers) at the same time as a GPU
training job on a single-GPU machine.

### Encoding details

- **Uniform:** clamp non-positive eigenvalues, then
  `NO_utilities.encode_eigenfrequency_uniform_torch` → constant `32×32` patches
  `ln(s)/100`.
- **Wavelet / FFT:** encode unique eigenvalues with
  `NO_utilities.embed_eigenfrequency_wavelet`, scatter to full
  `(N_struct, N_wv, N_band, 32, 32)`, spot-check with
  `extract_eigenfrequency_from_wavelet`.

Bulk re-encode existing folders with `encode_eigenfrequency_uniform_full.py` /
`encode_eigenfrequency_fft_full.py` if needed.

---

## 4) Per-batch generator: `generate_dispersion_dataset.py`

For each design:

1. Synthesize a correlated unit-cell geometry (`get_design2` / kernel props, `p4mm`).
2. Map to material fields (`convert_design`, `apply_steel_polymer_paradigm`).
3. Assemble `K`, `M` and solve the reduced Bloch eigenproblem per wavevector
   (`dispersion_with_matrix_save_opt`).
4. Store geometries, wavevector/band embeddings, eigenfrequencies, and complex
   displacement fields as real/imag channels.

Determinism: `design_number = struct_idx + rng_seed_offset`.

A transformation-matrix cache `precomputed_T_matrices.pkl` may be reused for a fixed
wavevector grid.

Core FEA modules live under `2d-dispersion-py/` (system matrices, elements, designs,
wavevectors, symmetry, etc.). Wavelet embeddings for wavevector/band channels use
`NO_utilities.py` at repo root.

---

## 5) Tensor inventory

Let `N_struct` = designs, `N_wv = 325`, `N_band = 6`, `N_pix = 32`, and
`n = N_struct × N_wv × N_band` before downselection.

### Raw `*_pt` bundle

Typical path: `OUTPUT/output_<timestamp>/<continuous|binarized>_<timestamp>_pt/`

| File | Shape | Role |
|------|-------|------|
| `geometries_full.pt` | `(N_struct, 32, 32)` | Geometry (model input channel 0) |
| `waveforms_full.pt` | `(N_wv, 32, 32)` | Wavevector wavelet embedding (input ch1) |
| `band_fft_full.pt` | `(N_band, 32, 32)` | Band wavelet embedding (input ch2) |
| `wavevectors_full.pt` | `(N_struct, N_wv, 2)` | Raw `(k_x, k_y)` |
| `eigenvalue_data_full.pt` | `(N_struct, N_wv, N_band)` | Scalar eigenfrequencies |
| `displacements_dataset.pt` | 4 × `(n, 32, 32)` | `u_x,u_y` real/imag |
| `reduced_indices.pt` | length `n` | `(design, wv, band)` sample map |
| `design_params_full.pt` | `(N_struct, …)` | Design metadata |

### Eigenfrequency patches

| File | Shape |
|------|-------|
| `eigenfrequency_uniform_full.pt` | `(N_struct, N_wv, N_band, 32, 32)` |
| `eigenfrequency_fft_full.pt` | `(N_struct, N_wv, N_band, 32, 32)` |

### Training tensors

Built by `build_inputs_outputs_from_reduced_indices.py`:

| File | Shape | Channels |
|------|-------|----------|
| `inputs.pt` | `(n, 3, 32, 32)` | geometry, wavevector embedding, band embedding |
| `outputs.pt` / displacement stacks | `(n, 5, …)` or split | ch0 reserved for eigenfrequency; ch1–4 displacements |
| `outputs_w_uniform.pt` / `outputs_w_fft.pt` | `(n, 5, 32, 32)` | optional stacked targets |

`train_from_disk.py` does **not** require `outputs_w_*.pt`. It reads `inputs.pt`,
displacement channels from `outputs.pt`, and fills output ch0 on the fly from
`eigenfrequency_*_full.pt` via `reduced_indices.pt`.

### Index downselection

`rename_and_downselect_indices.py` renames the full index list to `indices_full.pt`
and writes a thinner `reduced_indices.pt` (keeps `max(1, N_wv // 5)` wavevectors per
`(design, band)` with a fixed seed). Use this to control training set size.

### Alternate input encodings (ablations)

For non-wavelet input channels, build matching stacks with:

- `build_constant_input_tensors.py` → constant field embeddings
- `build_sinusoidal_input_tensors.py` → sinusoidal embeddings

Default training uses the wavelet stacks (`waveforms_full.pt`, `band_fft_full.pt`).

---

## 6) After generation

1. Place continuous and binary shards under `DATASETS/` with the
   `c_train_*` / `b_train_*` / `c_test` / `b_test` naming convention.
2. Ensure each latest `*_pt` folder has `inputs.pt`, `outputs.pt`,
   `reduced_indices.pt`, and the chosen `eigenfrequency_*_full.pt`.
3. Continue with [ML_TRAINING.md](ML_TRAINING.md).
