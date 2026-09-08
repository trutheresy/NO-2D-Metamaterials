# Inference Pipeline

Post-training evaluation for an I3O5 checkpoint on `c_test` and `b_test`: dense
prediction, scoring, and standard result figures (field cases, loss histograms,
dispersion overlays, boundary-length scatters).

**Related:** [ML_TRAINING.md](ML_TRAINING.md) · [DATA_GENERATION.md](DATA_GENERATION.md)

Call the Python entry points directly (no checked-in launcher script).

---

## 1) Output layout

Paths are resolved through `output_layout.py`:

```
INFERENCE/<model>/<dataset>/…     # predictions, CSVs, reports
PLOTS/<model>/<dataset>/…         # per-dataset figures
PLOTS/<model>/…                   # cross-dataset figures
```

`<model>` is the training run folder name (optionally with an epoch/best suffix in
filenames, e.g. `…_E12` or `…_best`). Pass `--model-name`, `--dataset`, and
`--output-subdir` (or `--output-dir`) consistently across steps.

---

## 2) Canonical steps

Run for each of `c_test` and `b_test` unless noted. Use GPU only for step 1; keep
scoring/plotting on CPU (`--device cpu`) unless you know the GPU is free.

| # | Script | Purpose | Typical output |
|---|--------|---------|----------------|
| 1 | `run_model_inference_gpu.py` | Dense predictions `(N, 5, 32, 32)` | `INFERENCE/<model>/<ds>/predictions_I3O5_<model>.pt` |
| 2 | (manual copy) | Place `eigenvalue_data_full.pt`, `geometries_full.pt`, `wavevectors_full.pt` beside predictions | same folder |
| 3 | `compare_inference_to_truth.py` | Per-channel / overall MAE & MSE vs truth | `loss_comparison_<ds>.csv` |
| 4 | `decode_predicted_eigenvalues.py` | Decode predicted ch0 → scalar eigenfrequencies | `eigenvalues_predictions_full.pt` |
| 5 | `per_sample_loss.py` | Per-sample scalar losses (match `--eigen-encoding`) | `PLOTS/…/<LOSS>_sample_case_plots/*.npy` |
| 6 | `plot_loss_histograms.py` | Log-scale histograms (all / displacement / frequency) | `PLOTS/…/{all,disp,freq} channel histograms/` |
| 7 | `plot_sample_cases.py` | Truth vs prediction at loss percentiles | `PLOTS/…/<LOSS>_sample_case_plots/` |
| 8 | `scatter_loss_vs_boundary.py` | **b_test:** loss vs interface length | `PLOTS/…/boundary_length_vs_loss/` |
| 9 | `scatter_loss_vs_boundary_by_band.py` | **b_test:** same, stratified by band | `PLOTS/…/boundary_length_vs_loss_by_band/` |
| 10 | `2d-dispersion-py/plot_dispersions_true_vs_pred.py` | Dispersion band overlays | `PLOTS/…/dispersion_overlay/` |
| 11 | `2d-dispersion-py/plot_dispersion_infer_eigenfrequencies.py --no-infer` | Design + truth dispersion panels | `PLOTS/…/dispersion_plots/` |

Boundary-length helpers live in `compute_boundary_length.py`. Shared inference
utilities: `model_inference_common.py`, `input_encodings.py`, `NO_utilities.py`,
`per_sample_loss.py`.

### Optional extras (not required for the main figure set)

| Script | Purpose |
|--------|---------|
| `DIAGNOSTICS/plot_per_pixel_relative_error.py --dataset-mode` | Per-pixel relative-error stacks |
| `DIAGNOSTICS/plot_high_loss_samples.py` | Worst-case field galleries |
| `DIAGNOSTICS/analyze_second_peak_waves.py` | Bimodal high-loss wave/band enrichment |
| `DIAGNOSTICS/plot_ibz_second_peak_waves.py` | IBZ map of second-peak wavevectors |

---

## 3) Encoding flags

Two independent encoding choices:

### Output channel 0 (eigenfrequency)

`--eigen-encoding uniform` (default) or `fft` (wavelet patches in
`eigenfrequency_fft_full.pt`). Encode/decode helpers are in `NO_utilities.py`.
Use the same choice as training’s `--eigen-ch0-encoding`.

### Input wavevector / band channels

On `run_model_inference_gpu.py` / `run_model_inference_cpu.py`:

| `--input-encoding` | Reads from the `*_pt` folder |
|--------------------|------------------------------|
| `wavelet` (default path) | `waveforms_full.pt`, `band_fft_full.pt` |
| `sinusoidal` | `waveforms_sinusoidal_full.pt`, `band_sinusoidal_full.pt` |
| `uniform` | `waveforms_constant_full.pt`, `band_constant_full.pt` (4-in models) |
| `auto` | From the run’s `resolved_config.json` |

Registry: `input_encodings.py`.

---

## 4) Running a new checkpoint

1. Confirm no other job holds the GPU (`MODELS/training_runs/*/run_metadata.json`
   should not show `"status": "running"` if you need CUDA).
2. Set `MODEL` to the training run directory name and pick a checkpoint
   (`_E12`, `_best`, …).
3. For `c_test` then `b_test`, run steps 1–7; add 8–9 on binary test; add 10–11 for
   dispersion figures.
4. Redirect long jobs to a log under `INFERENCE/` (e.g. `INFERENCE/_run_<tag>.log`).

CPU fallback for step 1: `run_model_inference_cpu.py`.

---

## 5) Encoding / methodology figures (optional)

Scripts that regenerate encoding diagnostics (similarity heatmaps, encode–decode
checks, palette previews):

- `inspect_wavelet_embeddings_ibz.py`, `inspect_band_embeddings.py`
- `plot_wavelet_embeddings_ibz.py`, `plot_encoding_figures_palette.py`
- `plot_1d_scalar_encode_decode_error.py`

These are independent of the per-checkpoint evaluation table above.
