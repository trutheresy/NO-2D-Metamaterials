# Inference Pipeline

Canonical post-training evaluation for an I3O5 (or I4O5) checkpoint on `c_test` and
`b_test`. Run the scripts below in order; there is no checked-in `_run_*_pipeline.ps1`
launcher anymore — call Python directly (or wrap locally if you need one).

**Related:** [RULES.md](RULES.md) · [ML_TRAINING.md](ML_TRAINING.md) · [DATA_GENERATION.md](DATA_GENERATION.md)

---

## Output layout

Scripts write through `output_layout.resolve_script_output_dir`:

```
INFERENCE/<model>/<dataset>/<step-folder>/   # tensors, CSVs, reports
PLOTS/<model>/<dataset>/<step-folder>/       # figures
PLOTS/<model>/<step-folder>/                 # cross-dataset figures (e.g. IBZ map)
```

`<model>` is the checkpoint / run name without a trailing timestamp
(e.g. `NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260711_E12`).
Every script accepts `--model-name`, `--dataset`, `--output-subdir` (and
`--output-dir` to override the layout).

---

## Pipeline steps (in order)

Per dataset (`c_test`, then `b_test`) unless noted.

| # | Script | Purpose | Default output |
|---|--------|---------|----------------|
| 1 | `run_model_inference_gpu.py` | Dense predictions `(N, 5, 32, 32)`. `--input-encoding {wavelet,sinusoidal,uniform,auto}` (`auto` reads `resolved_config.json`). | `INFERENCE/<model>/<ds>/predictions_I{3\|4}O5_<model>.pt` |
| 2 | (manual copy) | Copy `eigenvalue_data_full.pt`, `geometries_full.pt`, `wavevectors_full.pt` beside predictions. | `INFERENCE/<model>/<ds>/` |
| 3 | `compare_inference_to_truth.py` | Per-channel + overall MAE/MSE vs truth. `--eigen-encoding {uniform,fft}`. | `loss_comparison_<ds>.csv` |
| 4 | `decode_predicted_eigenvalues.py` | Decode prediction ch0 to scalar eigenvalues. | `eigenvalues_predictions_full.pt` |
| 5 | `plot_per_pixel_relative_error.py --dataset-mode` | Per-pixel relative-error stack + mean heatmap. | `relative_error_dataset/` |
| 6 | `analyze_second_peak_waves.py` | Bimodal log-NMAE split; enrichment tables + report. | `second_peak_analysis/` |
| 7 | `per_sample_loss.py` (× losses) | Per-sample scalar loss arrays. Match `--eigen-encoding`. | `PLOTS/.../<LOSS>_sample_case_plots/*.npy` |
| 8 | `plot_loss_histograms.py` | Log-scale histograms + KDE (all / disp / freq). | `PLOTS/.../{all,disp,freq} channel histograms/` |
| 9 | `plot_sample_cases.py` | Truth-vs-prediction fields at loss percentiles. | `PLOTS/.../<LOSS>_sample_case_plots/` |
| 10 | `plot_high_loss_samples.py` | Worst NMAE/NMSE field plots. | `PLOTS/.../high_loss_analysis/` |
| 11 | `scatter_loss_vs_boundary.py` (b_test) | Loss vs boundary length. | `PLOTS/.../boundary_length_vs_loss/` |
| 12 | `scatter_loss_vs_boundary_by_band.py` (b_test) | Same by band + CSV. | `PLOTS/.../boundary_length_vs_loss_by_band/` |
| 13 | `2d-dispersion-py/plot_dispersions_true_vs_pred.py` | Dispersion overlays. | `PLOTS/.../dispersion_overlay/` |
| 14 | `2d-dispersion-py/plot_dispersion_infer_eigenfrequencies.py --no-infer` | Design + truth dispersion plots. | `PLOTS/.../dispersion_plots/` |
| 15 | `plot_ibz_second_peak_waves.py` (once, after both datasets) | IBZ map of second-peak wavevectors. | `PLOTS/<model>/second_peak_analysis/` |

---

## Running for a new model

1. Confirm GPU is free ([RULES.md](RULES.md) §2).
2. Set `MODEL` to the training run folder name and pick checkpoint (`_E12`, `_best`, …).
3. For each of `c_test` and `b_test`, run steps 1–14; then step 15 once.
4. Redirect long jobs to a log under `INFERENCE/` (e.g. `INFERENCE/_run_<tag>.log`).

Conventions:

- CPU by default for scoring/plot steps (`--device cpu`); GPU only for step 1.
- `--tag <dataset>` sets output filenames; `--dataset <dataset>` sets the layout folder.
- Channel-0 eigenfrequency: `--eigen-encoding uniform` (default) or `fft` (wavelet;
  file `eigenfrequency_fft_full.pt`). Encode/decode live in `NO_utilities.py`.
- Input wavevector/band encoding (independent of ch0): `--input-encoding` on
  `run_model_inference_gpu.py` / `_cpu.py`. Files in each `*_pt` folder:
  - wavelet → `waveforms_full.pt` + `band_fft_full.pt`
  - sinusoidal → `waveforms_sinusoidal_full.pt` + `band_sinusoidal_full.pt`
  - uniform → `waveforms_constant_full.pt` + `band_constant_full.pt` (4 input channels)
  - `auto` → `resolved_config.json`
  Shared map: `input_encodings.py`.
