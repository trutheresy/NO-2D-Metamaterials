# Standard Inference Pipeline

The canonical post-training evaluation flow for an I3O5 model checkpoint, run against
both test datasets (`c_test`, `b_test`). The executable definition is the model's
`_run_*_pipeline.ps1` launcher (current reference: `_run_0705E12_pipeline.ps1`);
this document describes the steps, scripts, and output layout.

**Related docs:** [RULES.md](RULES.md) · [ML_TRAINING.md](ML_TRAINING.md) · [DATA_GENERATION.md](DATA_GENERATION.md)

---

## Output layout

All steps write through `output_layout.resolve_script_output_dir`:

```
INFERENCE/<model>/<dataset>/<step-folder>/   # data-like outputs (tensors, CSVs, reports)
PLOTS/<model>/<dataset>/<step-folder>/       # figure outputs
PLOTS/<model>/<step-folder>/                 # cross-dataset figures (e.g. IBZ map)
```

`<model>` is the checkpoint name without dataset prefix / trailing timestamp
(e.g. `NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best`).
Every script accepts `--model-name`, `--dataset`, `--output-subdir` (and
`--output-dir` to override the layout entirely).

## Pipeline steps, in order

Per dataset (`c_test`, then `b_test`) unless noted.

| # | Script | Purpose | Output (default subdir) |
|---|--------|---------|-------------------------|
| 1 | `run_model_inference_gpu.py` | Run the checkpoint over the dataset; save dense predictions `(N, 5, 32, 32)`. | `INFERENCE/<model>/<ds>/predictions_I3O5_<model>.pt` |
| 2 | (copy step in launcher) | Copy `eigenvalue_data_full.pt`, `geometries_full.pt`, `wavevectors_full.pt` beside predictions. | `INFERENCE/<model>/<ds>/` |
| 3 | `compare_inference_to_truth.py` | Per-channel + overall MAE/MSE statistics vs truth. | `loss_comparison_<ds>.csv` |
| 4 | `decode_predicted_eigenvalues.py` | Decode prediction ch0 (uniform encoding) to scalar eigenvalues. | `eigenvalues_predictions_full.pt` |
| 5 | `plot_per_pixel_relative_error.py --dataset-mode` | Per-pixel \|e\|/\|t\| stack `(N, 32, 32)` + mean-over-samples heatmap. | `INFERENCE/.../relative_error_dataset/` |
| 6 | `analyze_second_peak_waves.py` | Split bimodal log-NMAE distribution; per-wavevector / per-band second-peak enrichment tables + report. | `INFERENCE/.../second_peak_analysis/` |
| 7 | `per_sample_loss.py` (×6 losses) | Per-sample scalar loss arrays (mae, mse, rms, nmae, nmse, nrms). | `PLOTS/.../<LOSS>_sample_case_plots/*.npy` |
| 8 | `plot_loss_histograms.py` (×3 groups) | Log-scale histograms + KDE per channel group (all/disp/freq). | `PLOTS/.../{all,disp,freq} channel histograms/` |
| 9 | `plot_sample_cases.py` (×6 losses) | Truth-vs-prediction field plots at loss percentiles. | `PLOTS/.../<LOSS>_sample_case_plots/` |
| 10 | `plot_high_loss_samples.py` | Field plots of the worst NMAE/NMSE samples. | `PLOTS/.../high_loss_analysis/` |
| 11 | `scatter_loss_vs_boundary.py` (b_test only) | Per-sample loss vs geometry boundary length. | `PLOTS/.../boundary_length_vs_loss/` |
| 12 | `scatter_loss_vs_boundary_by_band.py` (b_test only) | Same scatter split per band, + per-geometry CSV. | `PLOTS/.../boundary_length_vs_loss_by_band/` |
| 13 | `2d-dispersion-py/plot_dispersions_true_vs_pred.py` | True-vs-predicted dispersion overlays. | `PLOTS/.../dispersion_overlay/` |
| 14 | `2d-dispersion-py/plot_dispersion_infer_eigenfrequencies.py --no-infer` | Design images + truth dispersion plots (prune `design_raw/` after). | `PLOTS/.../dispersion_plots/` |
| 15 | `plot_ibz_second_peak_waves.py` (once, after both datasets) | IBZ map highlighting wavevectors with >50% second-peak membership (c ∩ b overlap). | `PLOTS/<model>/second_peak_analysis/second_peak_ibz_map.png` |

## Running for a new model

Copy `_run_0705E12_pipeline.ps1`, update `$runDir` / `$ckpt` / `$MODEL`, and launch
detached with output to a log under `INFERENCE/`. The launcher variants:

- `_run_<model>_pipeline.ps1` — full pipeline including GPU inference (steps 1–15).
- `_run_<model>_postinfer.ps1` — steps 3–15 when predictions already exist.
- `_run_<model>_postinfer_resume.ps1` — ad-hoc resume from a mid-pipeline failure.

Conventions:

- CPU by default for scoring/plot steps (`--device cpu`); GPU only for step 1
  (see [RULES.md §2](RULES.md) for GPU coordination).
- `--tag <dataset>` sets output filenames; `--dataset <dataset>` sets the layout folder.
- Backfill helpers for adding a new step to old runs follow the pattern of
  `_run_relative_error_backfill.ps1` (skip-if-complete checks + structure verification).
