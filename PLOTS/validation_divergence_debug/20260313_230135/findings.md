# Findings: Validation-Train Loss Divergence Debug

## Dataset coverage
- Analyzed datasets: train=48, test=2, total=50.
- Per-dataset statistics are computed from sampled images (`n_sample_images` in `dataset_stats.csv`).

## Train-vs-test distribution similarity
- Mean KS distance across tracked metrics: `0.4258`.
- Max relative mean gap metric: `eigen_mean` with `rel_mean_gap=7.1864e-01`.
- Median relative mean gap across metrics: `4.3993e-03`.

## Output scale comparison
- `std_eigen_over_std_disp_all` = `1.996055e+01`.
- `std_eigen_over_std_disp_ch0` = `1.947368e+01`.
- `std_eigen_over_std_disp_ch1` = `2.066270e+01`.
- `std_eigen_over_std_disp_ch2` = `1.951866e+01`.
- `std_eigen_over_std_disp_ch3` = `2.044971e+01`.

## Validation loading audit
- Validation loader wiring checks all pass: `True`.
- Train/test shard overlap count: `0`.
- Train samples: `18720000`, test samples: `780000`.

## Recommendations
- If eigenfrequency/dispersion scale ratios are high, test channel-weighted loss or per-channel normalization.
- If train/test gaps are small yet val loss plateaus, test regularization/lr schedule and per-channel loss weighting.
- Keep this validation split audit as a guardrail when editing shard discovery and eval code.
