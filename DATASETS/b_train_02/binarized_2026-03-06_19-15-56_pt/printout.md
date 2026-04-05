# PT Stats for `binarized_2026-03-06_19-15-56_pt`

- Folder: `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_02\binarized_2026-03-06_19-15-56_pt`
- Files processed: `12`
- Sampling cap per file: `200000` values
- Note: stats/histograms are sampled when files exceed the cap.

| file | type | shape_or_structure | count | mean | std | min | max | non_finite | hist_png |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| band_fft_full.pt | tensor[torch.float16] | 6x32x32 | 6144 | 4.166115e-03 | 9.306568e-02 | -7.397461e-01 | 9.541016e-01 | 0 | hist_band_fft_full.png |
| design_params_full.pt | tensor[torch.float16] | 1000x1 | 1000 | 1.499500e+03 | 2.886750e+02 | 1.000000e+03 | 1.999000e+03 | 0 | hist_design_params_full.png |
| displacements_dataset.pt | TensorDataset | t0:1950000x32x32; t1:1950000x32x32; t2:1950000x32x32; t3:1950000x32x32 | 200000 | -9.364724e-05 | 1.563295e-02 | -2.325439e-01 | 2.172852e-01 | 0 | hist_displacements_dataset.png |
| eigenfrequency_fft_full.pt | tensor[torch.float16] | 1000x325x6x32x32 | 200000 | 4.097814e-04 | 3.115451e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_eigenfrequency_fft_full.png |
| eigenvalue_data_full.pt | tensor[torch.float16] | 1000x325x6 | 200000 | 4.871887e+02 | 3.921389e+02 | 0.000000e+00 | 3.912000e+03 | 0 | hist_eigenvalue_data_full.png |
| geometries_full.pt | tensor[torch.float16] | 1000x32x32 | 200000 | 4.990750e-01 | 4.999991e-01 | 0.000000e+00 | 1.000000e+00 | 0 | hist_geometries_full.png |
| indices_full.pt | list[int64] | (1950000, 3) | 200000 | 2.216455e+02 | 2.715488e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_indices_full.png |
| inputs.pt | tensor[torch.float16] | 390000x3x32x32 | 200000 | 1.665565e-01 | 3.941753e-01 | -9.755859e-01 | 1.000000e+00 | 0 | hist_inputs.png |
| outputs.pt | tensor[torch.float16] | 390000x5x32x32 | 200000 | -3.332961e-05 | 1.401247e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_outputs.png |
| reduced_indices.pt | list[int64] | (390000, 3) | 200000 | 2.207677e+02 | 2.710062e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_reduced_indices.png |
| waveforms_full.pt | tensor[torch.float16] | 325x32x32 | 200000 | 4.986227e-04 | 2.103123e-01 | -9.843750e-01 | 9.833984e-01 | 0 | hist_waveforms_full.png |
| wavevectors_full.pt | tensor[torch.float16] | 1000x325x2 | 200000 | 7.907855e-01 | 1.696178e+00 | -3.140625e+00 | 3.140625e+00 | 0 | hist_wavevectors_full.png |
