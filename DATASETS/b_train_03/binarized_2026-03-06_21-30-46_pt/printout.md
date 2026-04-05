# PT Stats for `binarized_2026-03-06_21-30-46_pt`

- Folder: `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_03\binarized_2026-03-06_21-30-46_pt`
- Files processed: `12`
- Sampling cap per file: `200000` values
- Note: stats/histograms are sampled when files exceed the cap.

| file | type | shape_or_structure | count | mean | std | min | max | non_finite | hist_png |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| band_fft_full.pt | tensor[torch.float16] | 6x32x32 | 6144 | 4.166115e-03 | 9.306568e-02 | -7.397461e-01 | 9.541016e-01 | 0 | hist_band_fft_full.png |
| design_params_full.pt | tensor[torch.float16] | 1000x1 | 1000 | 2.499500e+03 | 2.886775e+02 | 2.000000e+03 | 3.000000e+03 | 0 | hist_design_params_full.png |
| displacements_dataset.pt | TensorDataset | t0:1950000x32x32; t1:1950000x32x32; t2:1950000x32x32; t3:1950000x32x32 | 200000 | -7.998885e-05 | 1.582610e-02 | -3.051758e-01 | 3.950195e-01 | 0 | hist_displacements_dataset.png |
| eigenfrequency_fft_full.pt | tensor[torch.float16] | 1000x325x6x32x32 | 200000 | -9.226182e-04 | 3.123466e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_eigenfrequency_fft_full.png |
| eigenvalue_data_full.pt | tensor[torch.float16] | 1000x325x6 | 200000 | 4.688996e+02 | 3.668309e+02 | 0.000000e+00 | 2.634000e+03 | 0 | hist_eigenvalue_data_full.png |
| geometries_full.pt | tensor[torch.float16] | 1000x32x32 | 200000 | 4.942700e-01 | 4.999672e-01 | 0.000000e+00 | 1.000000e+00 | 0 | hist_geometries_full.png |
| indices_full.pt | list[int64] | (1950000, 3) | 200000 | 2.218681e+02 | 2.718189e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_indices_full.png |
| inputs.pt | tensor[torch.float16] | 390000x3x32x32 | 200000 | 1.668066e-01 | 3.937009e-01 | -9.760742e-01 | 1.000000e+00 | 0 | hist_inputs.png |
| outputs.pt | tensor[torch.float16] | 390000x5x32x32 | 200000 | -3.367932e-04 | 1.395459e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_outputs.png |
| reduced_indices.pt | list[int64] | (390000, 3) | 200000 | 2.218200e+02 | 2.715205e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_reduced_indices.png |
| waveforms_full.pt | tensor[torch.float16] | 325x32x32 | 200000 | -1.201640e-04 | 2.101187e-01 | -9.843750e-01 | 9.887695e-01 | 0 | hist_waveforms_full.png |
| wavevectors_full.pt | tensor[torch.float16] | 1000x325x2 | 200000 | 7.810646e-01 | 1.694753e+00 | -3.140625e+00 | 3.140625e+00 | 0 | hist_wavevectors_full.png |
