# PT Stats for `binarized_2026-03-06_23-40-44_pt`

- Folder: `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_04\binarized_2026-03-06_23-40-44_pt`
- Files processed: `12`
- Sampling cap per file: `200000` values
- Note: stats/histograms are sampled when files exceed the cap.

| file | type | shape_or_structure | count | mean | std | min | max | non_finite | hist_png |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| band_fft_full.pt | tensor[torch.float16] | 6x32x32 | 6144 | 4.166115e-03 | 9.306568e-02 | -7.397461e-01 | 9.541016e-01 | 0 | hist_band_fft_full.png |
| design_params_full.pt | tensor[torch.float16] | 1000x1 | 1000 | 3.499500e+03 | 2.886776e+02 | 3.000000e+03 | 4.000000e+03 | 0 | hist_design_params_full.png |
| displacements_dataset.pt | TensorDataset | t0:1950000x32x32; t1:1950000x32x32; t2:1950000x32x32; t3:1950000x32x32 | 200000 | -1.643771e-04 | 1.558081e-02 | -2.719727e-01 | 2.687988e-01 | 0 | hist_displacements_dataset.png |
| eigenfrequency_fft_full.pt | tensor[torch.float16] | 1000x325x6x32x32 | 200000 | -8.198820e-05 | 3.109827e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_eigenfrequency_fft_full.png |
| eigenvalue_data_full.pt | tensor[torch.float16] | 1000x325x6 | 200000 | 4.642288e+02 | 3.795314e+02 | 0.000000e+00 | 2.634000e+03 | 0 | hist_eigenvalue_data_full.png |
| geometries_full.pt | tensor[torch.float16] | 1000x32x32 | 200000 | 4.856250e-01 | 4.997933e-01 | 0.000000e+00 | 1.000000e+00 | 0 | hist_geometries_full.png |
| indices_full.pt | list[int64] | (1950000, 3) | 200000 | 2.215998e+02 | 2.718830e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_indices_full.png |
| inputs.pt | tensor[torch.float16] | 390000x3x32x32 | 200000 | 1.631896e-01 | 3.912168e-01 | -9.760742e-01 | 1.000000e+00 | 0 | hist_inputs.png |
| outputs.pt | tensor[torch.float16] | 390000x5x32x32 | 200000 | -7.776175e-04 | 1.412230e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_outputs.png |
| reduced_indices.pt | list[int64] | (390000, 3) | 200000 | 2.209794e+02 | 2.709900e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_reduced_indices.png |
| waveforms_full.pt | tensor[torch.float16] | 325x32x32 | 200000 | 7.559338e-04 | 2.100157e-01 | -9.843750e-01 | 9.887695e-01 | 0 | hist_waveforms_full.png |
| wavevectors_full.pt | tensor[torch.float16] | 1000x325x2 | 200000 | 7.846565e-01 | 1.697030e+00 | -3.140625e+00 | 3.140625e+00 | 0 | hist_wavevectors_full.png |
