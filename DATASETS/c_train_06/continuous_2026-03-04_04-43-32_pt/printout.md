# PT Stats for `continuous_2026-03-04_04-43-32_pt`

- Folder: `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_06\continuous_2026-03-04_04-43-32_pt`
- Files processed: `12`
- Sampling cap per file: `200000` values
- Note: stats/histograms are sampled when files exceed the cap.

| file | type | shape_or_structure | count | mean | std | min | max | non_finite | hist_png |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| band_fft_full.pt | tensor[torch.float16] | 6x32x32 | 6144 | 4.166115e-03 | 9.306568e-02 | -7.397461e-01 | 9.541016e-01 | 0 | hist_band_fft_full.png |
| design_params_full.pt | tensor[torch.float16] | 1000x1 | 1000 | 5.499500e+03 | 2.886828e+02 | 5.000000e+03 | 6.000000e+03 | 0 | hist_design_params_full.png |
| displacements_dataset.pt | TensorDataset | t0:1950000x32x32; t1:1950000x32x32; t2:1950000x32x32; t3:1950000x32x32 | 200000 | -6.259761e-05 | 1.563400e-02 | -4.345703e-01 | 6.557617e-01 | 0 | hist_displacements_dataset.png |
| eigenfrequency_fft_full.pt | tensor[torch.float16] | 1000x325x6x32x32 | 200000 | 3.197658e-05 | 3.120750e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_eigenfrequency_fft_full.png |
| eigenvalue_data_full.pt | tensor[torch.float16] | 1000x325x6 | 200000 | 1.666341e+03 | 8.007999e+02 | 0.000000e+00 | 3.802000e+03 | 0 | hist_eigenvalue_data_full.png |
| geometries_full.pt | tensor[torch.float16] | 1000x32x32 | 200000 | 5.085588e-01 | 2.771645e-01 | -5.006790e-04 | 1.000000e+00 | 0 | hist_geometries_full.png |
| indices_full.pt | list[int64] | (1950000, 3) | 200000 | 2.212883e+02 | 2.716178e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_indices_full.png |
| inputs.pt | tensor[torch.float16] | 390000x3x32x32 | 200000 | 1.707935e-01 | 3.164593e-01 | -9.760742e-01 | 1.000000e+00 | 0 | hist_inputs.png |
| outputs.pt | tensor[torch.float16] | 390000x5x32x32 | 200000 | 1.718962e-04 | 1.408374e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_outputs.png |
| reduced_indices.pt | list[int64] | (390000, 3) | 200000 | 2.209149e+02 | 2.708743e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_reduced_indices.png |
| waveforms_full.pt | tensor[torch.float16] | 325x32x32 | 200000 | 3.777949e-04 | 2.102721e-01 | -9.843750e-01 | 9.887695e-01 | 0 | hist_waveforms_full.png |
| wavevectors_full.pt | tensor[torch.float16] | 1000x325x2 | 200000 | 7.834119e-01 | 1.696247e+00 | -3.140625e+00 | 3.140625e+00 | 0 | hist_wavevectors_full.png |
