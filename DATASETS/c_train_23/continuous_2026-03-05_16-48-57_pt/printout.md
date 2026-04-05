# PT Stats for `continuous_2026-03-05_16-48-57_pt`

- Folder: `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_23\continuous_2026-03-05_16-48-57_pt`
- Files processed: `12`
- Sampling cap per file: `1500000` values
- Note: stats/histograms are sampled when files exceed the cap.

| file | type | shape_or_structure | count | mean | std | min | max | non_finite | hist_png |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| band_fft_full.pt | tensor[torch.float16] | 6x32x32 | 6144 | 4.166115e-03 | 9.306568e-02 | -7.397461e-01 | 9.541016e-01 | 0 | hist_band_fft_full.png |
| design_params_full.pt | tensor[torch.float16] | 1000x1 | 1000 | 2.249947e+04 | 2.886863e+02 | 2.200000e+04 | 2.299200e+04 | 0 | hist_design_params_full.png |
| displacements_dataset.pt | TensorDataset | t0:1950000x32x32; t1:1950000x32x32; t2:1950000x32x32; t3:1950000x32x32 | 200000 | -1.698252e-04 | 1.550726e-02 | -6.132812e-01 | 8.525391e-01 | 0 | hist_displacements_dataset.png |
| eigenfrequency_fft_full.pt | tensor[torch.float16] | 1000x325x6x32x32 | 1500000 | 2.470415e-04 | 3.120509e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_eigenfrequency_fft_full.png |
| eigenvalue_data_full.pt | tensor[torch.float16] | 1000x325x6 | 1500000 | 1.645964e+03 | 7.979894e+02 | 0.000000e+00 | 3.936000e+03 | 0 | hist_eigenvalue_data_full.png |
| geometries_full.pt | tensor[torch.float16] | 1000x32x32 | 1024000 | 4.961900e-01 | 2.789019e-01 | -5.006790e-04 | 1.000000e+00 | 0 | hist_geometries_full.png |
| indices_full.pt | list[int64] | (1950000, 3) | 1500000 | 2.212686e+02 | 2.714154e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_indices_full.png |
| inputs.pt | tensor[torch.float16] | 390000x3x32x32 | 1500000 | 1.670913e-01 | 3.129279e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_inputs.png |
| outputs.pt | tensor[torch.float16] | 390000x5x32x32 | 1500000 | -5.307633e-05 | 1.402938e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_outputs.png |
| reduced_indices.pt | list[int64] | (390000, 3) | 1170000 | 2.212711e+02 | 2.713878e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_reduced_indices.png |
| waveforms_full.pt | tensor[torch.float16] | 325x32x32 | 332800 | 3.803919e-04 | 2.100451e-01 | -9.843750e-01 | 9.887695e-01 | 0 | hist_waveforms_full.png |
| wavevectors_full.pt | tensor[torch.float16] | 1000x325x2 | 650000 | 7.844086e-01 | 1.695530e+00 | -3.140625e+00 | 3.140625e+00 | 0 | hist_wavevectors_full.png |
