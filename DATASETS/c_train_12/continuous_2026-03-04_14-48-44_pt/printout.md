# PT Stats for `continuous_2026-03-04_14-48-44_pt`

- Folder: `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_12\continuous_2026-03-04_14-48-44_pt`
- Files processed: `12`
- Sampling cap per file: `1500000` values
- Note: stats/histograms are sampled when files exceed the cap.

| file | type | shape_or_structure | count | mean | std | min | max | non_finite | hist_png |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| band_fft_full.pt | tensor[torch.float16] | 6x32x32 | 6144 | 4.166115e-03 | 9.306568e-02 | -7.397461e-01 | 9.541016e-01 | 0 | hist_band_fft_full.png |
| design_params_full.pt | tensor[torch.float16] | 1000x1 | 1000 | 1.149950e+04 | 2.886932e+02 | 1.100000e+04 | 1.200000e+04 | 0 | hist_design_params_full.png |
| displacements_dataset.pt | TensorDataset | t0:1950000x32x32; t1:1950000x32x32; t2:1950000x32x32; t3:1950000x32x32 | 200000 | -1.415873e-04 | 1.530823e-02 | -4.443359e-01 | 9.140625e-01 | 0 | hist_displacements_dataset.png |
| eigenfrequency_fft_full.pt | tensor[torch.float16] | 1000x325x6x32x32 | 1500000 | 9.208907e-05 | 3.117210e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_eigenfrequency_fft_full.png |
| eigenvalue_data_full.pt | tensor[torch.float16] | 1000x325x6 | 1500000 | 1.655841e+03 | 8.058213e+02 | 0.000000e+00 | 3.960000e+03 | 0 | hist_eigenvalue_data_full.png |
| geometries_full.pt | tensor[torch.float16] | 1000x32x32 | 1024000 | 5.025248e-01 | 2.805809e-01 | -5.006790e-04 | 1.000000e+00 | 0 | hist_geometries_full.png |
| indices_full.pt | list[int64] | (1950000, 3) | 1500000 | 2.211128e+02 | 2.713373e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_indices_full.png |
| inputs.pt | tensor[torch.float16] | 390000x3x32x32 | 1500000 | 1.693243e-01 | 3.153502e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_inputs.png |
| outputs.pt | tensor[torch.float16] | 390000x5x32x32 | 1500000 | 8.547732e-05 | 1.398584e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_outputs.png |
| reduced_indices.pt | list[int64] | (390000, 3) | 1170000 | 2.213806e+02 | 2.713617e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_reduced_indices.png |
| waveforms_full.pt | tensor[torch.float16] | 325x32x32 | 332800 | 3.803919e-04 | 2.100451e-01 | -9.843750e-01 | 9.887695e-01 | 0 | hist_waveforms_full.png |
| wavevectors_full.pt | tensor[torch.float16] | 1000x325x2 | 650000 | 7.844086e-01 | 1.695530e+00 | -3.140625e+00 | 3.140625e+00 | 0 | hist_wavevectors_full.png |
