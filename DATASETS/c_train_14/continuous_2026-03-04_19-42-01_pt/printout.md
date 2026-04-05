# PT Stats for `continuous_2026-03-04_19-42-01_pt`

- Folder: `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_14\continuous_2026-03-04_19-42-01_pt`
- Files processed: `12`
- Sampling cap per file: `1500000` values
- Note: stats/histograms are sampled when files exceed the cap.

| file | type | shape_or_structure | count | mean | std | min | max | non_finite | hist_png |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| band_fft_full.pt | tensor[torch.float16] | 6x32x32 | 6144 | 4.166115e-03 | 9.306568e-02 | -7.397461e-01 | 9.541016e-01 | 0 | hist_band_fft_full.png |
| design_params_full.pt | tensor[torch.float16] | 1000x1 | 1000 | 1.349950e+04 | 2.886932e+02 | 1.300000e+04 | 1.400000e+04 | 0 | hist_design_params_full.png |
| displacements_dataset.pt | TensorDataset | t0:1950000x32x32; t1:1950000x32x32; t2:1950000x32x32; t3:1950000x32x32 | 200000 | -1.286626e-04 | 1.588483e-02 | -6.762695e-01 | 9.462891e-01 | 0 | hist_displacements_dataset.png |
| eigenfrequency_fft_full.pt | tensor[torch.float16] | 1000x325x6x32x32 | 1500000 | 4.890545e-04 | 3.117171e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_eigenfrequency_fft_full.png |
| eigenvalue_data_full.pt | tensor[torch.float16] | 1000x325x6 | 1500000 | 1.643344e+03 | 8.093527e+02 | 0.000000e+00 | 3.778000e+03 | 0 | hist_eigenvalue_data_full.png |
| geometries_full.pt | tensor[torch.float16] | 1000x32x32 | 1024000 | 5.039570e-01 | 2.783871e-01 | -5.006790e-04 | 1.000000e+00 | 0 | hist_geometries_full.png |
| indices_full.pt | list[int64] | (1950000, 3) | 1500000 | 2.211604e+02 | 2.713508e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_indices_full.png |
| inputs.pt | tensor[torch.float16] | 390000x3x32x32 | 1500000 | 1.692715e-01 | 3.153260e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_inputs.png |
| outputs.pt | tensor[torch.float16] | 390000x5x32x32 | 1500000 | -8.503926e-05 | 1.403547e-01 | -9.843750e-01 | 1.000000e+00 | 0 | hist_outputs.png |
| reduced_indices.pt | list[int64] | (390000, 3) | 1170000 | 2.212959e+02 | 2.713682e+02 | 0.000000e+00 | 9.990000e+02 | 0 | hist_reduced_indices.png |
| waveforms_full.pt | tensor[torch.float16] | 325x32x32 | 332800 | 3.803919e-04 | 2.100451e-01 | -9.843750e-01 | 9.887695e-01 | 0 | hist_waveforms_full.png |
| wavevectors_full.pt | tensor[torch.float16] | 1000x325x2 | 650000 | 7.836234e-01 | 1.694863e+00 | -3.140625e+00 | 3.140625e+00 | 0 | hist_wavevectors_full.png |
