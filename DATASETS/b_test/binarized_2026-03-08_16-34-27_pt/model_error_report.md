# Model Error Report - b_test

- model: `D:/Research/NO-2D-Metamaterials/MODELS/training_runs/NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_20260311_175808_32860/NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_best.pth`
- dataset folder: `D:\Research\NO-2D-Metamaterials\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt`
- device: `cpu`
- samples evaluated: `1000`
- batch size: `64`

| channel | mae | rmse | max_abs_err | count |
|---|---:|---:|---:|---:|
| ch0 `eigenfrequency_fft` | 2.157803e-01 | 3.127706e-01 | 1.801600e+00 | 1024000 |
| ch1 `disp_x_real` | 9.347999e-03 | 1.582434e-02 | 1.801715e-01 | 1024000 |
| ch2 `disp_x_imag` | 8.726412e-03 | 1.545918e-02 | 1.685131e-01 | 1024000 |
| ch3 `disp_y_real` | 9.298450e-03 | 1.572912e-02 | 1.694095e-01 | 1024000 |
| ch4 `disp_y_imag` | 8.817029e-03 | 1.537037e-02 | 1.708241e-01 | 1024000 |

## Histogram Files
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt\model_error_hist_ch0_eigenfrequency_fft.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt\model_error_hist_ch1_disp_x_real.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt\model_error_hist_ch2_disp_x_imag.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt\model_error_hist_ch3_disp_y_real.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt\model_error_hist_ch4_disp_y_imag.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt\model_error_hist_all_channels.png`
