# Model Error Report - b_train_01

- model: `D:/Research/NO-2D-Metamaterials/MODELS/training_runs/NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_20260311_175808_32860/NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_best.pth`
- dataset folder: `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_01\binarized_2026-03-06_17-18-57_pt`
- device: `cpu`
- samples evaluated: `50000`
- batch size: `260`

| channel | mae | rmse | max_abs_err | count |
|---|---:|---:|---:|---:|
| ch0 `eigenfrequency_fft` | 2.039256e-01 | 3.057344e-01 | 1.907362e+00 | 51200000 |
| ch1 `disp_x_real` | 9.332714e-03 | 1.571555e-02 | 4.120183e-01 | 51200000 |
| ch2 `disp_x_imag` | 8.534668e-03 | 1.506340e-02 | 4.091350e-01 | 51200000 |
| ch3 `disp_y_real` | 9.298761e-03 | 1.602778e-02 | 4.106519e-01 | 51200000 |
| ch4 `disp_y_imag` | 8.786593e-03 | 1.538352e-02 | 4.095995e-01 | 51200000 |

## Histogram Files
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_01\binarized_2026-03-06_17-18-57_pt\model_error_hist_ch0_eigenfrequency_fft.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_01\binarized_2026-03-06_17-18-57_pt\model_error_hist_ch1_disp_x_real.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_01\binarized_2026-03-06_17-18-57_pt\model_error_hist_ch2_disp_x_imag.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_01\binarized_2026-03-06_17-18-57_pt\model_error_hist_ch3_disp_y_real.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_01\binarized_2026-03-06_17-18-57_pt\model_error_hist_ch4_disp_y_imag.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\b_train_01\binarized_2026-03-06_17-18-57_pt\model_error_hist_all_channels.png`
