# Model Error Report - c_test

- model: `D:/Research/NO-2D-Metamaterials/MODELS/training_runs/NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_20260311_175808_32860/NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_best.pth`
- dataset folder: `D:\Research\NO-2D-Metamaterials\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt`
- device: `cpu`
- samples evaluated: `50000`
- batch size: `260`

| channel | mae | rmse | max_abs_err | count |
|---|---:|---:|---:|---:|
| ch0 `eigenfrequency_fft` | 6.995433e-02 | 1.227210e-01 | 1.884744e+00 | 51200000 |
| ch1 `disp_x_real` | 9.005622e-03 | 1.523380e-02 | 9.899268e-01 | 51200000 |
| ch2 `disp_x_imag` | 8.851126e-03 | 1.500666e-02 | 9.945958e-01 | 51200000 |
| ch3 `disp_y_real` | 9.205336e-03 | 1.606025e-02 | 9.997993e-01 | 51200000 |
| ch4 `disp_y_imag` | 8.858094e-03 | 1.520160e-02 | 1.000231e+00 | 51200000 |

## Histogram Files
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt\model_error_hist_ch0_eigenfrequency_fft.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt\model_error_hist_ch1_disp_x_real.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt\model_error_hist_ch2_disp_x_imag.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt\model_error_hist_ch3_disp_y_real.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt\model_error_hist_ch4_disp_y_imag.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt\model_error_hist_all_channels.png`
