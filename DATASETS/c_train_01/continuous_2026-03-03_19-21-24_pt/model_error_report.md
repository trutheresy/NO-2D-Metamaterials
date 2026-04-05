# Model Error Report - c_train_01

- model: `D:/Research/NO-2D-Metamaterials/MODELS/training_runs/NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_20260311_175808_32860/NO_I3O5_BCF16_L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_260311_best.pth`
- dataset folder: `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_01\continuous_2026-03-03_19-21-24_pt`
- device: `cpu`
- samples evaluated: `50000`
- batch size: `260`

| channel | mae | rmse | max_abs_err | count |
|---|---:|---:|---:|---:|
| ch0 `eigenfrequency_fft` | 6.651981e-02 | 1.145641e-01 | 1.866326e+00 | 51200000 |
| ch1 `disp_x_real` | 9.060599e-03 | 1.520169e-02 | 9.902305e-01 | 51200000 |
| ch2 `disp_x_imag` | 8.808264e-03 | 1.500254e-02 | 9.887220e-01 | 51200000 |
| ch3 `disp_y_real` | 9.305284e-03 | 1.602423e-02 | 1.000414e+00 | 51200000 |
| ch4 `disp_y_imag` | 8.828643e-03 | 1.523230e-02 | 9.899780e-01 | 51200000 |

## Histogram Files
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_01\continuous_2026-03-03_19-21-24_pt\model_error_hist_ch0_eigenfrequency_fft.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_01\continuous_2026-03-03_19-21-24_pt\model_error_hist_ch1_disp_x_real.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_01\continuous_2026-03-03_19-21-24_pt\model_error_hist_ch2_disp_x_imag.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_01\continuous_2026-03-03_19-21-24_pt\model_error_hist_ch3_disp_y_real.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_01\continuous_2026-03-03_19-21-24_pt\model_error_hist_ch4_disp_y_imag.png`
- `D:\Research\NO-2D-Metamaterials\DATASETS\c_train_01\continuous_2026-03-03_19-21-24_pt\model_error_hist_all_channels.png`
