import numpy as np
import torch
from pathlib import Path

from per_sample_loss import compute_per_sample_losses, prepare_scoring_data, resolve_device

pt = Path("DATASETS/c_test/continuous_2026-03-05_20-07-34_pt")
pred_path = Path(
    "INFERENCE/C_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat_260530-023704/"
    "predictions_I3O5_NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat.pt"
)
predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
truth, n_geom, n_wv, n_bands, fh, fw, channels = prepare_scoring_data(pt, predictions)
device = resolve_device("cpu")
n = 4096
losses = compute_per_sample_losses(truth[:n], predictions[:n], channels, ["mae", "nmae"], device, 4096)
eigen = torch.load(pt / "eigenfrequency_uniform_full.pt", map_location="cpu", mmap=True, weights_only=True)
truth_ch0 = eigen.reshape(-1, fh, fw)[:n]
mae_ch0 = (predictions[:n, 0].float() - truth_ch0.float()).abs().mean(dim=(1, 2)).numpy()

print("channels used:", channels)
print("disp MAE mean", float(losses["mae"].mean()))
print("ch0  MAE mean", float(mae_ch0.mean()))
print("identical?", bool(np.allclose(losses["mae"], mae_ch0)))
print("max abs diff", float(np.max(np.abs(losses["mae"] - mae_ch0))))
print("disp NMAE mean", float(losses["nmae"].mean()))
