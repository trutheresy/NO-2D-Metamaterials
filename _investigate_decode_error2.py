"""Follow-up: per-band patch-domain error and the ln-encoding amplification law (c_test)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\Research\NO-2D-Metamaterials")
PT = ROOT / r"DATASETS\c_test\continuous_2026-03-05_20-07-34_pt"
MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
INF = ROOT / "INFERENCE" / MODEL / "c_test"

N_GEOM = 100
EPS = 1e-5


def main() -> None:
    eigen_true = torch.load(PT / "eigenvalue_data_full.pt", map_location="cpu", weights_only=True).float().numpy()
    n_geom, n_wv, n_bands = eigen_true.shape
    ef_uni = torch.load(PT / "eigenfrequency_uniform_full.pt", map_location="cpu", mmap=True, weights_only=True)
    preds = torch.load(INF / f"predictions_I3O5_{MODEL}.pt", map_location="cpu", mmap=True, weights_only=True)

    rows_per_geom = n_wv * n_bands
    n_rows = N_GEOM * rows_per_geom
    ch0 = preds[:n_rows, 0]

    # patch-domain: per-band MAE of model patch mean vs true pixel
    true_pix = ef_uni[:N_GEOM, :, :, 0, 0].float().numpy().reshape(-1, n_bands)  # ln(s)/100 truth
    pred_mean = ch0.float().mean(dim=(1, 2)).numpy().reshape(-1, n_bands)
    pred_corner = ch0[:, 0, 0].float().numpy().reshape(-1, n_bands)

    print("=== patch-domain per-band (ln(s)/100 units) ===")
    for b in range(n_bands):
        mae_m = np.abs(pred_mean[:, b] - true_pix[:, b]).mean()
        mae_c = np.abs(pred_corner[:, b] - true_pix[:, b]).mean()
        noise = np.abs(pred_corner[:, b] - pred_mean[:, b]).mean()
        print(f"b{b}: MAE(mean-decode src)={mae_m:.3e}  MAE(corner px)={mae_c:.3e}  "
              f"corner-vs-mean noise={noise:.3e}  -> pred freq rel err ~{100*mae_m*100:.2f}% / {100*mae_c*100:.2f}%")

    # amplification law check: per-sample rel err vs 100*patch err
    t = eigen_true[:N_GEOM].reshape(-1, n_bands)
    dec_mean = np.exp(100.0 * pred_mean.astype(np.float64))
    rel = np.abs(dec_mean - t) / (np.abs(t) + EPS)
    lin = 100.0 * np.abs(pred_mean - true_pix)
    print(f"\namplification law: corr(rel_err, 100*|patch_err|) = "
          f"{np.corrcoef(rel.ravel(), lin.ravel())[0,1]:.4f}")
    print(f"mean rel err={rel.mean():.4e}  mean 100*|patch err|={lin.mean():.4e}")

    # best geometry g360: patch-domain error
    sl = slice(360 * rows_per_geom, 361 * rows_per_geom)
    ch0g = preds[sl, 0].float()
    pm = ch0g.mean(dim=(1, 2)).numpy().reshape(-1, n_bands)
    tp = ef_uni[360, :, :, 0, 0].float().numpy().reshape(-1, n_bands)
    print(f"\ng360 patch MAE (mean) = {np.abs(pm-tp).mean():.3e} -> expected freq rel err ~{100*np.abs(pm-tp).mean()*100:.2f}%")

    # validation-metric context
    print("\nval_l1_loss_ch0 (0707 E12) = 4.635e-04 patch MAE -> expected freq rel err ~4.6%")
    print("c_test patch MAE (this run) = ", f"{np.abs(pred_mean-true_pix).mean():.3e}",
          f"-> ~{100*np.abs(pred_mean-true_pix).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
