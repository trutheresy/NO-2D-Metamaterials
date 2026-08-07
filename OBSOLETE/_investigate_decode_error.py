"""Isolate error sources in the dispersion overlay eigenvalue path (c_test).

Compares, per band and overall:
  A. encode/decode round-trip floor: decode(eigenfrequency_uniform_full) vs eigenvalue_data_full
  B. decoded predictions (pixel reduce, as pipeline used) vs truth
  C. decoded predictions (mean reduce) vs truth
  D. patch-domain ch0 NMAE (what training's val_loss_ch0 measures) on the same samples
  E. best-ranked geometry (g360) drill-down
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\Research\NO-2D-Metamaterials")
PT = ROOT / r"DATASETS\c_test\continuous_2026-03-05_20-07-34_pt"
MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
INF = ROOT / "INFERENCE" / MODEL / "c_test"

N_GEOM_SAMPLE = 100  # geometries to scan for patch/decode comparisons
EPS = 1e-5


def nmae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean() / (np.abs(true).mean() + EPS))


def per_band_nmae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    # pred/true: (..., n_bands); normalize each band by its own mean |true|
    ax = tuple(range(pred.ndim - 1))
    return np.abs(pred - true).mean(axis=ax) / (np.abs(true).mean(axis=ax) + EPS)


def main() -> None:
    eigen_true = torch.load(PT / "eigenvalue_data_full.pt", map_location="cpu", weights_only=True).float().numpy()
    n_geom, n_wv, n_bands = eigen_true.shape
    print(f"truth eigenvalues: {eigen_true.shape}")

    # ---------- A. encode/decode round-trip floor ----------
    ef_uni = torch.load(PT / "eigenfrequency_uniform_full.pt", map_location="cpu", mmap=True, weights_only=True)
    print(f"eigenfrequency_uniform_full: {tuple(ef_uni.shape)} {ef_uni.dtype}")
    g_sl = slice(0, N_GEOM_SAMPLE)
    # corner-pixel decode of TRUE patches
    true_pix = ef_uni[g_sl, :, :, 0, 0].float().numpy()  # (G, n_wv, n_bands)
    decoded_true = np.exp(100.0 * true_pix.astype(np.float64))
    t = eigen_true[g_sl]
    print("\n=== A. round-trip floor: decode(true uniform patch) vs eigenvalue_data ===")
    print(f"overall NMAE : {nmae(decoded_true, t):.6e}")
    pb = per_band_nmae(decoded_true.reshape(-1, n_bands), t.reshape(-1, n_bands))
    print("per-band NMAE:", " ".join(f"b{i}={v:.4e}" for i, v in enumerate(pb)))
    # where is it worst? check small eigenvalues (Gamma point, acoustic bands)
    small = t < 50
    if small.any():
        print(f"rows with true<50 rad/s: {small.sum()} ; their round-trip NMAE: "
              f"{nmae(decoded_true[small], t[small]):.4e}")

    # ---------- B/C. decoded predictions, pixel vs mean ----------
    preds = torch.load(INF / f"predictions_I3O5_{MODEL}.pt", map_location="cpu", mmap=True, weights_only=True)
    print(f"\npredictions: {tuple(preds.shape)} {preds.dtype}")
    rows_per_geom = n_wv * n_bands
    n_rows = N_GEOM_SAMPLE * rows_per_geom
    ch0 = preds[:n_rows, 0]  # (rows, 32, 32) float16, mmap

    corner = ch0[:, 0, 0].float().numpy()
    patch_mean = ch0.float().mean(dim=(1, 2)).numpy()
    dec_pixel = np.exp(100.0 * corner.astype(np.float64)).reshape(N_GEOM_SAMPLE, n_wv, n_bands)
    dec_mean = np.exp(100.0 * patch_mean.astype(np.float64)).reshape(N_GEOM_SAMPLE, n_wv, n_bands)

    print("\n=== B. model decoded (PIXEL reduce, pipeline default) vs truth ===")
    print(f"overall NMAE : {nmae(dec_pixel, t):.6e}")
    pb = per_band_nmae(dec_pixel.reshape(-1, n_bands), t.reshape(-1, n_bands))
    print("per-band NMAE:", " ".join(f"b{i}={v:.4e}" for i, v in enumerate(pb)))

    print("\n=== C. model decoded (MEAN reduce) vs truth ===")
    print(f"overall NMAE : {nmae(dec_mean, t):.6e}")
    pb = per_band_nmae(dec_mean.reshape(-1, n_bands), t.reshape(-1, n_bands))
    print("per-band NMAE:", " ".join(f"b{i}={v:.4e}" for i, v in enumerate(pb)))

    # patch non-uniformity of model output
    patch_std = ch0[: 20 * rows_per_geom].float().std(dim=(1, 2)).numpy()
    print(f"\nmodel ch0 patch spatial std: mean={patch_std.mean():.3e} p95={np.percentile(patch_std,95):.3e} "
          f"(true encoding std = 0)")
    print(f"corner vs patch-mean |diff|: mean={np.abs(corner-patch_mean).mean():.3e} "
          f"p95={np.percentile(np.abs(corner-patch_mean),95):.3e}")

    # ---------- D. patch-domain ch0 NMAE (training-style) on same samples ----------
    true_pix_full = ef_uni[g_sl].float().numpy().reshape(n_rows, 32, 32)  # truth patches
    pred_patches = ch0.float().numpy().reshape(n_rows, 32, 32)
    mae_patch = np.abs(pred_patches - true_pix_full).mean()
    denom_patch = np.abs(true_pix_full).mean()
    print("\n=== D. patch-domain ch0 (training metric style) ===")
    print(f"ch0 patch MAE  : {mae_patch:.6e}")
    print(f"ch0 patch NMAE : {mae_patch / (denom_patch + EPS):.6e}  (compare val_loss_ch0=7.66e-03)")
    print(f"log-amplification: exp(100*MAE)-1 = {np.exp(100.0 * mae_patch) - 1.0:.4%} expected freq rel err")

    # ---------- E. best-ranked geometry g360 ----------
    ev_pred_file = INF / "eigenvalues_predictions_full.pt"
    ev_pred = torch.load(ev_pred_file, map_location="cpu", weights_only=True).float().numpy()
    g = 360
    tg, pg = eigen_true[g], ev_pred[g]
    print(f"\n=== E. best-ranked geometry g{g} (from saved eigenvalues_predictions_full) ===")
    print(f"overall NMAE : {nmae(pg, tg):.6e}")
    pb = per_band_nmae(pg, tg)
    print("per-band NMAE:", " ".join(f"b{i}={v:.4e}" for i, v in enumerate(pb)))
    rel = np.abs(pg - tg) / (np.abs(tg) + EPS)
    print("per-band mean rel err:", " ".join(f"b{i}={rel[:, i].mean():.4e}" for i in range(n_bands)))
    print("per-band true mean   :", " ".join(f"b{i}={tg[:, i].mean():8.1f}" for i in range(n_bands)))
    # is saved file pixel or mean decode? compare with our two decodings for g360 if in sample
    if g < N_GEOM_SAMPLE:
        print("g360 in sampled range")
    else:
        sl = slice(g * rows_per_geom, (g + 1) * rows_per_geom)
        ch0g = preds[sl, 0]
        cg = np.exp(100.0 * ch0g[:, 0, 0].double().numpy()).reshape(n_wv, n_bands)
        mg = np.exp(100.0 * ch0g.double().mean(dim=(1, 2)).numpy()).reshape(n_wv, n_bands)
        print(f"saved-vs-pixel-decode max|diff|: {np.abs(pg - cg).max():.4e}")
        print(f"saved-vs-mean-decode  max|diff|: {np.abs(pg - mg).max():.4e}")
        print(f"g{g} NMAE pixel-decode: {nmae(cg, tg):.6e}  mean-decode: {nmae(mg, tg):.6e}")


if __name__ == "__main__":
    main()
