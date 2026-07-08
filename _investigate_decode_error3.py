"""Compare decoder variants for the uniform ln(s)/100 encoding on c_test.

Variants (all evaluated against eigenvalue_data_full truth, 200 geometries):
  1. corner pixel, float16 cast (current pipeline: NU.decode path)
  2. corner pixel, float32
  3. patch mean, float32
  4. patch median, float32
  5. trimmed mean (drop 10% tails), float32
  6. patch mean, float64
Also reports the float16 storage floor (decode of TRUE patches).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\Research\NO-2D-Metamaterials")
PT = ROOT / r"DATASETS\c_test\continuous_2026-03-05_20-07-34_pt"
MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
INF = ROOT / "INFERENCE" / MODEL / "c_test"

N_GEOM = 200
EPS = 1e-5


def nmae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean() / (np.abs(true).mean() + EPS))


def per_band(pred: np.ndarray, true: np.ndarray, n_bands: int) -> str:
    p = pred.reshape(-1, n_bands)
    t = true.reshape(-1, n_bands)
    vals = np.abs(p - t).mean(axis=0) / (np.abs(t).mean(axis=0) + EPS)
    return " ".join(f"b{i}={v:.4e}" for i, v in enumerate(vals))


def main() -> None:
    eigen_true = torch.load(PT / "eigenvalue_data_full.pt", map_location="cpu", weights_only=True).float().numpy()
    n_geom, n_wv, n_bands = eigen_true.shape
    t = eigen_true[:N_GEOM]

    preds = torch.load(INF / f"predictions_I3O5_{MODEL}.pt", map_location="cpu", mmap=True, weights_only=True)
    rows_per_geom = n_wv * n_bands
    ch0 = preds[: N_GEOM * rows_per_geom, 0]  # (rows, 32, 32) float16 mmap

    flat = ch0.reshape(ch0.shape[0], -1)

    def decode(pix: np.ndarray) -> np.ndarray:
        return np.exp(100.0 * pix.astype(np.float64)).reshape(N_GEOM, n_wv, n_bands)

    variants: dict[str, np.ndarray] = {}

    # 1. corner pixel with float16 cast (mimics NU.decode_eigenfrequency_uniform)
    c16 = ch0[:, 0, 0].numpy()  # already float16
    ln16 = (c16 * np.float16(100.0)).astype(np.float16)
    variants["corner px, f16 (pipeline)"] = np.exp(ln16).astype(np.float64).reshape(N_GEOM, n_wv, n_bands)

    # 2. corner pixel, float32 math
    variants["corner px, f32"] = decode(ch0[:, 0, 0].float().numpy())

    # 3. patch mean, float32
    variants["patch mean, f32"] = decode(flat.float().mean(dim=1).numpy())

    # 4. patch median, float32
    variants["patch median, f32"] = decode(flat.float().median(dim=1).values.numpy())

    # 5. trimmed mean (drop lowest/highest 10% of 1024 px)
    sorted_px = flat.float().sort(dim=1).values
    k = int(0.10 * sorted_px.shape[1])
    variants["trimmed mean 10%, f32"] = decode(sorted_px[:, k:-k].mean(dim=1).numpy())

    # 6. patch mean computed in float64
    variants["patch mean, f64"] = decode(flat.double().mean(dim=1).numpy())

    print(f"c_test, {N_GEOM} geometries; NMAE vs eigenvalue_data_full\n")
    for name, dec in variants.items():
        print(f"{name:28s} overall={nmae(dec, t):.6e}   {per_band(dec, t, n_bands)}")

    # storage floor: decode TRUE float16 patches
    ef_uni = torch.load(PT / "eigenfrequency_uniform_full.pt", map_location="cpu", mmap=True, weights_only=True)
    true_pix = ef_uni[:N_GEOM, :, :, 0, 0].float().numpy()
    floor = np.exp(100.0 * true_pix.astype(np.float64))
    print(f"\n{'float16 storage floor':28s} overall={nmae(floor, t):.6e}   {per_band(floor, t, n_bands)}")

    # theoretical float16 quantization: ulp near p~0.07 is 2^-14
    ulp = 2.0 ** -14
    print(f"\nfloat16 ulp near p=0.07: {ulp:.2e} -> expected rel err ~100*ulp/4 = {100*ulp/4:.4%}")


if __name__ == "__main__":
    main()
