"""
Phase 2: why do near-zero truth pixels concentrate on rows/cols 0 and 16,
and what is special about pixel (0,0)?

  A. Per-pixel frac-zero maps per channel: print rows/cols profiles.
  B. Pixel (0,0): per-channel |t| distribution (is Im exactly 0 by phase convention?)
  C. Wavevector structure: load wavevectors_full.pt, check which k make
     standing waves with nodes at x=0 / x=a/2 (cols 0/16).
  D. For a Gamma-point-like wavevector vs generic k: mean |Im u| map structure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from per_sample_loss import (
    DEFAULT_SCORING_CHANNELS,
    load_dataset_layout,
    load_truth_batch,
    open_scoring_sources,
)

ROOT = Path(r"d:\Research\NO-2D-Metamaterials")
PT = ROOT / r"DATASETS\b_test\binarized_2026-03-08_16-34-27_pt"
NPZ = ROOT / "INFERENCE" / "_investigate_cross_b.npz"
NAMES = ["eig", "dx_re", "dx_im", "dy_re", "dy_im"]


def profile(map2d: np.ndarray, label: str) -> None:
    rows = map2d.mean(axis=1)
    cols = map2d.mean(axis=0)
    top_rows = np.argsort(rows)[::-1][:4]
    top_cols = np.argsort(cols)[::-1][:4]
    print(f"  {label}: hottest rows {[(int(r), round(float(rows[r]), 3)) for r in top_rows]}")
    print(f"  {label}: hottest cols {[(int(c), round(float(cols[c]), 3)) for c in top_cols]}")


def main() -> None:
    d = np.load(NPZ)
    frac_zero = d["frac_zero"]
    mean_rel = d["mean_rel"]

    print("=== A. frac-zero row/col profiles per channel ===")
    for c in range(5):
        profile(frac_zero[c], f"fz[{NAMES[c]}]")
    print()
    for c in [2, 4]:
        fz = frac_zero[c]
        print(
            f"fz[{NAMES[c]}]  (0,0)={fz[0, 0]:.3f}  (0,16)={fz[0, 16]:.3f}  "
            f"(16,0)={fz[16, 0]:.3f}  (16,16)={fz[16, 16]:.3f}  interior(8,8)={fz[8, 8]:.3f}"
        )

    print("\n=== B. pixel (0,0) truth values per channel (5k stratified samples) ===")
    n_geom, n_wv, n_bands, fh, fw = load_dataset_layout(PT)
    total = n_geom * n_wv * n_bands
    sources = open_scoring_sources(PT, total, (fh, fw), True)
    channels = list(DEFAULT_SCORING_CHANNELS)
    idx = np.arange(0, total, 389)  # ~5000 samples
    vals = {c: [] for c in range(5)}
    corner = {c: [] for c in range(5)}
    center = {c: [] for c in range(5)}
    bs = 2048
    for start in range(0, len(idx), bs):
        sl = idx[start : start + bs]
        lo, hi = int(sl[0]), int(sl[-1]) + 1
        t = load_truth_batch(sources, channels, lo, hi).float()[sl - lo]
        for c in range(5):
            corner[c].append(t[:, c, 0, 0].numpy())
            center[c].append(t[:, c, 16, 16].numpy())
    for c in range(5):
        cv = np.concatenate(corner[c])
        ce = np.concatenate(center[c])
        print(
            f"{NAMES[c]:<6} (0,0): frac|t|<1e-6={np.mean(np.abs(cv) < 1e-6):.3f} "
            f"frac<1e-4={np.mean(np.abs(cv) < 1e-4):.3f} mean|t|={np.abs(cv).mean():.5f} | "
            f"(16,16): frac<1e-6={np.mean(np.abs(ce) < 1e-6):.3f} "
            f"frac<1e-4={np.mean(np.abs(ce) < 1e-4):.3f} mean|t|={np.abs(ce).mean():.5f}"
        )

    print("\n=== C. wavevector structure ===")
    wv = torch.load(PT / "wavevectors_full.pt", map_location="cpu", weights_only=True)
    wv = wv.float().numpy()
    print("wavevectors shape:", wv.shape)
    w = wv.reshape(-1, wv.shape[-1]) if wv.ndim > 2 else wv
    print("first 10:", np.round(w[:10], 4))
    print("kx range:", w[:, 0].min(), w[:, 0].max(), " ky range:", w[:, 1].min(), w[:, 1].max())
    # count exact-zero / high-symmetry components
    for name, arr in [("kx", w[:, 0]), ("ky", w[:, 1])]:
        uniq = np.unique(np.round(arr, 6))
        print(f"{name}: n_unique={len(uniq)} min={uniq[0]:.4f} max={uniq[-1]:.4f}")
        print(f"   frac {name}==0: {np.mean(arr == 0):.3f}   frac {name}==max: {np.mean(arr == arr.max()):.3f}")

    print("\n=== D. mean |Im ux| map for Gamma-like vs generic wavevector (geom 0..40) ===")
    # choose wave index with k ~ 0 and one generic
    knorm = np.linalg.norm(w[:325], axis=1)
    gamma_w = int(np.argmin(knorm))
    generic_w = int(np.argsort(knorm)[len(knorm) // 2])
    print(f"gamma-like wave={gamma_w} k={w[gamma_w]}, generic wave={generic_w} k={w[generic_w]}")
    disp_idx_gamma = []
    disp_idx_generic = []
    for g in range(40):
        for b in range(n_bands):
            disp_idx_gamma.append(g * (n_wv * n_bands) + gamma_w * n_bands + b)
            disp_idx_generic.append(g * (n_wv * n_bands) + generic_w * n_bands + b)
    for label, indices in [("gamma", disp_idx_gamma), ("generic", indices_g := disp_idx_generic)]:
        acc = np.zeros((fh, fw))
        for i in indices:
            t = load_truth_batch(sources, channels, i, i + 1).float()[0]
            acc += t[2].abs().numpy()
        acc /= len(indices)
        mask = np.zeros((fh, fw), bool)
        mask[0, :] = mask[:, 0] = mask[16, :] = mask[:, 16] = True
        print(
            f"{label}: |Im ux| cross={acc[mask].mean():.6f} off={acc[~mask].mean():.6f} "
            f"(0,0)={acc[0, 0]:.6f} (16,16)={acc[16, 16]:.6f} row0={acc[0, :].mean():.6f} "
            f"col0={acc[:, 0].mean():.6f} row16={acc[16, :].mean():.6f} col16={acc[:, 16].mean():.6f}"
        )


if __name__ == "__main__":
    main()
