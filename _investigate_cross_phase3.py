"""
Phase 3: quantify the symmetry mechanism.

Wavevector grid ('none' IBZ): kx in linspace(-pi/a, pi/a, 25), ky in linspace(0, pi/a, 13).
Classes:
  both  : kx in {0, +-pi/a} AND ky in {0, pi/a}  -> real matrices, modes real standing waves
  x-mir : kx in {0, +-pi/a} only                 -> mirror x->-x commutes; nodal lines cols 0/16
  y-mir : ky in {0, pi/a} only                   -> mirror y->-y commutes; nodal lines rows 0/16
  gen   : generic k                              -> no exact nodal-line constraint

For each class, on a stratified b_test subset, compute on-cross vs off-cross:
  frac(|t|<1e-4) and mean rel error (channel-averaged, disp channels only),
  plus mean |t| of imag channels to show real-mode collapse.
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
MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
PRED = ROOT / "INFERENCE" / MODEL / "b_test" / f"predictions_I3O5_{MODEL}.pt"
EPS = 1e-5


def main() -> None:
    n_geom, n_wv, n_bands, fh, fw = load_dataset_layout(PT)
    total = n_geom * n_wv * n_bands
    pred = torch.load(PRED, map_location="cpu", mmap=True, weights_only=True)
    sources = open_scoring_sources(PT, total, (fh, fw), True)
    channels = list(DEFAULT_SCORING_CHANNELS)

    wv = torch.load(PT / "wavevectors_full.pt", map_location="cpu", weights_only=True).float().numpy()
    k = wv[0]  # same grid for all geometries: (325, 2)
    kmax = np.abs(k).max()
    x_inv = np.isclose(np.abs(k[:, 0]), 0) | np.isclose(np.abs(k[:, 0]), kmax)
    y_inv = np.isclose(np.abs(k[:, 1]), 0) | np.isclose(np.abs(k[:, 1]), kmax)
    kclass = np.where(x_inv & y_inv, 0, np.where(x_inv, 1, np.where(y_inv, 2, 3)))
    names = {0: "both-mirror", 1: "x-mirror", 2: "y-mirror", 3: "generic"}
    for c in range(4):
        print(f"{names[c]:<12} n_k={np.sum(kclass == c):3d} / 325")

    cross = np.zeros((fh, fw), bool)
    cross[0, :] = cross[:, 0] = cross[16, :] = cross[:, 16] = True

    # accumulators per class: [rel_cross, rel_off, fz_cross, fz_off, imagt_all, n]
    acc = {c: np.zeros(5) for c in range(4)}
    cnt = {c: 0 for c in range(4)}

    idx = np.arange(0, total, 61)
    bs = 4096
    for start in range(0, len(idx), bs):
        sl = idx[start : start + bs]
        lo, hi = int(sl[0]), int(sl[-1]) + 1
        t = load_truth_batch(sources, channels, lo, hi).float()[sl - lo]
        p = pred[lo:hi].float()[sl - lo]
        at = t.abs()
        rel = ((p - t).abs() / (at + EPS))[:, 1:].mean(1).numpy()  # disp channels only
        fz = (at[:, 1:] < 1e-4).float().mean(1).numpy()
        imag_t = at[:, [2, 4]].mean(1).numpy()

        rem = sl % (n_wv * n_bands)
        w_idx = rem // n_bands
        kc = kclass[w_idx]
        for c in range(4):
            m = kc == c
            if not m.any():
                continue
            acc[c][0] += rel[m][:, cross].mean(1).sum()
            acc[c][1] += rel[m][:, ~cross].mean(1).sum()
            acc[c][2] += fz[m][:, cross].mean(1).sum()
            acc[c][3] += fz[m][:, ~cross].mean(1).sum()
            acc[c][4] += imag_t[m].mean(axis=(1, 2)).sum()
            cnt[c] += int(m.sum())

    print()
    print(f"{'class':<12} {'n':>6} {'rel cross':>10} {'rel off':>9} {'ratio':>6} {'fz cross':>9} {'fz off':>7} {'mean|Im t|':>11}")
    for c in range(4):
        n = max(cnt[c], 1)
        a = acc[c] / n
        print(
            f"{names[c]:<12} {cnt[c]:>6} {a[0]:>10.2f} {a[1]:>9.2f} {a[0] / max(a[1], 1e-9):>6.2f} "
            f"{a[2]:>9.3f} {a[3]:>7.3f} {a[4]:>11.6f}"
        )

    # contribution decomposition: what share of total on-cross rel error comes from each class?
    tot_cross = sum(acc[c][0] for c in range(4))
    print("\nShare of summed on-cross rel error by class:")
    for c in range(4):
        print(f"  {names[c]:<12} {acc[c][0] / tot_cross:6.1%}  (population share {cnt[c] / sum(cnt.values()):6.1%})")


if __name__ == "__main__":
    main()
