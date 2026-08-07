"""
Phase 4 (final): counterfactual tests.

  A. Masked relative error: exclude pixels with |t| < 1e-4 from the per-pixel mean.
     If the crosshair vanishes, the pattern is a denominator artifact, not model failure.
  B. Absolute-error map restricted to high-symmetry wavevectors: does the model
     genuinely do worse on the mirror lines in |e| terms?
  C. Corner pixel: rel error at (0,0) for dx_im (truth is exactly 0 there by phase
     convention) - show it equals |prediction|/eps.
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
ZT = 1e-4


def main() -> None:
    n_geom, n_wv, n_bands, fh, fw = load_dataset_layout(PT)
    total = n_geom * n_wv * n_bands
    pred = torch.load(PRED, map_location="cpu", mmap=True, weights_only=True)
    sources = open_scoring_sources(PT, total, (fh, fw), True)
    channels = list(DEFAULT_SCORING_CHANNELS)

    wv = torch.load(PT / "wavevectors_full.pt", map_location="cpu", weights_only=True).float().numpy()
    k = wv[0]
    kmax = np.abs(k).max()
    x_inv = np.isclose(np.abs(k[:, 0]), 0) | np.isclose(np.abs(k[:, 0]), kmax)
    y_inv = np.isclose(np.abs(k[:, 1]), 0) | np.isclose(np.abs(k[:, 1]), kmax)
    hi_sym = x_inv | y_inv

    cross = np.zeros((fh, fw), bool)
    cross[0, :] = cross[:, 0] = cross[16, :] = cross[:, 16] = True

    rel_sum = np.zeros((fh, fw))
    rel_cnt = np.zeros((fh, fw))
    abs_hi_sum = np.zeros((fh, fw))
    abs_gen_sum = np.zeros((fh, fw))
    n_hi = 0
    n_gen = 0
    corner_pred = []
    corner_rel = []

    idx = np.arange(0, total, 61)
    bs = 4096
    for start in range(0, len(idx), bs):
        sl = idx[start : start + bs]
        lo, hi = int(sl[0]), int(sl[-1]) + 1
        t = load_truth_batch(sources, channels, lo, hi).float()[sl - lo]
        p = pred[lo:hi].float()[sl - lo]
        at = t.abs()
        ae = (p - t).abs()

        # A. masked rel error over disp channels
        rel = (ae / (at + EPS))[:, 1:]
        valid = (at[:, 1:] >= ZT)
        rel_sum += (rel * valid).sum(dim=(0, 1)).numpy()
        rel_cnt += valid.sum(dim=(0, 1)).numpy()

        # B. abs error split by k symmetry class (disp channels)
        rem = sl % (n_wv * n_bands)
        w_idx = rem // n_bands
        hs = hi_sym[w_idx]
        ae_disp = ae[:, 1:].mean(1).numpy()
        abs_hi_sum += ae_disp[hs].sum(0)
        abs_gen_sum += ae_disp[~hs].sum(0)
        n_hi += int(hs.sum())
        n_gen += int((~hs).sum())

        # C. corner dx_im
        corner_pred.append(p[:, 2, 0, 0].numpy())
        corner_rel.append((ae[:, 2, 0, 0] / (at[:, 2, 0, 0] + EPS)).numpy())

    masked_rel = rel_sum / np.maximum(rel_cnt, 1)
    abs_hi = abs_hi_sum / n_hi
    abs_gen = abs_gen_sum / n_gen

    print("=== A. masked mean rel error (|t| >= 1e-4 only), disp channels ===")
    print(f"cross {masked_rel[cross].mean():.4f}   off {masked_rel[~cross].mean():.4f}   ratio {masked_rel[cross].mean() / masked_rel[~cross].mean():.2f}")
    print(f"corner(0,0) {masked_rel[0, 0]:.4f}  center(16,16) {masked_rel[16, 16]:.4f}  interior(8,8) {masked_rel[8, 8]:.4f}")
    rows = masked_rel.mean(1)
    print("hottest rows:", [(int(r), round(float(rows[r]), 3)) for r in np.argsort(rows)[::-1][:4]])

    print("\n=== B. abs error |e| (disp ch mean), high-symmetry vs generic k ===")
    print(f"high-sym  (n={n_hi}): cross {abs_hi[cross].mean():.5f} off {abs_hi[~cross].mean():.5f} ratio {abs_hi[cross].mean() / abs_hi[~cross].mean():.2f}")
    print(f"generic   (n={n_gen}): cross {abs_gen[cross].mean():.5f} off {abs_gen[~cross].mean():.5f} ratio {abs_gen[cross].mean() / abs_gen[~cross].mean():.2f}")

    cp = np.concatenate(corner_pred)
    cr = np.concatenate(corner_rel)
    print("\n=== C. corner (0,0) dx_im: truth==0 always ===")
    print(f"mean |pred| = {np.abs(cp).mean():.2e}   (model emits small nonzero values)")
    print(f"mean rel = {cr.mean():.1f}  ~= mean|pred|/eps = {np.abs(cp).mean() / EPS:.1f}")


if __name__ == "__main__":
    main()
