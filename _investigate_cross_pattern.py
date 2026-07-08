"""
Deep-dive: why is mean per-pixel relative error concentrated on the
"crosshair + top/left frame" (rows/cols 0 and 16 of the 32x32 field)?

Phase 1 (this script):
  A. Per-channel per-pixel truth-magnitude stats (mean |t|, fraction of samples
     with |t| ~ 0) on a stratified sample.
  B. Per-channel per-pixel absolute-error stats (is the model actually worse there?)
  C. Per-pixel median vs mean of channel-averaged relative error (tail artifact test).
  D. Wavevector/band breakdown: which (wave, band) combos produce near-zero |t|
     on the cross lines.

Writes an .npz with all maps + prints a compact report.
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
DATASET = "b_test"
PT = ROOT / r"DATASETS\b_test\binarized_2026-03-08_16-34-27_pt"
MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
PRED = ROOT / "INFERENCE" / MODEL / DATASET / f"predictions_I3O5_{MODEL}.pt"
OUT_NPZ = ROOT / "INFERENCE" / "_investigate_cross_b.npz"

EPS = 1e-5
ZERO_THRESH = 1e-4  # float16 displacement scale is ~1e-2, so 1e-4 is "essentially zero"
NAMES = ["eig", "dx_re", "dx_im", "dy_re", "dy_im"]


def main() -> None:
    n_geom, n_wv, n_bands, fh, fw = load_dataset_layout(PT)
    total = n_geom * n_wv * n_bands
    pred = torch.load(PRED, map_location="cpu", mmap=True, weights_only=True)
    sources = open_scoring_sources(PT, total, (fh, fw), True)
    channels = list(DEFAULT_SCORING_CHANNELS)
    C = len(channels)

    # Stratified subset: step 61 is coprime with n_bands=6 and n_wv=325,
    # so it cycles through all wave/band combinations across geometries.
    idx = np.arange(0, total, 61)
    n = 0
    sum_abs_t = np.zeros((C, fh, fw))
    sum_abs_e = np.zeros((C, fh, fw))
    sum_rel = np.zeros((C, fh, fw))
    cnt_zero = np.zeros((C, fh, fw))
    # channel-averaged rel error samples at probe pixels for median test
    probe_pixels = [(0, 0), (0, 8), (16, 16), (16, 8), (8, 8), (24, 24), (8, 16)]
    probe_vals: list[np.ndarray] = []

    # per (wave, band): mean |t| on cross for the imag channels
    cross_mask = np.zeros((fh, fw), bool)
    cross_mask[0, :] = cross_mask[:, 0] = cross_mask[16, :] = cross_mask[:, 16] = True
    wb_sum_t_cross = np.zeros((n_wv, n_bands))
    wb_sum_t_off = np.zeros((n_wv, n_bands))
    wb_cnt = np.zeros((n_wv, n_bands))

    bs = 4096
    for start in range(0, len(idx), bs):
        sl = idx[start : start + bs]
        lo, hi = int(sl[0]), int(sl[-1]) + 1
        t = load_truth_batch(sources, channels, lo, hi).float()[sl - lo]
        p = pred[lo:hi].float()[sl - lo]
        at = t.abs()
        ae = (p - t).abs()
        rel = ae / (at + EPS)

        sum_abs_t += at.sum(0).numpy()
        sum_abs_e += ae.sum(0).numpy()
        sum_rel += rel.sum(0).numpy()
        cnt_zero += (at < ZERO_THRESH).float().sum(0).numpy()

        rel_hw = rel.mean(1).numpy()  # channel-averaged (B, H, W)
        probe_vals.append(np.stack([rel_hw[:, y, x] for (y, x) in probe_pixels], axis=1))

        # imag-channel truth magnitude on/off cross per (wave, band)
        rem = sl % (n_wv * n_bands)
        w_idx = rem // n_bands
        b_idx = rem % n_bands
        imag_t = at[:, [2, 4]].mean(1).numpy()  # (B, H, W)
        tc = imag_t[:, cross_mask].mean(1)
        to = imag_t[:, ~cross_mask].mean(1)
        np.add.at(wb_sum_t_cross, (w_idx, b_idx), tc)
        np.add.at(wb_sum_t_off, (w_idx, b_idx), to)
        np.add.at(wb_cnt, (w_idx, b_idx), 1)
        n += len(sl)

    mean_t = sum_abs_t / n
    mean_e = sum_abs_e / n
    mean_rel = sum_rel / n
    frac_zero = cnt_zero / n
    probe = np.concatenate(probe_vals, axis=0)  # (n, n_probe)
    wb_ratio = wb_sum_t_cross / np.maximum(wb_sum_t_off, 1e-12)

    np.savez(
        OUT_NPZ,
        mean_t=mean_t,
        mean_e=mean_e,
        mean_rel=mean_rel,
        frac_zero=frac_zero,
        probe_pixels=np.array(probe_pixels),
        probe=probe.astype(np.float32),
        wb_ratio=wb_ratio,
        wb_cnt=wb_cnt,
        n=n,
    )

    mask = cross_mask
    print(f"n={n} stratified samples ({DATASET})")
    hdr = f"{'ch':<6} {'|t| cross':>10} {'|t| off':>9} {'|e| cross':>10} {'|e| off':>9} {'rel cross':>10} {'rel off':>8} {'fz cross':>9} {'fz off':>7}"
    print(hdr)
    for c in range(C):
        print(
            f"{NAMES[c]:<6} {mean_t[c][mask].mean():10.5f} {mean_t[c][~mask].mean():9.5f} "
            f"{mean_e[c][mask].mean():10.5f} {mean_e[c][~mask].mean():9.5f} "
            f"{mean_rel[c][mask].mean():10.2f} {mean_rel[c][~mask].mean():8.2f} "
            f"{frac_zero[c][mask].mean():9.3f} {frac_zero[c][~mask].mean():7.3f}"
        )

    print("\nProbe pixels: channel-averaged rel error distribution")
    print(f"{'pixel':<10} {'mean':>8} {'p50':>8} {'p90':>8} {'p99':>9}")
    for j, (y, x) in enumerate(probe_pixels):
        v = probe[:, j]
        print(
            f"({y:2d},{x:2d})   {v.mean():8.3f} {np.median(v):8.3f} "
            f"{np.percentile(v, 90):8.3f} {np.percentile(v, 99):9.3f}"
        )

    print("\nImag-|t| cross/off ratio by band (mean over wavevectors):")
    print("  band:", " ".join(f"{b}:{wb_ratio[:, b].mean():.3f}" for b in range(n_bands)))
    flat = wb_ratio.ravel()
    order = np.argsort(flat)
    print("\n10 most-nodal (wave, band) [lowest cross/off imag |t| ratio]:")
    for k in order[:10]:
        w, b = divmod(int(k), n_bands)
        print(f"  wave={w:3d} band={b} ratio={flat[k]:.4f}")
    print("\n10 least-nodal (wave, band):")
    for k in order[-10:]:
        w, b = divmod(int(k), n_bands)
        print(f"  wave={w:3d} band={b} ratio={flat[k]:.4f}")


if __name__ == "__main__":
    main()
