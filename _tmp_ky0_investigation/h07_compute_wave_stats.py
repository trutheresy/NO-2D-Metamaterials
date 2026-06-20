"""Compute per-wave / per-channel NMAE stats from existing predictions (read-only)."""

from __future__ import annotations

import sys

import numpy as np
import torch
from tqdm import tqdm

from common import CACHE, DATASETS, NMAE_EPS, PRED, classify_waves, flat_indices, load_kxy, save_json
from per_sample_loss import prepare_scoring_data, resolve_device
from second_peak_analysis import second_peak_mask

BATCH = 8192
CHANNELS = [0, 1, 2, 3, 4]
CH_NAMES = ["eig", "ux_r", "ux_i", "uy_r", "uy_i"]


def per_channel_nmae_batches(truth, pred, device, batch_size):
    """Return (n, 5) per-channel NMAE."""
    n = truth.shape[0]
    out = np.empty((n, 5), dtype=np.float64)
    for start in tqdm(range(0, n, batch_size), desc="Per-ch NMAE", unit="batch"):
        end = min(start + batch_size, n)
        t = truth[start:end].to(device, dtype=torch.float32)
        p = pred[start:end].to(device, dtype=torch.float32)
        err = (p - t).abs().mean(dim=(2, 3))
        denom = t.abs().mean(dim=(2, 3)) + NMAE_EPS
        out[start:end] = (err / denom).double().cpu().numpy()
    return out


def group_mean(vals: np.ndarray, wave: np.ndarray, idxs: np.ndarray) -> dict[int, float]:
    return {int(w): float(vals[wave == w].mean()) for w in idxs}


def summarize_by_wave(nmae_group: np.ndarray, wave: np.ndarray, bands: np.ndarray, idxs: np.ndarray):
    rows = []
    for w in idxs:
        m = wave == w
        rows.append(
            {
                "wave": int(w),
                "n": int(m.sum()),
                "nmae_mean": float(nmae_group[m].mean()),
                "nmae_median": float(np.median(nmae_group[m])),
                "second_peak_pct": float(100.0 * m.sum() / max(m.sum(), 1)),  # placeholder
                "by_band": {
                    int(b): float(nmae_group[m & (bands == b)].mean())
                    for b in range(int(bands.max()) + 1)
                },
            }
        )
    return rows


def run_tag(tag: str) -> dict:
    cfg = DATASETS[tag]
    pt_dir, inf_dir = cfg["pt_dir"], cfg["inf_dir"]
    pred_path = inf_dir / PRED
    device = resolve_device("auto")
    predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
    truth_flat, n_geom, n_wv, n_bands, *_ = prepare_scoring_data(pt_dir, predictions, CHANNELS)

    ch_nmae = per_channel_nmae_batches(truth_flat, predictions[:, CHANNELS], device, BATCH)
    # combined group weighting
    disp_mean = ch_nmae[:, 1:].mean(axis=1)
    group_nmae = 0.5 * ch_nmae[:, 0] + 0.5 * disp_mean

    geom, wave, band = flat_indices(n_geom, n_wv, n_bands)
    second, peak_info = second_peak_mask(group_nmae)

    kxy = load_kxy(pt_dir)
    groups = classify_waves(kxy)
    groups["kx0_only"] = np.setdiff1d(groups["kx0"], groups["ky0"])

    def agg(idxs: np.ndarray) -> dict:
        m = np.isin(wave, idxs)
        if not m.any():
            return {"n_samples": 0}
        return {
            "n_samples": int(m.sum()),
            "n_waves": int(len(idxs)),
            "group_nmae_mean": float(group_nmae[m].mean()),
            "group_nmae_median": float(np.median(group_nmae[m])),
            "second_peak_pct": float(100.0 * second[m].mean()),
            "ch0_nmae_mean": float(ch_nmae[m, 0].mean()),
            "disp_nmae_mean": float(disp_mean[m].mean()),
            "ch_means": {CH_NAMES[i]: float(ch_nmae[m, i].mean()) for i in range(5)},
        }

    per_wave = []
    for w in range(n_wv):
        m = wave == w
        per_wave.append(
            {
                "wave": w,
                "kx": float(kxy[w, 0]),
                "ky": float(kxy[w, 1]),
                "n": int(m.sum()),
                "group_nmae": float(group_nmae[m].mean()),
                "second_peak_pct": float(100.0 * second[m].mean()),
                "ch0": float(ch_nmae[m, 0].mean()),
                "disp": float(disp_mean[m].mean()),
            }
        )

    by_group_band = {}
    for gname, idxs in groups.items():
        m_wave = np.isin(wave, idxs)
        by_group_band[gname] = {}
        for b in range(n_bands):
            m = m_wave & (band == b)
            by_group_band[gname][str(b)] = {
                "n": int(m.sum()),
                "nmae_mean": float(group_nmae[m].mean()) if m.any() else None,
            }

    # H13: geometry-wide ky=0 failure rate
    ky0_set = groups["ky0"]
    geom_ky0 = []
    for g in range(n_geom):
        m = (geom == g) & np.isin(wave, ky0_set)
        geom_ky0.append(
            {
                "geom": int(g),
                "ky0_second_peak_pct": float(100.0 * second[m].mean()),
                "ky0_nmae_mean": float(group_nmae[m].mean()),
            }
        )
    geom_ky0_sorted = sorted(geom_ky0, key=lambda r: r["ky0_second_peak_pct"], reverse=True)
    geom_ky0_summary = {
        "mean_ky0_second_pct": float(np.mean([r["ky0_second_peak_pct"] for r in geom_ky0])),
        "max_ky0_second_pct": float(max(r["ky0_second_peak_pct"] for r in geom_ky0)),
        "n_geoms_above_80pct_ky0_second": int(sum(1 for r in geom_ky0 if r["ky0_second_peak_pct"] > 80)),
        "top5_geoms": geom_ky0_sorted[:5],
    }

    return {
        "tag": tag,
        "peak_info": peak_info,
        "global_second_peak_pct": float(100.0 * second.mean()),
        "groups": {k: agg(v) for k, v in groups.items()},
        "by_group_band": by_group_band,
        "geom_ky0_summary": geom_ky0_summary,
        "per_wave": per_wave,
    }


def main() -> None:
    for tag in sys.argv[1:] or list(DATASETS):
        print(f"\n=== {tag} ===")
        res = run_tag(tag)
        out = CACHE / f"wave_stats_{tag}.json"
        save_json(out, res)
        print(f"Wrote {out}")
        for gname, g in res["groups"].items():
            if g.get("n_samples"):
                print(
                    f"  {gname:10s}  n={g['n_samples']:7d}  nmae={g['group_nmae_mean']:.4f}  "
                    f"2nd%={g['second_peak_pct']:.1f}  ch0={g['ch0_nmae_mean']:.4f}  "
                    f"disp={g['disp_nmae_mean']:.4f}"
                )


if __name__ == "__main__":
    main()
