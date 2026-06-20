"""Utilities for second-peak detection and wave/band correlation tables."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2_contingency


def flat_indices(n_geom: int, n_wv: int, n_bands: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (geom, wave, band) index arrays for C-order flat layout."""
    total = n_geom * n_wv * n_bands
    combined = np.arange(total, dtype=np.int64)
    per_geom = n_wv * n_bands
    geom = combined // per_geom
    wave = (combined % per_geom) // n_bands
    band = combined % n_bands
    return geom, wave, band


def find_two_peaks(log_vals: np.ndarray, n_bins: int = 400, smooth_sigma: float = 2.0) -> dict | None:
    """Return split point and peak locations in log10 space, or None."""
    lo, hi = float(log_vals.min()), float(log_vals.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(log_vals, bins=edges)
    sm = gaussian_filter1d(hist.astype(float), smooth_sigma)

    peaks: list[tuple[int, float, float]] = []
    for i in range(1, len(sm) - 1):
        if sm[i] > sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > sm.max() * 0.02:
            peaks.append((i, sm[i], centers[i]))
    peaks.sort(key=lambda t: t[1], reverse=True)
    if len(peaks) < 2:
        return None

    top2 = sorted(peaks[:2], key=lambda t: t[2])
    i_lo, _, log_lo = top2[0]
    i_hi, _, log_hi = top2[1]
    valley_slice = sm[i_lo : i_hi + 1]
    valley_i = i_lo + int(np.argmin(valley_slice))
    return {
        "log_split": float(centers[valley_i]),
        "split_value": float(10 ** centers[valley_i]),
        "peak_logs": [float(log_lo), float(log_hi)],
        "valley_log": float(centers[valley_i]),
    }


def second_peak_mask(loss: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """Boolean mask True where sample is in the high-loss (second) peak."""
    finite = np.isfinite(loss) & (loss > 0)
    pos = loss[finite]
    if pos.size < 2:
        return np.zeros_like(loss, dtype=bool), None
    info = find_two_peaks(np.log10(pos))
    if info is None:
        return np.zeros_like(loss, dtype=bool), None
    split = info["split_value"]
    return finite & (loss > split), info


def band_table(
    band: np.ndarray,
    second: np.ndarray,
    n_bands: int,
    reference_loss: str = "nmae",
) -> list[dict]:
    """Per-band second-peak rates and enrichment vs uniform 1/n_bands."""
    rows = []
    pop_rate = float(second.mean())
    uniform = 1.0 / n_bands
    for b in range(n_bands):
        mask = band == b
        n = int(mask.sum())
        n_sec = int((second & mask).sum())
        rate = n_sec / max(n, 1)
        rows.append(
            {
                "band": b,
                "n_total": n,
                "n_second": n_sec,
                "frac_second_pct": 100.0 * rate,
                "uniform_pct": 100.0 * uniform,
                "enrichment": rate / max(uniform, 1e-12),
                "share_of_second_peak_pct": 100.0 * n_sec / max(second.sum(), 1),
            }
        )
    tab = np.array([[r["n_second"], r["n_total"] - r["n_second"]] for r in rows])
    chi2, pval, _, _ = chi2_contingency(tab + 1e-9)
    for r in rows:
        r["chi2_pooled_p"] = pval
    return rows


def wave_table(
    wave: np.ndarray,
    second: np.ndarray,
    kxy: np.ndarray,
    n_geom: int,
    n_bands: int,
    top_n: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Full per-wave stats and top-N rows by second-peak rate (for markdown table)."""
    n_wv = kxy.shape[0]
    pop_rate = float(second.mean())
    full: list[dict] = []
    for w in range(n_wv):
        mask = wave == w
        n = int(mask.sum())
        n_sec = int((second & mask).sum())
        rate = n_sec / max(n, 1)
        kx, ky = float(kxy[w, 0]), float(kxy[w, 1])
        full.append(
            {
                "wave": w,
                "kx": kx,
                "ky": ky,
                "n_total": n,
                "n_second": n_sec,
                "frac_second_pct": 100.0 * rate,
                "enrichment_vs_pop": rate / max(pop_rate, 1e-12),
                "share_of_second_peak_pct": 100.0 * n_sec / max(second.sum(), 1),
            }
        )
    ranked = sorted(full, key=lambda r: r["frac_second_pct"], reverse=True)
    return full, ranked[:top_n]
