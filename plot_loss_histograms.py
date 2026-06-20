"""
Plot per-sample loss histograms (100 bins) with a fitted density curve overlaid, one
figure per loss criterion.

Consumes the same inputs as ``per_sample_loss.py``:
--dataset-pt-dir : dataset *_pt folder with ``displacements_dataset.pt`` (truth) and
                   ``eigenfrequency_uniform_full.pt`` (layout).
--inference      : dense prediction tensor from ``run_model_inference.py`` with shape
                   ``(n_geom*n_wv*n_bands, out_channels, H, W)``.

By default the per-sample loss uses channels 0–4 with **group weighting**:
50% eigenfrequency (ch0) + 50% mean of displacement channels 1–4 (not 1/5 per channel).
For each requested loss it computes that scalar and
renders a histogram with a Gaussian-KDE curve scaled to match relative bin heights
when normalization is enabled (each sample weighted by 1/N, with N the full per-loss
sample count before excluding non-positive or out-of-range values for plotting).
The y-axis is autoscaled. By default the x-axis is log-scaled with a major tick at
every order of magnitude in the data range.

Supported losses: mae, mse, nmae, nmse, nrms (aliases l1, l2, nl1, nl2, nrmse). NMAE/NMSE/NRMS use the
eps-stabilized denominators from per_sample_loss.py (defaults 1e-5 / 1e-5).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator
from scipy.stats import gaussian_kde

from per_sample_loss import (
    compute_per_sample_losses,
    normalize_channel_weighting,
    normalize_loss_name,
    parse_channels,
    parse_index_list,
    prepare_scoring_data,
    resolve_device,
)
from second_peak_analysis import flat_indices


LOSS_XLABEL = {
    "mae": "Per-sample MAE",
    "mse": "Per-sample MSE",
    "nmae": "Per-sample NMAE",
    "nmse": "Per-sample NMSE",
    "nrms": "Per-sample NRMS",
}


def fit_curve(
    values: np.ndarray,
    bin_edges: np.ndarray,
    n_samples_for_kde: int,
    seed: int,
    normalize: bool = True,
    normalize_denom: int | None = None,
):
    """Return (x, y) of a Gaussian-KDE density scaled to histogram height, or None."""
    finite = values[np.isfinite(values)]
    if finite.size < 2 or float(finite.std()) == 0.0:
        return None
    rng = np.random.default_rng(seed)
    sample = finite if finite.size <= n_samples_for_kde else rng.choice(finite, n_samples_for_kde, replace=False)
    try:
        kde = gaussian_kde(sample)
    except np.linalg.LinAlgError:
        return None
    bin_width = float(bin_edges[1] - bin_edges[0])
    xs = np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 512)
    denom = float(normalize_denom if normalize_denom is not None else finite.size) if normalize else 1.0
    ys = kde(xs) * finite.size * bin_width / denom
    return xs, ys


def fit_curve_logx(
    values: np.ndarray,
    bin_edges: np.ndarray,
    n_samples_for_kde: int,
    seed: int,
    normalize: bool = True,
    normalize_denom: int | None = None,
):
    """Return (x, y) of a log-space KDE scaled to log-bin height, or None."""
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0.0]
    if finite.size < 2:
        return None
    logv = np.log10(finite)
    if float(logv.std()) == 0.0:
        return None
    rng = np.random.default_rng(seed)
    sample = logv if logv.size <= n_samples_for_kde else rng.choice(logv, n_samples_for_kde, replace=False)
    try:
        kde = gaussian_kde(sample)
    except np.linalg.LinAlgError:
        return None
    log_edges = np.log10(bin_edges)
    dlog = float(log_edges[1] - log_edges[0])
    us = np.linspace(float(log_edges[0]), float(log_edges[-1]), 512)
    denom = float(normalize_denom if normalize_denom is not None else finite.size) if normalize else 1.0
    ys = kde(us) * finite.size * dlog / denom
    return np.power(10.0, us), ys


def sample_weights(
    values: np.ndarray,
    normalize: bool,
    normalize_denom: int | None = None,
) -> np.ndarray | None:
    """Per-sample histogram weights; None means raw counts."""
    if not normalize:
        return None
    denom = normalize_denom if normalize_denom is not None else values.size
    return np.full(values.shape, 1.0 / denom)


def log_decade_limits(lo: float, hi: float) -> tuple[float, float]:
    """Expand (lo, hi) outward to the nearest enclosing power-of-10 decades."""
    xmin = 10.0 ** math.floor(math.log10(lo))
    xmax = 10.0 ** math.ceil(math.log10(hi))
    if xmax <= xmin:
        xmax = xmin * 10.0
    return xmin, xmax


def set_log_decade_ticks(ax, xmin: float, xmax: float) -> None:
    """Place a major tick at every integer power of 10 between xmin and xmax."""
    exp_lo = int(math.floor(math.log10(xmin)))
    exp_hi = int(math.ceil(math.log10(xmax)))
    tick_vals = np.power(10.0, np.arange(exp_lo, exp_hi + 1, dtype=float))
    ax.set_xlim(xmin, xmax)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(tick_vals))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(NullLocator())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-pt-dir", required=True, help="Dataset *_pt folder with displacements_dataset.pt.")
    p.add_argument("--inference", required=True, help="Dense prediction tensor (.pt), shape (n_geom*n_wv*n_bands, C, H, W).")
    p.add_argument("--losses", nargs="+", required=True, help="Loss criteria: mae mse nmae nmse nrms (aliases l1, l2, nl1, nl2, nrmse).")
    p.add_argument("--channels", default="0,1,2,3,4", help="Comma-separated prediction channels (default: 0,1,2,3,4).")
    p.add_argument(
        "--channel-weighting",
        default="group",
        choices=("uniform", "group"),
        help="How to combine per-channel losses (default: group = 50%% ch0 + 50%% mean(ch1-4)).",
    )
    p.add_argument("--title", default="", help="Optional figure title (blank by default).")
    p.add_argument("--output-dir", default="", help="Folder for the histogram PNGs (default: inference file's folder).")
    p.add_argument("--out-prefix", default="loss_histogram", help="Output filename prefix.")
    p.add_argument("--tag", default="", help="Optional tag in filenames (e.g. dataset name).")
    p.add_argument("--bins", type=int, default=200, help="Number of histogram bins (default 200).")
    p.add_argument("--clip-percentile", type=float, default=100.0,
                   help="Clip the x-range at this upper percentile of the loss to keep heavy tails readable (default 100 = no clip).")
    p.add_argument("--linear-x", action="store_true",
                   help="Use linear-spaced bins and a linear x-axis. By default the x-axis is log-scaled "
                        "(log-spaced bins, KDE fit in log10 space) since losses span many orders of magnitude; "
                        "in log mode only strictly-positive losses are shown.")
    p.add_argument("--raw-count", action="store_true",
                   help="Use raw sample counts on the y-axis instead of relative count (1/N weights).")
    p.add_argument("--no-curve", action="store_true", help="Disable the fitted density curve overlay.")
    p.add_argument("--kde-samples", type=int, default=200000, help="Max samples used to fit the KDE curve (default 200000).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nmae-eps", type=float, default=1e-5, help="Epsilon added to mean(|t|) denominator for nmae (default 1e-5).")
    p.add_argument("--nmse-eps", type=float, default=1e-5, help="Epsilon added to mean(t^2) denominator for nmse (default 1e-5).")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument(
        "--exclude-wave-indices",
        default="",
        help="Comma-separated wavevector indices to omit from histograms (all bands/geometries).",
    )
    args = p.parse_args()
    log_x = not args.linear_x
    normalize = not args.raw_count

    losses = []
    for name in args.losses:
        ln = normalize_loss_name(name)
        if ln not in losses:
            losses.append(ln)

    channels = parse_channels(args.channels)
    channel_weighting = normalize_channel_weighting(args.channel_weighting)
    exclude_waves = parse_index_list(args.exclude_wave_indices, "--exclude-wave-indices")
    device = resolve_device(args.device)
    dataset_pt_dir = Path(args.dataset_pt_dir)
    infer_path = Path(args.inference)
    out_dir = Path(args.output_dir) if args.output_dir else infer_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = torch.load(infer_path, map_location="cpu", mmap=True, weights_only=True)
    truth_flat, n_geom, n_wv, n_bands, field_h, field_w, channels = prepare_scoring_data(
        dataset_pt_dir, predictions, channels
    )
    total = n_geom * n_wv * n_bands

    print(f"Dataset   : {dataset_pt_dir}")
    print(f"Truth     : eigenfrequency_uniform + displacements  shape={tuple(truth_flat.shape)} dtype={truth_flat.dtype}")
    print(f"Inference : {infer_path}  shape={tuple(predictions.shape)} dtype={predictions.dtype}")
    print(f"Channels  : {channels}   weighting: {channel_weighting}   Field: {field_h}x{field_w}   Device: {device}")
    if exclude_waves:
        print(f"Excluding wave indices: {exclude_waves}  ({len(exclude_waves)} wavevectors)")
    print(f"Samples   : {total}   Losses: {losses}")
    print(f"Output dir: {out_dir}")

    per_sample = compute_per_sample_losses(
        truth_flat=truth_flat,
        predictions=predictions,
        channels=channels,
        losses=losses,
        device=device,
        batch_size=args.batch_size,
        nmae_eps=args.nmae_eps,
        nmse_eps=args.nmse_eps,
        channel_weighting=channel_weighting,
    )

    if exclude_waves:
        _, wave_idx, _ = flat_indices(n_geom, n_wv, n_bands)
        keep = ~np.isin(wave_idx, np.array(exclude_waves, dtype=np.int64))
        n_kept = int(keep.sum())
        print(f"Histogram sample count after wave exclusion: {n_kept} / {total}")
        per_sample = {loss: arr[keep] for loss, arr in per_sample.items()}

    tag = f"_{args.tag}" if args.tag else ""
    for loss in losses:
        vals = per_sample[loss]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            print(f"[skip] {loss}: no finite values.")
            continue

        mean_v = float(finite.mean())
        median_v = float(np.median(finite))
        norm_denom = finite.size

        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

        if log_x:
            pos = finite[finite > 0.0]
            if pos.size < 2:
                print(f"[skip] {loss}: fewer than 2 positive values for log-x.")
                plt.close(fig)
                continue
            hi = float(np.percentile(pos, args.clip_percentile)) if args.clip_percentile < 100.0 else float(pos.max())
            lo = float(pos.min())
            if hi <= lo:
                hi = lo * 10.0
            xmin, xmax = log_decade_limits(lo, hi)
            plot_vals = pos[(pos >= xmin) & (pos <= xmax)]
            bin_edges = np.logspace(math.log10(xmin), math.log10(xmax), args.bins + 1)
            weights = sample_weights(plot_vals, normalize, norm_denom)
            ax.hist(
                plot_vals, bins=bin_edges, weights=weights,
                color="steelblue", edgecolor="white", linewidth=0.3, alpha=0.85,
            )
            set_log_decade_ticks(ax, xmin, xmax)
            if not args.no_curve:
                curve = fit_curve_logx(
                    plot_vals, bin_edges, args.kde_samples, args.seed, normalize, norm_denom,
                )
                if curve is not None:
                    ax.plot(curve[0], curve[1], color="crimson", linewidth=1.8, label="KDE fit (log-x)")
            if xmin <= mean_v <= xmax and mean_v > 0.0:
                ax.axvline(mean_v, color="darkorange", linestyle="--", linewidth=1.4, label=f"mean = {mean_v:.3e}")
            if xmin <= median_v <= xmax and median_v > 0.0:
                ax.axvline(median_v, color="seagreen", linestyle=":", linewidth=1.4, label=f"median = {median_v:.3e}")
        else:
            hi = float(np.percentile(finite, args.clip_percentile)) if args.clip_percentile < 100.0 else float(finite.max())
            lo = float(finite.min())
            if hi <= lo:
                hi = lo + 1e-12
            plot_vals = finite[(finite >= lo) & (finite <= hi)]
            weights = sample_weights(plot_vals, normalize, norm_denom)
            _, bin_edges, _ = ax.hist(
                plot_vals, bins=args.bins, range=(lo, hi), weights=weights,
                color="steelblue", edgecolor="white", linewidth=0.3, alpha=0.85,
            )
            if not args.no_curve:
                curve = fit_curve(plot_vals, bin_edges, args.kde_samples, args.seed, normalize, norm_denom)
                if curve is not None:
                    ax.plot(curve[0], curve[1], color="crimson", linewidth=1.8, label="KDE fit")
            if lo <= mean_v <= hi:
                ax.axvline(mean_v, color="darkorange", linestyle="--", linewidth=1.4, label=f"mean = {mean_v:.3e}")
            if lo <= median_v <= hi:
                ax.axvline(median_v, color="seagreen", linestyle=":", linewidth=1.4, label=f"median = {median_v:.3e}")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        ax.set_xlabel(LOSS_XLABEL.get(loss, f"Per-sample {loss}"))
        ax.set_ylabel("Relative Count" if normalize else "Sample Count")
        if args.title:
            ax.set_title(args.title)
        ax.legend()

        clip_tag = "" if args.clip_percentile >= 100.0 else f"_clip{args.clip_percentile:g}"
        scale_tag = "" if log_x else "_linear"
        out_path = out_dir / f"{args.out_prefix}_{loss}{tag}{scale_tag}{clip_tag}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {loss:<5} N={finite.size}  mean={mean_v:.6e}  median={median_v:.6e}  "
              f"min={float(finite.min()):.6e}  max={float(finite.max()):.6e} -> {out_path.name}")


if __name__ == "__main__":
    main()
