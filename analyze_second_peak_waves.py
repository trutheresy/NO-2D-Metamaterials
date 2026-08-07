"""
Second-peak wavevector/band correlation analysis for an inference run.

Computes per-sample NMAE (default: group channel weighting, 50% eigenfrequency ch0 +
50% mean displacement ch1-4, matching the loss-histogram default), splits the
log-scaled distribution at the valley between its two modes
(``second_peak_analysis.find_two_peaks``), then reports which wavevectors and bands
are enriched in the high-loss (second) peak.

Outputs (to INFERENCE/<model>/<dataset>/second_peak_analysis/ by default):
  {prefix}_{tag}_wave_table.csv   : per-wavevector second-peak stats (all 325 waves)
  {prefix}_{tag}_band_table.csv   : per-band second-peak stats
  {prefix}_{tag}_report.md        : markdown report (>50% wave catalog + band table)

Usage:
  python analyze_second_peak_waves.py --dataset-pt-dir <pt> --predictions <pred.pt>
      --model-name <MODEL> --dataset c_test --tag c_test
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from output_layout import resolve_script_output_dir
from per_sample_loss import (
    compute_per_sample_losses,
    load_dataset_layout,
    normalize_channel_weighting,
    open_scoring_sources,
    parse_channels,
    resolve_device,
)
from NO_utilities import EIGENFREQUENCY_ENCODING_FILES, resolve_eigenfrequency_encoding
from second_peak_analysis import band_table, flat_indices, second_peak_mask, wave_table


def k_label(kx: float, ky: float, kmax: float) -> str:
    """Human-readable (kx, ky) as multiples of pi, e.g. '(-11pi/12, 0)'."""

    def comp(v: float) -> str:
        r = Fraction(v / kmax).limit_denominator(24)
        if r == 0:
            return "0"
        num, den = r.numerator, r.denominator
        sign = "-" if num < 0 else "+"
        num = abs(num)
        if num == 1 and den == 1:
            return f"{sign}pi"
        if den == 1:
            return f"{sign}{num}pi"
        if num == 1:
            return f"{sign}pi/{den}"
        return f"{sign}{num}pi/{den}"

    return f"({comp(kx)}, {comp(ky)})"


def bz_tag(kx: float, ky: float, kmax: float) -> str:
    tags = []
    if abs(ky) < 1e-9:
        tags.append("ky=0 (Gamma-X)")
    if abs(kx) < 1e-9:
        tags.append("kx=0")
    if np.isclose(abs(kx), kmax):
        tags.append("|kx|=pi")
    if np.isclose(abs(ky), kmax):
        tags.append("|ky|=pi")
    return ", ".join(tags) if tags else "generic"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-pt-dir", required=True)
    p.add_argument(
        "--eigen-encoding",
        choices=tuple(EIGENFREQUENCY_ENCODING_FILES),
        default="uniform",
        help="Channel-0 truth encoding: uniform or fft (wavelet). Default: uniform.",
    )
    p.add_argument("--predictions", required=True)
    p.add_argument("--loss", default="nmae", help="Loss criterion (default: nmae).")
    p.add_argument("--channels", default="0,1,2,3,4")
    p.add_argument("--channel-weighting", default="group", choices=("uniform", "group"),
                   help="group = 50%% ch0 + 50%% mean(ch1-4) (default, matches histograms).")
    p.add_argument("--threshold-pct", type=float, default=50.0,
                   help="Report waves with second-peak membership above this percent (default 50).")
    p.add_argument("--output-dir", default="", help="Explicit output folder (overrides layout).")
    p.add_argument("--model-name", default="", help="Model name for INFERENCE/<model>/<dataset>/<subdir>.")
    p.add_argument("--dataset", default="", help="Dataset tag for the layout (default: --tag).")
    p.add_argument("--output-subdir", default="second_peak_analysis")
    p.add_argument("--category", default="inference", choices=("plots", "inference"))
    p.add_argument("--tag", default="", help="Tag for output filenames.")
    p.add_argument("--out-prefix", default="second_peak")
    p.add_argument("--nmae-eps", type=float, default=1e-5)
    p.add_argument("--nmse-eps", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="cpu", choices=("auto", "cuda", "cpu"))
    args = p.parse_args()

    dataset_pt_dir = Path(args.dataset_pt_dir)
    pred_path = Path(args.predictions)
    out_dir = resolve_script_output_dir(
        explicit=args.output_dir or None,
        category=args.category,
        model_name=args.model_name or None,
        dataset=args.dataset or args.tag,
        subdir=args.output_subdir,
        fallback=pred_path.parent,
    )

    channels = parse_channels(args.channels)
    weighting = normalize_channel_weighting(args.channel_weighting)
    device = resolve_device(args.device)

    predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
    eigen_encoding = resolve_eigenfrequency_encoding(args.eigen_encoding)
    n_geom, n_wv, n_bands, fh, fw = load_dataset_layout(dataset_pt_dir, eigen_encoding)
    total = n_geom * n_wv * n_bands
    sources = open_scoring_sources(
        dataset_pt_dir, total, (fh, fw), any(c >= 1 for c in channels), eigen_encoding=eigen_encoding
    )

    print(f"Dataset    : {dataset_pt_dir}")
    print(f"Eigen enc  : {eigen_encoding}")
    print(f"Predictions: {pred_path}")
    print(f"Samples    : {total} (geom={n_geom} wv={n_wv} bands={n_bands})")
    print(f"Loss       : {args.loss}  channels={channels}  weighting={weighting}")
    print(f"Output dir : {out_dir}")

    per_sample = compute_per_sample_losses(
        truth_flat=None,
        predictions=predictions,
        channels=channels,
        losses=[args.loss],
        device=device,
        batch_size=args.batch_size,
        nmae_eps=args.nmae_eps,
        nmse_eps=args.nmse_eps,
        channel_weighting=weighting,
        sources=sources,
    )
    loss_vals = per_sample[args.loss]

    second, info = second_peak_mask(loss_vals)
    if info is None:
        raise SystemExit("No bimodal structure found in the loss distribution; nothing to report.")
    pop_rate = float(second.mean())
    print(f"\nSecond-peak split: {info['split_value']:.4f} "
          f"(log10 peaks ~ {info['peak_logs'][0]:.3f}, {info['peak_logs'][1]:.3f})")
    print(f"Global second-peak rate: {100 * pop_rate:.2f}%")

    _, wave_idx, band_idx = flat_indices(n_geom, n_wv, n_bands)
    kxy = torch.load(dataset_pt_dir / "wavevectors_full.pt", map_location="cpu",
                     weights_only=True).float().numpy()[0]
    kmax = float(np.abs(kxy).max())

    full_waves, _ = wave_table(wave_idx, second, kxy, n_geom, n_bands, top_n=10)
    bands = band_table(band_idx, second, n_bands)

    tag = args.tag or "run"
    wave_csv = out_dir / f"{args.out_prefix}_{tag}_wave_table.csv"
    with wave_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(full_waves[0].keys()))
        w.writeheader()
        w.writerows(full_waves)

    band_csv = out_dir / f"{args.out_prefix}_{tag}_band_table.csv"
    with band_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bands[0].keys()))
        w.writeheader()
        w.writerows(bands)

    hot = [r for r in full_waves if r["frac_second_pct"] > args.threshold_pct]
    hot.sort(key=lambda r: r["frac_second_pct"], reverse=True)

    lines = [
        f"# Second-peak wave/band analysis: {args.model_name or pred_path.parent.name} / {tag}",
        "",
        f"- Loss: **{args.loss}** (channels {channels}, {weighting} weighting; eps={args.nmae_eps:g})",
        f"- Global second-peak rate: **{100 * pop_rate:.2f}%**",
        f"- Split threshold: **{info['split_value']:.4f}** "
        f"(log10 peaks ~ {info['peak_logs'][0]:.3f}, {info['peak_logs'][1]:.3f})",
        f"- Waves above {args.threshold_pct:g}%: **{len(hot)}** - "
        f"`{', '.join(str(r['wave']) for r in hot)}`",
        "",
        "| Wave | k | BZ tag | N second / N total | % in 2nd peak | Enrichment |",
        "|-----:|:--|:-------|-------------------:|--------------:|-----------:|",
    ]
    for r in hot:
        lines.append(
            f"| {r['wave']} | {k_label(r['kx'], r['ky'], kmax)} | {bz_tag(r['kx'], r['ky'], kmax)} "
            f"| {r['n_second']:,} / {r['n_total']:,} | {r['frac_second_pct']:.2f} "
            f"| {r['enrichment_vs_pop']:.2f}x |"
        )
    lines += ["", "## Bands", "", "| Band | % in 2nd peak | Enrichment vs uniform |", "|-----:|--------------:|----------------------:|"]
    for r in bands:
        lines.append(f"| {r['band']} | {r['frac_second_pct']:.2f} | {r['enrichment']:.2f}x |")

    report = out_dir / f"{args.out_prefix}_{tag}_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWaves >{args.threshold_pct:g}% second-peak membership: {len(hot)}")
    for r in hot[:12]:
        print(f"  wave={r['wave']:3d}  k={k_label(r['kx'], r['ky'], kmax):<18} "
              f"{r['frac_second_pct']:6.2f}%  ({bz_tag(r['kx'], r['ky'], kmax)})")
    if len(hot) > 12:
        print(f"  ... and {len(hot) - 12} more (see report)")
    print(f"\nWrote: {wave_csv}\n       {band_csv}\n       {report}")


if __name__ == "__main__":
    main()
