"""
Band-resolved variant of scatter_loss_vs_boundary.py (binary geometries only).

For each geometry and each band, the per-sample losses are averaged over the wavevectors,
giving one point per (geometry, band). Every geometry therefore contributes 6 points (one
per band) at the same x = boundary length, colored by band. One figure per loss criterion.

The legend has one entry per band reading "band <b>, rho = <Spearman>" where the Spearman
correlation is computed across the 1000 geometries for that band.

Inputs match per_sample_loss.py:
  --dataset-pt-dir  dataset *_pt folder with displacements_dataset.pt (truth)
  --inference       dense prediction tensor    (n_geom*n_wv*n_bands, C, H, W)
  --geometries      discrete geometries tensor (n_geom, H, W), binary {0,1}

Outputs (to --output-dir, default <inference-dir>/boundary_length_vs_loss_by_band):
  - one scatter PNG per loss
  - per_geometry_band_boundary_vs_loss.csv (geom_idx, boundary_length, <loss>_b0..b5 ...)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from per_sample_loss import (
    compute_per_sample_losses,
    normalize_loss_name,
    parse_channels,
    prepare_scoring_data,
    resolve_device,
)
from compute_boundary_length import load_geometries, to_binary, boundary_length
from output_layout import resolve_script_output_dir


LOSS_YLABEL = {
    "mae": "Mean (over wavevectors) MAE",
    "mse": "Mean (over wavevectors) MSE",
    "nmae": "Mean (over wavevectors) NMAE",
    "nmse": "Mean (over wavevectors) NMSE",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-pt-dir", required=True, help="Dataset *_pt folder with displacements_dataset.pt.")
    p.add_argument("--inference", required=True, help="Dense prediction tensor (.pt), shape (n_geom*n_wv*n_bands, C, H, W).")
    p.add_argument("--geometries", required=True, help="Discrete geometries tensor (.pt), shape (n_geom, H, W), binary {0,1}.")
    p.add_argument("--losses", nargs="+", default=["mae", "mse", "nmae", "nmse"], help="Loss criteria (default: all four).")
    p.add_argument("--channels", default="0,1,2,3,4", help="Comma-separated prediction channels to average (default: 0,1,2,3,4).")
    p.add_argument("--output-dir", default="", help="Explicit output folder (overrides the model/dataset layout below).")
    p.add_argument("--model-name", default="", help="Model name for the PLOTS/<model>/<dataset>/<subdir> layout.")
    p.add_argument("--dataset", default="", help="Dataset folder for the layout (default: --tag).")
    p.add_argument("--output-subdir", default="boundary_length_vs_loss_by_band", help="Script output folder name under PLOTS/<model>/<dataset> (default: boundary_length_vs_loss_by_band).")
    p.add_argument("--tag", default="", help="Dataset tag for filenames (e.g. b_test).")
    p.add_argument("--periodic", action="store_true", help="Use periodic (wrap-around) boundary length.")
    p.add_argument("--linear-y", action="store_true", help="Use a linear loss (y) axis (default is log-scaled).")
    p.add_argument("--cmap", default="tab10", help="Matplotlib colormap for per-band colors (default tab10).")
    p.add_argument("--nmae-eps", type=float, default=1e-5)
    p.add_argument("--nmse-eps", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="cpu", choices=("auto", "cuda", "cpu"),
                   help="Compute device (default: cpu). Use 'cuda' or 'auto' to opt into GPU.")
    args = p.parse_args()

    log_y = not args.linear_y
    losses = []
    for name in args.losses:
        ln = normalize_loss_name(name)
        if ln not in losses:
            losses.append(ln)

    channels = parse_channels(args.channels)
    device = resolve_device(args.device)
    dataset_pt_dir = Path(args.dataset_pt_dir)
    infer_path = Path(args.inference)
    out_dir = resolve_script_output_dir(
        explicit=args.output_dir or None,
        category="plots",
        model_name=args.model_name or None,
        dataset=args.dataset or args.tag,
        subdir=args.output_subdir,
        fallback=infer_path.parent / "boundary_length_vs_loss_by_band",
    )

    predictions = torch.load(infer_path, map_location="cpu", mmap=True, weights_only=True)
    truth_flat, n_geom, n_wv, n_bands, field_h, field_w, channels = prepare_scoring_data(
        dataset_pt_dir, predictions, channels
    )

    gb = to_binary(load_geometries(Path(args.geometries)), None)
    if int(gb.shape[0]) != n_geom:
        raise ValueError(f"geometry count {int(gb.shape[0])} != dataset n_geom {n_geom}.")
    blen = boundary_length(gb, args.periodic, 1.0)  # (n_geom,)

    print(f"Dataset    : {dataset_pt_dir}")
    print(f"Truth      : eigenfrequency_uniform + displacements  shape={tuple(truth_flat.shape)}")
    print(f"Inference  : {infer_path}  shape={tuple(predictions.shape)}")
    print(f"Channels   : {channels}")
    print(f"Geometries : {n_geom}   n_wv={n_wv}   n_bands={n_bands}")
    print(f"Losses     : {losses}   Output: {out_dir}")

    per_sample = compute_per_sample_losses(
        truth_flat=truth_flat, predictions=predictions, channels=channels,
        losses=losses, device=device, batch_size=args.batch_size,
        nmae_eps=args.nmae_eps, nmse_eps=args.nmse_eps,
    )

    # combined index is (geom, wv, band) row-major -> reshape and average over wavevector axis
    tag = f"_{args.tag}" if args.tag else ""
    mode = "periodic" if args.periodic else "interior"
    cmap = plt.get_cmap(args.cmap)
    geom_idx = np.arange(n_geom)

    per_geom_band = {}
    for loss in losses:
        per_geom_band[loss] = per_sample[loss].reshape(n_geom, n_wv, n_bands).mean(axis=1)  # (n_geom, n_bands)

    # save combined table
    header = ["geom_idx", "boundary_length"]
    cols = [geom_idx.astype(np.float64), blen]
    for loss in losses:
        for b in range(n_bands):
            header.append(f"{loss}_b{b}")
            cols.append(per_geom_band[loss][:, b])
    table = np.stack(cols, axis=1)
    csv_path = out_dir / f"per_geometry_band_boundary_vs_loss{tag}.csv"
    np.savetxt(csv_path, table, delimiter=",", header=",".join(header), comments="", fmt="%.10g")
    print(f"\nBoundary length ({mode}): mean={blen.mean():.2f} min={blen.min():.0f} max={blen.max():.0f}")
    print(f"Wrote table: {csv_path}")

    for loss in losses:
        ydata = per_geom_band[loss]  # (n_geom, n_bands)
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        print(f"  {loss}:")
        for b in range(n_bands):
            yb = ydata[:, b]
            finite = np.isfinite(yb) & np.isfinite(blen)
            rho = spearmanr(blen[finite], yb[finite])[0] if finite.sum() > 2 else float("nan")
            color = cmap(b % cmap.N)
            ax.scatter(blen[finite], yb[finite], s=12, alpha=0.5, color=color,
                       edgecolors="none", label=f"band {b}, \u03c1 = {rho:.3f}")
            print(f"    band {b}: Spearman rho = {rho:.4f}")
        ax.set_xlabel(f"Boundary length ({mode} edges)")
        ax.set_ylabel(LOSS_YLABEL.get(loss, f"Mean (over wavevectors) {loss}"))
        if log_y:
            ax.set_yscale("log")
        else:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend(loc="best", framealpha=0.9)
        out_path = out_dir / f"boundary_vs_{loss}_by_band{tag}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"    -> {out_path.name}")


if __name__ == "__main__":
    main()
