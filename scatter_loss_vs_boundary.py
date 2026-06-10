"""
Dataset-level correlation between geometry boundary length and model loss (binary
geometries only).

For each of the 1000 geometries, the per-sample losses (one per wavevector x band) are
averaged to a single per-geometry mean loss, for each requested loss criterion
(mae, mse, nmae, nmse). These are scattered against the 0/1 boundary length of the
geometry (from compute_boundary_length.py), one figure per loss, with Pearson/Spearman
correlations annotated.

Inputs match per_sample_loss.py:
  --truth      ground-truth field tensor (n_geom, n_wv, n_bands, H, W)
  --inference  dense prediction tensor    (n_geom*n_wv*n_bands, C, H, W)
  --geometries discrete geometries tensor (n_geom, H, W), binary {0,1}

Outputs (to --output-dir, default <inference-dir>/boundary_length_vs_loss):
  - one scatter PNG per loss
  - per_geometry_boundary_vs_loss.csv with columns
    geom_idx, boundary_length, <loss>_mean ... and the correlation summary printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from per_sample_loss import compute_per_sample_losses, normalize_loss_name, resolve_device
from compute_boundary_length import load_geometries, to_binary, boundary_length


LOSS_YLABEL = {
    "mae": "Mean per-geometry MAE",
    "mse": "Mean per-geometry MSE",
    "nmae": "Mean per-geometry NMAE",
    "nmse": "Mean per-geometry NMSE",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", required=True, help="Ground-truth field tensor (.pt), shape (n_geom, n_wv, n_bands, H, W).")
    p.add_argument("--inference", required=True, help="Dense prediction tensor (.pt), shape (n_geom*n_wv*n_bands, C, H, W).")
    p.add_argument("--geometries", required=True, help="Discrete geometries tensor (.pt), shape (n_geom, H, W), binary {0,1}.")
    p.add_argument("--losses", nargs="+", default=["mae", "mse", "nmae", "nmse"], help="Loss criteria (default: all four).")
    p.add_argument("--channel", type=int, default=0, help="Prediction channel to compare (default 0 = eigenfrequency).")
    p.add_argument("--output-dir", default="", help="Output folder (default: <inference-dir>/boundary_length_vs_loss).")
    p.add_argument("--tag", default="", help="Dataset tag for filenames (e.g. b_test).")
    p.add_argument("--periodic", action="store_true", help="Use periodic (wrap-around) boundary length.")
    p.add_argument("--linear-y", action="store_true", help="Use a linear loss (y) axis (default is log-scaled).")
    p.add_argument("--nmae-eps", type=float, default=1e-5)
    p.add_argument("--nmse-eps", type=float, default=1e-7)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = p.parse_args()

    log_y = not args.linear_y
    losses = []
    for name in args.losses:
        ln = normalize_loss_name(name)
        if ln not in losses:
            losses.append(ln)

    device = resolve_device(args.device)
    truth_path = Path(args.truth)
    infer_path = Path(args.inference)
    out_dir = Path(args.output_dir) if args.output_dir else infer_path.parent / "boundary_length_vs_loss"
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = torch.load(truth_path, map_location="cpu", mmap=True, weights_only=True)
    predictions = torch.load(infer_path, map_location="cpu", mmap=True, weights_only=True)
    if truth.ndim != 5:
        raise ValueError(f"Expected truth shape (n_geom, n_wv, n_bands, H, W); got {tuple(truth.shape)}.")
    n_geom, n_wv, n_bands, field_h, field_w = (int(s) for s in truth.shape)
    per_geom = n_wv * n_bands
    total = n_geom * per_geom
    if predictions.shape[0] != total:
        raise ValueError(f"Prediction rows {predictions.shape[0]} != n_geom*n_wv*n_bands ({total}).")

    # boundary length per geometry
    gb = to_binary(load_geometries(Path(args.geometries)), None)
    if int(gb.shape[0]) != n_geom:
        raise ValueError(f"geometry count {int(gb.shape[0])} != truth n_geom {n_geom}.")
    blen = boundary_length(gb, args.periodic, 1.0)  # (n_geom,)

    truth_flat = truth.reshape(total, field_h, field_w)
    print(f"Truth      : {truth_path}  shape={tuple(truth.shape)}")
    print(f"Inference  : {infer_path}  shape={tuple(predictions.shape)}")
    print(f"Geometries : {n_geom}   per-geometry samples (n_wv*n_bands) = {per_geom}")
    print(f"Losses     : {losses}   Output: {out_dir}")

    per_sample = compute_per_sample_losses(
        truth_flat=truth_flat, predictions=predictions, channel=args.channel,
        losses=losses, device=device, batch_size=args.batch_size,
        nmae_eps=args.nmae_eps, nmse_eps=args.nmse_eps,
    )

    # average per geometry: combined index is geom-major (geom slowest), so reshape (n_geom, per_geom)
    tag = f"_{args.tag}" if args.tag else ""
    mode = "periodic" if args.periodic else "interior"
    geom_idx = np.arange(n_geom)
    per_geom_means = {}
    for loss in losses:
        per_geom_means[loss] = per_sample[loss].reshape(n_geom, per_geom).mean(axis=1)

    # save combined table
    header = ["geom_idx", "boundary_length"] + [f"{l}_mean" for l in losses]
    cols = [geom_idx.astype(np.float64), blen] + [per_geom_means[l] for l in losses]
    table = np.stack(cols, axis=1)
    csv_path = out_dir / f"per_geometry_boundary_vs_loss{tag}.csv"
    np.savetxt(csv_path, table, delimiter=",", header=",".join(header), comments="", fmt="%.10g")
    print(f"\nBoundary length ({mode}): mean={blen.mean():.2f} min={blen.min():.0f} max={blen.max():.0f}")
    print(f"Wrote table: {csv_path}")

    for loss in losses:
        y = per_geom_means[loss]
        finite = np.isfinite(y) & np.isfinite(blen)
        x_f, y_f = blen[finite], y[finite]
        pr = pearsonr(x_f, y_f) if x_f.size > 2 else (float("nan"), float("nan"))
        sr = spearmanr(x_f, y_f) if x_f.size > 2 else (float("nan"), float("nan"))

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        stats_label = f"Pearson r = {pr[0]:.3f}\nSpearman \u03c1 = {sr[0]:.3f}"
        ax.scatter(x_f, y_f, s=12, alpha=0.45, color="steelblue", edgecolors="none", label=stats_label)
        ax.set_xlabel(f"Boundary length ({mode} edges)")
        ax.set_ylabel(LOSS_YLABEL.get(loss, f"Mean per-geometry {loss}"))
        if log_y:
            ax.set_yscale("log")
        else:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend(loc="best", handlelength=0, handletextpad=0, markerscale=0, frameon=True)
        scale_tag = "" if log_y else "_linear"
        out_path = out_dir / f"boundary_vs_{loss}{tag}{scale_tag}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  {loss:<5} Pearson r={pr[0]:.4f} (p={pr[1]:.2e})  Spearman rho={sr[0]:.4f} (p={sr[1]:.2e}) -> {out_path.name}")


if __name__ == "__main__":
    main()
