"""
Per-pixel relative error heatmaps: |p - t| / (|t| + eps).

Single-sample mode (default):
  Saves (C, H, W) per-channel maps, a multi-panel figure, and summary stats.

Dataset mode (--dataset-mode):
  Loads stacked predictions/truth for all samples, builds a (N, H, W) tensor by
  averaging per-pixel relative error across output channels (uniform mean over ch0-4),
  saves the stack, and writes one (H, W) heatmap that averages each pixel across
  all samples. No per-sample figures are written.

Outputs default to INFERENCE/<model-name>/<dataset>/relative_error_dataset/ when
--model-name is set; override with --output-dir or --category plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from output_layout import resolve_script_output_dir
from per_sample_loss import (
    DEFAULT_SCORING_CHANNELS,
    load_dataset_layout,
    load_displacements_dataset,
    open_scoring_sources,
    load_truth_batch,
)

DEFAULT_NMAE_EPS = 1e-5
CHANNEL_LABELS = [
    "ch0_eigenfrequency",
    "ch1_disp_x",
    "ch2_disp_x_imag",
    "ch3_disp_y",
    "ch4_disp_y_imag",
]


def resolve_sample_from_loss_array(loss_path: Path, percentile: int) -> tuple[int, int, int, int, float]:
    arr = np.load(loss_path)
    losses = arr[:, 4]
    order = np.argsort(losses, kind="stable")
    n = order.shape[0]
    rank = int(round((1.0 - percentile / 100.0) * (n - 1)))
    row = int(order[rank])
    return (
        int(arr[row, 0]),
        int(arr[row, 1]),
        int(arr[row, 2]),
        int(arr[row, 3]),
        float(arr[row, 4]),
    )


def load_truth_single(
    dataset_pt_dir: Path,
    geom: int,
    wave: int,
    band: int,
    combined: int,
    out_channels: int,
) -> torch.Tensor:
    eigen = torch.load(
        dataset_pt_dir / "eigenfrequency_uniform_full.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    chans = [eigen[geom, wave, band].float()]
    if out_channels > 1:
        displacements = load_displacements_dataset(dataset_pt_dir)
        chans.extend(displacements.tensors[i][combined].float() for i in range(out_channels - 1))
    return torch.stack(chans, dim=0)


def compute_relative_error(pred: torch.Tensor, truth: torch.Tensor, eps: float) -> torch.Tensor:
    err = (pred - truth).abs()
    denom = truth.abs() + eps
    return err / denom


def combine_channels_uniform(rel: torch.Tensor) -> torch.Tensor:
    """Average per-pixel relative error across channel dim: (..., C, H, W) -> (..., H, W)."""
    return rel.mean(dim=-3)


def compute_nmae_per_channel(pred: torch.Tensor, truth: torch.Tensor, eps: float) -> np.ndarray:
    err = (pred - truth).abs()
    mae = err.mean(dim=(-2, -1))
    denom = truth.abs().mean(dim=(-2, -1)) + eps
    return (mae / denom).numpy()


def save_heatmap(
    data: np.ndarray,
    png_path: Path,
    title: str,
    *,
    log_scale: bool = False,
    vmax_percentile: float = 99.0,
) -> None:
    positive = data[np.isfinite(data) & (data > 0)]
    vmax = float(np.percentile(positive, vmax_percentile)) if positive.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(4.5, 4.0), constrained_layout=True)
    if log_scale:
        plot_data = np.log10(np.clip(data, 1e-8, None))
        im = ax.imshow(plot_data, cmap="magma", vmin=np.log10(1e-8), vmax=np.log10(vmax))
        cbar_label = "log10(mean |e|/|t|)"
    else:
        im = ax.imshow(data, cmap="magma", vmin=0.0, vmax=vmax)
        cbar_label = "mean |e| / |t|"
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def resolve_out_dir(args: argparse.Namespace, pred_path: Path) -> Path:
    return resolve_script_output_dir(
        explicit=args.output_dir or None,
        category=args.category,
        model_name=args.model_name or None,
        dataset=args.dataset or args.tag,
        subdir=args.output_subdir,
        fallback=pred_path.parent,
    )


def run_single_sample(args: argparse.Namespace) -> None:
    dataset_pt_dir = Path(args.dataset_pt_dir)
    pred_path = Path(args.predictions)
    out_dir = resolve_out_dir(args, pred_path)

    predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
    out_channels = int(predictions.shape[1])
    loss_scalar = float("nan")

    if args.loss_array:
        combined, geom, wave, band, loss_scalar = resolve_sample_from_loss_array(
            Path(args.loss_array), args.percentile
        )
    else:
        if args.combined_idx is None or args.geom is None or args.wave is None or args.band is None:
            raise SystemExit("Provide --loss-array or all of --combined-idx --geom --wave --band.")
        combined, geom, wave, band = args.combined_idx, args.geom, args.wave, args.band

    truth = load_truth_single(dataset_pt_dir, geom, wave, band, combined, out_channels)
    pred = predictions[combined].float()
    rel = compute_relative_error(pred, truth, args.eps).numpy()
    nmae_ch = compute_nmae_per_channel(pred, truth, args.eps)

    stem = f"{args.tag}_comb{combined}_g{geom}_w{wave}_b{band}"
    npz_path = out_dir / f"{stem}_per_pixel_rel_error.npz"
    np.savez_compressed(
        npz_path,
        relative_error=rel.astype(np.float32),
        nmae_per_channel=nmae_ch.astype(np.float32),
        combined_idx=combined,
        geom_idx=geom,
        wave_idx=wave,
        band_idx=band,
        eps=args.eps,
    )

    n_ch = rel.shape[0]
    fig, axes = plt.subplots(1, n_ch, figsize=(3.2 * n_ch, 3.2), constrained_layout=True, squeeze=False)
    for ch in range(n_ch):
        ax = axes[0, ch]
        data = rel[ch]
        positive = data[data > 0]
        vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
        vmax = max(vmax, 1e-6)
        if args.log_heatmap:
            plot_data = np.log10(np.clip(data, 1e-8, None))
            im = ax.imshow(plot_data, cmap="magma", vmin=np.log10(1e-8), vmax=np.log10(vmax))
            cbar_label = "log10(|e|/|t|)"
        else:
            im = ax.imshow(data, cmap="magma", vmin=0.0, vmax=vmax)
            cbar_label = "|e| / |t|"
        ax.set_title(CHANNEL_LABELS[ch] if ch < len(CHANNEL_LABELS) else f"ch{ch}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.suptitle(
        f"Per-pixel relative error  {args.tag}  comb={combined}  (g,w,b)=({geom},{wave},{band})",
        fontsize=10,
    )
    png_path = out_dir / f"{stem}_per_pixel_rel_error.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"Sample         : comb={combined}  (g,w,b)=({geom},{wave},{band})")
    if args.loss_array:
        print(f"Loss array     : {args.loss_array}")
        print(f"Percentile     : p{args.percentile:02d}  scalar loss={loss_scalar:.6e}")
    print(f"Saved array    : {npz_path}")
    print(f"Saved heatmap  : {png_path}")
    print(f"eps            : {args.eps:g}")
    print()
    print(f"{'channel':<22} {'mean(rel)':>12} {'median(rel)':>14} {'NMAE':>12}")
    for ch in range(n_ch):
        label = CHANNEL_LABELS[ch] if ch < len(CHANNEL_LABELS) else f"ch{ch}"
        flat = rel[ch].ravel()
        print(
            f"{label:<22} {flat.mean():12.6f} {np.median(flat):14.6f} {nmae_ch[ch]:12.6f}"
        )
    all_flat = rel.ravel()
    disp_flat = rel[1:].ravel() if n_ch > 1 else all_flat
    print()
    print(f"Mean all pixels/channels     : {all_flat.mean():.6f}  ({100 * all_flat.mean():.2f}%)")
    print(f"Median all pixels/channels   : {np.median(all_flat):.6f}  ({100 * np.median(all_flat):.2f}%)")
    print(f"Mean displacement ch only    : {disp_flat.mean():.6f}  ({100 * disp_flat.mean():.2f}%)")
    print(f"Median displacement ch only  : {np.median(disp_flat):.6f}  ({100 * np.median(disp_flat):.2f}%)")
    print(f"NMAE mean ch0-4              : {nmae_ch.mean():.6f}  ({100 * nmae_ch.mean():.2f}%)")
    print(f"NMAE mean disp ch1-4         : {nmae_ch[1:].mean():.6f}  ({100 * nmae_ch[1:].mean():.2f}%)")


def run_dataset_mode(args: argparse.Namespace) -> None:
    dataset_pt_dir = Path(args.dataset_pt_dir)
    pred_path = Path(args.predictions)
    out_dir = resolve_out_dir(args, pred_path)

    channels = list(DEFAULT_SCORING_CHANNELS)
    predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
    n_geom, n_wv, n_bands, field_h, field_w = load_dataset_layout(dataset_pt_dir)
    total = n_geom * n_wv * n_bands
    if predictions.shape[0] != total:
        raise ValueError(
            f"Prediction count {predictions.shape[0]} != n_geom*n_wv*n_bands ({total})."
        )
    if int(predictions.shape[1]) != len(channels):
        raise ValueError(f"Expected {len(channels)} prediction channels; got {predictions.shape[1]}.")

    print(f"Dataset mode   : {args.tag}")
    print(f"Dataset        : {dataset_pt_dir}")
    print(f"Predictions    : {pred_path}  shape={tuple(predictions.shape)}")
    print(f"Samples        : {total}  field={field_h}x{field_w}")
    print(f"Channels       : {channels} (uniform mean -> one HxW map per sample)")
    print(f"Output dir     : {out_dir}")

    sources = open_scoring_sources(
        dataset_pt_dir, total, (field_h, field_w), need_displacements=True
    )

    stack_path = out_dir / f"{args.tag}_per_pixel_rel_error_stack.npy"
    stack = np.lib.format.open_memmap(
        stack_path,
        mode="w+",
        dtype=np.float32,
        shape=(total, field_h, field_w),
    )
    pixel_sum = np.zeros((field_h, field_w), dtype=np.float64)

    batch_size = args.batch_size
    for start in tqdm(range(0, total, batch_size), desc="Relative error", unit="batch"):
        end = min(start + batch_size, total)
        pred_b = predictions[start:end].float()
        truth_b = load_truth_batch(sources, channels, start, end).float()
        rel_b = compute_relative_error(pred_b, truth_b, args.eps)
        rel_hw = combine_channels_uniform(rel_b).cpu().numpy().astype(np.float32, copy=False)
        stack[start:end] = rel_hw
        pixel_sum += rel_hw.sum(axis=0, dtype=np.float64)

    stack.flush()
    mean_map = (pixel_sum / total).astype(np.float32)

    mean_path = out_dir / f"{args.tag}_per_pixel_rel_error_mean_over_samples.npy"
    meta_path = out_dir / f"{args.tag}_per_pixel_rel_error_dataset.npz"
    np.save(mean_path, mean_map)
    np.savez_compressed(
        meta_path,
        mean_over_samples=mean_map,
        eps=np.float32(args.eps),
        n_samples=np.int64(total),
        field_h=np.int32(field_h),
        field_w=np.int32(field_w),
        channels=np.array(channels, dtype=np.int32),
        stack_npy=str(stack_path.name),
    )

    png_path = out_dir / f"{args.tag}_per_pixel_rel_error_mean_over_samples.png"
    save_heatmap(
        mean_map,
        png_path,
        title=f"{args.tag}: mean per-pixel |e|/|t| over {total:,} samples",
        log_scale=args.log_heatmap,
    )

    finite = stack[:].reshape(-1)
    finite = finite[np.isfinite(finite)]
    print()
    print(f"Saved stack    : {stack_path}  shape=({total}, {field_h}, {field_w})")
    print(f"Saved mean map : {mean_path}")
    print(f"Saved metadata : {meta_path}")
    print(f"Saved heatmap  : {png_path}")
    print(f"Stack mean(all pixels,samples): {finite.mean():.6f}  ({100 * finite.mean():.2f}%)")
    print(f"Stack median(all pixels,samples): {np.median(finite):.6f}  ({100 * np.median(finite):.2f}%)")
    print(f"Mean-map mean(pixels)         : {mean_map.mean():.6f}  ({100 * mean_map.mean():.2f}%)")
    print(f"Mean-map median(pixels)       : {np.median(mean_map):.6f}  ({100 * np.median(mean_map):.2f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-pt-dir", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument(
        "--output-dir",
        default="",
        help="Explicit output folder (overrides --model-name / --output-subdir layout).",
    )
    p.add_argument(
        "--model-name",
        default="",
        help="Model name for INFERENCE/<model>/<dataset>/<subdir> (or PLOTS/... with --category plots).",
    )
    p.add_argument("--dataset", default="", help="Dataset tag for the layout (default: --tag).")
    p.add_argument(
        "--output-subdir",
        default="relative_error_dataset",
        help="Script output folder under <model>/<dataset> (default: relative_error_dataset).",
    )
    p.add_argument(
        "--category",
        default="inference",
        choices=("plots", "inference"),
        help="Top-level root: inference (data, default) or plots.",
    )
    p.add_argument(
        "--dataset-mode",
        action="store_true",
        help="Process all stacked samples; save (N,H,W) tensor + one mean-over-samples heatmap.",
    )
    p.add_argument("--combined-idx", type=int, default=None, help="Flat sample index into predictions.")
    p.add_argument("--loss-array", default="", help="Optional per_sample_loss .npy to pick a percentile case.")
    p.add_argument("--percentile", type=int, default=50, help="Performance percentile when using --loss-array.")
    p.add_argument("--geom", type=int, default=None)
    p.add_argument("--wave", type=int, default=None)
    p.add_argument("--band", type=int, default=None)
    p.add_argument("--eps", type=float, default=DEFAULT_NMAE_EPS)
    p.add_argument("--tag", default="sample", help="Prefix for output filenames.")
    p.add_argument("--log-heatmap", action="store_true", help="Use log10 color scale for heatmaps.")
    p.add_argument("--batch-size", type=int, default=8192, help="Batch size for --dataset-mode.")
    args = p.parse_args()

    if args.dataset_mode:
        run_dataset_mode(args)
    else:
        run_single_sample(args)


if __name__ == "__main__":
    main()
