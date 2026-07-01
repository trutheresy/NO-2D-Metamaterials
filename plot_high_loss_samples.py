"""
Find, visualize, and characterize the highest-loss samples for one or more loss arrays
produced by per_sample_loss.py (columns: combined_idx, geom_idx, wave_idx, band_idx, loss).

Motivation: surface the highest per-sample loss cases (ranked by per_sample_loss.py,
which by default averages each loss over displacement channels 1–4).

For each loss array it:
  - plots the top-N samples with NO_utilities.visualize_sample (input panel + target/output/
    diff grid), saved to <output-dir>/<LOSS>_high_loss_samples/
  - prints a per-sample table (loss, mean displacement MAE/MSE, mean|t|, mean t^2, max|t|
    averaged over the four displacement truth fields) and an aggregate pattern summary
    over the top --analyze-top samples (band counts, repeated geometries, denominator
    distribution).

Inputs mirror plot_sample_cases.py (needs the dataset *_pt folder and dense predictions).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import NO_utilities
from per_sample_loss import DEFAULT_SCORING_CHANNELS, parse_channels
from output_layout import resolve_script_output_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-pt-dir", required=True, help="Dataset *_pt folder (geometries/waveforms/band_fft/eigenfrequency_uniform/displacements).")
    p.add_argument("--predictions", required=True, help="Dense prediction tensor (.pt), shape (n_geom*n_wv*n_bands, C, H, W).")
    p.add_argument("--loss-array", nargs=2, action="append", metavar=("NAME", "PATH"), required=True,
                   help="Loss name and .npy path from per_sample_loss.py. Repeatable.")
    p.add_argument("--output-dir", default="", help="Explicit base folder (overrides the model/dataset layout below); a <LOSS>_high_loss_samples subfolder is created inside it.")
    p.add_argument("--model-name", default="", help="Model name for the PLOTS/<model>/<dataset>/<subdir> layout.")
    p.add_argument("--dataset", default="", help="Dataset folder for the layout (default: --tag).")
    p.add_argument("--output-subdir", default="high_loss_analysis", help="Script output folder name under PLOTS/<model>/<dataset> (default: high_loss_analysis).")
    p.add_argument("--tag", default="", help="Dataset tag for filenames (e.g. c_test).")
    p.add_argument("--top", type=int, default=12, help="Number of highest-loss samples to plot (default 12).")
    p.add_argument("--analyze-top", type=int, default=200, help="Number of highest-loss samples for the pattern summary (default 200).")
    p.add_argument("--channels", default=",".join(str(c) for c in DEFAULT_SCORING_CHANNELS),
                   help="Prediction channels the loss was averaged over (default: 0,1,2,3,4).")
    p.add_argument("--field-cmap", default="RdBu_r")
    p.add_argument("--diverge-center", type=float, default=0.0)
    p.add_argument("--unified-colorbar", action="store_true", default=True)
    p.add_argument("--show-eigfreq", action=argparse.BooleanOptionalAction, default=False,
                   help="Show the eigenfrequency channel column (default: off).")
    args = p.parse_args()

    channels = parse_channels(args.channels)
    pt = Path(args.dataset_pt_dir)
    base_out = resolve_script_output_dir(
        explicit=args.output_dir or None,
        category="plots",
        model_name=args.model_name or None,
        dataset=args.dataset or args.tag,
        subdir=args.output_subdir,
        fallback=Path(args.predictions).parent,
    )

    geometries = torch.load(pt / "geometries_full.pt", weights_only=False)
    waveforms = torch.load(pt / "waveforms_full.pt", weights_only=False)
    band_ffts = torch.load(pt / "band_fft_full.pt", weights_only=False)
    eigenfrequency_uniform = torch.load(pt / "eigenfrequency_uniform_full.pt", weights_only=False)
    displacements = torch.load(pt / "displacements_dataset.pt", weights_only=False)
    predictions = torch.load(args.predictions, map_location="cpu", mmap=True, weights_only=True)

    n_wv = int(waveforms.shape[0])
    n_band = int(band_ffts.shape[0])
    full_rows = int(geometries.shape[0]) * n_wv * n_band
    disp_rows = int(displacements.tensors[0].shape[0])
    if disp_rows != full_rows:
        raise ValueError(
            f"displacements rows {disp_rows} != full grid {full_rows}; "
            f"this script requires dense full-dataset layout."
        )
    if int(predictions.shape[0]) != full_rows:
        raise ValueError(
            f"predictions rows {predictions.shape[0]} != full grid {full_rows}; "
            f"inference must be a dense full-dataset run."
        )

    def stack_target_channels(d: int, w: int, b: int, combined: int) -> torch.Tensor:
        ch0 = eigenfrequency_uniform[d, w, b]
        chans = [displacements.tensors[i][combined] for i in range(4)]
        return torch.stack([ch0, *chans], dim=0)

    def displacement_field_stats(combined: int):
        abs_means = []
        sq_means = []
        max_abs = 0.0
        for i in range(4):
            t = displacements.tensors[i][combined].float()
            abs_means.append(float(t.abs().mean()))
            sq_means.append(float(t.square().mean()))
            max_abs = max(max_abs, float(t.abs().max()))
        return float(np.mean(abs_means)), float(np.mean(sq_means)), max_abs

    def displacement_mae_mse(out: torch.Tensor, tgt: torch.Tensor):
        maes = []
        mses = []
        for ch in channels:
            err = out[ch] - tgt[ch]
            maes.append(float(err.abs().mean()))
            mses.append(float(err.square().mean()))
        return float(np.mean(maes)), float(np.mean(mses))

    tag = args.tag or pt.parent.name
    for loss_name, loss_path in args.loss_array:
        arr = np.load(loss_path)
        losses = arr[:, 4]
        order = np.argsort(losses, kind="stable")[::-1]  # descending: worst first
        n = order.shape[0]
        out_dir = base_out / f"{loss_name.upper()}_high_loss_samples"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {tag}  loss={loss_name}  (worst {args.top} of {n}) ===")
        print(f"{'rank':>4}{'comb':>10}{'g':>6}{'w':>5}{'b':>4}{'loss':>14}{'mae':>13}{'mse':>13}{'mean|t|':>12}{'mean t^2':>13}{'max|t|':>12}")
        for rank in range(min(args.top, n)):
            row = int(order[rank])
            comb = int(arr[row, 0]); d = int(arr[row, 1]); w = int(arr[row, 2]); b = int(arr[row, 3])
            val = float(arr[row, 4])

            tgt = stack_target_channels(d, w, b, comb).float()
            out = predictions[comb].float()
            mae, mse = displacement_mae_mse(out, tgt)
            ma, ms, mx = displacement_field_stats(comb)
            print(f"{rank:>4}{comb:>10}{d:>6}{w:>5}{b:>4}{val:>14.4e}{mae:>13.4e}{mse:>13.4e}{ma:>12.4e}{ms:>13.4e}{mx:>12.4e}")

            input_tensor = torch.stack([geometries[d], waveforms[w], band_ffts[b]], dim=0).float()
            plt.close("all")
            NO_utilities.visualize_sample(
                input_tensor.cpu(), out.cpu(), tgt.cpu(),
                unified_colorbar=args.unified_colorbar, field_cmap=args.field_cmap,
                diverge_center=args.diverge_center, show_eigfreq=args.show_eigfreq,
            )
            nums = plt.get_fignums()
            suptitle = (f"{tag} | {loss_name.upper()} worst#{rank} | comb={comb} (g={d},w={w},b={b}) | "
                        f"{loss_name}={val:.3e} | mean|t|={ma:.2e}")
            plt.figure(nums[0]).suptitle(suptitle + "  [inputs]", y=1.02, fontsize=10)
            plt.figure(nums[1]).suptitle(suptitle + "  [target / output / diff]", y=0.965, fontsize=11)
            base = f"{tag}_{loss_name}_worst{rank:02d}_g{d}_w{w}_b{b}"
            plt.figure(nums[0]).savefig(out_dir / f"{base}_input.png", dpi=150, bbox_inches="tight")
            plt.figure(nums[1]).savefig(out_dir / f"{base}_fields.png", dpi=150, bbox_inches="tight")
            plt.close("all")

        # ---- pattern summary over the worst K ----
        k = min(args.analyze_top, n)
        top_rows = order[:k]
        bands = arr[top_rows, 3].astype(int)
        geoms = arr[top_rows, 1].astype(int)
        denom_abs = np.array([displacement_field_stats(int(arr[r, 0]))[0] for r in top_rows])
        band_counts = Counter(bands.tolist())
        geom_counts = Counter(geoms.tolist())
        print(f"\n  --- pattern over worst {k} '{loss_name}' samples ---")
        print(f"  band distribution (band: count): " + ", ".join(f"{bb}:{band_counts.get(bb,0)}" for bb in range(n_band)))
        top_geoms = geom_counts.most_common(5)
        print(f"  most repeated geometries (geom: count): " + ", ".join(f"{g}:{c}" for g, c in top_geoms))
        print(f"  unique geometries among worst {k}: {len(geom_counts)}")
        print(f"  displacement truth mean|t| over worst {k}: median={np.median(denom_abs):.3e}  "
              f"mean={denom_abs.mean():.3e}  max={denom_abs.max():.3e}")
        print(f"    (worst-{k} median mean|t| vs whole-array context: smaller => near-zero true field drives blow-up)")
        print(f"  -> plotted top {min(args.top, n)} to {out_dir}")


if __name__ == "__main__":
    main()
