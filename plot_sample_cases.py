"""
Plot per-sample-loss cases at performance percentiles p0/p01/.../p99/p100 using the
same visualization as ``figures_continuous_I3O5_efUniform.ipynb``
(NO_utilities.visualize_sample). Performance is the opposite of loss: p99 is the
best/lowest-loss sample and p01 the worst/highest-loss sample.

For each loss array produced by ``per_sample_loss.py`` (columns:
combined_idx, geom_idx, wave_idx, band_idx, loss), the percentile samples are
selected with the same nearest-rank ordering used by that script. By default the
loss is averaged over all five prediction channels 0–4 (eigenfrequency + displacements). For each case:

    - Inputs  : stack([geometry[d], waveform[w], band_fft[b]])            -> (3, H, W)
    - Output  : saved dense prediction tensor[combined]                   -> (C, H, W)
    - Target  : ch0 = eigenfrequency_uniform_full[d, w, b],
                ch1-4 = displacements_dataset.tensors[i][combined]        -> (C, H, W)

``visualize_sample`` emits two figures (the 1x3 input panel and the
target/output/diff grid); both are saved to the output folder with filenames that
encode dataset, loss, case, combined index and (g, w, b).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import NO_utilities


# Percentiles are in PERFORMANCE (opposite of loss): higher p = better performance =
# lower loss. Performance percentile p maps to loss rank round((1 - p/100) * (n-1)) on
# the ascending-sorted losses (rank 0 = lowest loss = best performance).
PERFORMANCE_PERCENTILES = [0, 1, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99, 100]


def performance_case_label(p: int) -> str:
    if p == 0:
        return "p0"
    if p == 100:
        return "p100"
    return f"p{p:02d}"


CASE_RANKS = [
    (performance_case_label(p), (lambda n, _p=p: int(round((1.0 - _p / 100.0) * (n - 1)))))
    for p in PERFORMANCE_PERCENTILES
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-pt-dir", required=True, help="Dataset *_pt folder with geometries/waveforms/band_fft/eigenfrequency_uniform/displacements.")
    p.add_argument("--predictions", required=True, help="Dense prediction tensor (.pt), shape (n_geom*n_wv*n_bands, C, H, W).")
    p.add_argument("--loss-array", nargs=2, action="append", metavar=("NAME", "PATH"), required=True,
                   help="Loss name and .npy path from per_sample_loss.py. Repeatable, e.g. --loss-array mae a.npy --loss-array mse b.npy")
    p.add_argument("--output-dir", required=True, help="Folder to save the case plots.")
    p.add_argument("--tag", default="", help="Dataset tag for filenames (e.g. c_test).")
    p.add_argument("--field-cmap", default="RdBu_r")
    p.add_argument("--diverge-center", type=float, default=0.0)
    p.add_argument("--unified-colorbar", action="store_true", default=True)
    p.add_argument("--show-eigfreq", action=argparse.BooleanOptionalAction, default=False,
                   help="Show the eigenfrequency channel (channel 0) column. Default: off (4-channel output).")
    p.add_argument("--title", action=argparse.BooleanOptionalAction, default=False,
                   help="Add a suptitle to each figure. Default: off (no title).")
    args = p.parse_args()

    pt = Path(args.dataset_pt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometries = torch.load(pt / "geometries_full.pt", weights_only=False)
    waveforms = torch.load(pt / "waveforms_full.pt", weights_only=False)
    band_ffts = torch.load(pt / "band_fft_full.pt", weights_only=False)
    eigenfrequency_uniform = torch.load(pt / "eigenfrequency_uniform_full.pt", weights_only=False)
    displacements = torch.load(pt / "displacements_dataset.pt", weights_only=False)
    predictions = torch.load(args.predictions, map_location="cpu", mmap=True, weights_only=True)

    n_wv = int(waveforms.shape[0])
    n_band = int(band_ffts.shape[0])
    disp_rows = int(displacements.tensors[0].shape[0])
    full_rows = int(geometries.shape[0]) * n_wv * n_band
    if disp_rows != full_rows:
        raise ValueError(f"displacements rows {disp_rows} != full {full_rows}; this script assumes full layout.")

    def stack_target_channels(d: int, w: int, b: int, combined: int) -> torch.Tensor:
        ch0 = eigenfrequency_uniform[d, w, b]
        chans = [displacements.tensors[i][combined] for i in range(4)]
        return torch.stack([ch0, *chans], dim=0)

    tag = args.tag or pt.parent.name
    for loss_name, loss_path in args.loss_array:
        arr = np.load(loss_path)
        losses = arr[:, 4]
        order = np.argsort(losses, kind="stable")
        n = order.shape[0]
        print(f"\n=== {tag}  loss={loss_name}  ({n} samples) ===")
        for case, rank_fn in CASE_RANKS:
            row = int(order[rank_fn(n)])
            comb = int(arr[row, 0]); d = int(arr[row, 1]); w = int(arr[row, 2]); b = int(arr[row, 3])
            val = float(arr[row, 4])

            input_tensor = torch.stack([geometries[d], waveforms[w], band_ffts[b]], dim=0).float()
            output_tensor = predictions[comb].float()
            target_tensor = stack_target_channels(d, w, b, comb).float()

            plt.close("all")
            NO_utilities.visualize_sample(
                input_tensor.cpu(),
                output_tensor.cpu(),
                target_tensor.cpu(),
                unified_colorbar=args.unified_colorbar,
                field_cmap=args.field_cmap,
                diverge_center=args.diverge_center,
                show_eigfreq=args.show_eigfreq,
            )
            nums = plt.get_fignums()
            if len(nums) < 2:
                raise RuntimeError(f"Expected 2 figures from visualize_sample, got {len(nums)}.")
            input_fig = plt.figure(nums[0])
            fields_fig = plt.figure(nums[1])
            if args.title:
                suptitle = (f"{tag} | {loss_name.upper()} {case} | combined={comb} "
                            f"(g={d}, w={w}, b={b}) | loss={val:.6e}")
                input_fig.suptitle(suptitle + "  [inputs]", y=1.02, fontsize=11)
                fields_fig.suptitle(suptitle + "  [target / output / diff]", y=0.965, fontsize=12)

            base = f"{tag}_{loss_name}_{case}_g{d}_w{w}_b{b}"
            input_path = out_dir / f"{base}_input.png"
            fields_path = out_dir / f"{base}_fields.png"
            input_fig.savefig(input_path, dpi=150, bbox_inches="tight")
            fields_fig.savefig(fields_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"  {case:<5} comb={comb:>8} (g={d},w={w},b={b}) loss={val:.6e} -> {input_path.name}, {fields_path.name}")


if __name__ == "__main__":
    main()
