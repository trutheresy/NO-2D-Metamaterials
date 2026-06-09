"""
DIAGNOSTIC SCRIPT - safe to delete on cleanup passes (not part of the core pipeline).

Diagnostic for compute_boundary_length.py: run on the first N geometries of a discrete
dataset, print the 0/1 boundary-length count for each, and save an image per geometry with
the boundary edges (interfaces between 0 and 1 pixels) highlighted on top of the geometry.

The number of drawn segments equals the interior boundary-length count, so the picture is a
visual cross-check of the computed value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from compute_boundary_length import load_geometries, to_binary, boundary_length

B_TEST_GEOM = r"DATASETS\b_test\binarized_2026-03-08_16-34-27_pt\geometries_full.pt"


def boundary_segments(g: np.ndarray, periodic: bool):
    """Line segments (in imshow data coords) lying on every 0/1 interface edge."""
    segs = []
    # vertical interface at x=j+0.5 between columns j and j+1
    ii, jj = np.nonzero(g[:, :-1] != g[:, 1:])
    segs += [[(j + 0.5, i - 0.5), (j + 0.5, i + 0.5)] for i, j in zip(ii, jj)]
    # horizontal interface at y=i+0.5 between rows i and i+1
    ii, jj = np.nonzero(g[:-1, :] != g[1:, :])
    segs += [[(j - 0.5, i + 0.5), (j + 0.5, i + 0.5)] for i, j in zip(ii, jj)]
    if periodic:
        h, w = g.shape
        for i in np.nonzero(g[:, 0] != g[:, -1])[0]:
            segs.append([(-0.5, i - 0.5), (-0.5, i + 0.5)])
        for j in np.nonzero(g[0, :] != g[-1, :])[0]:
            segs.append([(j - 0.5, -0.5), (j + 0.5, -0.5)])
    return segs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometries", default=B_TEST_GEOM, help="Discrete geometries tensor (.pt). Default: b_test.")
    p.add_argument("--num", type=int, default=40, help="Number of leading geometries to diagnose (default 40).")
    p.add_argument("--output-dir", default="", help="Folder for the PNGs (default: <geometries-dir>/boundary_diagnostic).")
    p.add_argument("--periodic", action="store_true", help="Also draw/count wrap-around border edges.")
    p.add_argument("--title", action="store_true", help="Add a title to each plot (default: no title).")
    p.add_argument("--threshold", type=float, default=None, help="Binarize a non-binary field at this threshold.")
    args = p.parse_args()

    geom_path = Path(args.geometries)
    g_all = to_binary(load_geometries(geom_path), args.threshold)
    n = min(args.num, int(g_all.shape[0]))
    g_sel = g_all[:n]
    lengths = boundary_length(g_sel, args.periodic, 1.0)

    out_dir = Path(args.output_dir) if args.output_dir else geom_path.parent / "boundary_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "periodic" if args.periodic else "interior"
    print(f"Geometries : {geom_path}")
    print(f"First {n} geometries  (mode: {mode} edges)")
    for i in range(n):
        gi = g_sel[i].cpu().numpy()
        segs = boundary_segments(gi, args.periodic)
        count = int(round(float(lengths[i])))
        assert len(segs) == count, f"segment/count mismatch geom {i}: {len(segs)} vs {count}"
        print(f"  geom {i:3d}:  boundary length = {count}")

        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.imshow(gi, cmap="gray", vmin=0, vmax=1, interpolation="nearest", origin="upper")
        ax.add_collection(LineCollection(segs, colors="red", linewidths=2.0))
        if args.title:
            ax.set_title(f"geom {i}  -  boundary length = {count}")
        ax.set_xticks([]); ax.set_yticks([])
        out_path = out_dir / f"geom{i:03d}_boundary_{count}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Saved {n} images to {out_dir}")


if __name__ == "__main__":
    main()
