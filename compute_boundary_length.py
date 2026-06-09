"""
Compute the total 0/1 boundary (interface) length of discrete (binarized) geometries.

For each geometry the boundary length is the number of grid edges separating a 0-pixel
from a 1-pixel: every pair of 4-connected adjacent pixels whose values differ contributes
one unit of length (scaled by --pixel-size).

Input geometries tensor (.pt) is expected to have shape (n_geom, H, W) with binary values
{0, 1}. Non-binary inputs can be thresholded with --threshold.

By default only interior edges are counted (the literal "edges between adjacent pixels in
the image"). Use --periodic to additionally count wrap-around edges between opposite
borders, which is the physically meaningful interface length for a periodic unit cell.

Outputs a (n_geom, 2) .npy array of columns (geom_idx, boundary_length) and prints summary
statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_geometries(path: Path) -> torch.Tensor:
    """Load a geometries tensor of shape (n_geom, H, W) from common container types."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        g = obj
    elif hasattr(obj, "tensors") and len(obj.tensors) > 0:  # TensorDataset
        g = obj.tensors[0]
    elif isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
        g = torch.stack(list(obj), dim=0)
    else:
        raise TypeError(f"Unsupported geometries container: {type(obj)}")
    if g.ndim == 2:
        g = g.unsqueeze(0)
    if g.ndim != 3:
        raise ValueError(f"Expected geometries of shape (n_geom, H, W); got {tuple(g.shape)}.")
    return g


def to_binary(g: torch.Tensor, threshold: float | None) -> torch.Tensor:
    """Return a {0,1} int8 tensor, validating binarity unless a threshold is given."""
    gf = g.float()
    if threshold is not None:
        return (gf >= threshold).to(torch.int8)
    uniq = torch.unique(gf)
    extra = uniq[(uniq != 0) & (uniq != 1)]
    if extra.numel() > 0:
        raise ValueError(
            f"Geometries are not binary (found values {extra[:8].tolist()}...). "
            f"Pass --threshold to binarize a continuous field."
        )
    return gf.to(torch.int8)


def boundary_length(g: torch.Tensor, periodic: bool, pixel_size: float) -> np.ndarray:
    """Per-geometry 0/1 interface length. g is (n_geom, H, W) binary int8."""
    diff_h = (g[:, :, 1:] != g[:, :, :-1]).sum(dim=(1, 2))   # edges between horizontally-adjacent pixels
    diff_v = (g[:, 1:, :] != g[:, :-1, :]).sum(dim=(1, 2))   # edges between vertically-adjacent pixels
    edges = diff_h + diff_v
    if periodic:
        wrap_h = (g[:, :, 0] != g[:, :, -1]).sum(dim=1)      # left<->right wrap
        wrap_v = (g[:, 0, :] != g[:, -1, :]).sum(dim=1)      # top<->bottom wrap
        edges = edges + wrap_h + wrap_v
    return (edges.to(torch.float64) * pixel_size).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometries", required=True, help="Discrete geometries tensor (.pt), shape (n_geom, H, W), binary {0,1}.")
    p.add_argument("--output", default="", help="Output .npy path (default: <geometries-dir>/boundary_length[_periodic].npy).")
    p.add_argument("--periodic", action="store_true", help="Also count wrap-around edges between opposite borders (periodic unit cell).")
    p.add_argument("--pixel-size", type=float, default=1.0, help="Length of one pixel edge (default 1.0 = count of edges).")
    p.add_argument("--threshold", type=float, default=None, help="Binarize a non-binary field at this threshold (g >= threshold -> 1).")
    args = p.parse_args()

    geom_path = Path(args.geometries)
    g = load_geometries(geom_path)
    gb = to_binary(g, args.threshold)
    n_geom, h, w = (int(s) for s in gb.shape)

    lengths = boundary_length(gb, args.periodic, args.pixel_size)
    geom_idx = np.arange(n_geom, dtype=np.float64)
    out_arr = np.stack([geom_idx, lengths], axis=1)

    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "_periodic" if args.periodic else ""
        out_path = geom_path.parent / f"boundary_length{suffix}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out_arr)

    print(f"Geometries : {geom_path}  shape=({n_geom}, {h}, {w})")
    print(f"Mode       : {'periodic (with wrap edges)' if args.periodic else 'interior edges only'}   pixel_size={args.pixel_size}")
    print(f"Output     : {out_path}  shape={out_arr.shape}  (columns: geom_idx, boundary_length)")
    pct = np.percentile(lengths, [0, 25, 50, 75, 100])
    print(
        f"Boundary length  mean={lengths.mean():.4f}  std={lengths.std():.4f}\n"
        f"  min={pct[0]:.4f}  p25={pct[1]:.4f}  median={pct[2]:.4f}  p75={pct[3]:.4f}  max={pct[4]:.4f}"
    )
    imin = int(np.argmin(lengths)); imax = int(np.argmax(lengths))
    print(f"  argmin: geom {imin} (length {lengths[imin]:.4f})   argmax: geom {imax} (length {lengths[imax]:.4f})")


if __name__ == "__main__":
    main()
