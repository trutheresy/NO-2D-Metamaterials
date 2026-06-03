"""
Overlay TRUE vs PREDICTED dispersion bands on the IBZ contour.

Same contour/interpolation logic as ``plot_dispersions.py``, but each figure
overlays two band sets for the same geometry:

- TRUE bands       : solid lines, one color per band (6 bands -> 6 colors).
- PREDICTED bands  : dashed lines, the SAME color as the corresponding band.

Inputs
------
--true : folder (or *_pt subfolder) holding the ground-truth dataset, providing
         ``eigenvalue_data_full.pt`` (TRUE eigenvalues), ``geometries_full.pt``,
         and ``wavevectors_full.pt``.
--pred : predicted eigenvalues -- either a .pt file or a folder. When a folder,
         ``eigenvalues_predictions_full.pt`` is preferred, else ``eigenvalue_data_full.pt``.

Both eigenvalue tensors are shape ``(N_struct, N_wv, N_eig)`` and share the
ground-truth wavevector grid. One overlay PNG per structure is written, named by
geometry index.

Units note: values are plotted as stored (no Hz/rad-s conversion), matching
``plot_dispersions.py``. The dataset eigenvalues are angular frequency (rad/s),
so the y-axis defaults to ``Frequency [rad/s]`` (override with --ylabel).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.distance import cdist

from plot_dispersion_with_eigenfrequencies_reduced_set import get_IBZ_contour_wavevectors


# One color per band index; TRUE (solid) and PRED (dashed) share the band's color.
BAND_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
]


def resolve_pt_dir_with(folder: Path, required: str) -> Path:
    """Return folder (or a *_pt subfolder) that contains ``required``."""
    if (folder / required).exists():
        return folder
    pt_dirs = [p for p in folder.iterdir() if p.is_dir() and p.name.endswith("_pt") and (p / required).exists()]
    if not pt_dirs:
        raise FileNotFoundError(f"Could not find {required} in {folder} or a *_pt subfolder.")
    return max(pt_dirs, key=lambda p: p.stat().st_mtime)


def load_true_dir(true_arg: str):
    folder = Path(true_arg)
    pt_dir = resolve_pt_dir_with(folder, "eigenvalue_data_full.pt")
    geometries = torch.load(pt_dir / "geometries_full.pt", map_location="cpu", weights_only=False).to(torch.float32).numpy()
    wavevectors = torch.load(pt_dir / "wavevectors_full.pt", map_location="cpu", weights_only=False).to(torch.float32).numpy()
    eigen_true = torch.load(pt_dir / "eigenvalue_data_full.pt", map_location="cpu", weights_only=False).to(torch.float32).numpy()
    return geometries, wavevectors, eigen_true, pt_dir


def load_pred(pred_arg: str) -> np.ndarray:
    p = Path(pred_arg)
    if p.is_dir():
        for name in ("eigenvalues_predictions_full.pt", "eigenvalue_data_full.pt"):
            if (p / name).exists():
                p = p / name
                break
        else:
            raise FileNotFoundError(
                f"No eigenvalues_predictions_full.pt or eigenvalue_data_full.pt in {pred_arg}"
            )
    return torch.load(p, map_location="cpu", weights_only=False).to(torch.float32).numpy(), p


def interp_to_contour(wavevectors: np.ndarray, frequencies: np.ndarray, contour_wv: np.ndarray) -> np.ndarray:
    """Interpolate per-band frequencies onto the contour (linear + nearest fallback).

    Mirrors plot_dispersions.py: scatteredInterpolant-style linear interpolation
    with a nearest-neighbor fill outside the convex hull.
    """
    n_bands = frequencies.shape[1]
    out = np.zeros((len(contour_wv), n_bands), dtype=np.float32)
    wv32 = wavevectors.astype(np.float32)
    cwv32 = contour_wv.astype(np.float32)
    for b in range(n_bands):
        interp = LinearNDInterpolator(wv32, frequencies[:, b].astype(np.float32), fill_value=np.nan)
        vals = np.asarray(interp(cwv32), dtype=np.float32)
        nan_mask = np.isnan(vals)
        if np.any(nan_mask):
            nan_idx = np.where(nan_mask)[0]
            dist = cdist(cwv32[nan_idx], wv32)
            nearest = np.argmin(dist, axis=1)
            vals[nan_idx] = frequencies[nearest, b].astype(np.float32)
        out[:, b] = vals
    return out


def plot_overlay(ax, contour_info, true_c, pred_c, title, ylabel, mark_points):
    x = contour_info["wavevector_parameter"]
    n_bands = min(true_c.shape[1], pred_c.shape[1])
    for b in range(n_bands):
        color = BAND_COLORS[b % len(BAND_COLORS)]
        ax.plot(x, true_c[:, b], "-", color=color, linewidth=2)
        ax.plot(x, pred_c[:, b], "--", color=color, linewidth=2)
        if mark_points:
            ax.plot(x, true_c[:, b], "o", color=color, markersize=3, markeredgewidth=0.5, markeredgecolor="white")

    for i in range(contour_info["N_segment"] + 1):
        ax.axvline(i, color="k", linestyle="--", alpha=0.3, linewidth=1)

    if contour_info.get("vertex_labels"):
        vertex_positions = np.linspace(0, contour_info["N_segment"], len(contour_info["vertex_labels"]))
        ax.set_xticks(vertex_positions)
        ax.set_xticklabels(contour_info["vertex_labels"])

    ax.set_xlabel("Wavevector Contour Parameter", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color="k", linestyle="-", linewidth=2, label="True"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=2, label="Predicted"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)


def main(
    true_dir: str,
    pred: str,
    n_structs: int | None = None,
    title: str = "",
    output_dir: str | None = None,
    ylabel: str = "Frequency [rad/s]",
    mark_points: bool = False,
) -> None:
    geometries, wavevectors, eigen_true, true_pt = load_true_dir(true_dir)
    eigen_pred, pred_path = load_pred(pred)

    n_true = eigen_true.shape[0]
    n_pred = eigen_pred.shape[0]
    if n_true != n_pred:
        print(f"Warning: true N_struct ({n_true}) != predicted N_struct ({n_pred}); using the minimum.")
    n_total = min(n_true, n_pred, geometries.shape[0])
    n_plot = n_total if n_structs is None else min(int(n_structs), n_total)

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = Path.cwd() / "PLOTS" / f"{Path(true_dir).name}_true_vs_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"True eigenvalues : {true_pt / 'eigenvalue_data_full.pt'}  shape={eigen_true.shape}")
    print(f"Pred eigenvalues : {pred_path}  shape={eigen_pred.shape}")
    print(f"Plotting {n_plot} structures -> {out_dir}")

    contour_wv, contour_info = get_IBZ_contour_wavevectors(10, 1.0, "p4mm")

    for struct_idx in range(n_plot):
        wv = wavevectors[struct_idx]
        true_c = interp_to_contour(wv, eigen_true[struct_idx], contour_wv)
        pred_c = interp_to_contour(wv, eigen_pred[struct_idx], contour_wv)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        plot_overlay(ax, contour_info, true_c, pred_c, title, ylabel, mark_points)
        fig.savefig(out_dir / f"{struct_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay true (solid) vs predicted (dashed) dispersion bands.")
    parser.add_argument("--true", required=True, help="Ground-truth dataset folder (or *_pt) with eigenvalue_data_full.pt, geometries_full.pt, wavevectors_full.pt.")
    parser.add_argument("--pred", required=True, help="Predicted eigenvalues: a .pt file or a folder containing eigenvalues_predictions_full.pt / eigenvalue_data_full.pt.")
    parser.add_argument("-n", "--n-structs", type=int, default=None, help="Number of structures to plot (default: all).")
    parser.add_argument("-t", "--title", type=str, default="", help="Custom plot title (default: blank).")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output folder for plots.")
    parser.add_argument("--ylabel", type=str, default="Frequency [rad/s]", help="Y-axis label (data is angular frequency, rad/s).")
    parser.add_argument("--mark-points", action="store_true", help="Add markers on the true bands.")
    args = parser.parse_args()
    main(args.true, args.pred, args.n_structs, args.title, args.output_dir, args.ylabel, args.mark_points)
