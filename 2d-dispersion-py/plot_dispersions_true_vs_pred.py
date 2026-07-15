"""
Overlay TRUE vs PREDICTED dispersion bands on the IBZ contour.

Same contour logic as ``plot_dispersions.py``, but bands are sampled only at wavevectors
from the dataset grid that lie on the p4mm Γ–X–M–Γ path (no interpolation onto
synthetic contour points). Each figure overlays two band sets for the same geometry:

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
ground-truth wavevector grid. One overlay PNG per structure is written with a
filename encoding two loss rankings plus geometry index, e.g.
``NMAE001_NMSE042_g123.png`` (by default ranked by NMAE then NMSE).

Also writes ``dispersion_losses.csv`` in the same output folder with per-geometry
MAE, MSE, NMAE, and NMSE plus rank columns (1 = lowest loss / best). Losses are
computed over all wavevector x band points on the full grid (not the IBZ contour
used for plotting).

Units note: values are plotted as stored (no Hz/rad-s conversion), matching
``plot_dispersions.py``. The dataset eigenvalues are angular frequency (rad/s),
so the y-axis defaults to ``Frequency [rad/s]`` (override with --ylabel).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from output_layout import resolve_script_output_dir

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_dispersion_with_eigenfrequencies_reduced_set import select_p4mm_contour_from_grid


# One color per band index; TRUE (solid) and PRED (dashed) share the band's color.
BAND_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
]

LOSS_NAMES = ("mae", "mse", "nmae", "nmse")
DEFAULT_RANK_LOSS_PRIMARY = "nmae"
DEFAULT_RANK_LOSS_SECONDARY = "nmse"
DEFAULT_NMSE_EPS = 1e-5


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


def eigenvalues_on_contour(eigenvalues: np.ndarray, contour_indices: np.ndarray) -> np.ndarray:
    """Gather per-band eigenvalues at contour wavevector indices."""
    return eigenvalues[contour_indices].astype(np.float32, copy=False)


def compute_geometry_dispersion_losses(
    eigen_true: np.ndarray,
    eigen_pred: np.ndarray,
    n_structs: int,
    nmae_eps: float = 1e-5,
    nmse_eps: float = DEFAULT_NMSE_EPS,
) -> dict[str, np.ndarray]:
    """Per-geometry MAE, MSE, NMAE, NMSE over all wavevector x band scalar points."""
    true_g = eigen_true[:n_structs].astype(np.float32, copy=False)
    pred_g = eigen_pred[:n_structs].astype(np.float32, copy=False)
    err = pred_g - true_g
    flat_true = true_g.reshape(n_structs, -1)
    flat_err = err.reshape(n_structs, -1)

    mae = np.abs(flat_err, dtype=np.float32).mean(axis=1)
    mse = np.square(flat_err, dtype=np.float32).mean(axis=1)
    nmae = mae / (np.abs(flat_true, dtype=np.float32).mean(axis=1) + np.float32(nmae_eps))
    nmse = mse / (np.square(flat_true, dtype=np.float32).mean(axis=1) + np.float32(nmse_eps))
    return {
        "mae": mae.astype(np.float32),
        "mse": mse.astype(np.float32),
        "nmae": nmae.astype(np.float32),
        "nmse": nmse.astype(np.float32),
    }


def compute_loss_ranks(losses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return per-loss ranks (1 = lowest loss / best) using stable sort."""
    ranks: dict[str, np.ndarray] = {}
    for name in LOSS_NAMES:
        vals = losses[name]
        n = int(vals.shape[0])
        order = np.argsort(vals, kind="stable")
        rank = np.empty(n, dtype=np.int32)
        rank[order] = np.arange(1, n + 1, dtype=np.int32)
        ranks[name] = rank
    return ranks


def validate_rank_loss(name: str) -> str:
    key = name.lower()
    if key not in LOSS_NAMES:
        raise ValueError(f"Unknown rank loss {name!r}; choose from {', '.join(LOSS_NAMES)}.")
    return key


def plot_filename(
    geom_idx: int,
    ranks: dict[str, np.ndarray],
    rank_primary: str,
    rank_secondary: str,
) -> str:
    """Build overlay PNG name, e.g. NMAE001_NMSE042_g123.png."""
    return (
        f"{rank_primary.upper()}{ranks[rank_primary][geom_idx]:03d}_"
        f"{rank_secondary.upper()}{ranks[rank_secondary][geom_idx]:03d}_"
        f"g{geom_idx:03d}.png"
    )


def save_dispersion_losses_csv(
    out_dir: Path,
    losses: dict[str, np.ndarray],
    ranks: dict[str, np.ndarray],
) -> Path:
    """Write geometry_index, four losses, and per-loss ranks to dispersion_losses.csv."""
    n = int(losses["mae"].shape[0])
    csv_path = out_dir / "dispersion_losses.csv"
    header = ["geometry_index", *LOSS_NAMES, *[f"rank_{name}" for name in LOSS_NAMES]]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for geom_idx in range(n):
            writer.writerow(
                [geom_idx]
                + [f"{float(losses[name][geom_idx]):.12e}" for name in LOSS_NAMES]
                + [int(ranks[name][geom_idx]) for name in LOSS_NAMES]
            )
    return csv_path


def plot_overlay(
    ax,
    contour_info,
    true_c,
    pred_c,
    title,
    ylabel,
    mark_points,
    xlabel: str = "Wavevector Contour Parameter",
    label_fontsize: float = 12,
    title_fontsize: float = 14,
    legend_fontsize: float = 10,
    tick_fontsize: float = 10,
    legend_loc: str = "upper right",
    legend_ncol: int = 1,
):
    x = contour_info["wavevector_parameter"]
    n_bands = min(true_c.shape[1], pred_c.shape[1])
    for b in range(n_bands):
        color = BAND_COLORS[b % len(BAND_COLORS)]
        ax.plot(x, true_c[:, b], "-", color=color, linewidth=2)
        ax.plot(x, pred_c[:, b], "--", color=color, linewidth=2)
        if mark_points:
            ax.plot(x, true_c[:, b], "o", color=color, markersize=3, markeredgewidth=0.5, markeredgecolor="white")

    for i in contour_info.get("segment_vertex_params", np.arange(contour_info["N_segment"] + 1)):
        ax.axvline(float(i), color="k", linestyle="--", alpha=0.3, linewidth=1)

    if contour_info.get("vertex_labels"):
        vertex_positions = contour_info.get("segment_vertex_params")
        if vertex_positions is None:
            vertex_positions = np.linspace(0, contour_info["N_segment"], len(contour_info["vertex_labels"]))
        ax.set_xticks(vertex_positions)
        ax.set_xticklabels(contour_info["vertex_labels"])

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color="k", linestyle="-", linewidth=2, label="True"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=2, label="Predicted"),
    ]
    legend_kw = dict(handles=legend_handles, loc=legend_loc, fontsize=legend_fontsize, ncol=legend_ncol)
    if legend_loc == "lower center":
        # Sit legend at the center-bottom of the axes (inside the plot area).
        legend_kw["bbox_to_anchor"] = (0.5, 0.02)
        legend_kw["framealpha"] = 0.92
    ax.legend(**legend_kw)


def main(
    true_dir: str,
    pred: str,
    n_structs: int | None = None,
    title: str = "",
    output_dir: str | None = None,
    ylabel: str = "Frequency [rad/s]",
    mark_points: bool = False,
    nmae_eps: float = 1e-5,
    nmse_eps: float = DEFAULT_NMSE_EPS,
    rank_primary: str = DEFAULT_RANK_LOSS_PRIMARY,
    rank_secondary: str = DEFAULT_RANK_LOSS_SECONDARY,
    square: bool = False,
    xlabel: str = "Wavevector Contour Parameter",
    larger_fonts: bool = False,
    legend_loc: str = "upper right",
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

    rank_primary = validate_rank_loss(rank_primary)
    rank_secondary = validate_rank_loss(rank_secondary)
    if rank_primary == rank_secondary:
        raise ValueError("rank-primary and rank-secondary must be different losses.")

    losses = compute_geometry_dispersion_losses(
        eigen_true, eigen_pred, n_total, nmae_eps=nmae_eps, nmse_eps=nmse_eps
    )
    ranks = compute_loss_ranks(losses)
    csv_path = save_dispersion_losses_csv(out_dir, losses, ranks)
    print(f"Saved losses CSV : {csv_path}  ({n_total} geometries)")
    for name in LOSS_NAMES:
        vals = losses[name]
        print(
            f"  {name.upper():4s} mean={vals.mean():.6e}, min={vals.min():.6e}, max={vals.max():.6e}"
        )

    contour_indices, _, contour_info = select_p4mm_contour_from_grid(wavevectors)
    print(
        f"Contour grid pts : {contour_info['n_contour_points']} along path "
        f"({contour_info['n_unique_contour_points']} unique k of {wavevectors.shape[1]})"
    )

    # Square default 8x8; narrowed width ≈0.8*8 → 6.5 (keep square proportions).
    figsize = (6.5, 6.5) if square else (10.0, 6.0)
    label_fs = 18.0 if larger_fonts else 12.0
    title_fs = 17.0 if larger_fonts else 14.0
    legend_fs = 16.0 if larger_fonts else 10.0
    tick_fs = 16.0 if larger_fonts else 10.0
    legend_ncol = 2 if legend_loc == "lower center" else 1

    for struct_idx in range(n_plot):
        true_c = eigenvalues_on_contour(eigen_true[struct_idx], contour_indices)
        pred_c = eigenvalues_on_contour(eigen_pred[struct_idx], contour_indices)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plot_overlay(
            ax,
            contour_info,
            true_c,
            pred_c,
            title,
            ylabel,
            mark_points,
            xlabel=xlabel,
            label_fontsize=label_fs,
            title_fontsize=title_fs,
            legend_fontsize=legend_fs,
            tick_fontsize=tick_fs,
            legend_loc=legend_loc,
            legend_ncol=legend_ncol,
        )
        png_name = plot_filename(struct_idx, ranks, rank_primary, rank_secondary)
        fig.savefig(out_dir / png_name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay true (solid) vs predicted (dashed) dispersion bands.")
    parser.add_argument("--true", required=True, help="Ground-truth dataset folder (or *_pt) with eigenvalue_data_full.pt, geometries_full.pt, wavevectors_full.pt.")
    parser.add_argument("--pred", required=True, help="Predicted eigenvalues: a .pt file or a folder containing eigenvalues_predictions_full.pt / eigenvalue_data_full.pt.")
    parser.add_argument("-n", "--n-structs", type=int, default=None, help="Number of structures to plot (default: all).")
    parser.add_argument("-t", "--title", type=str, default="", help="Custom plot title (default: blank).")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Explicit output folder (overrides the model/dataset layout below).")
    parser.add_argument("--model-name", type=str, default="", help="Model name for the PLOTS/<model>/<dataset>/<subdir> layout.")
    parser.add_argument("--dataset", type=str, default="", help="Dataset folder for the layout (e.g. c_test / b_test).")
    parser.add_argument("--output-subdir", type=str, default="dispersion_overlay", help="Script output folder name under PLOTS/<model>/<dataset> (default: dispersion_overlay).")
    parser.add_argument("--ylabel", type=str, default="Frequency [rad/s]", help="Y-axis label (data is angular frequency, rad/s).")
    parser.add_argument("--nmae-eps", type=float, default=1e-5, help="Epsilon added to mean(|true|) in per-geometry eigenvalue NMAE (default 1e-5).")
    parser.add_argument("--nmse-eps", type=float, default=DEFAULT_NMSE_EPS, help="Epsilon added to mean(true^2) in per-geometry eigenvalue NMSE (default 1e-5).")
    parser.add_argument(
        "--rank-primary",
        type=str,
        default=DEFAULT_RANK_LOSS_PRIMARY,
        choices=LOSS_NAMES,
        help="Loss used for the first rank segment in PNG filenames (default: nmae).",
    )
    parser.add_argument(
        "--rank-secondary",
        type=str,
        default=DEFAULT_RANK_LOSS_SECONDARY,
        choices=LOSS_NAMES,
        help="Loss used for the second rank segment in PNG filenames (default: nmse).",
    )
    parser.add_argument("--mark-points", action="store_true", help="Add markers on the true bands.")
    parser.add_argument("--square", action="store_true", help="Use a square figure aspect ratio.")
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Wavevector Contour Parameter",
        help='X-axis label (default: "Wavevector Contour Parameter").',
    )
    parser.add_argument(
        "--larger-fonts",
        action="store_true",
        help="Slightly larger axis/title/legend/tick fonts.",
    )
    parser.add_argument(
        "--legend-loc",
        type=str,
        default="upper right",
        choices=("upper right", "lower center", "upper left", "lower left", "lower right", "best"),
        help="Legend location (default: upper right).",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None and args.model_name:
        output_dir = str(resolve_script_output_dir(
            explicit=None,
            category="plots",
            model_name=args.model_name,
            dataset=args.dataset,
            subdir=args.output_subdir,
        ))
    main(
        args.true,
        args.pred,
        args.n_structs,
        args.title,
        output_dir,
        args.ylabel,
        args.mark_points,
        args.nmae_eps,
        args.nmse_eps,
        args.rank_primary,
        args.rank_secondary,
        args.square,
        args.xlabel,
        args.larger_fonts,
        args.legend_loc,
    )
