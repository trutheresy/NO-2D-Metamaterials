"""
Smoke test for SISSI on 32x32 four-quadrant +/-1 patterns.

Builds a composite figure: for each of the 16 sign combinations, shows target vs
prediction side-by-side (match and sign-flip rows) with SISSI in each panel subtitle.

Default smoke-test constants: C1 = C2 = 1e-6 (override via --c1 / --c2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sissi_loss import kernel_size_from_window_radius, sissi_index, sigma_for_window_radius

SMOKE_C1 = 1e-6
SMOKE_C2 = 1e-6
DEFAULT_WINDOW_RADIUS = 5
DEFAULT_OUTPUT_DIR = Path("PLOTS")
DEFAULT_OUTPUT_STEM = "sissi_quadrant_smoke"


def quadrant_pattern_32(signs: tuple[int, int, int, int]) -> torch.Tensor:
    """32x32 field with four 16x16 quadrants taking values in {+1, -1}."""
    s1, s2, s3, s4 = signs
    top = torch.cat(
        [torch.full((16, 16), float(s1)), torch.full((16, 16), float(s2))],
        dim=1,
    )
    bot = torch.cat(
        [torch.full((16, 16), float(s3)), torch.full((16, 16), float(s4))],
        dim=1,
    )
    return torch.cat([top, bot], dim=0)


def pattern_label(signs: tuple[int, int, int, int]) -> str:
    return "".join("+" if s > 0 else "-" for s in signs)


def all_quadrant_signs() -> list[tuple[int, int, int, int]]:
    return [
        tuple(1 if (bits >> (3 - k)) & 1 else -1 for k in range(4))  # type: ignore[return-value]
        for bits in range(16)
    ]


def output_path_for_radius(
    output_dir: Path,
    output_stem: str,
    window_radius: int,
) -> Path:
    return output_dir / f"{output_stem}_r{window_radius}.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SISSI quadrant-pattern smoke test plot")
    p.add_argument("--c1", type=float, default=SMOKE_C1, help="SISSI stability constant C1")
    p.add_argument("--c2", type=float, default=SMOKE_C2, help="SISSI stability constant C2")
    p.add_argument(
        "--window-radius",
        type=int,
        default=None,
        help="Single Gaussian window radius in pixels (kernel size = 2*radius+1)",
    )
    p.add_argument(
        "--window-radii",
        type=int,
        nargs="+",
        default=None,
        help="Multiple window radii to plot (e.g. --window-radii 2 3 4)",
    )
    p.add_argument(
        "--comparison",
        choices=("flip", "match", "both"),
        default="both",
        help="pred vs target relation (default: both match and sign-flip)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output PNGs",
    )
    p.add_argument(
        "--output-stem",
        type=str,
        default=DEFAULT_OUTPUT_STEM,
        help="Filename stem; window radius is appended as _r{N}",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output path (single --window-radius only; no _r suffix added)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-pattern score tables",
    )
    return p.parse_args()


def resolve_window_radii(args: argparse.Namespace) -> list[int]:
    if args.window_radii is not None:
        if args.window_radius is not None:
            raise SystemExit("Use only one of --window-radius or --window-radii")
        return list(args.window_radii)
    if args.window_radius is not None:
        return [args.window_radius]
    return [DEFAULT_WINDOW_RADIUS]


def plot_case(
    ax: plt.Axes,
    target: torch.Tensor,
    pred: torch.Tensor,
    *,
    pattern: str,
    comparison: str,
    c1: float,
    c2: float,
    window_radius: int,
) -> float:
    """Draw target | pred on ax; return SISSI score."""
    t = target.view(1, 1, 32, 32)
    p = pred.view(1, 1, 32, 32)
    score = float(
        sissi_index(
            p,
            t,
            c1=c1,
            c2=c2,
            window_radius=window_radius,
        )
    )

    combined = np.concatenate(
        [target.detach().cpu().numpy(), pred.detach().cpu().numpy()],
        axis=1,
    )
    ax.imshow(combined, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.axvline(31.5, color="k", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    comp_label = "pred = target" if comparison == "match" else "pred = -target"
    ax.set_title(
        f"{pattern}  |  {comp_label}\nSISSI = {score:.4f}",
        fontsize=8,
    )
    ax.text(16, 33, "target", ha="center", va="bottom", fontsize=7, transform=ax.transData)
    ax.text(48, 33, "pred", ha="center", va="bottom", fontsize=7, transform=ax.transData)
    return score


def run_smoke_plot(
    *,
    window_radius: int,
    output: Path,
    c1: float,
    c2: float,
    comparison: str,
    quiet: bool,
) -> list[tuple[str, str, float]]:
    signs_list = all_quadrant_signs()
    comparisons: list[str]
    if comparison == "both":
        comparisons = ["match", "flip"]
    else:
        comparisons = [comparison]

    kernel_size = kernel_size_from_window_radius(window_radius)
    sigma = sigma_for_window_radius(window_radius)

    nrows = 4 * len(comparisons)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.2))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    comp_line = (
        "match (top) / sign-flip (bottom) per pattern"
        if len(comparisons) == 2
        else comparisons[0]
    )
    fig.suptitle(
        f"SISSI quadrant smoke test  (C1={c1:.1e}, C2={c2:.1e})\n"
        f"Gaussian window radius={window_radius} ({kernel_size}x{kernel_size}, sigma={sigma:.3f})  —  {comp_line}",
        fontsize=11,
        y=0.995,
    )

    scores: list[tuple[str, str, float]] = []
    for pattern_idx, signs in enumerate(signs_list):
        col = pattern_idx % ncols
        block = pattern_idx // ncols
        label = pattern_label(signs)
        target = quadrant_pattern_32(signs)
        for comp_i, comp in enumerate(comparisons):
            row = block * len(comparisons) + comp_i
            pred = target if comp == "match" else -target
            score = plot_case(
                axes[row, col],
                target,
                pred,
                pattern=label,
                comparison=comp,
                c1=c1,
                c2=c2,
                window_radius=window_radius,
            )
            scores.append((label, comp, score))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output.resolve()}")
    print(
        f"Window: radius={window_radius}, kernel={kernel_size}x{kernel_size}, sigma={sigma:.4f}"
    )
    if not quiet:
        print(f"{'pattern':>8}  {'comparison':>8}  {'SISSI':>10}")
        for label, comp, score in scores:
            print(f"{label:>8}  {comp:>8}  {score:10.4f}")
    return scores


def main() -> None:
    args = parse_args()
    radii = resolve_window_radii(args)

    if args.output is not None:
        if len(radii) != 1:
            raise SystemExit("--output requires exactly one window radius")
        outputs = [args.output]
    else:
        outputs = [
            output_path_for_radius(args.output_dir, args.output_stem, r) for r in radii
        ]

    for window_radius, output in zip(radii, outputs):
        run_smoke_plot(
            window_radius=window_radius,
            output=output,
            c1=args.c1,
            c2=args.c2,
            comparison=args.comparison,
            quiet=args.quiet and len(radii) > 1,
        )


if __name__ == "__main__":
    main()
