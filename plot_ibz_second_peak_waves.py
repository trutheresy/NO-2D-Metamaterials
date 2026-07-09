"""
Plot the IBZ wavevector grid with high-second-peak wavevectors highlighted.

Reads the per-wavevector tables produced by ``analyze_second_peak_waves.py``
(``second_peak_<dataset>_wave_table.csv``) for one or more datasets of a model,
marks wavevectors whose second-peak membership exceeds ``--threshold-pct`` in
*all* datasets (overlap, red) or in a single dataset only (per-dataset colors),
and writes one IBZ map PNG.

Defaults to the standard layout:
  input : INFERENCE/<model>/<dataset>/second_peak_analysis/second_peak_<dataset>_wave_table.csv
  output: PLOTS/<model>/second_peak_analysis/second_peak_ibz_map.png

Usage:
  python plot_ibz_second_peak_waves.py --model-name <MODEL> --datasets c_test b_test
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from output_layout import INFERENCE_ROOT, resolve_output_dir

SINGLE_COLORS = ["#9b59b6", "#e67e22", "#16a085", "#2c3e50"]


def load_wave_table(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = int(row["wave"])
            rows[w] = {
                "kx": float(row["kx"]),
                "ky": float(row["ky"]),
                "frac_second_pct": float(row["frac_second_pct"]),
            }
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", required=True, help="Model folder name under INFERENCE/ and PLOTS/.")
    p.add_argument("--datasets", nargs="+", default=["c_test", "b_test"],
                   help="Dataset tags to combine (default: c_test b_test).")
    p.add_argument("--wave-tables", nargs="*", default=[],
                   help="Explicit wave-table CSV paths (overrides the standard layout; one per dataset).")
    p.add_argument("--threshold-pct", type=float, default=50.0,
                   help="Highlight waves above this second-peak membership percent (default 50).")
    p.add_argument("--output-dir", default="", help="Explicit output folder (overrides PLOTS layout).")
    p.add_argument("--output-subdir", default="second_peak_analysis")
    p.add_argument("--out-name", default="second_peak_ibz_map.png")
    args = p.parse_args()

    if args.wave_tables:
        if len(args.wave_tables) != len(args.datasets):
            raise SystemExit("--wave-tables must have one path per dataset.")
        table_paths = [Path(t) for t in args.wave_tables]
    else:
        table_paths = [
            INFERENCE_ROOT / args.model_name / ds / "second_peak_analysis" / f"second_peak_{ds}_wave_table.csv"
            for ds in args.datasets
        ]
    for tp in table_paths:
        if not tp.exists():
            raise SystemExit(f"Missing wave table: {tp}\nRun analyze_second_peak_waves.py first.")

    tables = {ds: load_wave_table(tp) for ds, tp in zip(args.datasets, table_paths)}
    waves = sorted(next(iter(tables.values())).keys())
    kx = np.array([tables[args.datasets[0]][w]["kx"] for w in waves])
    ky = np.array([tables[args.datasets[0]][w]["ky"] for w in waves])
    kmax = float(np.abs(np.concatenate([kx, ky])).max())
    kx_n, ky_n = kx / kmax, ky / kmax

    hot_sets = {
        ds: {w for w in waves if tables[ds][w]["frac_second_pct"] > args.threshold_pct}
        for ds in args.datasets
    }
    overlap = set.intersection(*hot_sets.values()) if len(hot_sets) > 1 else set()
    singles = {ds: hot_sets[ds] - overlap for ds in args.datasets}
    plain = [w for w in waves if w not in overlap and all(w not in s for s in singles.values())]

    out_dir = Path(args.output_dir) if args.output_dir else resolve_output_dir(
        "plots", args.model_name, "", args.output_subdir
    )
    out_path = out_dir / args.out_name

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.add_patch(Rectangle((-1, 0), 2, 1, fill=False, linewidth=2, edgecolor="black", zorder=1))

    idx = {w: i for i, w in enumerate(waves)}
    plain_i = np.array([idx[w] for w in plain], dtype=int)
    ax.scatter(kx_n[plain_i], ky_n[plain_i], s=18, c="#b0b0b0", alpha=0.7, linewidths=0,
               label=f"Other wavevectors ({len(plain)})", zorder=2)

    for j, ds in enumerate(args.datasets):
        s = sorted(singles[ds])
        if not s:
            continue
        si = np.array([idx[w] for w in s], dtype=int)
        ax.scatter(kx_n[si], ky_n[si], s=70, c=SINGLE_COLORS[j % len(SINGLE_COLORS)],
                   edgecolors="black", linewidths=0.4,
                   label=f"{ds} only >{args.threshold_pct:g}% ({len(s)})", zorder=4)

    if overlap:
        oi = np.array([idx[w] for w in sorted(overlap)], dtype=int)
        joined = " \u2229 ".join(args.datasets)
        ax.scatter(kx_n[oi], ky_n[oi], s=55, c="#e74c3c", edgecolors="black", linewidths=0.35,
                   label=f"Overlap >{args.threshold_pct:g}% {joined} ({len(overlap)})", zorder=5)

    ky0 = np.abs(ky_n) < 1e-3
    kx0 = np.abs(kx_n) < 1e-3
    ax.plot(kx_n[ky0], ky_n[ky0], color="#3498db", linewidth=1.5, alpha=0.5, zorder=3, label="ky=0 (Γ–X)")
    ax.plot(kx_n[kx0], ky_n[kx0], color="#2ecc71", linewidth=1.5, alpha=0.5, zorder=3, label="kx=0")

    # Annotate high-symmetry corners and Gamma when they are highlighted
    all_hot = overlap | set().union(*singles.values())
    for w in sorted(all_hot):
        x, y = kx_n[idx[w]], ky_n[idx[w]]
        on_corner = np.isclose(abs(x), 1) and (np.isclose(y, 0) or np.isclose(y, 1))
        gamma = np.isclose(x, 0) and np.isclose(y, 0)
        x_pt = np.isclose(x, 0) and np.isclose(y, 1)
        if on_corner or gamma or x_pt:
            ax.annotate(("Γ\n" if gamma else "") + f"w{w}", (x, y),
                        textcoords="offset points", xytext=(4, 4), fontsize=8, color="black")

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.06, 1.08)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"$k_x / \pi$")
    ax.set_ylabel(r"$k_y / \pi$")
    ax.set_title(
        f"{args.model_name}\nIBZ wavevector grid ({len(waves)} points), "
        f">{args.threshold_pct:g}% second-peak membership"
    )
    ax.legend(loc="center", bbox_to_anchor=(2 / 3, 0.5), fontsize=8, framealpha=0.92)
    ax.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels([r"$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$",
                        r"$0$", r"$+\frac{\pi}{4}$", r"$+\frac{\pi}{2}$", r"$+\frac{3\pi}{4}$", r"$+\pi$"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([r"$0$", r"$+\frac{\pi}{4}$", r"$+\frac{\pi}{2}$", r"$+\frac{3\pi}{4}$", r"$+\pi$"])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    for ds in args.datasets:
        print(f"{ds}: {len(hot_sets[ds])} waves > {args.threshold_pct:g}%")
    if len(args.datasets) > 1:
        print(f"overlap: {len(overlap)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
