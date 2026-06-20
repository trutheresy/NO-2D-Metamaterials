"""Export embedded wavelet patches (waveforms_full) grouped by IBZ region."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import NO_utilities as NU

ROOT = Path(__file__).resolve().parent
PT_DIR = ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt"
OUT_DIR = ROOT / "PLOTS/wavelet_embeddings_ibz"

PI = math.pi
TOL = 1e-3
CORNERS = {0, 24, 300, 324}

# Mutually exclusive folders (priority: corner > ky=0 > kx=0 > other_edges > interior)
FOLDERS = ("corner", "ky=0", "kx=0", "other_edges", "interior")


def k_to_label(val: float) -> str:
    r = val / PI
    n = round(r * 12)
    if abs(r - n / 12) > 2e-3:
        return f"{r:.4f}π"
    if n == 0:
        return "0"
    sign = "-" if n < 0 else "+"
    n = abs(n)
    g = math.gcd(n, 12)
    n, d = n // g, 12 // g
    if n == 1:
        return f"{sign}π/{d}"
    if n == d:
        return f"{sign}π"
    return f"{sign}{n}π/{d}"


def assign_region(kx: float, ky: float, wave: int) -> str:
    if wave in CORNERS or (
        abs(abs(kx) - PI) < TOL and (abs(ky) < TOL or abs(ky - PI) < TOL)
    ):
        return "corner"
    if abs(ky) < TOL:
        return "ky=0"
    if abs(kx) < TOL:
        return "kx=0"
    if abs(abs(kx) - PI) < TOL or abs(ky - PI) < TOL:
        return "other_edges"
    return "interior"


def load_waveforms() -> tuple[np.ndarray, np.ndarray]:
    kxy = torch.load(PT_DIR / "wavevectors_full.pt", map_location="cpu", weights_only=False)[0].numpy().astype(
        np.float64
    )
    wf_stored = torch.load(PT_DIR / "waveforms_full.pt", map_location="cpu", weights_only=False).numpy().astype(
        np.float32
    )
    wf_recomputed = NU.embed_2const_wavelet(kxy[:, 0], kxy[:, 1], size=32, verbose=False).astype(np.float32)
    max_diff = float(np.max(np.abs(wf_stored - wf_recomputed)))
    if max_diff > 1e-3:
        print(f"WARNING: stored vs recomputed embed max diff = {max_diff:.4g}")
    return kxy, wf_stored


def save_patch(fig_path: Path, patch: np.ndarray, wave: int, kx: float, ky: float, region: str) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=120)
    vmax = max(float(np.abs(patch).max()), 1e-6)
    im = ax.imshow(patch, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    k_str = f"({k_to_label(kx)}, {k_to_label(ky)})"
    ax.set_title(f"w{wave}  {k_str}\n{region}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def save_folder_grid(folder: Path, region: str, entries: list[tuple[int, np.ndarray, float, float]]) -> None:
    if not entries:
        return
    n = len(entries)
    ncol = min(5, n) if region != "ky=0" else 5
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow), dpi=140)
    axes = np.atleast_2d(axes)
    for idx, (wave, patch, kx, ky) in enumerate(entries):
        r, c = divmod(idx, ncol)
        ax = axes[r, c]
        vmax = max(float(np.abs(patch).max()), 1e-6)
        ax.imshow(patch, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(f"w{wave}\n{k_to_label(kx)}, {k_to_label(ky)}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(n, nrow * ncol):
        r, c = divmod(idx, ncol)
        axes[r, c].axis("off")
    fig.suptitle(f"Embedded wavelets — {region} ({n} wavevectors)", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / f"_grid_{region.replace('=', '')}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    kxy, waveforms = load_waveforms()
    n_wv = kxy.shape[0]

    if OUT_DIR.exists():
        for p in OUT_DIR.rglob("*.png"):
            p.unlink()
    for name in FOLDERS:
        (OUT_DIR / name).mkdir(parents=True, exist_ok=True)

    by_region: dict[str, list[tuple[int, np.ndarray, float, float]]] = {k: [] for k in FOLDERS}
    assignments: list[dict] = []

    for w in range(n_wv):
        kx, ky = float(kxy[w, 0]), float(kxy[w, 1])
        region = assign_region(kx, ky, w)
        patch = waveforms[w]
        by_region[region].append((w, patch, kx, ky))
        fname = f"w{w:03d}_kx{k_to_label(kx).replace('/', 'd')}_ky{k_to_label(ky).replace('/', 'd')}.png"
        save_patch(OUT_DIR / region / fname, patch, w, kx, ky, region)
        assignments.append({"wave": w, "kx": kx, "ky": ky, "region": region})

    for region in FOLDERS:
        entries = sorted(by_region[region], key=lambda t: t[0])
        save_folder_grid(OUT_DIR / region, region, entries)

    print(f"Wrote {n_wv} wavelet PNGs to {OUT_DIR}")
    for region in FOLDERS:
        print(f"  {region}: {len(by_region[region])} images")


if __name__ == "__main__":
    main()
