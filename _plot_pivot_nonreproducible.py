"""
Examples of 'correct eigenvector, not reproducible' under PLOTS/ky0_pivot_explainer/.

  same_mode_four_phases.png
      One dead-mode field; multiply by exp(i*theta) for four theta values.
      All are the same physical solution; |u| unchanged; Re/Im look unrelated.

  two_stored_solves_aligned.png
      Real dataset: same geometry+band at +k and -k (two independent pivot draws).
      Stored Re/Im disagree; after one global phase rotation they match.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = r"d:\Research\NO-2D-Metamaterials"
PT = ROOT + r"\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt"
OUT = ROOT + r"\PLOTS\ky0_pivot_explainer"
os.makedirs(OUT, exist_ok=True)

N_KX, N_KY, N_BANDS = 25, 13, 6
N_WV = N_KX * N_KY


def wave_idx(i_kx: int, i_ky: int) -> int:
    return i_ky * N_KX + i_kx


def flat(g: int, w: int, b: int) -> int:
    return g * (N_WV * N_BANDS) + w * N_BANDS + b


disp = torch.load(PT + r"\displacements_dataset.pt", map_location="cpu", weights_only=False)
DT = disp.tensors

W_P = wave_idx(18, 0)
W_M = wave_idx(6, 0)
G = 1
B_DEAD = 2


def load_uy(g: int, w: int, b: int) -> np.ndarray:
    f = flat(g, w, b)
    re = DT[2][f].numpy().astype(np.float64)
    im = DT[3][f].numpy().astype(np.float64)
    return re + 1j * im


def sym(ax, arr, title, cmap="RdBu_r"):
    v = np.abs(arr).max()
    v = max(v, 1e-12)
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=-v, vmax=v)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def mag(ax, arr, title, cmap="magma"):
    im = ax.imshow(np.abs(arr), origin="lower", cmap=cmap, vmin=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# --- Figure 1: four global phases, one underlying mode ---------------------------------
uy0 = load_uy(G, W_P, B_DEAD)
thetas_deg = [0, 47, 94, 141]
variants = [uy0 * np.exp(1j * np.radians(t)) for t in thetas_deg]

fig, axes = plt.subplots(3, 4, figsize=(11.5, 8.8), constrained_layout=True)
for j, (t, u) in enumerate(zip(thetas_deg, variants)):
    sym(axes[0, j], u.real, f"Re(u_y) after rotate {t}\u00b0")
    sym(axes[1, j], u.imag, f"Im(u_y) after rotate {t}\u00b0")
    mag(axes[2, j], u, f"|u_y|  (identical)")
fig.suptitle(
    "Same physical eigenvector, four equally valid global phases\n"
    f"Dead shear mode, geometry {G}, k = (+\u03c0/2, 0), band {B_DEAD}\n"
    "Only the Re/Im bookkeeping changes; magnitude |u| is unchanged.",
    fontsize=11,
)
fig.savefig(os.path.join(OUT, "same_mode_four_phases.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# --- Figure 2: two stored solves (+k vs -k), align by global phase -----------------
uy_p = load_uy(G, W_P, B_DEAD)
uy_m = load_uy(G, W_M, B_DEAD)
inner = (uy_p * np.conj(uy_m)).sum()
phi_align = np.angle(inner)
uy_m_aligned = uy_m * np.exp(1j * phi_align)
phase_deg = np.degrees(phi_align)
max_re_diff = np.abs(uy_p.real - uy_m_aligned.real).max()
max_im_diff = np.abs(uy_p.imag - uy_m_aligned.imag).max()
max_mag_diff = np.abs(np.abs(uy_p) - np.abs(uy_m)).max()

fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.2), constrained_layout=True)
sym(axes[0, 0], uy_p.real, f"STORED Re(u_y)\nk = (+\u03c0/2, 0)")
sym(axes[0, 1], uy_m.real, f"STORED Re(u_y)\nk = (\u2212\u03c0/2, 0)\n(different pivot draw)")
sym(axes[0, 2], uy_p.real - uy_m.real, "Re difference\n(looks wrong)")
sym(axes[1, 0], uy_p.imag, "STORED Im(u_y)\n+k")
sym(axes[1, 1], uy_m.imag, "STORED Im(u_y)\n\u2212k")
sym(axes[1, 2], uy_p.imag - uy_m.imag, "Im difference\n(looks wrong)")
mag(axes[2, 0], uy_p, "|u_y| at +k")
mag(axes[2, 1], uy_m, "|u_y| at \u2212k")
axes[2, 2].imshow(
    np.abs(uy_p) - np.abs(uy_m),
    origin="lower",
    cmap="RdBu_r",
    vmin=-np.abs(uy_p).max(),
    vmax=np.abs(uy_p).max(),
)
axes[2, 2].set_title(f"|u| difference\nmax = {max_mag_diff:.2e}\n(physics identical)")
axes[2, 2].set_xticks([])
axes[2, 2].set_yticks([])

fig.suptitle(
    f"Two stored 'solves' for the same geometry {G}, band {B_DEAD} (time-reversal pair on ky=0)\n"
    "Stored targets disagree in Re/Im, but |u| matches \u2014 both are correct, convention is not reproducible.",
    fontsize=11,
)
fig.savefig(os.path.join(OUT, "two_stored_solves_raw.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
sym(axes[0, 0], uy_p.real, "Re(u_y) at +k  (reference)")
sym(axes[0, 1], uy_m_aligned.real, f"Re(u_y) at \u2212k\nafter global rotate {phase_deg:.0f}\u00b0")
sym(axes[0, 2], uy_p.real - uy_m_aligned.real,
    f"Re residual\nmax = {max_re_diff:.2e}\n(now match)")
sym(axes[1, 0], uy_p.imag, "Im(u_y) at +k")
sym(axes[1, 1], uy_m_aligned.imag, f"Im(u_y) at \u2212k\nafter same rotate")
sym(axes[1, 2], uy_p.imag - uy_m_aligned.imag,
    f"Im residual\nmax = {max_im_diff:.2e}\n(now match)")
fig.suptitle(
    "After multiplying the \u2212k field by one global phase, it coincides with +k\n"
    "\u2192 same eigenvector; only the arbitrary pivot phase differed between saves.",
    fontsize=11,
)
fig.savefig(os.path.join(OUT, "two_stored_solves_aligned.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

print("Wrote:")
print(" ", os.path.join(OUT, "same_mode_four_phases.png"))
print(" ", os.path.join(OUT, "two_stored_solves_raw.png"))
print(" ", os.path.join(OUT, "two_stored_solves_aligned.png"))
print(f"Alignment phase for geom {G}: {phase_deg:.1f} deg")
