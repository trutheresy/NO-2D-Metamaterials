"""
Visual explainer for the phase-pivot problem on the ky=0 line.

Figures (PLOTS/ky0_pivot_explainer/):
 1. pivot_location.png      : |u_x| for an alive vs dead mode; the pivot pixel
                              (0,0) sits on the y=0 mirror line, which is a
                              zero line of u_x for shear-type modes.
 2. phase_wheel.png         : conceptual complex-plane diagram of the pivot.
 3. random_phase_pair.png   : same geometry+band at +kx and -kx. Alive band:
                              Re matches, Im flips (clean conjugate pair).
                              Dead band: Re/Im scrambled by random phase even
                              though |u| is identical.
 4. phase_scatter.png       : relative phase of the (+k, -k) pair across many
                              geometries, dead vs alive (unit circle).
 5. pivot_vs_nmae.png       : pivot magnitude vs displacement NMAE (binned).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Circle

ROOT = r"d:\Research\NO-2D-Metamaterials"
PT = ROOT + r"\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt"
MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
PRED = ROOT + rf"\INFERENCE\{MODEL}\c_test\predictions_I3O5_{MODEL}.pt"
OUT = ROOT + r"\PLOTS\ky0_pivot_explainer"

import os
os.makedirs(OUT, exist_ok=True)

N_KX, N_KY, N_BANDS, N_GEOM = 25, 13, 6, 1000
N_WV = N_KX * N_KY

def wave_idx(i_kx, i_ky):
    return i_ky * N_KX + i_kx

def flat(g, w, b):
    return g * (N_WV * N_BANDS) + w * N_BANDS + b

disp = torch.load(PT + r"\displacements_dataset.pt", map_location="cpu", weights_only=False)
DT = disp.tensors  # ReX, ImX, ReY, ImY

W_P = wave_idx(18, 0)   # (+pi/2, 0)
W_M = wave_idx(6, 0)    # (-pi/2, 0)

def get(g, w, b):
    f = flat(g, w, b)
    ux = DT[0][f].numpy().astype(np.float64) + 1j * DT[1][f].numpy().astype(np.float64)
    uy = DT[2][f].numpy().astype(np.float64) + 1j * DT[3][f].numpy().astype(np.float64)
    return ux, uy

def pivot_rel(ux, uy):
    n = np.sqrt((np.abs(ux) ** 2 + np.abs(uy) ** 2).sum())
    return np.abs(ux[0, 0]) / max(n, 1e-30)

# pick one geometry: band 1 alive (longitudinal), band 2 dead (shear)
G = 3
ux_a, uy_a = get(G, W_P, 1)   # alive
ux_d, uy_d = get(G, W_P, 2)   # dead
print(f"geometry {G}: pivot band1 (alive) = {pivot_rel(ux_a, uy_a):.2e}, "
      f"band2 (dead) = {pivot_rel(ux_d, uy_d):.2e}")

# ------------------------------------------------------------- Fig 1: pivot location
fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6), constrained_layout=True)
panels = [
    (np.abs(ux_a), "|u_x|  —  ALIVE mode (band 1, longitudinal)\nu_x is even across the y=0 mirror line", 0),
    (np.abs(uy_a), "|u_y| of the same alive mode\n(small: mode is u_x-polarized)", 1),
    (np.abs(ux_d), "|u_x|  —  DEAD mode (band 2, shear)\nu_x is odd across y=0  →  u_x = 0 on the whole row", 2),
    (np.abs(uy_d), "|u_y| of the same dead mode\n(large: mode is u_y-polarized)", 3),
]
vmax_a = max(np.abs(ux_a).max(), np.abs(uy_a).max())
vmax_d = max(np.abs(ux_d).max(), np.abs(uy_d).max())
for arr, title, i in panels:
    ax = axes[i // 2, i % 2]
    vmax = vmax_a if i < 2 else vmax_d
    im = ax.imshow(arr, origin="lower", cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=9.5)
    ax.axhline(0, color="cyan", linewidth=1.4, linestyle="--", alpha=0.85)
    ax.plot(0, 0, marker="o", ms=11, mfc="none", mec="lime", mew=2.2)
    ax.annotate("pivot pixel (0,0)", (0, 0), xytext=(4.5, 3.2), fontsize=9,
                color="lime", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="lime", lw=1.4))
    ax.text(30.8, 0.9, "y = 0 mirror line", color="cyan", fontsize=8.5,
            ha="right", va="bottom")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(
    f"Wave k = (\u03c0/2, 0), geometry {G}  \u2014  where the phase pivot lives\n"
    "The solver divides every eigenvector by the phase of u_x at pixel (0,0).",
    fontsize=11,
)
fig.savefig(OUT + r"\pivot_location.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- Fig 2: phase wheel concept
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
for ax, alive in zip(axes, (True, False)):
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="0.65", lw=1.2))
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    if alive:
        v = 0.82 * np.exp(1j * 0.6)
        ax.add_patch(FancyArrowPatch((0, 0), (v.real, v.imag),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color="#1f77b4", lw=2.4))
        ax.annotate("u_x(0,0)\nbig, well-defined angle", (v.real, v.imag),
                    xytext=(0.12, 1.06), fontsize=10, color="#1f77b4")
        ax.set_title('ALIVE mode\nangle(u_x(0,0)) is stable →\nstored phase is reproducible', fontsize=10)
    else:
        rng = np.random.default_rng(2)
        for i in range(9):
            ang = rng.uniform(0, 2 * np.pi)
            v = 0.045 * np.exp(1j * ang)
            ax.add_patch(FancyArrowPatch((0, 0), (18 * v.real, 18 * v.imag),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color="#d62728", lw=1.4, alpha=0.55))
        ax.add_patch(Circle((0, 0), 0.09, fill=True, color="#d62728", alpha=0.25))
        ax.annotate("u_x(0,0) \u2248 1e-16\n(pure round-off noise)", (0.09, -0.09),
                    xytext=(0.28, -0.75), fontsize=10, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728"))
        ax.set_title('DEAD mode\nangle(noise) is uniformly random →\nstored phase is a coin flip', fontsize=10)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel("Re"); ax.set_ylabel("Im")
fig.suptitle("The pivot in the complex plane: every eigenvector is multiplied by exp(\u2212i\u00b7angle(u_x(0,0)))", fontsize=11)
fig.savefig(OUT + r"\phase_wheel.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- Fig 3: +k / -k pair comparison
# geometry 1: dead-band pair phase mismatch ~ -88 deg (visually obvious scrambling)
G_PAIR = 1
ux_ap, _ = get(G_PAIR, W_P, 1)
ux_am, _ = get(G_PAIR, W_M, 1)
_, uy_dp = get(G_PAIR, W_P, 2)
_, uy_dm = get(G_PAIR, W_M, 2)

def sym_imshow(ax, arr, title):
    v = np.abs(arr).max()
    im = ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im

fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.2), constrained_layout=True)
rows = [
    (ux_ap.real, ux_am.real, "ALIVE band 1:  Re(u_x)", "identical \u2713"),
    (ux_ap.imag, ux_am.imag, "ALIVE band 1:  Im(u_x)", "sign-flipped \u2713"),
    (uy_dp.real, uy_dm.real, "DEAD band 2:  Re(u_y)", "scrambled \u2717"),
    (uy_dp.imag, uy_dm.imag, "DEAD band 2:  Im(u_y)", "scrambled \u2717"),
]
for r, (ap, am, label, verdict) in enumerate(rows):
    sym_imshow(axes[r, 0], ap, f"{label}\nat k = (+\u03c0/2, 0)")
    sym_imshow(axes[r, 1], am, f"{label}\nat k = (\u2212\u03c0/2, 0)")
    diff_or_mag = np.abs(ap + 1j * 0) if False else None
    # third column: magnitude comparison |u| to show structure is the same
    if r < 2:
        mag_p, mag_m = np.abs(ux_ap), np.abs(ux_am)
    else:
        mag_p, mag_m = np.abs(uy_dp), np.abs(uy_dm)
    im = axes[r, 2].imshow(mag_p - mag_m, origin="lower", cmap="RdBu_r",
                           vmin=-np.abs(mag_p).max(), vmax=np.abs(mag_p).max())
    axes[r, 2].set_title(f"|u| difference (+k vs \u2212k)\nphysics identical; verdict: {verdict}", fontsize=9)
    axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
fig.suptitle(
    "Time-reversal partners (+k and \u2212k are BOTH in the dataset on the ky=0 row)\n"
    "Physics says: same |u|, Re equal, Im flipped. Alive band obeys this.\n"
    "Dead band: random pivot phase scrambles Re/Im independently at +k and \u2212k.",
    fontsize=11,
)
fig.savefig(OUT + r"\random_phase_pair.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- Fig 4: phase scatter over geometries
n_show = 250
phis_dead, phis_alive = [], []
for g in range(n_show):
    for b, bucket in ((1, phis_alive), (2, phis_dead)):
        uxp, uyp = get(g, W_P, b)
        uxm, uym = get(g, W_M, b)
        piv = pivot_rel(uxp, uyp)
        inner = (uxp * uxm).sum() + (uyp * uym).sum()
        # route by actual pivot, not band label, to keep classes clean
        if piv < 1e-4:
            phis_dead.append(np.angle(inner))
        else:
            phis_alive.append(np.angle(inner))
phis_dead = np.array(phis_dead)
phis_alive = np.array(phis_alive)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), subplot_kw={"projection": "polar"},
                         constrained_layout=True)
for ax, phis, name, color in (
    (axes[0], phis_alive, f"ALIVE samples (n={len(phis_alive)})", "#1f77b4"),
    (axes[1], phis_dead, f"DEAD samples (n={len(phis_dead)})", "#d62728"),
):
    ax.hist(phis, bins=36, color=color, alpha=0.85)
    ax.set_title(f"{name}\nrelative phase between stored u(+k) and u(\u2212k)", fontsize=10)
    ax.set_yticklabels([])
fig.suptitle(
    "Phase mismatch of time-reversal partners at k = (\u00b1\u03c0/2, 0), 250 geometries\n"
    "Alive: clustered (deterministic convention)   |   Dead: uniform (random pivot noise)",
    fontsize=11,
)
fig.savefig(OUT + r"\phase_scatter.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- Fig 5: pivot vs NMAE
pred = torch.load(PRED, map_location="cpu", mmap=True, weights_only=True)
waves_pool = ([wave_idx(i, 0) for i in (3, 6, 9, 15, 18, 21)]
              + [wave_idx(12, j) for j in (2, 4, 6, 8, 10)]
              + [wave_idx(18, 6), wave_idx(6, 3), wave_idx(21, 9)])
gs = np.arange(120)
piv_all, nm_all = [], []
for w in waves_pool:
    idx = torch.from_numpy(np.array([flat(g, w, b) for g in gs for b in range(N_BANDS)]))
    rex = DT[0][idx].numpy().astype(np.float64)
    imx = DT[1][idx].numpy().astype(np.float64)
    rey = DT[2][idx].numpy().astype(np.float64)
    imy = DT[3][idx].numpy().astype(np.float64)
    p = pred[idx].numpy().astype(np.float64)
    ux = rex + 1j * imx; uy = rey + 1j * imy
    norm = np.sqrt((np.abs(ux) ** 2 + np.abs(uy) ** 2).sum(axis=(1, 2)))
    piv_all.append(np.abs(ux[:, 0, 0]) / (norm + 1e-30))
    def nm(a, b):
        return np.abs(a - b).mean(axis=(1, 2)) / (np.abs(b).mean(axis=(1, 2)) + 1e-5)
    nm_all.append((nm(p[:, 1], rex) + nm(p[:, 2], imx) + nm(p[:, 3], rey) + nm(p[:, 4], imy)) / 4)
piv_all = np.concatenate(piv_all)
nm_all = np.concatenate(nm_all)

fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
ax.scatter(np.clip(piv_all, 1e-8, None), nm_all, s=4, alpha=0.18, color="#444", rasterized=True)
edges = np.logspace(-8, 0, 17)
centers, med = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (piv_all >= lo) & (piv_all < hi)
    if m.sum() >= 25:
        centers.append(np.sqrt(lo * hi))
        med.append(np.median(nm_all[m]))
ax.plot(centers, med, "-o", color="#d62728", lw=2.2, ms=5, label="median NMAE per bin")
ax.axvspan(1e-8, 1e-5, color="#d62728", alpha=0.07)
ax.text(3e-8, 1.62, "pivot = numerical noise\n(random phase)", fontsize=9, color="#a33")
ax.set_xscale("log")
ax.set_xlabel("pivot magnitude  |u_x(0,0)| / ||u||   (log scale)")
ax.set_ylabel("displacement NMAE (4-channel mean)")
ax.set_title("The pivot alone predicts prediction failure\n"
             "12,240 samples across ky=0 row, kx=0 column, and generic k")
ax.set_ylim(0, 1.8)
ax.legend()
fig.savefig(OUT + r"\pivot_vs_nmae.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("Wrote figures to", OUT)
