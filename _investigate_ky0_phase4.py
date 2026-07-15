"""
Phase 4: close the loop.

 1. Pooled pivot-vs-NMAE curve: pivot magnitude predicts displacement NMAE
    sample-by-sample, across all k lines.
 2. Mode-class census: on ky=0 vs kx=0, what fraction of the 6 lowest bands is
    u_y-dominant (transverse for x-prop / longitudinal for y-prop), and how
    does that align with the pivot-dead fraction?
 3. Phase determinism: for pivot-dead vs pivot-alive samples, compare the
    stored field of the SAME (geom, band) at the +kx / -kx partners:
    |cos| of complex overlap magnitude (structure) vs phase spread.
"""
from __future__ import annotations

import numpy as np
import torch

ROOT = r"d:\Research\NO-2D-Metamaterials"
PT = ROOT + r"\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt"
MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
PRED = ROOT + rf"\INFERENCE\{MODEL}\c_test\predictions_I3O5_{MODEL}.pt"

N_KX, N_KY, N_BANDS, N_GEOM = 25, 13, 6, 1000
N_WV = N_KX * N_KY

def wave_idx(i_kx, i_ky):
    return i_ky * N_KX + i_kx

def flat(g, w, b):
    return g * (N_WV * N_BANDS) + w * N_BANDS + b

disp = torch.load(PT + r"\displacements_dataset.pt", map_location="cpu", weights_only=False)
DT = disp.tensors
pred = torch.load(PRED, map_location="cpu", mmap=True, weights_only=True)

N_G = 120
gs = np.arange(N_G)

def get_fields(w):
    idx = torch.from_numpy(np.array([flat(g, w, b) for g in gs for b in range(N_BANDS)]))
    rex = DT[0][idx].numpy().astype(np.float64)
    imx = DT[1][idx].numpy().astype(np.float64)
    rey = DT[2][idx].numpy().astype(np.float64)
    imy = DT[3][idx].numpy().astype(np.float64)
    p = pred[idx].numpy().astype(np.float64)
    ux = rex + 1j * imx
    uy = rey + 1j * imy
    return ux, uy, p, rex, imx, rey, imy

def disp_nmae(p, rex, imx, rey, imy):
    def nm(a, b):
        return np.abs(a - b).mean(axis=(1, 2)) / (np.abs(b).mean(axis=(1, 2)) + 1e-5)
    return (nm(p[:, 1], rex) + nm(p[:, 2], imx) + nm(p[:, 3], rey) + nm(p[:, 4], imy)) / 4

# -------------------------------------------------- 1: pooled pivot vs NMAE
print("=" * 72)
print("1) POOLED pivot magnitude -> displacement NMAE (all lines mixed)")
print("=" * 72)
pool_waves = ([wave_idx(i, 0) for i in (3, 6, 9, 15, 18, 21)]        # ky=0
              + [wave_idx(12, j) for j in (2, 4, 6, 8, 10)]           # kx=0
              + [wave_idx(i, 12) for i in (6, 9, 18)]                 # ky=pi
              + [wave_idx(18, 6), wave_idx(6, 3), wave_idx(21, 9)])   # generic
piv_all, nm_all = [], []
for w in pool_waves:
    ux, uy, p, rex, imx, rey, imy = get_fields(w)
    norm = np.sqrt((np.abs(ux) ** 2 + np.abs(uy) ** 2).sum(axis=(1, 2)))
    piv_all.append(np.abs(ux[:, 0, 0]) / (norm + 1e-30))
    nm_all.append(disp_nmae(p, rex, imx, rey, imy))
piv_all = np.concatenate(piv_all)
nm_all = np.concatenate(nm_all)
edges = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
print(f"{'pivot bin':>18} {'n':>6} {'mean NMAE':>10} {'median NMAE':>12}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (piv_all >= lo) & (piv_all < hi)
    if m.sum() == 0:
        continue
    print(f"[{lo:.0e}, {hi:.0e}) {m.sum():>6d} {nm_all[m].mean():>10.3f} {np.median(nm_all[m]):>12.3f}")
from scipy.stats import spearmanr
rho, pv = spearmanr(piv_all, nm_all)
print(f"\nSpearman(pivot, NMAE) = {rho:.3f} (p={pv:.1e}, n={len(nm_all)})")

# -------------------------------------------------- 2: mode class census
print()
print("=" * 72)
print("2) MODE CLASS CENSUS on ky=0 vs kx=0 (bands 0-5, 120 geoms)")
print("=" * 72)
def census(w, name):
    ux, uy, p, rex, imx, rey, imy = get_fields(w)
    norm2 = (np.abs(ux) ** 2 + np.abs(uy) ** 2).sum(axis=(1, 2))
    ey = (np.abs(uy) ** 2).sum(axis=(1, 2)) / (norm2 + 1e-30)   # u_y energy share
    piv = np.abs(ux[:, 0, 0]) / (np.sqrt(norm2) + 1e-30)
    dead = piv < 1e-4
    nm = disp_nmae(p, rex, imx, rey, imy)
    print(f"{name:<18} dead={dead.mean():.3f}  "
          f"u_y-share[dead]={ey[dead].mean() if dead.any() else float('nan'):.3f}  "
          f"u_y-share[alive]={ey[~dead].mean():.3f}  "
          f"NMAE dead/alive = {nm[dead].mean() if dead.any() else float('nan'):.3f}/{nm[~dead].mean():.3f}")
    # per-band dead fraction
    deadb = dead.reshape(N_G, N_BANDS).mean(axis=0)
    print(f"{'':<18} per-band dead frac: " + "  ".join(f"b{b}:{deadb[b]:.2f}" for b in range(N_BANDS)))

census(wave_idx(18, 0), "ky=0  (pi/2, 0)")
census(wave_idx(6, 0), "ky=0  (-pi/2, 0)")
census(wave_idx(12, 6), "kx=0  (0, pi/2)")
census(wave_idx(12, 3), "kx=0  (0, pi/4)")
census(wave_idx(18, 12), "ky=pi (pi/2, pi)")
census(wave_idx(18, 6), "gen   (pi/2, pi/2)")

# -------------------------------------------------- 3: phase determinism across +-kx
print()
print("=" * 72)
print("3) PHASE CONSISTENCY of (+kx,0) vs (-kx,0) stored fields")
print("=" * 72)
print("u(-k) should equal conj(u(k)) up to the pivot-fixed global phase.")
print("Report |overlap| (structure match) and circular spread of overlap phase.")
for i_ky, label in ((0, "ky=0"), (6, "ky=pi/2 [not conjugate pair]"), (12, "ky=pi")):
    wp, wm = wave_idx(18, i_ky), wave_idx(6, i_ky)   # +-pi/2
    uxp, uyp, pp, *_ = get_fields(wp)
    uxm, uym, pm, *_ = get_fields(wm)
    normp = np.sqrt((np.abs(uxp) ** 2 + np.abs(uyp) ** 2).sum(axis=(1, 2)))
    normm = np.sqrt((np.abs(uxm) ** 2 + np.abs(uym) ** 2).sum(axis=(1, 2)))
    pivp = np.abs(uxp[:, 0, 0]) / (normp + 1e-30)
    # overlap of u(+k) with conj(u(-k))
    inner = (uxp * uxm).sum(axis=(1, 2)) + (uyp * uym).sum(axis=(1, 2))  # <conj(u-)|u+> if u- = conj
    mag = np.abs(inner) / (normp * normm + 1e-30)
    ang = np.angle(inner)
    dead = pivp < 1e-4
    for cls, m in (("dead ", dead), ("alive", ~dead)):
        if m.sum() < 5:
            continue
        # circular std of phase
        R = np.abs(np.exp(1j * ang[m]).mean())
        circ_std = np.sqrt(-2 * np.log(max(R, 1e-12)))
        print(f"  {label:<28} {cls} n={m.sum():4d}  |overlap|={mag[m].mean():.3f}  "
              f"phase circ-std={circ_std:.3f} rad  (0=deterministic, ~2.4=uniform)")

print("\nDone.")
