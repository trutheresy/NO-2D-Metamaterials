"""
Phase 3: test the phase-pivot hypothesis.

Solver convention: eig_vecs *= exp(-1j*angle(eig_vecs[0,:])), pivot = u_x at
node (0,0), which lies ON the mirror lines x=0 and y=0.

Hypothesis: on mirror-invariant k lines through Gamma (ky=0 row, kx=0 col), one
parity class has u_x(0,0) = 0 exactly -> global phase of the stored target is
numerical noise -> Re/Im displacement channels unpredictable, independent of
band gap. Lines ky=pi / kx=pi have a *twisted* mirror (projective, e^{iG.r})
that does not force a node at the origin -> unaffected.

Tests:
 1. Distribution of pivot |u_x(0,0)| / field_norm per wave line.
 2. NMAE split by pivot size (small-pivot class vs large-pivot class).
 3. Global-phase-aligned NMAE: min_phi ||p - e^{i phi} t|| -- if aligned NMAE
    drops to generic levels, the error is a pure random global phase.
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
EPS = 1e-9

def wave_idx(i_kx, i_ky):
    return i_ky * N_KX + i_kx

def flat(g, w, b):
    return g * (N_WV * N_BANDS) + w * N_BANDS + b

disp = torch.load(PT + r"\displacements_dataset.pt", map_location="cpu", weights_only=False)
DT = disp.tensors                      # ReX, ImX, ReY, ImY  each (N,32,32)
pred = torch.load(PRED, map_location="cpu", mmap=True, weights_only=True)

WAVES = {
    "ky=0    (pi/2, 0)":    wave_idx(18, 0),
    "ky=0    (-pi/2, 0)":   wave_idx(6, 0),
    "ky=0    (3pi/4, 0)":   wave_idx(21, 0),
    "kx=0    (0, pi/2)":    wave_idx(12, 6),
    "kx=0    (0, pi/4)":    wave_idx(12, 3),
    "kx=0    (0, 3pi/4)":   wave_idx(12, 9),
    "ky=pi   (pi/2, pi)":   wave_idx(18, 12),
    "ky=pi   (-pi/2, pi)":  wave_idx(6, 12),
    "kx=pi   (pi, pi/2)":   wave_idx(24, 6),
    "kx=-pi  (-pi, pi/2)":  wave_idx(0, 6),
    "generic (pi/2, pi/2)": wave_idx(18, 6),
    "generic (3pi/4, pi/4)": wave_idx(21, 3),
}

N_G = 150
gs = np.arange(N_G)

def analyze_wave(w: int):
    idx = np.array([flat(g, w, b) for g in gs for b in range(N_BANDS)])
    t = torch.from_numpy(idx)
    rex = DT[0][t].numpy().astype(np.float64)
    imx = DT[1][t].numpy().astype(np.float64)
    rey = DT[2][t].numpy().astype(np.float64)
    imy = DT[3][t].numpy().astype(np.float64)
    p = pred[t].numpy().astype(np.float64)          # (n,5,32,32)

    ux = rex + 1j * imx
    uy = rey + 1j * imy
    norm = np.sqrt((np.abs(ux) ** 2 + np.abs(uy) ** 2).sum(axis=(1, 2)))
    pivot = np.abs(ux[:, 0, 0]) / (norm + EPS)      # relative pivot magnitude

    # raw disp NMAE (uniform over 4 disp channels, per-channel normalized)
    def nmae(pr, tr):
        return np.abs(pr - tr).mean(axis=(1, 2)) / (np.abs(tr).mean(axis=(1, 2)) + 1e-5)
    raw = (nmae(p[:, 1], rex) + nmae(p[:, 2], imx) + nmae(p[:, 3], rey) + nmae(p[:, 4], imy)) / 4

    # global-phase-aligned NMAE: rotate TRUTH by e^{i phi}, phi from complex overlap
    pux = p[:, 1] + 1j * p[:, 2]
    puy = p[:, 3] + 1j * p[:, 4]
    inner = (pux * np.conj(ux)).sum(axis=(1, 2)) + (puy * np.conj(uy)).sum(axis=(1, 2))
    phi = np.angle(inner)
    ph = np.exp(1j * phi)[:, None, None]
    ux_r, uy_r = ux * ph, uy * ph
    aligned = (nmae(p[:, 1], ux_r.real) + nmae(p[:, 2], ux_r.imag)
               + nmae(p[:, 3], uy_r.real) + nmae(p[:, 4], uy_r.imag)) / 4

    return pivot, raw, aligned, np.abs(phi)

print(f"{'wave set':<24} {'frac pivot<1e-3':>15} {'median pivot':>13} | {'raw NMAE':>9} {'aligned':>8} | "
      f"{'NMAE sml-piv':>12} {'NMAE lrg-piv':>12} | {'med |phi|':>9}")
for name, w in WAVES.items():
    pivot, raw, aligned, absphi = analyze_wave(w)
    small = pivot < 1e-3
    large = ~small
    nm_s = raw[small].mean() if small.any() else float("nan")
    nm_l = raw[large].mean() if large.any() else float("nan")
    print(f"{name:<24} {small.mean():>15.3f} {np.median(pivot):>13.2e} | {raw.mean():>9.3f} {aligned.mean():>8.3f} | "
          f"{nm_s:>12.3f} {nm_l:>12.3f} | {np.median(absphi):>9.3f}")

print()
print("Interpretation:")
print(" - frac pivot<1e-3      : share of samples whose phase pivot u_x(0,0) ~ 0")
print(" - aligned              : NMAE after optimal global phase rotation of truth")
print(" - NMAE sml-piv/lrg-piv : raw NMAE within pivot classes")
print(" - med |phi|            : median |optimal phase| (0 = phases already agree)")
