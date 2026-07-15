"""
Classify 0705 high-loss wavevectors as shear vs longitudinal by polarization,
and check whether poor performers include non-shear modes.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"d:\Research\NO-2D-Metamaterials")
PT = ROOT / "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt"
MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
WAVE_CSV = (
    ROOT
    / "INFERENCE"
    / MODEL
    / "c_test/second_peak_analysis/second_peak_c_test_wave_table.csv"
)

N_KX, N_KY, N_BANDS, N_GEOM = 25, 13, 6, 1000
N_WV = N_KX * N_KY
PI = np.pi


def wave_idx(i_kx: int, i_ky: int) -> int:
    return i_ky * N_KX + i_kx


def flat(g: int, w: int, b: int) -> int:
    return g * (N_WV * N_BANDS) + w * N_BANDS + b


def k_label(kx: float, ky: float) -> str:
    def c(v: float) -> str:
        r = v / PI
        n = round(r * 12)
        if abs(r - n / 12) > 2e-3:
            return f"{r:.3f}pi"
        if n == 0:
            return "0"
        sign = "-" if n < 0 else "+"
        n = abs(n)
        g = np.gcd(n, 12)
        n, d = n // g, 12 // g
        if n == 1 and d == 1:
            return f"{sign}pi"
        if n == 1:
            return f"{sign}pi/{d}"
        if n == d:
            return f"{sign}pi"
        return f"{sign}{n}pi/{d}"

    return f"({c(kx)}, {c(ky)})"


rows = {int(r["wave"]): r for r in csv.DictReader(WAVE_CSV.open())}
kxy = torch.load(PT / "wavevectors_full.pt", map_location="cpu", weights_only=False)[0].numpy()
disp = torch.load(PT / "displacements_dataset.pt", map_location="cpu", weights_only=False)
DT = disp.tensors

# Sample many geometries for polarization stats
gs = np.arange(80)


def polarization_stats(w: int) -> dict:
    """Per-band u_y energy share and dead-pivot fraction."""
    uy_share = np.zeros(N_BANDS)
    dead = np.zeros(N_BANDS)
    for b in range(N_BANDS):
        shares, pivs = [], []
        for g in gs:
            f = flat(g, w, b)
            ux = DT[0][f].numpy().astype(np.float64) + 1j * DT[1][f].numpy().astype(np.float64)
            uy = DT[2][f].numpy().astype(np.float64) + 1j * DT[3][f].numpy().astype(np.float64)
            ex = (np.abs(ux) ** 2).sum()
            ey = (np.abs(uy) ** 2).sum()
            n2 = ex + ey + 1e-30
            shares.append(ey / n2)
            pivs.append(np.abs(ux[0, 0]) / (np.sqrt(n2) + 1e-30))
        uy_share[b] = np.mean(shares)
        dead[b] = np.mean(np.array(pivs) < 1e-4)
    return {"uy_share": uy_share, "dead": dead}


# Hot waves from report (>50%)
hot = sorted(
    [w for w, r in rows.items() if float(r["frac_second_pct"]) > 50.0],
    key=lambda w: -float(rows[w]["frac_second_pct"]),
)

print("=" * 88)
print("0705 c_test waves with >50% second-peak membership")
print("=" * 88)
print(
    f"{'wave':>4} {'k':<18} {'%2nd':>6} | "
    f"{'shear bands (uy>0.5)':<22} {'dead-piv bands':<18} {'n_shear':>7} {'n_dead':>6} | note"
)

for w in hot:
    r = rows[w]
    kx, ky = float(r["kx"]), float(r["ky"])
    st = polarization_stats(w)
    shear_bands = [b for b in range(N_BANDS) if st["uy_share"][b] > 0.5]
    dead_bands = [b for b in range(N_BANDS) if st["dead"][b] > 0.5]
    # Classification of the WAVEVECTOR (line type)
    if abs(ky) < 1e-6 and abs(kx) < 1e-6:
        note = "Gamma (TRIM)"
    elif abs(ky) < 1e-6 and abs(abs(kx) - PI) < 0.05:
        note = "X point (TRIM)"
    elif abs(abs(ky) - PI) < 0.05 and abs(abs(kx) - PI) < 0.05:
        note = "M corner (TRIM)"
    elif abs(kx) < 1e-6 and abs(abs(ky) - PI) < 0.05:
        note = "(0,pi) TRIM"
    elif abs(ky) < 1e-6:
        note = "ky=0 line (mix shear+long)"
    elif abs(kx) < 1e-6:
        note = "kx=0 line"
    else:
        note = "generic"
    print(
        f"{w:4d} {k_label(kx, ky):<18} {float(r['frac_second_pct']):6.1f} | "
        f"{str(shear_bands):<22} {str(dead_bands):<18} {len(shear_bands):7d} {len(dead_bands):6d} | {note}"
    )

# Per-band detail for representative waves
print()
print("=" * 88)
print("Per-band polarization (uy energy share) and dead-pivot rate at representative k")
print("=" * 88)
reps = {
    "ky=0 (pi/2,0)": wave_idx(18, 0),
    "ky=0 (pi/4,0)": wave_idx(15, 0),
    "Gamma (0,0)": wave_idx(12, 0),
    "kx=0 (0,pi/2)": wave_idx(12, 6),
    "kx=0 (0,pi/12) wave37": wave_idx(12, 1),
    "generic (pi/2,pi/2)": wave_idx(18, 6),
}
for name, w in reps.items():
    st = polarization_stats(w)
    frac = float(rows[w]["frac_second_pct"])
    print(f"\n{name}  wave={w}  second-peak={frac:.1f}%")
    print(f"  band: " + "  ".join(f"{b:>6d}" for b in range(N_BANDS)))
    print(f"  uy% : " + "  ".join(f"{100*st['uy_share'][b]:5.1f}%" for b in range(N_BANDS)))
    print(f"  dead: " + "  ".join(f"{100*st['dead'][b]:5.1f}%" for b in range(N_BANDS)))
    labels = []
    for b in range(N_BANDS):
        if st["uy_share"][b] > 0.5:
            labels.append("SHEAR")
        else:
            labels.append("LONG ")
    print(f"  class:" + "  ".join(f"{lab:>6s}" for lab in labels))

# Broader: among all waves with elevated loss (>30%), which are NOT on ky=0?
print()
print("=" * 88)
print("Elevated waves (>30% second-peak) that are NOT on ky=0")
print("=" * 88)
elev = [
    (w, float(r["frac_second_pct"]), float(r["kx"]), float(r["ky"]))
    for w, r in rows.items()
    if float(r["frac_second_pct"]) > 30.0 and abs(float(r["ky"])) > 1e-6
]
elev.sort(key=lambda t: -t[1])
print(f"{'wave':>4} {'k':<18} {'%2nd':>6} {'line':<12} shear_bands dead_bands")
for w, frac, kx, ky in elev:
    st = polarization_stats(w)
    shear_bands = [b for b in range(N_BANDS) if st["uy_share"][b] > 0.5]
    dead_bands = [b for b in range(N_BANDS) if st["dead"][b] > 0.5]
    line = "kx=0" if abs(kx) < 1e-6 else ("|k|=pi edge" if abs(abs(kx) - PI) < 0.05 or abs(abs(ky) - PI) < 0.05 else "other")
    print(
        f"{w:4d} {k_label(kx, ky):<18} {frac:6.1f} {line:<12} "
        f"{shear_bands} {dead_bands}"
    )

print()
print("Done.")
