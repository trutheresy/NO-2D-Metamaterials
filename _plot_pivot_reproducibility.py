"""
Demonstrate "correct eigenvector, but not reproducible".

Re-runs the actual FEA dispersion solver 3 times on the SAME p4mm geometry at
k = (pi/2, 0). ARPACK starts from a random vector, so the raw eigenvector comes
back with an arbitrary global phase each run; the pivot convention
(exp(-i*angle(u_x at node (0,0)))) is supposed to cancel that.

 - ALIVE band (longitudinal, pivot large): all 3 runs give identical Re/Im.
 - DEAD band (shear, pivot = 0 by symmetry): each run returns a different
   Re/Im split -- yet all runs agree after phase alignment and have identical
   |u| and eigenfrequency (i.e. each answer is CORRECT, just not reproducible).

Output: PLOTS/ky0_pivot_explainer/resolve_reproducibility.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"d:\Research\NO-2D-Metamaterials")
sys.path.insert(0, str(ROOT / "2d-dispersion-py"))

from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt
from get_design2 import get_design2
from design_parameters import DesignParameters
from design_conversion import convert_design, apply_steel_rubber_paradigm

OUT = ROOT / "PLOTS" / "ky0_pivot_explainer"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- build const
const = {
    "N_ele": 1,
    "N_pix": 32,
    "N_eig": 6,
    "sigma_eig": 1e-2,
    "a": 1.0,
    "design_scale": "linear",
    "isUseGPU": False,
    "isUseImprovement": False,   # non-VEC assembly (VEC path needs numpy>=2)
    "isUseSecondImprovement": False,
    "isUseParallel": False,
    "isSaveEigenvectors": True,
    "isComputeGroupVelocity": False,
    "isSaveKandM": False,
    "E_min": 200e6,
    "E_max": 200e9,
    "rho_min": 8e2,
    "rho_max": 8e3,
    "poisson_min": 0.0,
    "poisson_max": 0.5,
    "t": 1.0,
}

# same design pipeline as the dataset generator (continuous, p4mm-symmetric)
dp = DesignParameters(1)
dp.property_coupling = "coupled"
dp.design_style = "kernel"
dp.design_options = {
    "kernel": "periodic",
    "sigma_f": 1.0,
    "sigma_l": 1.0,
    "symmetry_type": "p4mm",
    "N_value": np.inf,
}
dp.N_pix = [32, 32]
dp.design_number = 3
dp = dp.prepare()

design = get_design2(dp)
design = convert_design(design, "linear", "linear",
                        const["E_min"], const["E_max"],
                        const["rho_min"], const["rho_max"])
design = apply_steel_rubber_paradigm(design, const)
const["design"] = np.asarray(design, dtype=np.float16)
g = np.asarray(design)[:, :, 0].astype(np.float64)
print(f"design symmetric? |g-flipx|={np.abs(g - g[:, ::-1]).max():.2e} "
      f"|g-flipy|={np.abs(g - g[::-1, :]).max():.2e}")

K_POINT = np.array([[np.pi / 2, 0.0]])
N_RUNS = 3

runs = []
for r in range(N_RUNS):
    np.random.seed(1000 + r)   # different ARPACK starting vectors
    wv, fr, ev, *_ = dispersion_with_matrix_save_opt(const, K_POINT.copy())
    runs.append((np.asarray(fr)[0], np.asarray(ev)[:, 0, :]))  # (6,), (2048, 6)
    print(f"run {r}: eigenfrequencies = {np.round(np.asarray(fr)[0], 3)}")

def panes(vec):
    """Split interleaved DOF vector into complex u_x, u_y 32x32 fields."""
    ux = vec[0::2].reshape(32, 32)
    uy = vec[1::2].reshape(32, 32)
    return ux, uy

# find alive/dead bands by pivot magnitude in run 0
pivots = []
for b in range(6):
    ux, uy = panes(runs[0][1][:, b])
    n = np.sqrt((np.abs(ux) ** 2 + np.abs(uy) ** 2).sum())
    pivots.append(np.abs(ux[0, 0]) / n)
pivots = np.array(pivots)
print("pivot magnitudes per band:", np.array2string(pivots, precision=2))
alive_b = int(np.argmax(pivots))
dead_b = int(np.argmin(pivots))
print(f"alive band = {alive_b} (pivot {pivots[alive_b]:.2e}), "
      f"dead band = {dead_b} (pivot {pivots[dead_b]:.2e})")

def dominant_pane(b):
    ux, uy = panes(runs[0][1][:, b])
    return ("u_x", 0) if np.abs(ux).sum() >= np.abs(uy).sum() else ("u_y", 1)

fig, axes = plt.subplots(2, 5, figsize=(16.4, 7.2), constrained_layout=True)
for row, (b, label) in enumerate(((alive_b, "ALIVE"), (dead_b, "DEAD"))):
    pane_name, pane_i = dominant_pane(b)
    fields = []
    for r in range(N_RUNS):
        ux, uy = panes(runs[r][1][:, b])
        fields.append(ux if pane_i == 0 else uy)
    ref = fields[0]
    vmax = max(np.abs(f.real).max() for f in fields)

    # phases relative to run 0
    phis = []
    for f in fields:
        inner = (f * np.conj(ref)).sum()
        phis.append(np.degrees(np.angle(inner)))

    for r in range(N_RUNS):
        ax = axes[row, r]
        ax.imshow(fields[r].real, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label} band {b}: Re({pane_name})\nrun {r + 1}   "
                     f"(phase vs run 1: {phis[r]:+.0f}\u00b0)", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])

    # column 4: run 3 after global phase alignment to run 1
    aligned = fields[2] * np.exp(-1j * np.radians(phis[2]))
    ax = axes[row, 3]
    ax.imshow(aligned.real, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"run 3 rotated back by {-phis[2]:+.0f}\u00b0\n= run 1 again "
                 f"(same physics)", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])

    # column 5: |u| difference run1 vs run3
    d = np.abs(fields[2]) - np.abs(ref)
    ax = axes[row, 4]
    ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"|{pane_name}| run 3 \u2212 run 1\n(zero: magnitude identical)", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])

    f0 = runs[0][0][b]
    axes[row, 0].set_ylabel(
        f"{label} band {b}\npivot = {pivots[b]:.1e}\nf = {f0:.1f} Hz (all runs)",
        fontsize=10,
    )

fig.suptitle(
    "Same geometry, same wavevector k = (\u03c0/2, 0), FEA solver re-run 3 times\n"
    "Top: alive mode \u2014 pivot fixes the phase, every run returns the same Re/Im.  "
    "Bottom: dead mode \u2014 pivot is 0, every run returns a different Re/Im split of the same correct eigenvector.",
    fontsize=11.5,
)
fig.savefig(OUT / "resolve_reproducibility.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("Wrote", OUT / "resolve_reproducibility.png")
