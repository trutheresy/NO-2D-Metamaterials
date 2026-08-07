"""
Phase 2: test the symmetry / band-crossing hypothesis.

A. Are geometries mirror-symmetric (x-flip, y-flip)?
B. Are eigenvalue scalars exactly equal for (+kx, ky) vs (-kx, ky)?
C. Mirror parity of eigenvector fields on the kx=0 and ky=0 lines.
D. Band-gap statistics: are exact/near degeneracies more common on the
   symmetry lines than at generic k?
E. Does per-sample displacement error correlate with small band gap?
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

def wave_idx(i_kx: int, i_ky: int) -> int:
    return i_ky * N_KX + i_kx

def flat(g: int, w: int, b: int) -> int:
    return g * (N_WV * N_BANDS) + w * N_BANDS + b

def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))

# ---------------------------------------------------------------- A: geometry symmetry
print("=" * 70)
print("A) GEOMETRY MIRROR SYMMETRY")
print("=" * 70)
geoms = torch.load(PT + r"\geometries_full.pt", map_location="cpu", weights_only=False)
g = geoms if isinstance(geoms, torch.Tensor) else geoms.tensors[0]
g = g.float()
print("geometries shape:", tuple(g.shape))
gg = g.reshape(g.shape[0], 32, 32) if g.ndim != 3 else g
sample = gg[:50].numpy().astype(np.float64)
dx = np.abs(sample - sample[:, :, ::-1]).mean()       # flip x (last axis)
dy = np.abs(sample - sample[:, ::-1, :]).mean()       # flip y
dT = np.abs(sample - np.transpose(sample, (0, 2, 1))).mean()  # diagonal
scale = np.abs(sample).mean()
print(f"mean|g|={scale:.4f}  |g - flipx(g)|={dx:.6f}  |g - flipy(g)|={dy:.6f}  |g - g^T|={dT:.6f}")
print("-> symmetric under a flip if diff ~ 0")

# ---------------------------------------------------------------- B: eigenvalue equality
print()
print("=" * 70)
print("B) EIGENVALUE SCALARS: omega(+kx,ky) vs omega(-kx,ky)")
print("=" * 70)
ev = torch.load(PT + r"\eigenvalue_data_full.pt", map_location="cpu", weights_only=False)
if isinstance(ev, dict):
    print("keys:", list(ev.keys()))
    ev = next(iter(ev.values()))
ev = ev if isinstance(ev, torch.Tensor) else torch.as_tensor(ev)
print("eigenvalue tensor shape:", tuple(ev.shape))
evn = ev.numpy().astype(np.float64)
# expected (n_geom, n_wv, n_bands) or flat
if evn.ndim == 1:
    evn = evn.reshape(N_GEOM, N_WV, N_BANDS)
elif evn.ndim == 2:
    evn = evn.reshape(N_GEOM, N_WV, N_BANDS)

for i_ky in (0, 1, 6, 12):
    diffs, rels = [], []
    for i_kx in range(12):
        wp, wm = wave_idx(24 - i_kx, i_ky), wave_idx(i_kx, i_ky)
        a, b = evn[:200, wp, :], evn[:200, wm, :]
        diffs.append(np.abs(a - b).max())
        rels.append((np.abs(a - b) / (np.abs(a) + 1e-30)).max())
    print(f"i_ky={i_ky:2d}: max |w+ - w-| = {max(diffs):.6e}   max rel = {max(rels):.6e}")

# ---------------------------------------------------------------- C: mirror parity of eigenfields
print()
print("=" * 70)
print("C) MIRROR PARITY of displacement fields on symmetry lines")
print("=" * 70)
disp = torch.load(PT + r"\displacements_dataset.pt", map_location="cpu", weights_only=False)
DT = disp.tensors

rng = np.random.default_rng(1)
gs = rng.choice(N_GEOM, size=8, replace=False)

def parity_score(w: int, axis: str) -> list[float]:
    """|cos| between complex u_x field and its mirror image (flip along axis)."""
    out = []
    for g_i in gs:
        for b in range(N_BANDS):
            f = flat(g_i, w, b)
            re = DT[0][f].numpy().astype(np.float64)
            im = DT[1][f].numpy().astype(np.float64)
            u = re + 1j * im
            if axis == "x":
                m = u[:, ::-1]
            else:
                m = u[::-1, :]
            num = np.abs(np.vdot(m, u))
            den = np.linalg.norm(u.ravel()) * np.linalg.norm(m.ravel())
            out.append(num / max(den, 1e-30))
    return out

tests = [
    ("(pi/2, 0)  vs y-flip", wave_idx(18, 0), "y"),
    ("(pi/2, 0)  vs x-flip", wave_idx(18, 0), "x"),
    ("(0, pi/2)  vs x-flip", wave_idx(12, 6), "x"),
    ("(0, pi/2)  vs y-flip", wave_idx(12, 6), "y"),
    ("(pi/2, pi/2) vs x-flip", wave_idx(18, 6), "x"),
    ("(pi/2, pi/2) vs y-flip", wave_idx(18, 6), "y"),
]
for name, w, ax in tests:
    sc = parity_score(w, ax)
    print(f"  {name:<24} mean |overlap| = {np.mean(sc):.4f}  (1.0 = mirror eigenstate)")

# ---------------------------------------------------------------- D: band-gap statistics
print()
print("=" * 70)
print("D) MIN BAND GAP (relative) at symmetry lines vs generic k")
print("=" * 70)
freqs = np.sqrt(np.clip(evn, 0, None))  # eigenvalues are omega^2? report both
def min_rel_gap(vals: np.ndarray) -> np.ndarray:
    """vals (G, B) sorted -> min over adjacent-band gaps of (w[b+1]-w[b])/w[b+1]."""
    v = np.sort(vals, axis=1)
    gaps = (v[:, 1:] - v[:, :-1]) / (v[:, 1:] + 1e-30)
    return gaps.min(axis=1)

groups = {
    "ky=0 row (kx=+-pi/2..3pi/4)": [wave_idx(i, 0) for i in (6, 9, 15, 18)],
    "kx=0 col (ky=pi/4..3pi/4)": [wave_idx(12, j) for j in (3, 6, 9)],
    "ky=pi row (same kx)": [wave_idx(i, 12) for i in (6, 9, 15, 18)],
    "generic (kx=+-pi/2, ky=pi/2..)": [wave_idx(18, 6), wave_idx(6, 6), wave_idx(18, 3), wave_idx(6, 9)],
}
print(f"{'group':<32} {'median gap':>11} {'p10 gap':>9} {'frac<1e-3':>10} {'frac<1e-2':>10}")
for name, ws in groups.items():
    allg = np.concatenate([min_rel_gap(freqs[:, w, :]) for w in ws])
    print(f"{name:<32} {np.median(allg):>11.5f} {np.percentile(allg,10):>9.5f} "
          f"{np.mean(allg < 1e-3):>10.4f} {np.mean(allg < 1e-2):>10.4f}")

# ---------------------------------------------------------------- E: error vs gap
print()
print("=" * 70)
print("E) PREDICTION ERROR vs BAND GAP (ky=0 row, kx=+-pi/2)")
print("=" * 70)
pred = torch.load(PRED, map_location="cpu", mmap=True, weights_only=True)
eig = torch.load(PT + r"\eigenfrequency_uniform_full.pt", map_location="cpu", mmap=True, weights_only=True)
eig_flat = eig.reshape(-1, 32, 32)

def disp_nmae_and_gap(w: int, n_g: int = 200):
    gs2 = np.arange(n_g)
    nmaes, gaps = [], []
    for g_i in gs2:
        v = np.sort(freqs[g_i, w, :])
        for b in range(N_BANDS):
            f = flat(g_i, w, b)
            p = pred[f].numpy().astype(np.float64)
            err, mag = 0.0, 0.0
            for c in range(1, 5):
                t = DT[c - 1][f].numpy().astype(np.float64)
                err += np.abs(p[c] - t).mean()
                mag += np.abs(t).mean()
            nmaes.append(err / max(mag, 1e-9))
            # gap of this band to nearest neighbor band
            wb = freqs[g_i, w, b]
            others = np.delete(freqs[g_i, w, :], b)
            gaps.append(np.min(np.abs(others - wb)) / max(wb, 1e-30))
    return np.array(nmaes), np.array(gaps)

for name, w in (("ky=0 (pi/2,0)", wave_idx(18, 0)), ("generic (pi/2,pi/2)", wave_idx(18, 6))):
    nm, gp = disp_nmae_and_gap(w)
    lo = gp < np.percentile(gp, 25)
    hi = gp > np.percentile(gp, 75)
    from scipy.stats import spearmanr
    rho, pv = spearmanr(gp, nm)
    print(f"{name}: spearman(gap, nmae) = {rho:.3f} (p={pv:.1e})  "
          f"nmae[gap Q1]={nm[lo].mean():.3f} nmae[gap Q4]={nm[hi].mean():.3f}")

print("\nDone.")
