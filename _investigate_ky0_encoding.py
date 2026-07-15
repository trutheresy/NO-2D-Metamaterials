"""
Deep dive: why is the ky=0 row categorically worse than the kx=0 column?

Tests, on c_test + 0705 model:
 1. Embedding geometry: for ky=0, is embed(+kx,0) related to embed(-kx,0) by a
    trivial transform (x-reflection * sign)? Cosine similarity raw and after
    x-flip, per ky row.
 2. Truth conjugacy: verify Re fields equal / Im fields negated for (+kx,0) vs
    (-kx,0) pairs (time reversal), and NOT for ky>0 rows.
 3. Prediction ambiguity: is pred(+kx,0) ~ pred(-kx,0)? Compare pair distance of
    predictions vs pair distance of truths, per channel, per ky row.
 4. Per-channel NMAE by ky row and by kx column (Re vs Im channels).
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
EPS = 1e-5

def wave_idx(i_kx: int, i_ky: int) -> int:
    return i_ky * N_KX + i_kx

def flat(g: int, w: int, b: int) -> int:
    return g * (N_WV * N_BANDS) + w * N_BANDS + b

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))

# ---------------------------------------------------------------- embeddings
print("=" * 70)
print("1) EMBEDDING: +kx vs -kx relations per ky row")
print("=" * 70)
wf = torch.load(PT + r"\waveforms_full.pt", map_location="cpu", weights_only=False).numpy().astype(np.float64)
kxy = torch.load(PT + r"\wavevectors_full.pt", map_location="cpu", weights_only=False)[0].numpy().astype(np.float64)
print(f"waveforms {wf.shape}, k range kx [{kxy[:,0].min():.3f},{kxy[:,0].max():.3f}] ky [{kxy[:,1].min():.3f},{kxy[:,1].max():.3f}]")

print(f"\n{'i_ky':>4} {'ky':>8} | mean cos(e+, e-) | mean cos(e+, -flipx(e-)) | max abs cos")
for i_ky in range(N_KY):
    ky = kxy[wave_idx(0, i_ky), 1]
    raw, refl = [], []
    for i_kx in range(12):           # pair i_kx <-> 24-i_kx (kx = -pi+i*pi/12)
        wp, wm = wave_idx(24 - i_kx, i_ky), wave_idx(i_kx, i_ky)
        ep, em = wf[wp], wf[wm]
        raw.append(cos_sim(ep, em))
        refl.append(cos_sim(ep, -em[:, ::-1]))   # x-reflection * sign
    raw, refl = np.array(raw), np.array(refl)
    print(f"{i_ky:>4} {ky/np.pi:>7.3f}pi | {np.nanmean(raw):>16.4f} | {np.nanmean(refl):>24.4f} | {max(np.nanmax(np.abs(raw)), np.nanmax(np.abs(refl))):>10.4f}")

# nearest-neighbor distinctiveness of ky=0 row vs kx=0 column (spatial cosine)
def row_col_distinct():
    rows = {}
    ky0 = [wave_idx(i, 0) for i in range(N_KX)]
    kx0 = [wave_idx(12, j) for j in range(N_KY)]
    for name, idxs in (("ky=0 row (25)", ky0), ("kx=0 col (13)", kx0)):
        sims = []
        for a in idxs:
            best = max(
                abs(cos_sim(wf[a], wf[b])) for b in idxs if b != a
            )
            sims.append(best)
        rows[name] = (np.mean(sims), np.max(sims))
    return rows

print("\nWithin-set max |cos| to nearest neighbor (spatial):")
for name, (m, mx) in row_col_distinct().items():
    print(f"  {name}: mean {m:.4f}  max {mx:.4f}")

# ---------------------------------------------------------------- truth conjugacy
print()
print("=" * 70)
print("2) TRUTH: time-reversal conjugacy of (+kx, ky) vs (-kx, ky)")
print("=" * 70)
disp = torch.load(PT + r"\displacements_dataset.pt", map_location="cpu", weights_only=False)
DT = disp.tensors  # 4 x (N, 32, 32): ch1 Re(ux), ch2 Im(ux), ch3 Re(uy), ch4 Im(uy)
eig = torch.load(PT + r"\eigenfrequency_uniform_full.pt", map_location="cpu", mmap=True, weights_only=True)
eig_flat = eig.reshape(-1, 32, 32)

rng = np.random.default_rng(0)
geoms = rng.choice(N_GEOM, size=12, replace=False)
bands = [0, 2, 4]

def pair_relation(i_ky: int) -> dict:
    """For pairs (+kx,-kx) at row i_ky: cos of Re-Re, Im-Im, and -Im-Im."""
    re_cos, im_cos, im_neg_cos, ef_cos = [], [], [], []
    for i_kx in range(3, 12, 4):     # a few kx values
        wp, wm = wave_idx(24 - i_kx, i_ky), wave_idx(i_kx, i_ky)
        for g in geoms[:6]:
            for b in bands:
                fp, fm = flat(g, wp, b), flat(g, wm, b)
                re_x_p = DT[0][fp].numpy().astype(np.float64)
                re_x_m = DT[0][fm].numpy().astype(np.float64)
                im_x_p = DT[1][fp].numpy().astype(np.float64)
                im_x_m = DT[1][fm].numpy().astype(np.float64)
                re_cos.append(cos_sim(re_x_p, re_x_m))
                im_cos.append(cos_sim(im_x_p, im_x_m))
                im_neg_cos.append(cos_sim(im_x_p, -im_x_m))
                ef_cos.append(cos_sim(eig_flat[fp].numpy(), eig_flat[fm].numpy()))
    return {
        "re": np.nanmean(re_cos), "im": np.nanmean(im_cos),
        "im_neg": np.nanmean(im_neg_cos), "ef": np.nanmean(ef_cos),
    }

print(f"{'i_ky':>4} {'ky':>8} | cos(ReX+,ReX-) | cos(ImX+,ImX-) | cos(ImX+,-ImX-) | cos(EF+,EF-)")
for i_ky in (0, 1, 6, 12):
    r = pair_relation(i_ky)
    ky = kxy[wave_idx(0, i_ky), 1]
    print(f"{i_ky:>4} {ky/np.pi:>7.3f}pi | {r['re']:>14.4f} | {r['im']:>14.4f} | {r['im_neg']:>15.4f} | {r['ef']:>12.4f}")

# Im magnitude vs Re magnitude along special lines
print("\nMean |Im| / mean |Re| of u_x truth per wave (12 geoms x 3 bands):")
special = {
    "Gamma (0,0)": wave_idx(12, 0),
    "X (pi,0)": wave_idx(24, 0),
    "(pi/2, 0)": wave_idx(18, 0),
    "(0, pi/12)": wave_idx(12, 1),
    "(0, pi/2)": wave_idx(12, 6),
    "(pi/2, pi/2)": wave_idx(18, 6),
    "(0, pi)": wave_idx(12, 12),
}
for name, w in special.items():
    ratios = []
    for g in geoms:
        for b in bands:
            f = flat(g, w, b)
            im = np.abs(DT[1][f].numpy()).mean() + np.abs(DT[3][f].numpy()).mean()
            re = np.abs(DT[0][f].numpy()).mean() + np.abs(DT[2][f].numpy()).mean()
            ratios.append(im / max(re, 1e-9))
    print(f"  {name:<14} |Im|/|Re| = {np.mean(ratios):.4f}")

# ---------------------------------------------------------------- prediction ambiguity
print()
print("=" * 70)
print("3) PREDICTIONS: does the model distinguish +kx from -kx?")
print("=" * 70)
pred = torch.load(PRED, map_location="cpu", mmap=True, weights_only=True)

def pair_pred_stats(i_ky: int) -> dict:
    """Distance between predictions of the +- pair vs distance between truths."""
    dp_im, dt_im, dp_re, dt_re = [], [], [], []
    pred_sim_im, truth_sim_im = [], []
    for i_kx in range(3, 12, 4):
        wp, wm = wave_idx(24 - i_kx, i_ky), wave_idx(i_kx, i_ky)
        for g in geoms[:6]:
            for b in bands:
                fp, fm = flat(g, wp, b), flat(g, wm, b)
                pp = pred[fp].numpy().astype(np.float64)   # (5,32,32)
                pm = pred[fm].numpy().astype(np.float64)
                t_im_p = DT[1][fp].numpy().astype(np.float64)
                t_im_m = DT[1][fm].numpy().astype(np.float64)
                t_re_p = DT[0][fp].numpy().astype(np.float64)
                t_re_m = DT[0][fm].numpy().astype(np.float64)
                dp_im.append(np.abs(pp[2] - pm[2]).mean())
                dt_im.append(np.abs(t_im_p - t_im_m).mean())
                dp_re.append(np.abs(pp[1] - pm[1]).mean())
                dt_re.append(np.abs(t_re_p - t_re_m).mean())
                pred_sim_im.append(cos_sim(pp[2], pm[2]))
                truth_sim_im.append(cos_sim(t_im_p, t_im_m))
    return {
        "dp_im": np.mean(dp_im), "dt_im": np.mean(dt_im),
        "dp_re": np.mean(dp_re), "dt_re": np.mean(dt_re),
        "psim": np.nanmean(pred_sim_im), "tsim": np.nanmean(truth_sim_im),
    }

print(f"{'i_ky':>4} | pair dist Im: pred vs truth | pair dist Re: pred vs truth | cos Im: pred / truth")
for i_ky in (0, 1, 6, 12):
    s = pair_pred_stats(i_ky)
    print(f"{i_ky:>4} | {s['dp_im']:>10.4f} vs {s['dt_im']:>10.4f}  | {s['dp_re']:>10.4f} vs {s['dt_re']:>10.4f}  | {s['psim']:>7.3f} / {s['tsim']:>7.3f}")

# ---------------------------------------------------------------- per-channel NMAE rows/cols
print()
print("=" * 70)
print("4) PER-CHANNEL NMAE: ky rows vs kx columns")
print("=" * 70)

def nmae_wave(w: int, n_g: int = 40) -> np.ndarray:
    """Per-channel NMAE for wave w over n_g geometries, all bands. -> (5,)"""
    gs = rng.choice(N_GEOM, size=n_g, replace=False)
    idx = np.array([flat(g, w, b) for g in gs for b in range(N_BANDS)])
    out = np.zeros(5)
    p = pred[torch.from_numpy(idx)].numpy().astype(np.float64)
    t0 = eig_flat[torch.from_numpy(idx)].numpy().astype(np.float64)
    out[0] = np.mean(np.abs(p[:, 0] - t0).mean(axis=(1, 2)) / (np.abs(t0).mean(axis=(1, 2)) + EPS))
    for c in range(1, 5):
        tc = DT[c - 1][torch.from_numpy(idx)].numpy().astype(np.float64)
        out[c] = np.mean(np.abs(p[:, c] - tc).mean(axis=(1, 2)) / (np.abs(tc).mean(axis=(1, 2)) + EPS))
    return out

rows_to_show = [
    ("ky=0 row (mean of kx=+-pi/2,+-3pi/4)", [wave_idx(i, 0) for i in (6, 9, 15, 18)]),
    ("ky=pi/2 row (same kx)", [wave_idx(i, 6) for i in (6, 9, 15, 18)]),
    ("kx=0 col (ky=pi/4..3pi/4)", [wave_idx(12, j) for j in (3, 6, 9)]),
    ("kx=pi/2 col (same ky)", [wave_idx(18, j) for j in (3, 6, 9)]),
]
print(f"{'set':<38} {'ch0 EF':>8} {'ReX':>8} {'ImX':>8} {'ReY':>8} {'ImY':>8}")
for name, ws in rows_to_show:
    vals = np.mean([nmae_wave(w) for w in ws], axis=0)
    print(f"{name:<38} {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} {vals[3]:>8.3f} {vals[4]:>8.3f}")

print("\nDone.")
