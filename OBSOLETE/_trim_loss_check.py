import numpy as np
import torch

PT = r"d:\Research\NO-2D-Metamaterials\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt"
MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
PRED = rf"d:\Research\NO-2D-Metamaterials\INFERENCE\{MODEL}\c_test\predictions_I3O5_{MODEL}.pt"
N_KX, N_KY, N_BANDS = 25, 13, 6
N_WV = N_KX * N_KY
EPS = 1e-5


def flat(g, w, b):
    return g * (N_WV * N_BANDS) + w * N_BANDS + b


def wi(ix, iy):
    return iy * N_KX + ix


disp = torch.load(PT + r"\displacements_dataset.pt", map_location="cpu", weights_only=False)
DT = disp.tensors
pred = torch.load(PRED, map_location="cpu", mmap=True, weights_only=True)
gs = np.arange(40)
waves = {
    "Gamma (0,0)": wi(12, 0),
    "X (pi,0)": wi(24, 0),
    "(0,pi)": wi(12, 12),
    "M (pi,pi)": wi(24, 12),
    "ky=0 (pi/2,0)": wi(18, 0),
    "generic (pi/2,pi/2)": wi(18, 6),
}
print(f"{'set':<22} {'mean|Re|':>9} {'mean|Im|':>9} {'Im/Re':>8} {'NMAE ReX':>9} {'NMAE ImX':>9} {'frac Im~0':>10}")
for name, w in waves.items():
    re_m, im_m, nr, ni, tiny = [], [], [], [], []
    for g in gs:
        for b in range(N_BANDS):
            f = flat(g, w, b)
            rx = DT[0][f].numpy().astype(np.float64)
            ix = DT[1][f].numpy().astype(np.float64)
            ry = DT[2][f].numpy().astype(np.float64)
            iy = DT[3][f].numpy().astype(np.float64)
            rem = 0.5 * (np.abs(rx).mean() + np.abs(ry).mean())
            imm = 0.5 * (np.abs(ix).mean() + np.abs(iy).mean())
            re_m.append(rem)
            im_m.append(imm)
            tiny.append(imm < 1e-4)
            p = pred[f].numpy().astype(np.float64)
            nr.append(np.abs(p[1] - rx).mean() / (np.abs(rx).mean() + EPS))
            ni.append(np.abs(p[2] - ix).mean() / (np.abs(ix).mean() + EPS))
    rem, imm = np.mean(re_m), np.mean(im_m)
    print(
        f"{name:<22} {rem:9.4f} {imm:9.4f} {imm/max(rem,1e-12):8.4f} "
        f"{np.mean(nr):9.3f} {np.mean(ni):9.3f} {np.mean(tiny):10.3f}"
    )
