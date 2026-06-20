import math
import torch
import numpy as np

def one(x):
    r = x / np.pi
    for d in [1, 2, 3, 4, 6, 12, 24]:
        n = round(r * d)
        if abs(r - n / d) < 1e-3:
            if n == 0:
                return "0"
            sign = "-" if n < 0 else ""
            n = abs(n)
            if n == d:
                return f"{sign}π"
            g = math.gcd(n, d)
            n, d = n // g, d // g
            if n == 1:
                return f"{sign}π/{d}"
            return f"{sign}{n}π/{d}"
    return f"{x/np.pi:.4f}π"

def k_as_pi_frac(kx, ky):
    if abs(ky) < 1e-3:
        return f"({one(kx)}, 0)"
    return f"({one(kx)}, {one(ky)})"

k = torch.load(
    "DATASETS/c_test/continuous_2026-03-05_20-07-34_pt/wavevectors_full.pt",
    map_location="cpu",
    weights_only=False,
)[0].numpy()
rows = [(11, 2), (12, 22), (13, 10), (14, 14), (15, 3), (16, 8), (17, 21), (18, 16), (19, 9), (20, 15)]
c_pct = {2: 67.6, 22: 67.4, 10: 67.2, 14: 66.7, 3: 66.3, 8: 66.0, 21: 65.8, 16: 65.6, 9: 65.5, 15: 65.5}
b_pct = {2: 75.3, 22: 75.7, 10: 69.0, 14: 68.5, 3: 73.2, 21: 73.1, 8: 66.7, 16: 65.6, 9: 66.0, 15: 65.5}
print("rank|wave|k (π frac)|c_test %|b_test %")
for rank, w in rows:
    kx, ky = k[w]
    bp = b_pct.get(w, float("nan"))
    print(f"{rank}|w{w}|{k_as_pi_frac(kx, ky)}|{c_pct[w]:.1f}|{bp:.1f}")
