from sweep_wavelet_embed_params import run_experiments
import json

exps = [
    {"name": "fs39_c9_13_s35", "freq_scale": 39, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.35},
    {"name": "fs39_c9_13_s38", "freq_scale": 39, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.38},
    {"name": "fs39_c10_14_s40", "freq_scale": 39, "kx_cycles": 10, "ky_cycles": 14, "sigma_numerator": 0.40},
    {"name": "fs40_c9_13_s40", "freq_scale": 40, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.40},
]
r = run_experiments(exps, "Round 12 — fs39 neighborhood")
best = min(r, key=lambda x: (x["max_offdiag"], x["pairs_ge_0.90"]))
print("BEST:", json.dumps(best, indent=2))
