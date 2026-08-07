from sweep_wavelet_embed_params import run_experiments
import json

exps = [
    {"name": "fs38_c9_13_s35", "freq_scale": 38, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.35},
    {"name": "fs37_c9_13_s40", "freq_scale": 37, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.40},
    {"name": "fs39_c9_13_s40", "freq_scale": 39, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.40},
    {"name": "fs32_sig35_c9_13", "freq_scale": 32, "sigma_numerator": 0.35, "kx_cycles": 9, "ky_cycles": 13},
    {"name": "fs32_sig35_c8_11", "freq_scale": 32, "sigma_numerator": 0.35, "kx_cycles": 8, "ky_cycles": 11},
    {"name": "fs35_c9_13_s35", "freq_scale": 35, "kx_cycles": 9, "ky_cycles": 13, "sigma_numerator": 0.35},
    {"name": "fs38_c8_11_s35", "freq_scale": 38, "kx_cycles": 8, "ky_cycles": 11, "sigma_numerator": 0.35},
]
r = run_experiments(exps, "Round 11 — final polish")
best = min(r, key=lambda x: (x["max_offdiag"], x["pairs_ge_0.90"]))
print("BEST:", json.dumps(best, indent=2))
