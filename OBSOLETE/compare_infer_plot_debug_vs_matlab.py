"""
Compare infer-plot debug intermediates (Python) against MATLAB debug artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio


def _mean_rel_pct(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(a - b)
    den = max(float(np.mean(np.abs(a))), 1e-12)
    return float(np.mean(d) / den * 100.0)


def _cmp(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"name": name, "shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    d = np.abs(a - b)
    return {
        "name": name,
        "shape_match": True,
        "max_abs": float(np.max(d)),
        "mean_abs": float(np.mean(d)),
        "mean_rel_pct": _mean_rel_pct(a, b),
    }


def _remap(mwv: np.ndarray, pwv: np.ndarray) -> np.ndarray:
    key = lambda r: (round(float(r[0]), 12), round(float(r[1]), 12))
    pmap = {key(r): i for i, r in enumerate(pwv)}
    out = []
    for r in mwv:
        k = key(r)
        if k in pmap:
            out.append(pmap[k])
        else:
            d2 = np.sum((pwv - r[None, :]) ** 2, axis=1)
            out.append(int(np.argmin(d2)))
    return np.asarray(out, dtype=np.int64)


def _triplet_to_map(tri: np.ndarray, matlab_1_based: bool = False) -> dict[tuple[int, int], complex]:
    out: dict[tuple[int, int], complex] = {}
    for r, c, re, im in tri:
        rr = int(r) - 1 if matlab_1_based else int(r)
        cc = int(c) - 1 if matlab_1_based else int(c)
        out[(rr, cc)] = complex(float(re), float(im))
    return out


def _cmp_triplet(name: str, m_tri: np.ndarray, p_tri: np.ndarray) -> dict:
    dm = _triplet_to_map(m_tri, matlab_1_based=True)
    dp = _triplet_to_map(p_tri, matlab_1_based=False)
    keys = sorted(set(dm.keys()).union(dp.keys()))
    vm = np.asarray([dm.get(k, 0.0 + 0.0j) for k in keys], dtype=np.complex128)
    vp = np.asarray([dp.get(k, 0.0 + 0.0j) for k in keys], dtype=np.complex128)
    return _cmp(name, vm, vp) | {"nnz_matlab": len(dm), "nnz_python": len(dp), "union_nnz": len(keys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-npz", required=True)
    ap.add_argument("--matlab-intermediate-mat", required=True)
    ap.add_argument("--matlab-plot-debug-mat", required=True)
    ap.add_argument("--report-json", required=True)
    args = ap.parse_args()

    py = np.load(args.python_npz)
    mi = sio.loadmat(args.matlab_intermediate_mat, squeeze_me=True, struct_as_record=False)
    mp = sio.loadmat(args.matlab_plot_debug_mat, squeeze_me=True, struct_as_record=False)["plot_debug"]

    m_design_input = np.asarray(mi["design_input"], dtype=np.float64)
    m_design3 = np.asarray(mi["design3"], dtype=np.float64)
    m_wv = np.asarray(mi["wv"], dtype=np.float64)
    m_fr = np.real(np.asarray(mi["fr"])).astype(np.float64)
    m_k = np.asarray(mi["K_triplet"], dtype=np.float64)
    m_m = np.asarray(mi["M_triplet"], dtype=np.float64)
    m_tdiag = np.asarray(mi["T_diag"], dtype=np.int64)

    p_design_raw = np.asarray(py["design_raw_single"], dtype=np.float64)
    p_design_norm = np.asarray(py["design_normalized"], dtype=np.float64)
    p_wv = np.asarray(py["wavevectors_raw"], dtype=np.float64)
    p_fr = np.asarray(py["frequencies_recon_raw"], dtype=np.float64)
    p_k = np.asarray(py["K_triplet"], dtype=np.float64)
    p_m = np.asarray(py["M_triplet"], dtype=np.float64)
    p_tdiag = np.asarray(py["T_diag"], dtype=np.int64)

    remap = _remap(m_wv, p_wv)

    comps = []
    comps.append(_cmp("design_raw_single_vs_matlab_input_ch1", m_design_input[:, :, 0], p_design_raw))
    comps.append(_cmp("design_normalized_vs_matlab_design3", m_design3, p_design_norm))
    comps.append(_cmp("wavevectors_raw", m_wv, p_wv))
    comps.append(_cmp("wavevectors_aligned", m_wv, p_wv[remap, :]))
    comps.append(_cmp("frequencies_recon_raw", m_fr, p_fr[remap, :]))
    comps.append(_cmp_triplet("K_triplet", m_k, p_k))
    comps.append(_cmp_triplet("M_triplet", m_m, p_m))
    comps.append(_cmp("T_diag", m_tdiag, p_tdiag))

    m_contour_wv = np.asarray(mp.contour_wavevectors, dtype=np.float64)
    m_contour_param = np.asarray(mp.contour_parameter, dtype=np.float64).reshape(-1)
    m_fcont = np.asarray(mp.frequencies_contour, dtype=np.float64)
    p_contour_wv = np.asarray(py["contour_wavevectors"], dtype=np.float64)
    p_contour_param = np.asarray(py["contour_parameter"], dtype=np.float64).reshape(-1)
    p_fcont = np.asarray(py["frequencies_contour"], dtype=np.float64)

    comps.append(_cmp("contour_wavevectors", m_contour_wv, p_contour_wv))
    comps.append(_cmp("contour_parameter", m_contour_param, p_contour_param))
    comps.append(_cmp("frequencies_contour", m_fcont, p_fcont))
    comps.append(_cmp("plot_points_y", m_fcont, np.asarray(py["plot_y"], dtype=np.float64)))

    out = {"comparisons": comps}
    rp = Path(args.report_json)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
