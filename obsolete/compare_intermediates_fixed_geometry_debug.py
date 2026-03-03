"""
DEBUG ONLY: delete after debugging is complete.

Exact intermediate comparison for fixed-geometry MATLAB vs Python debug flows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio


def _mean_rel_pct(a: np.ndarray, b: np.ndarray) -> float:
    abs_diff = np.abs(a - b)
    denom = max(float(np.mean(np.abs(a))), 1e-12)
    return float(np.mean(abs_diff) / denom * 100.0)


def _compare_dense(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"name": name, "shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    abs_diff = np.abs(a - b)
    return {
        "name": name,
        "shape_match": True,
        "max_abs": float(np.max(abs_diff)),
        "mean_abs": float(np.mean(abs_diff)),
        "mean_rel_pct": _mean_rel_pct(a, b),
    }


def _triplet_to_dict(tri: np.ndarray) -> dict[tuple[int, int], complex]:
    out: dict[tuple[int, int], complex] = {}
    if tri.size == 0:
        return out
    for r, c, re, im in tri:
        out[(int(r), int(c))] = complex(float(re), float(im))
    return out


def _normalize_matlab_triplet_indices(tri: np.ndarray) -> np.ndarray:
    if tri.size == 0:
        return tri
    out = tri.copy()
    out[:, 0] = out[:, 0] - 1.0
    out[:, 1] = out[:, 1] - 1.0
    return out


def _compare_triplets(name: str, ta: np.ndarray, tb: np.ndarray) -> dict:
    da = _triplet_to_dict(ta)
    db = _triplet_to_dict(tb)
    keys = sorted(set(da.keys()).union(db.keys()))
    if not keys:
        return {"name": name, "nnz_a": 0, "nnz_b": 0, "max_abs": 0.0, "mean_abs": 0.0, "mean_rel_pct": 0.0}
    va = np.asarray([da.get(k, 0.0 + 0.0j) for k in keys], dtype=np.complex128)
    vb = np.asarray([db.get(k, 0.0 + 0.0j) for k in keys], dtype=np.complex128)
    abs_diff = np.abs(va - vb)
    denom = max(float(np.mean(np.abs(va))), 1e-12)
    return {
        "name": name,
        "nnz_a": len(da),
        "nnz_b": len(db),
        "union_nnz": len(keys),
        "max_abs": float(np.max(abs_diff)),
        "mean_abs": float(np.mean(abs_diff)),
        "mean_rel_pct": float(np.mean(abs_diff) / denom * 100.0),
    }


def _build_wavevector_remap(mw: np.ndarray, pw: np.ndarray) -> tuple[np.ndarray, dict]:
    key = lambda row: (round(float(row[0]), 12), round(float(row[1]), 12))
    pmap = {key(r): i for i, r in enumerate(pw)}
    remap, fallback, dists = [], 0, []
    for r in mw:
        k = key(r)
        if k in pmap:
            remap.append(pmap[k])
            dists.append(0.0)
        else:
            d2 = np.sum((pw - r[None, :]) ** 2, axis=1)
            idx = int(np.argmin(d2))
            remap.append(idx)
            dists.append(float(np.sqrt(d2[idx])))
            fallback += 1
    remap_arr = np.asarray(remap, dtype=np.int64)
    diag = {
        "fallback_count": int(fallback),
        "fallback_fraction": float(fallback / max(len(mw), 1)),
        "max_nn_distance": float(np.max(dists) if dists else 0.0),
        "mean_nn_distance": float(np.mean(dists) if dists else 0.0),
        "all_unique": bool(len(np.unique(remap_arr)) == len(remap_arr)),
        "identity_fraction": float(np.mean(remap_arr == np.arange(len(remap_arr)))),
    }
    return remap_arr, diag


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-geometry Python/MATLAB intermediates.")
    parser.add_argument("--python-debug-dir", required=True)
    parser.add_argument("--matlab-stage-mat", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    py_dir = Path(args.python_debug_dir)
    m_stage = Path(args.matlab_stage_mat)
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    py_design_in = np.load(py_dir / "design_input.npy")
    py_design_map = np.load(py_dir / "design_after_map.npy")
    py_wv = np.load(py_dir / "wavevectors.npy").astype(np.float64)
    py_fr = np.load(py_dir / "frequencies.npy").astype(np.float64)
    py_k = np.load(py_dir / "K_triplet.npy")
    py_m = np.load(py_dir / "M_triplet.npy")

    m = sio.loadmat(str(m_stage), squeeze_me=False, struct_as_record=False)
    md_in = np.asarray(m["design_input"], dtype=np.float64)
    md_map = np.asarray(m["design3"], dtype=np.float64)
    mwv = np.asarray(m["wv"], dtype=np.float64)
    mfr_complex = np.asarray(m["fr"])
    mfr_real = np.real(mfr_complex).astype(np.float64)
    mfr_imag_max = float(np.max(np.abs(np.imag(mfr_complex))))
    mk = np.asarray(m["K_triplet"], dtype=np.float64)
    mm = np.asarray(m["M_triplet"], dtype=np.float64)
    mk = _normalize_matlab_triplet_indices(mk)
    mm = _normalize_matlab_triplet_indices(mm)

    remap, remap_diag = _build_wavevector_remap(mwv, py_wv)
    py_fr_aligned = py_fr[remap, :]

    comparisons = [
        _compare_dense("design_input", md_in, py_design_in),
        _compare_dense("design_mapped", md_map, py_design_map),
        _compare_dense("wavevectors_raw", mwv, py_wv),
        _compare_dense("frequencies_aligned", mfr_real, py_fr_aligned),
        _compare_triplets("K_triplet", mk, py_k),
        _compare_triplets("M_triplet", mm, py_m),
    ]
    if (py_dir / "T_diag.npy").exists() and "T_diag" in m:
        py_tdiag = np.load(py_dir / "T_diag.npy")
        m_tdiag = np.asarray(m["T_diag"], dtype=np.int64)
        comparisons.append(_compare_dense("T_diag", m_tdiag, py_tdiag))

    alerts = []
    for c in comparisons:
        if not c.get("shape_match", True):
            alerts.append({"quantity": c["name"], "type": "logic", "reason": "shape mismatch"})
            continue
        mr = c.get("mean_rel_pct", 0.0)
        if c["name"] == "wavevectors_raw":
            if remap_diag["fallback_fraction"] > 0.0 or remap_diag["identity_fraction"] < 1.0:
                alerts.append(
                    {
                        "quantity": "wavevectors",
                        "type": "logic",
                        "reason": "grid traversal/indexing mismatch",
                        "fallback_fraction": remap_diag["fallback_fraction"],
                        "identity_fraction": remap_diag["identity_fraction"],
                    }
                )
            continue
        if mr >= 1.0:
            alerts.append({"quantity": c["name"], "type": "logic", "mean_rel_pct": mr})
        elif mr > 0.0:
            alerts.append({"quantity": c["name"], "type": "precision", "mean_rel_pct": mr})

    report = {
        "python_debug_dir": str(py_dir),
        "matlab_stage_mat": str(m_stage),
        "wavevector_remap_diagnostics": remap_diag,
        "matlab_fr_imag_max_abs": mfr_imag_max,
        "comparisons": comparisons,
        "deviation_alerts": alerts,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
