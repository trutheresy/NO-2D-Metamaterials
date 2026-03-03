"""
Compare Python/MATLAB plot-debug intermediates and final plot points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio


def _mean_rel_pct(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(a - b)
    denom = max(float(np.mean(np.abs(a))), 1e-12)
    return float(np.mean(d) / denom * 100.0)


def _compare(name: str, a: np.ndarray, b: np.ndarray) -> dict:
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


def _load_mat_struct(path: Path) -> dict:
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    s = m["plot_debug"]
    return {
        "wavevectors_raw": np.asarray(s.wavevectors_raw, dtype=np.float64),
        "frequencies_raw": np.asarray(s.frequencies_raw, dtype=np.float64),
        "contour_wavevectors": np.asarray(s.contour_wavevectors, dtype=np.float64),
        "contour_parameter": np.asarray(s.contour_parameter, dtype=np.float64).reshape(-1),
        "frequencies_contour": np.asarray(s.frequencies_contour, dtype=np.float64),
        "plot_x": np.asarray(s.plot_x, dtype=np.float64).reshape(-1),
        "plot_y": np.asarray(s.plot_y, dtype=np.float64),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--python-debug-dir", required=True)
    p.add_argument("--matlab-debug-dir", required=True)
    p.add_argument("--n-structs", type=int, default=1)
    p.add_argument("--report-json", required=True)
    args = p.parse_args()

    py_dir = Path(args.python_debug_dir)
    m_dir = Path(args.matlab_debug_dir)
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    per_struct = []
    for s in range(args.n_structs):
        py = np.load(py_dir / f"struct_{s}_plot_debug.npz")
        md = _load_mat_struct(m_dir / f"struct_{s}_plot_debug.mat")

        pwv = np.asarray(py["wavevectors_raw"], dtype=np.float64)
        mwv = md["wavevectors_raw"]
        remap = _remap(mwv, pwv)
        pfr = np.asarray(py["frequencies_raw"], dtype=np.float64)
        pfr_aligned = pfr[remap, :]

        comps = [
            _compare("wavevectors_raw", mwv, pwv),
            _compare("wavevectors_aligned", mwv, pwv[remap, :]),
            _compare("frequencies_raw", md["frequencies_raw"], pfr),
            _compare("frequencies_aligned", md["frequencies_raw"], pfr_aligned),
            _compare("contour_wavevectors", md["contour_wavevectors"], np.asarray(py["contour_wavevectors"], dtype=np.float64)),
            _compare("contour_parameter", md["contour_parameter"], np.asarray(py["contour_parameter"], dtype=np.float64)),
            _compare("frequencies_contour", md["frequencies_contour"], np.asarray(py["frequencies_contour"], dtype=np.float64)),
            _compare(
                "frequencies_contour_scattered",
                md["frequencies_contour"],
                np.asarray(py["frequencies_contour_scattered"], dtype=np.float64)
                if "frequencies_contour_scattered" in py
                else np.asarray(py["frequencies_contour"], dtype=np.float64),
            ),
            _compare("plot_x", md["plot_x"], np.asarray(py["plot_x"], dtype=np.float64)),
            _compare("plot_y", md["plot_y"], np.asarray(py["plot_y"], dtype=np.float64)),
        ]
        per_struct.append({"struct_idx": s, "comparisons": comps})

    out_obj = {"n_structs": int(args.n_structs), "per_struct": per_struct}
    out.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(json.dumps(out_obj, indent=2))


if __name__ == "__main__":
    main()
