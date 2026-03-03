"""
DEBUG ONLY: delete after debugging is complete.

Compare full saved Python PT artifacts from fixed-debug generation
against MATLAB saved equivalents (direct or derived).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def _mean_rel_pct(a: np.ndarray, b: np.ndarray) -> float:
    abs_diff = np.abs(a - b)
    denom = max(float(np.mean(np.abs(a))), 1e-12)
    return float(np.mean(abs_diff) / denom * 100.0)


def _compare(name: str, a: np.ndarray, b: np.ndarray) -> dict:
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


def _build_remap(mw: np.ndarray, pw: np.ndarray) -> np.ndarray:
    key = lambda row: (round(float(row[0]), 12), round(float(row[1]), 12))
    pmap = {key(r): i for i, r in enumerate(pw)}
    remap = []
    for r in mw:
        k = key(r)
        if k in pmap:
            remap.append(pmap[k])
        else:
            d2 = np.sum((pw - r[None, :]) ** 2, axis=1)
            remap.append(int(np.argmin(d2)))
    return np.asarray(remap, dtype=np.int64)


def _expected_reduced_indices(n_struct: int, n_wv: int, n_eig: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for d in range(n_struct):
        for w in range(n_wv):
            for b in range(n_eig):
                out.append((d, w, b))
    return out


def _compare_displacements(mat_ev: np.ndarray, py_ds, remap: np.ndarray) -> dict:
    # MATLAB: (N_dof, N_wv, N_eig, N_struct), here fixed N_struct=1
    ev = mat_ev[..., 0]
    n_dof, n_wv, n_eig = ev.shape
    n_pix = int(np.sqrt(n_dof // 2))
    ev_t = np.transpose(ev, (1, 2, 0))  # (N_wv,N_eig,N_dof) in MATLAB order
    ex = ev_t[..., 0::2].reshape(n_wv * n_eig, n_pix, n_pix)
    ey = ev_t[..., 1::2].reshape(n_wv * n_eig, n_pix, n_pix)

    xr, xi, yr, yi = py_ds.tensors
    xr = xr.detach().cpu().numpy().astype(np.float64)
    xi = xi.detach().cpu().numpy().astype(np.float64)
    yr = yr.detach().cpu().numpy().astype(np.float64)
    yi = yi.detach().cpu().numpy().astype(np.float64)

    # Python dataset sample order is python-wavevector major; align to MATLAB order first.
    py_wb_indices = []
    for w_mat in range(n_wv):
        w_py = int(remap[w_mat])
        for b in range(n_eig):
            py_wb_indices.append(w_py * n_eig + b)
    py_wb_indices_arr = np.asarray(py_wb_indices, dtype=np.int64)
    xr_al = xr[py_wb_indices_arr]
    xi_al = xi[py_wb_indices_arr]
    yr_al = yr[py_wb_indices_arr]
    yi_al = yi[py_wb_indices_arr]

    px = xr_al + 1j * xi_al
    py = yr_al + 1j * yi_al
    mx = ex.astype(np.complex128)
    my = ey.astype(np.complex128)

    # Per-(wavevector,band) global phase alignment for eigenvectors.
    px_al = np.empty_like(px, dtype=np.complex128)
    py_al = np.empty_like(py, dtype=np.complex128)
    for i in range(px.shape[0]):
        mvec = np.concatenate([mx[i].ravel(), my[i].ravel()])
        pvec = np.concatenate([px[i].ravel(), py[i].ravel()])
        inner = np.vdot(pvec, mvec)
        if np.abs(inner) > 0:
            alpha = inner / np.abs(inner)
        else:
            alpha = 1.0 + 0.0j
        px_al[i] = px[i] * alpha
        py_al[i] = py[i] * alpha

    return {
        "x_real": _compare("displacements_x_real", mx.real.astype(np.float64), px_al.real.astype(np.float64)),
        "x_imag": _compare("displacements_x_imag", mx.imag.astype(np.float64), px_al.imag.astype(np.float64)),
        "y_real": _compare("displacements_y_real", my.real.astype(np.float64), py_al.real.astype(np.float64)),
        "y_imag": _compare("displacements_y_imag", my.imag.astype(np.float64), py_al.imag.astype(np.float64)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full saved PT bundle to MATLAB equivalents.")
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--mat-out", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    mat_out = Path(args.mat_out)
    rep = Path(args.report_json)
    rep.parent.mkdir(parents=True, exist_ok=True)

    m = sio.loadmat(str(mat_out), squeeze_me=False, struct_as_record=False)
    m_wv_all = np.asarray(m["WAVEVECTOR_DATA"], dtype=np.float64)
    m_fr_all = np.asarray(m["EIGENVALUE_DATA"], dtype=np.float64)
    m_designs_all = np.asarray(m["designs"], dtype=np.float64)
    m_ev = np.asarray(m["EIGENVECTOR_DATA"])
    if m_wv_all.ndim == 2:
        m_wv_all = np.expand_dims(m_wv_all, axis=2)
    if m_fr_all.ndim == 2:
        m_fr_all = np.expand_dims(m_fr_all, axis=2)
    if m_designs_all.ndim == 3:
        m_designs_all = np.expand_dims(m_designs_all, axis=3)
    if m_ev.ndim == 3:
        m_ev = np.expand_dims(m_ev, axis=3)
    m_wv = m_wv_all[:, :, 0]
    m_fr = m_fr_all[:, :, 0]
    m_designs = m_designs_all[:, :, :, 0]
    n_struct = 1
    n_wv = m_wv.shape[0]
    n_eig = m_fr.shape[1]

    py_geom = torch.load(pt_dir / "geometries_full.pt", map_location="cpu").numpy().astype(np.float64)
    py_wv = torch.load(pt_dir / "wavevectors_full.pt", map_location="cpu").numpy().astype(np.float64)
    py_fr = torch.load(pt_dir / "eigenvalue_data_full.pt", map_location="cpu").numpy().astype(np.float64)
    py_waveforms = torch.load(pt_dir / "waveforms_full.pt", map_location="cpu").numpy().astype(np.float64)
    py_band_fft = torch.load(pt_dir / "band_fft_full.pt", map_location="cpu").numpy().astype(np.float64)
    py_design_params = torch.load(pt_dir / "design_params_full.pt", map_location="cpu").numpy().astype(np.float64)
    py_indices = torch.load(pt_dir / "reduced_indices.pt", map_location="cpu")
    py_ds = torch.load(pt_dir / "displacements_dataset.pt", map_location="cpu")

    remap = _build_remap(m_wv, py_wv[0])
    py_fr_aligned = py_fr[0][remap, :]

    # Derived equivalents for waveforms / bands using MATLAB-side inputs.
    try:
        import NO_utils_multiple  # type: ignore

        waveforms_from_mat = NO_utils_multiple.embed_2const_wavelet(m_wv[:, 0], m_wv[:, 1], size=m_designs.shape[0])
        bands_from_mat = NO_utils_multiple.embed_integer_wavelet(np.arange(1, n_eig + 1), size=m_designs.shape[0])
    except Exception:
        waveforms_from_mat = np.zeros_like(py_waveforms)
        bands_from_mat = np.zeros_like(py_band_fft)

    comp = []
    comp.append(_compare("geometries_full", m_designs[:, :, 0][None, :, :], py_geom))
    comp.append(_compare("wavevectors_full_raw", m_wv[None, :, :], py_wv))
    comp.append(_compare("wavevectors_full_aligned", m_wv, py_wv[0][remap, :]))
    comp.append(_compare("eigenvalue_data_full_aligned", m_fr, py_fr_aligned))
    comp.append(_compare("waveforms_full_raw", waveforms_from_mat, py_waveforms))
    comp.append(_compare("waveforms_full_aligned", waveforms_from_mat, py_waveforms[remap, :, :]))
    comp.append(_compare("band_fft_full_from_n_eig", bands_from_mat, py_band_fft))
    comp.append(_compare("design_params_full_design_number", np.array([[0.0]], dtype=np.float64), py_design_params))

    expected_idx = _expected_reduced_indices(n_struct, n_wv, n_eig)
    indices_exact = list(py_indices) == expected_idx
    comp.append(
        {
            "name": "reduced_indices",
            "shape_match": True,
            "exact_match": bool(indices_exact),
            "count_python": int(len(py_indices)),
            "count_expected": int(len(expected_idx)),
        }
    )

    disp_cmp = _compare_displacements(m_ev, py_ds, remap)

    report = {
        "pt_dir": str(pt_dir),
        "mat_out": str(mat_out),
        "comparisons": comp,
        "displacements_dataset_comparison": disp_cmp,
    }
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
