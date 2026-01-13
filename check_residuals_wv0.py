#!/usr/bin/env python3
"""
Compute Rayleigh eigenvalues and residuals for wv0 first bands.
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import h5py
import sys
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
from system_matrices import get_transformation_matrix


def main():
    mat = Path("data/out_test_10_mat_original/out_binarized_1.mat")
    with h5py.File(mat, "r") as f:
        eigvec = np.array(f["EIGENVECTOR_DATA"])
        if eigvec.dtype.names:
            eigvec = eigvec["real"] + 1j * eigvec["imag"]
        wave = np.array(f["WAVEVECTOR_DATA"])
        designs = np.array(f["designs"])
        const_raw = {k: np.array(f["const"][k]) for k in f["const"]}

    struct_idx = 0
    wv_idx = 0
    design = designs[struct_idx].transpose(1, 2, 0)
    const = {
        "N_pix": int(np.array(const_raw["N_pix"]).item()),
        "N_ele": int(np.array(const_raw["N_ele"]).item()),
        "a": float(np.array(const_raw["a"]).item()),
        "E_min": float(np.array(const_raw["E_min"]).item()),
        "E_max": float(np.array(const_raw["E_max"]).item()),
        "rho_min": float(np.array(const_raw["rho_min"]).item()),
        "rho_max": float(np.array(const_raw["rho_max"]).item()),
        "poisson_min": float(
            np.array(const_raw.get("nu_min", const_raw.get("poisson_min", 0.3))).item()
        ),
        "poisson_max": float(
            np.array(const_raw.get("nu_max", const_raw.get("poisson_max", 0.3))).item()
        ),
        "t": float(np.array(const_raw.get("t", 1.0)).item()) if "t" in const_raw else 1.0,
        "design_scale": "linear",
        "isUseImprovement": True,
        "isUseSecondImprovement": False,
        "design": design,
    }

    K, M = get_system_matrices_VEC(const)
    wv = wave[struct_idx][:, wv_idx].astype(np.float32)
    T = sp.csr_matrix(get_transformation_matrix(wv, const))
    K = sp.csr_matrix(K)
    M = sp.csr_matrix(M)
    Kr = T.conj().T @ K @ T
    Mr = T.conj().T @ M @ T

    def eval_band(b):
        v = eigvec[struct_idx, b, wv_idx, :].astype(np.complex128)
        lhs = Kr @ v
        rhs = Mr @ v
        lhs = lhs.toarray().ravel() if sp.issparse(lhs) else lhs.ravel()
        rhs = rhs.toarray().ravel() if sp.issparse(rhs) else rhs.ravel()
        lam = v.conj().dot(lhs) / v.conj().dot(rhs)
        res = np.linalg.norm(lhs - lam * rhs) / np.linalg.norm(lhs)
        freq = np.sqrt(np.real(lam)) / (2 * np.pi)
        return freq, res

    for b in range(4):
        freq, res = eval_band(b)
        print(f"band {b}: freq={freq:.6e}, residual={res:.3e}")


if __name__ == "__main__":
    main()

