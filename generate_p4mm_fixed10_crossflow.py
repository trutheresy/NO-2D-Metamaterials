from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def _add_py_lib(repo: Path) -> None:
    import sys

    py_lib = repo / "2d-dispersion-py"
    if str(py_lib) not in sys.path:
        sys.path.insert(0, str(py_lib))


def _const() -> dict:
    n_wv_0 = 25
    return {
        "N_ele": 1,
        "N_pix": 32,
        "N_wv": [n_wv_0, int(np.ceil(n_wv_0 / 2))],
        "N_eig": 6,
        "sigma_eig": 1e-2,
        "a": 1.0,
        "design_scale": "linear",
        "E_min": 200e6,
        "E_max": 200e9,
        "rho_min": 8e2,
        "rho_max": 8e3,
        "poisson_min": 0.0,
        "poisson_max": 0.5,
        "t": 1.0,
        "isUseGPU": False,
        "isUseImprovement": True,
        "isUseSecondImprovement": False,
        "isUseParallel": False,
        "isSaveEigenvectors": False,
        "isSaveKandM": False,
        "isSaveMesh": False,
        "symmetry_type": "none",
    }


def main() -> None:
    repo = Path(__file__).resolve().parent
    _add_py_lib(repo)

    from design_parameters import DesignParameters  # type: ignore
    from get_design2 import get_design2  # type: ignore
    from wavevectors import get_IBZ_wavevectors  # type: ignore
    from design_conversion import apply_steel_rubber_paradigm  # type: ignore
    from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt  # type: ignore

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = repo / "fixed10_p4mm_outputs" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    n_struct = 10
    n_pix = 32

    # Build deterministic p4mm geometry bank (single-channel values in [0,1]).
    geometries = np.zeros((n_struct, n_pix, n_pix), dtype=np.float64)
    dp = DesignParameters(n_struct)
    dp.property_coupling = "coupled"
    dp.design_style = "kernel"
    dp.design_options = {
        "kernel": "periodic",
        "sigma_f": 1.0,
        "sigma_l": 1.0,
        "symmetry_type": "p4mm",
        "N_value": np.inf,
    }
    dp.N_pix = [n_pix, n_pix]
    dp = dp.prepare()

    for i in range(n_struct):
        dp.design_number = i
        dp = dp.prepare()
        d = np.asarray(get_design2(dp), dtype=np.float64)
        if d.ndim == 3:
            geometries[i] = d[:, :, 0]
        else:
            geometries[i] = d
        geometries[i] = np.clip(geometries[i], 0.0, 1.0)

    geom_mat = out_root / "geometries_full_p4mm_10.mat"
    sio.savemat(str(geom_mat), {"geometries_full": geometries})

    # Python prescribed-geometries generation.
    const = _const()
    const["wavevectors"] = np.asarray(get_IBZ_wavevectors(const["N_wv"], const["a"], "none"), dtype=np.float16)
    n_wv = int(np.prod(const["N_wv"]))
    n_eig = int(const["N_eig"])

    wavevector_data = np.zeros((n_struct, n_wv, 2), dtype=np.float16)
    eigenvalue_data = np.zeros((n_struct, n_wv, n_eig), dtype=np.float16)

    for i in range(n_struct):
        g = geometries[i].astype(np.float16)
        design3 = np.stack([g, g, g], axis=2).astype(np.float16)
        design3 = np.asarray(apply_steel_rubber_paradigm(design3, const), dtype=np.float16)
        const["design"] = design3
        wv, fr, *_ = dispersion_with_matrix_save_opt(const, const["wavevectors"])
        wavevector_data[i] = np.asarray(wv, dtype=np.float16)
        eigenvalue_data[i] = np.asarray(np.real(fr), dtype=np.float16)

    py_pt_dir = out_root / "python_pt_dataset"
    py_pt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(geometries.astype(np.float16)), py_pt_dir / "geometries_full.pt")
    torch.save(torch.from_numpy(wavevector_data), py_pt_dir / "wavevectors_full.pt")
    torch.save(torch.from_numpy(eigenvalue_data), py_pt_dir / "eigenvalue_data_full.pt")

    manifest = {
        "output_root": str(out_root),
        "geometry_mat": str(geom_mat),
        "python_pt_dir": str(py_pt_dir),
        "n_struct": n_struct,
        "symmetry_type": "p4mm",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
