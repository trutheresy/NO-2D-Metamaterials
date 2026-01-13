#!/usr/bin/env python3
"""
Quick check: reconstruct eigenvalues with Rayleigh quotient and compare to stored.

Uses a small subset (3 wavevectors x 3 bands) for efficiency.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys
import h5py

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
from system_matrices import get_transformation_matrix


def load_mat(mat_path: Path):
    with h5py.File(str(mat_path), 'r') as f:
        eigvec_data = np.array(f['EIGENVECTOR_DATA'])
        eigval_data = np.array(f['EIGENVALUE_DATA'])
        wavevectors = np.array(f['WAVEVECTOR_DATA'])
        designs = np.array(f['designs'])
        const = {k: np.array(f['const'][k]) for k in f['const']}
    if eigvec_data.dtype.names and 'real' in eigvec_data.dtype.names:
        eigvec_data = eigvec_data['real'] + 1j * eigvec_data['imag']
    return eigvec_data, eigval_data, wavevectors, designs, const


def main():
    mat_path = Path("data/out_test_10_mat_original/out_binarized_1.mat")
    if not mat_path.exists():
        print(f"File not found: {mat_path}")
        return

    eigvec_data, eigval_data, wavevectors, designs, const_raw = load_mat(mat_path)

    struct_idx = 0
    design = designs[struct_idx].transpose(1, 2, 0)  # (H, W, 3)
    wavevectors_struct = wavevectors[struct_idx].T  # (n_wv, 2)

    const = {
        'N_pix': int(np.array(const_raw['N_pix']).item()),
        'N_ele': int(np.array(const_raw['N_ele']).item()),
        'a': float(np.array(const_raw['a']).item()),
        'E_min': float(np.array(const_raw['E_min']).item()),
        'E_max': float(np.array(const_raw['E_max']).item()),
        'rho_min': float(np.array(const_raw['rho_min']).item()),
        'rho_max': float(np.array(const_raw['rho_max']).item()),
        'poisson_min': float(np.array(const_raw.get('nu_min', const_raw.get('poisson_min', 0.3))).item()),
        'poisson_max': float(np.array(const_raw.get('nu_max', const_raw.get('poisson_max', 0.3))).item()),
        't': float(np.array(const_raw.get('t', 1.0)).item()) if 't' in const_raw else 1.0,
        'design_scale': 'linear',
        'isUseImprovement': True,
        'isUseSecondImprovement': False,
        'design': design
    }

    print("=" * 80)
    print("Rayleigh Quotient Eigenvalue Check (subset: 3 wavevectors x 3 bands)")
    print("=" * 80)
    print(f"N_pix={const['N_pix']}, N_ele={const['N_ele']}, design shape={design.shape}")

    # Compute K, M once
    K, M = get_system_matrices_VEC(const)
    print(f"K shape={K.shape}, M shape={M.shape}")

    n_wv = min(3, wavevectors_struct.shape[0])
    n_bands = min(3, eigvec_data.shape[1])

    for wv_idx in range(n_wv):
        wv = wavevectors_struct[wv_idx].astype(np.float32)
        T = get_transformation_matrix(wv, const)
        T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.complex64))
        K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
        M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
        Kr = T_sparse.conj().T @ K_sparse @ T_sparse
        Mr = T_sparse.conj().T @ M_sparse @ T_sparse

        print(f"\nWavevector {wv_idx}: {wv}")
        for band_idx in range(n_bands):
            eigvec = eigvec_data[struct_idx, band_idx, wv_idx, :].astype(np.complex128)
            # Rayleigh quotient
            num = eigvec.conj().dot(Kr @ eigvec)
            den = eigvec.conj().dot(Mr @ eigvec)
            eigval_rayleigh = num / den
            freq_rayleigh = np.sqrt(np.real(eigval_rayleigh)) / (2 * np.pi)

            stored = eigval_data[struct_idx, band_idx, wv_idx]
            rel_err = abs(freq_rayleigh - stored) / max(abs(stored), 1e-12)

            print(f"  Band {band_idx}: freq_rayleigh={freq_rayleigh:.6e}, stored={stored:.6e}, rel_err={rel_err:.3e}")


if __name__ == "__main__":
    main()

