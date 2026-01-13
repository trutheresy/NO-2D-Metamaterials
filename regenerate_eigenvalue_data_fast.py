#!/usr/bin/env python3
"""
Fast version: Regenerate EIGENVALUE_DATA for first structure only (for testing).
"""

import numpy as np
import h5py
import scipy.sparse as sp
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
from system_matrices import get_transformation_matrix

def regenerate_eigenvalues_fast(mat_file_path, output_path=None, n_structs=1):
    """Regenerate for first n_structs only."""
    mat_file_path = Path(mat_file_path)
    if output_path is None:
        output_path = mat_file_path.parent / f"{mat_file_path.stem}_regenerated_fast.mat"
    else:
        output_path = Path(output_path)
    
    print("=" * 80)
    print(f"Regenerating EIGENVALUE_DATA (FAST: first {n_structs} structures only)")
    print("=" * 80)
    
    with h5py.File(str(mat_file_path), 'r') as f:
        eigvec_data = np.array(f['EIGENVECTOR_DATA'])
        designs = np.array(f['designs'])
        wavevectors = np.array(f['WAVEVECTOR_DATA'])
        const_dict = {key: np.array(f['const'][key]) for key in f['const']}
        
        n_designs = min(n_structs, designs.shape[0])
        n_bands = eigvec_data.shape[1]
        
        if wavevectors.ndim == 3:
            n_wavevectors = wavevectors.shape[2]
        else:
            n_wavevectors = wavevectors.shape[0] if wavevectors.ndim == 2 else wavevectors.shape[1]
        
        print(f"   Processing {n_designs} structures, {n_bands} bands, {n_wavevectors} wavevectors")
        
        if eigvec_data.dtype.names and 'real' in eigvec_data.dtype.names:
            eigvec_data = eigvec_data['real'] + 1j * eigvec_data['imag']
        
        EIGENVALUE_DATA_new = np.zeros((n_designs, n_bands, n_wavevectors), dtype=np.float64)
        
        const_base = {
            'N_pix': int(np.array(const_dict['N_pix']).item()),
            'N_ele': int(np.array(const_dict['N_ele']).item()),
            'a': float(np.array(const_dict['a']).item()),
            'E_min': float(np.array(const_dict['E_min']).item()),
            'E_max': float(np.array(const_dict['E_max']).item()),
            'rho_min': float(np.array(const_dict['rho_min']).item()),
            'rho_max': float(np.array(const_dict['rho_max']).item()),
            'poisson_min': float(np.array(const_dict.get('nu_min', const_dict.get('poisson_min', 0.3))).item()),
            'poisson_max': float(np.array(const_dict.get('nu_max', const_dict.get('poisson_max', 0.3))).item()),
            't': float(np.array(const_dict.get('t', 1.0)).item()) if 't' in const_dict else 1.0,
            'design_scale': 'linear',
            'isUseImprovement': True,
            'isUseSecondImprovement': False
        }
        
        for struct_idx in range(n_designs):
            print(f"\nProcessing structure {struct_idx + 1}/{n_designs}...")
            start_time = time.time()
            
            design = designs[struct_idx, :, :, :].transpose(1, 2, 0)
            const = const_base.copy()
            const['design'] = design
            
            K, M = get_system_matrices_VEC(const)
            K_sparse = sp.csr_matrix(K) if not sp.issparse(K) else K
            M_sparse = sp.csr_matrix(M) if not sp.issparse(M) else M
            
            if wavevectors.ndim == 3:
                wv_struct = wavevectors[struct_idx, :, :].T
            else:
                wv_struct = wavevectors
            
            for wv_idx in range(n_wavevectors):
                wv = wv_struct[wv_idx, :].astype(np.float32)
                T = get_transformation_matrix(wv, const)
                T_sparse = sp.csr_matrix(T) if not sp.issparse(T) else T
                
                Kr = T_sparse.conj().T @ K_sparse @ T_sparse
                Mr = T_sparse.conj().T @ M_sparse @ T_sparse
                
                for band_idx in range(n_bands):
                    eigvec = eigvec_data[struct_idx, band_idx, wv_idx, :].astype(np.complex128)
                    Kr_eigvec = Kr @ eigvec
                    Mr_eigvec = Mr @ eigvec
                    
                    if sp.issparse(Kr_eigvec):
                        Kr_eigvec = Kr_eigvec.toarray().flatten()
                    if sp.issparse(Mr_eigvec):
                        Mr_eigvec = Mr_eigvec.toarray().flatten()
                    
                    numerator = np.dot(eigvec.conj(), Kr_eigvec)
                    denominator = np.dot(eigvec.conj(), Mr_eigvec)
                    
                    if np.abs(denominator) < 1e-15:
                        eigval = 0.0
                    else:
                        eigval = numerator / denominator
                    
                    freq = np.sqrt(np.real(eigval)) / (2 * np.pi)
                    EIGENVALUE_DATA_new[struct_idx, band_idx, wv_idx] = freq
                
                if (wv_idx + 1) % 20 == 0:
                    print(f"  Processed {wv_idx + 1}/{n_wavevectors} wavevectors...")
            
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.2f} seconds")
        
        print(f"\nSaving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(str(output_path), 'w') as f_out:
            with h5py.File(str(mat_file_path), 'r') as f_in:
                for key in f_in.keys():
                    if key != 'EIGENVALUE_DATA':
                        f_in.copy(key, f_out)
            
            f_out.create_dataset('EIGENVALUE_DATA', data=EIGENVALUE_DATA_new, dtype=np.float64)
        
        print(f"Done! Saved to {output_path}")
        return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n-structs', type=int, default=1)
    args = parser.parse_args()
    
    regenerate_eigenvalues_fast(args.input_file, args.output, args.n_structs)

