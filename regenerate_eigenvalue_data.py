#!/usr/bin/env python3
"""
Regenerate EIGENVALUE_DATA from K, M, T matrices using Rayleigh quotient.

This script loads a .mat file, regenerates all eigenvalues using the correct
Rayleigh quotient formula, and saves the updated .mat file.
"""

import numpy as np
import h5py
import scipy.sparse as sp
from pathlib import Path
import sys
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))
from system_matrices_vec import get_system_matrices_VEC
from system_matrices import get_transformation_matrix

def regenerate_eigenvalues(mat_file_path, output_path=None):
    """
    Regenerate EIGENVALUE_DATA using Rayleigh quotient.
    
    Parameters
    ----------
    mat_file_path : str or Path
        Path to input .mat file
    output_path : str or Path, optional
        Path to output .mat file (default: overwrites input with _regenerated suffix)
    """
    mat_file_path = Path(mat_file_path)
    if output_path is None:
        output_path = mat_file_path.parent / f"{mat_file_path.stem}_regenerated.mat"
    else:
        output_path = Path(output_path)
    
    print("=" * 80)
    print("Regenerating EIGENVALUE_DATA using Rayleigh Quotient")
    print("=" * 80)
    print(f"Input:  {mat_file_path}")
    print(f"Output: {output_path}")
    
    # Load data
    print(f"\n1. Loading data...")
    with h5py.File(str(mat_file_path), 'r') as f:
        # Load all data
        eigvec_data = np.array(f['EIGENVECTOR_DATA'])
        designs = np.array(f['designs'])
        wavevectors = np.array(f['WAVEVECTOR_DATA'])
        const_dict = {key: np.array(f['const'][key]) for key in f['const']}
        
        # Get dimensions
        n_designs = designs.shape[0]
        n_bands = eigvec_data.shape[1]
        
        # Handle wavevector shape
        if wavevectors.ndim == 3:
            n_wavevectors = wavevectors.shape[2]
        else:
            n_wavevectors = wavevectors.shape[0] if wavevectors.ndim == 2 else wavevectors.shape[1]
        
        print(f"   n_designs: {n_designs}")
        print(f"   n_bands: {n_bands}")
        print(f"   n_wavevectors: {n_wavevectors}")
        
        # Convert eigenvector data
        if eigvec_data.dtype.names and 'real' in eigvec_data.dtype.names:
            eigvec_data = eigvec_data['real'] + 1j * eigvec_data['imag']
        
        # Initialize new eigenvalue data
        EIGENVALUE_DATA_new = np.zeros((n_designs, n_bands, n_wavevectors), dtype=np.float64)
        
        # Extract const parameters
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
        
        # Process each structure
        for struct_idx in range(n_designs):
            print(f"\n2. Processing structure {struct_idx + 1}/{n_designs}...")
            start_time = time.time()
            
            # Get design
            design = designs[struct_idx, :, :, :].transpose(1, 2, 0)  # (H, W, 3)
            const = const_base.copy()
            const['design'] = design
            
            # Compute K and M matrices (once per structure)
            print(f"   Computing K and M matrices...")
            K, M = get_system_matrices_VEC(const)
            K_sparse = sp.csr_matrix(K) if not sp.issparse(K) else K
            M_sparse = sp.csr_matrix(M) if not sp.issparse(M) else M
            
            # Get wavevectors for this structure
            if wavevectors.ndim == 3:
                wv_struct = wavevectors[struct_idx, :, :].T  # (n_wavevectors, 2)
            else:
                wv_struct = wavevectors  # (n_wavevectors, 2)
            
            # Process each wavevector
            for wv_idx in range(n_wavevectors):
                wv = wv_struct[wv_idx, :].astype(np.float32)
                
                # Compute T matrix
                T = get_transformation_matrix(wv, const)
                T_sparse = sp.csr_matrix(T) if not sp.issparse(T) else T
                
                # Compute reduced matrices
                Kr = T_sparse.conj().T @ K_sparse @ T_sparse
                Mr = T_sparse.conj().T @ M_sparse @ T_sparse
                
                # Process each band
                for band_idx in range(n_bands):
                    # Get eigenvector
                    eigvec = eigvec_data[struct_idx, band_idx, wv_idx, :].astype(np.complex128)
                    
                    # Compute eigenvalue using Rayleigh quotient
                    # eigval = (eigvec^H * Kr * eigvec) / (eigvec^H * Mr * eigvec)
                    Kr_eigvec = Kr @ eigvec
                    Mr_eigvec = Mr @ eigvec
                    
                    if sp.issparse(Kr_eigvec):
                        Kr_eigvec = Kr_eigvec.toarray().flatten()
                    if sp.issparse(Mr_eigvec):
                        Mr_eigvec = Mr_eigvec.toarray().flatten()
                    
                    numerator = np.dot(eigvec.conj(), Kr_eigvec)
                    denominator = np.dot(eigvec.conj(), Mr_eigvec)
                    
                    if np.abs(denominator) < 1e-15:
                        print(f"      WARNING: Small denominator for struct {struct_idx}, band {band_idx}, wv {wv_idx}")
                        eigval = 0.0
                    else:
                        eigval = numerator / denominator
                    
                    # Convert to frequency
                    freq = np.sqrt(np.real(eigval)) / (2 * np.pi)
                    EIGENVALUE_DATA_new[struct_idx, band_idx, wv_idx] = freq
                
                if (wv_idx + 1) % 10 == 0:
                    print(f"      Processed {wv_idx + 1}/{n_wavevectors} wavevectors...")
            
            elapsed = time.time() - start_time
            print(f"   Completed in {elapsed:.2f} seconds")
        
        # Save new .mat file
        print(f"\n3. Saving regenerated data to {output_path}...")
        with h5py.File(str(output_path), 'w') as f_out:
            # Copy all original data
            with h5py.File(str(mat_file_path), 'r') as f_in:
                for key in f_in.keys():
                    if key != 'EIGENVALUE_DATA':
                        f_in.copy(key, f_out)
            
            # Write new eigenvalue data
            f_out.create_dataset('EIGENVALUE_DATA', data=EIGENVALUE_DATA_new, dtype=np.float64)
        
        print(f"   Done! Regenerated eigenvalues saved to {output_path}")
        
        return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Regenerate EIGENVALUE_DATA using Rayleigh quotient')
    parser.add_argument('input_file', type=str, help='Input .mat file path')
    parser.add_argument('--output', type=str, default=None, help='Output .mat file path (default: input_regenerated.mat)')
    args = parser.parse_args()
    
    regenerate_eigenvalues(args.input_file, args.output)

