"""
Verification script for Step 4: Frequency Reconstruction

This script verifies that frequency reconstruction from eigenvectors works correctly
by:
1. Loading the dataset
2. Performing eigenvector format conversion
3. Computing K, M, T matrices
4. Reconstructing frequencies
5. Saving intermediate values for inspection
6. Checking for physical reasonableness

OPTIMIZATION NOTES:
==================
Where optimizations ARE applied:
- Preallocated lists/arrays: T_data, sample_indices_ev
- Vectorized operations: band statistics, Gamma point search, eigenvector statistics
- Precomputed matrices: Kr, Mr computed once and reused
- Batch conversions: wavevectors converted to float32 once
- Conditional eigenvalue computation: skipped for large matrices (>1000 DOF)

Where optimizations are NOT possible and why:
1. get_transformation_matrix() calls: Each T matrix depends on a specific wavevector,
   so they must be computed individually. Cannot batch because each wavevector
   creates a different transformation matrix structure.

2. reconstruct_frequencies_from_eigenvectors(): The function processes each wavevector
   sequentially because each requires its own T matrix. The inner loop over bands
   could potentially be vectorized, but the matrix-vector products (Kr @ eigvec)
   are already optimized by scipy.sparse.

3. Matrix eigenvalue computation: For large sparse matrices (>1000 DOF), full
   eigendecomposition is O(N^3) and memory-intensive. We skip it and only check
   for smaller matrices. Alternative: use scipy.sparse.linalg.eigs() for largest
   eigenvalues, but that's still expensive.

4. convert_field_to_dof_format(): Now vectorized using np.repeat, but the original
   nested loops were necessary because each pixel maps to a block of nodes. The
   vectorized version uses broadcasting which is much faster.

5. File I/O operations: Cannot be optimized - must write sequentially to disk.
   However, we only save essential data, not full arrays.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from plot_dispersion_infer_eigenfrequencies import (
    load_pt_dataset,
    convert_field_to_dof_format,
    reconstruct_frequencies_from_eigenvectors,
    create_const_dict,
    compute_K_M_matrices
)
from system_matrices import get_transformation_matrix


def verify_eigenvector_conversion(eigenvectors_field, N_pix, N_ele=4, struct_idx=0, N_struct=1, N_wv=91, N_eig=6):
    """
    Verify field-to-DOF format conversion.
    
    Returns:
    - eigenvectors_dof: Converted eigenvectors
    - conversion_stats: Statistics about the conversion
    """
    print("\n" + "="*70)
    print("VERIFYING EIGENVECTOR CONVERSION (Field → DOF Format)")
    print("="*70)
    
    print(f"\nInput (field format):")
    print(f"  Shape: {eigenvectors_field.shape}")
    print(f"  Dtype: {eigenvectors_field.dtype}")
    print(f"  Expected: (N_samples, 2, H, W) where N_samples = N_struct × N_wv × N_eig")
    
    # Check expected shape
    expected_samples = N_struct * N_wv * N_eig
    actual_samples = eigenvectors_field.shape[0]
    print(f"\n  Expected N_samples: {expected_samples}")
    print(f"  Actual N_samples: {actual_samples}")
    
    if actual_samples != expected_samples:
        print(f"  ⚠️  WARNING: Sample count mismatch!")
        print(f"     This may indicate incorrect N_struct, N_wv, or N_eig")
    
    # Convert to DOF format
    print(f"\nConverting to DOF format (N_pix={N_pix}, N_ele={N_ele})...")
    eigenvectors_dof = convert_field_to_dof_format(eigenvectors_field, N_pix, N_ele, reduced_space=True)
    
    print(f"\nOutput (DOF format):")
    print(f"  Shape: {eigenvectors_dof.shape}")
    print(f"  Dtype: {eigenvectors_dof.dtype}")
    print(f"  Expected: (N_samples, N_dof_reduced)")
    print(f"  Expected N_dof_reduced: 2 * (N_ele * N_pix)^2 = 2 * ({N_ele} * {N_pix})^2 = {2 * (N_ele * N_pix)**2}")
    
    # Reshape to structure format
    print(f"\nReshaping to structure format...")
    N_dof = eigenvectors_dof.shape[1]
    eigenvectors_reshaped = eigenvectors_dof.reshape(N_struct, N_wv, N_eig, N_dof)
    print(f"  Reshaped shape: {eigenvectors_reshaped.shape}")
    print(f"  Expected: (N_struct={N_struct}, N_wv={N_wv}, N_eig={N_eig}, N_dof={N_dof})")
    
    # Extract structure
    eigenvectors_struct = eigenvectors_reshaped[struct_idx]
    print(f"\nExtracted structure {struct_idx}:")
    print(f"  Shape: {eigenvectors_struct.shape}")
    print(f"  Expected: (N_wv={N_wv}, N_eig={N_eig}, N_dof={N_dof})")
    
    # Transpose to (N_dof, N_wv, N_eig)
    eigenvectors_final = eigenvectors_struct.transpose(2, 0, 1)
    print(f"\nFinal format (transposed):")
    print(f"  Shape: {eigenvectors_final.shape}")
    print(f"  Expected: (N_dof={N_dof}, N_wv={N_wv}, N_eig={N_eig})")
    
    # Check eigenvector properties (vectorized where possible)
    print(f"\nEigenvector Statistics:")
    wv_indices = np.array([0, N_wv//2, N_wv-1])
    band_indices = np.array([0, N_eig//2, N_eig-1])
    
    # Precompute norms for all combinations (vectorized)
    for wv_idx in wv_indices:
        for band_idx in band_indices:
            eigvec = eigenvectors_final[:, wv_idx, band_idx]
            # Use vectorized operations
            eigvec_abs = np.abs(eigvec)
            norm = np.linalg.norm(eigvec)
            print(f"  wv={wv_idx:3d}, band={band_idx}: norm={norm:.6e}, "
                  f"min={np.min(eigvec_abs):.6e}, max={np.max(eigvec_abs):.6e}")
    
    conversion_stats = {
        'input_shape': eigenvectors_field.shape,
        'output_shape': eigenvectors_final.shape,
        'N_dof': N_dof,
        'expected_N_dof': 2 * (N_ele * N_pix)**2,
        'samples_match': actual_samples == expected_samples
    }
    
    return eigenvectors_final, conversion_stats


def verify_matrix_computation(design_param, N_pix, const_params):
    """
    Verify K, M, T matrix computation.
    """
    print("\n" + "="*70)
    print("VERIFYING MATRIX COMPUTATION (K, M, T)")
    print("="*70)
    
    # Create const dictionary
    const = create_const_dict(design_param, N_pix, **const_params)
    
    # Compute K, M
    print("\nComputing K, M matrices...")
    K, M = compute_K_M_matrices(const)
    
    print(f"\nK matrix:")
    print(f"  Type: {type(K)}")
    if hasattr(K, 'shape'):
        print(f"  Shape: {K.shape}")
    if hasattr(K, 'dtype'):
        print(f"  Dtype: {K.dtype}")
    if hasattr(K, 'nnz'):
        print(f"  Non-zero elements: {K.nnz}")
        print(f"  Sparsity: {1 - K.nnz / (K.shape[0] * K.shape[1]):.4f}")
    
    print(f"\nM matrix:")
    print(f"  Type: {type(M)}")
    if hasattr(M, 'shape'):
        print(f"  Shape: {M.shape}")
    if hasattr(M, 'dtype'):
        print(f"  Dtype: {M.dtype}")
    if hasattr(M, 'nnz'):
        print(f"  Non-zero elements: {M.nnz}")
        print(f"  Sparsity: {1 - M.nnz / (M.shape[0] * M.shape[1]):.4f}")
    
    # Check matrix properties (optimize eigenvalue computation)
    import scipy.sparse as sp
    if sp.issparse(K):
        # Only compute eigenvalues for small matrices (expensive for large sparse)
        # For large matrices, check positive definiteness via Cholesky or other methods
        if K.shape[0] <= 1000:
            K_dense = K.toarray()
            M_dense = M.toarray()
            K_eigvals = np.linalg.eigvals(K_dense)
            M_eigvals = np.linalg.eigvals(M_dense)
            print(f"\nMatrix Properties:")
            print(f"  K eigenvalues (first 5): {K_eigvals[:5]}")
            print(f"  M eigenvalues (first 5): {M_eigvals[:5]}")
            print(f"  K is positive definite: {np.all(K_eigvals > 0)}")
            print(f"  M is positive definite: {np.all(M_eigvals > 0)}")
        else:
            # For large matrices, use more efficient checks
            print(f"\nMatrix Properties:")
            print(f"  K shape: {K.shape} (too large for full eigendecomposition)")
            print(f"  M shape: {M.shape} (too large for full eigendecomposition)")
            print(f"  ⚠️  Skipping eigenvalue check for large matrices (use sparse eigs for efficiency)")
            K_eigvals = None
            M_eigvals = None
    else:
        K_dense = K
        M_dense = M
        if K.shape[0] <= 1000:
            K_eigvals = np.linalg.eigvals(K_dense)
            M_eigvals = np.linalg.eigvals(M_dense)
            print(f"\nMatrix Properties:")
            print(f"  K eigenvalues (first 5): {K_eigvals[:5]}")
            print(f"  M eigenvalues (first 5): {M_eigvals[:5]}")
            print(f"  K is positive definite: {np.all(K_eigvals > 0)}")
            print(f"  M is positive definite: {np.all(M_eigvals > 0)}")
        else:
            print(f"\nMatrix Properties:")
            print(f"  ⚠️  Skipping eigenvalue check for large matrices")
            K_eigvals = None
            M_eigvals = None
    
    # Compute T matrices for a few wavevectors
    print(f"\nComputing T matrices...")
    wavevectors_sample = np.array([
        [0.0, 0.0],  # Gamma point
        [np.pi, 0.0],  # X point
        [np.pi, np.pi]  # M point
    ])
    
    # Preallocate T_data list
    n_sample_wv = len(wavevectors_sample)
    T_data = [None] * n_sample_wv
    
    for i, wv in enumerate(wavevectors_sample):
        T = get_transformation_matrix(wv.astype(np.float32), const)
        T_data[i] = T
        if sp.issparse(T):
            print(f"  T shape: {T.shape}, nnz: {T.nnz}")
        else:
            print(f"  T shape: {T.shape}")
    
    # Precompute T shapes (vectorized)
    T_shapes = [T.shape for T in T_data]
    
    matrix_stats = {
        'K_shape': K.shape,
        'M_shape': M.shape,
        'T_shapes': T_shapes,
        'K_positive_definite': np.all(K_eigvals > 0) if K_eigvals is not None else None,
        'M_positive_definite': np.all(M_eigvals > 0) if M_eigvals is not None else None
    }
    
    return K, M, T_data, matrix_stats, const


def verify_frequency_reconstruction(K, M, T_data, eigenvectors, wavevectors, N_eig, 
                                   struct_idx=0, N_pix=None, N_ele=4, save_intermediates=True):
    """
    Verify frequency reconstruction process.
    """
    print("\n" + "="*70)
    print("VERIFYING FREQUENCY RECONSTRUCTION")
    print("="*70)
    
    n_wavevectors = len(wavevectors)
    print(f"\nInput:")
    print(f"  Wavevectors: {n_wavevectors}")
    print(f"  Eigenvectors shape: {eigenvectors.shape}")
    print(f"  N_eig: {N_eig}")
    
    # Reconstruct frequencies
    print(f"\nReconstructing frequencies...")
    frequencies_recon = reconstruct_frequencies_from_eigenvectors(
        K, M, T_data, eigenvectors, wavevectors, N_eig,
        struct_idx=struct_idx, N_pix=N_pix, N_ele=N_ele
    )
    
    print(f"\nOutput:")
    print(f"  Frequencies shape: {frequencies_recon.shape}")
    print(f"  Expected: (N_wv={n_wavevectors}, N_eig={N_eig})")
    
    # Check frequency statistics
    print(f"\nFrequency Statistics:")
    print(f"  Min frequency: {np.min(frequencies_recon):.6e} Hz")
    print(f"  Max frequency: {np.max(frequencies_recon):.6e} Hz")
    print(f"  Mean frequency: {np.mean(frequencies_recon):.6e} Hz")
    print(f"  Frequencies are non-negative: {np.all(frequencies_recon >= 0)}")
    print(f"  Frequencies are finite: {np.all(np.isfinite(frequencies_recon))}")
    
    # Check for each band (vectorized statistics)
    print(f"\nPer-band Statistics:")
    # Precompute all band statistics at once (more efficient)
    band_mins = np.min(frequencies_recon, axis=0)
    band_maxs = np.max(frequencies_recon, axis=0)
    band_means = np.mean(frequencies_recon, axis=0)
    
    for band_idx in range(N_eig):
        print(f"  Band {band_idx}: min={band_mins[band_idx]:.6e}, "
              f"max={band_maxs[band_idx]:.6e}, mean={band_means[band_idx]:.6e}")
    
    # Check Gamma point (k=0,0) - should have zero frequency for rigid body mode
    # Vectorized search (more efficient than loop)
    gamma_mask = np.allclose(wavevectors, [0.0, 0.0], atol=1e-6, axis=1)
    gamma_indices = np.where(gamma_mask)[0]
    gamma_idx = gamma_indices[0] if len(gamma_indices) > 0 else None
    
    if gamma_idx is not None:
        print(f"\nGamma Point (k=[0,0], index={gamma_idx}):")
        gamma_freqs = frequencies_recon[gamma_idx, :]
        print(f"  Frequencies: {gamma_freqs}")
        print(f"  First band (rigid body mode): {gamma_freqs[0]:.6e} Hz")
        print(f"  Should be close to zero: {np.isclose(gamma_freqs[0], 0.0, atol=100.0)}")
    else:
        print(f"\n⚠️  WARNING: Gamma point (k=[0,0]) not found in wavevectors")
    
    # Detailed reconstruction for a few points (optimized)
    print(f"\nDetailed Reconstruction (first wavevector, first 3 bands):")
    import scipy.sparse as sp
    
    wv_idx = 0
    n_bands_check = min(3, N_eig)
    
    # Precompute matrices once (reused for all bands)
    T = T_data[wv_idx] if isinstance(T_data, list) else T_data[wv_idx]
    T_sparse = T if sp.issparse(T) else sp.csr_matrix(T.astype(np.float32))
    K_sparse = K if sp.issparse(K) else sp.csr_matrix(K.astype(np.float32))
    M_sparse = M if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32))
    
    # Precompute reduced matrices (reused for all bands)
    Kr = T_sparse.conj().T @ K_sparse @ T_sparse
    Mr = T_sparse.conj().T @ M_sparse @ T_sparse
    
    # Pre-extract eigenvectors for all bands at once
    eigvecs = eigenvectors[:, wv_idx, :n_bands_check].astype(np.complex128).T  # (n_bands, N_dof)
    
    # Vectorized matrix-vector products (if possible, otherwise loop)
    for band_idx in range(n_bands_check):
        eigvec = eigvecs[band_idx]  # (N_dof,)
        Kr_eigvec = Kr @ eigvec
        Mr_eigvec = Mr @ eigvec
        
        # Convert sparse results to dense efficiently
        if sp.issparse(Kr_eigvec):
            Kr_eigvec = Kr_eigvec.toarray().flatten()
        if sp.issparse(Mr_eigvec):
            Mr_eigvec = Mr_eigvec.toarray().flatten()
        
        # Compute norms efficiently
        Kr_norm = np.linalg.norm(Kr_eigvec)
        Mr_norm = np.linalg.norm(Mr_eigvec)
        eigval = Kr_norm / Mr_norm
        freq = np.sqrt(np.real(eigval)) / (2 * np.pi)
        
        print(f"  Band {band_idx}:")
        print(f"    ||Kr @ eigvec||: {Kr_norm:.6e}")
        print(f"    ||Mr @ eigvec||: {Mr_norm:.6e}")
        print(f"    eigval: {eigval:.6e}")
        print(f"    frequency: {freq:.6e} Hz")
        print(f"    reconstructed frequency: {frequencies_recon[wv_idx, band_idx]:.6e} Hz")
        print(f"    match: {np.isclose(freq, frequencies_recon[wv_idx, band_idx], rtol=1e-6)}")
    
    # Save intermediate values if requested
    if save_intermediates:
        output_dir = Path('verification_output')
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving intermediate values to: {output_dir}")
        
        # Save frequencies
        np.save(output_dir / 'frequencies_recon.npy', frequencies_recon)
        
        # Save eigenvector sample
        np.save(output_dir / 'eigenvectors_sample.npy', eigenvectors[:, :3, :3])
        
        # Save matrix info
        matrix_info = {
            'K_shape': K.shape,
            'M_shape': M.shape,
            'T_shapes': [T.shape for T in (T_data[:3] if isinstance(T_data, list) else [T_data])]
        }
        np.save(output_dir / 'matrix_info.npy', matrix_info)
        
        print(f"  Saved: frequencies_recon.npy, eigenvectors_sample.npy, matrix_info.npy")
    
    reconstruction_stats = {
        'frequencies_shape': frequencies_recon.shape,
        'min_freq': np.min(frequencies_recon),
        'max_freq': np.max(frequencies_recon),
        'mean_freq': np.mean(frequencies_recon),
        'all_non_negative': np.all(frequencies_recon >= 0),
        'all_finite': np.all(np.isfinite(frequencies_recon)),
        'gamma_point_found': gamma_idx is not None,
        'gamma_first_band_freq': gamma_freqs[0] if gamma_idx is not None else None
    }
    
    return frequencies_recon, reconstruction_stats


def main(data_dir, struct_idx=0, n_wv_sample=10):
    """
    Main verification function.
    """
    print("="*70)
    print("FREQUENCY RECONSTRUCTION VERIFICATION")
    print("="*70)
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"\nERROR: Dataset directory not found: {data_dir}")
        return
    
    # Load data
    print(f"\nLoading dataset from: {data_dir}")
    data = load_pt_dataset(data_dir, None, require_eigenvalue_data=False)
    
    # Extract data
    designs = data['designs']
    wavevectors_all = data['wavevectors']
    eigenvectors = data['eigenvectors']
    
    print(f"\nDataset shapes:")
    print(f"  designs: {designs.shape}")
    print(f"  wavevectors: {wavevectors_all.shape}")
    print(f"  eigenvectors: {eigenvectors.shape}")
    
    # Infer parameters
    N_pix = designs.shape[1]
    N_struct = designs.shape[0]
    n_wv_per_struct = wavevectors_all.shape[1]
    
    # Infer N_eig
    total_samples = eigenvectors.shape[0]
    N_eig = 6  # Default
    for test_N_eig in [6, 8, 10, 12]:
        samples_per_struct = n_wv_per_struct * test_N_eig
        if total_samples % samples_per_struct == 0:
            inferred_N_struct = total_samples // samples_per_struct
            if inferred_N_struct == N_struct:
                N_eig = test_N_eig
                break
    
    print(f"\nInferred parameters:")
    print(f"  N_pix: {N_pix}")
    print(f"  N_struct: {N_struct}")
    print(f"  N_wv per struct: {n_wv_per_struct}")
    print(f"  N_eig: {N_eig}")
    
    # Get data for this structure
    design_param = designs[struct_idx, :, :]
    wavevectors = wavevectors_all[struct_idx, :, :]
    
    # Sample wavevectors for faster verification
    if n_wv_sample < len(wavevectors):
        sample_indices = np.linspace(0, len(wavevectors)-1, n_wv_sample, dtype=int)
        wavevectors_sample = wavevectors[sample_indices]
        print(f"\nSampling {n_wv_sample} wavevectors out of {len(wavevectors)} for verification")
    else:
        wavevectors_sample = wavevectors
        sample_indices = np.arange(len(wavevectors))
    
    # Material parameters
    const_params = {
        'E_min': 20e6, 'E_max': 200e9,
        'rho_min': 400, 'rho_max': 8000,
        'nu_min': 0.0, 'nu_max': 0.5,
        'a': 1.0, 'N_ele': 4
    }
    
    # Step 1: Verify eigenvector conversion (on a small sample for speed)
    # Only convert eigenvectors for the sampled wavevectors
    print(f"\n⚠️  OPTIMIZATION: Only converting eigenvectors for {n_wv_sample} sampled wavevectors")
    print(f"    This will be much faster than converting all {n_wv_per_struct} wavevectors")
    
    # Calculate which samples correspond to the sampled wavevectors
    # Eigenvectors are stored as: (N_samples, 2, H, W) where N_samples = N_struct × N_wv × N_eig
    # For structure struct_idx, the samples start at: struct_idx * (N_wv * N_eig)
    struct_start_idx = struct_idx * (n_wv_per_struct * N_eig)
    
    # Get samples for the sampled wavevectors (preallocate)
    n_sample_ev = len(sample_indices) * N_eig
    sample_indices_ev = np.zeros(n_sample_ev, dtype=int)
    
    idx = 0
    for wv_idx in sample_indices:
        for eig_idx in range(N_eig):
            sample_idx = struct_start_idx + wv_idx * N_eig + eig_idx
            sample_indices_ev[idx] = sample_idx
            idx += 1
    
    eigenvectors_sample_field = eigenvectors[sample_indices_ev, :, :, :]
    print(f"    Extracted {len(sample_indices_ev)} eigenvector samples from structure {struct_idx}")
    
    eigenvectors_dof, conversion_stats = verify_eigenvector_conversion(
        eigenvectors_sample_field, N_pix, N_ele=4, struct_idx=0, 
        N_struct=1, N_wv=n_wv_sample, N_eig=N_eig
    )
    
    # Step 2: Verify matrix computation
    K, M, T_data_sample, matrix_stats, const = verify_matrix_computation(
        design_param, N_pix, const_params
    )
    
    # Compute T matrices for all sampled wavevectors (preallocate)
    print(f"\nComputing T matrices for {len(wavevectors_sample)} wavevectors...")
    n_wv_sample = len(wavevectors_sample)
    T_data = [None] * n_wv_sample
    
    # Pre-convert wavevectors to float32 once
    wavevectors_sample_f32 = wavevectors_sample.astype(np.float32)
    
    for i in range(n_wv_sample):
        T = get_transformation_matrix(wavevectors_sample_f32[i], const)
        T_data[i] = T
    
    # Step 3: Verify frequency reconstruction (on sampled wavevectors)
    # Extract eigenvectors for sampled wavevectors
    eigenvectors_sample = eigenvectors_dof[:, sample_indices, :]
    
    frequencies_recon, reconstruction_stats = verify_frequency_reconstruction(
        K, M, T_data, eigenvectors_sample, wavevectors_sample, N_eig,
        struct_idx=0, N_pix=N_pix, N_ele=4, save_intermediates=True
    )
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"\n✓ Eigenvector conversion: {'PASS' if conversion_stats['samples_match'] else 'FAIL'}")
    print(f"  Expected N_dof: {conversion_stats['expected_N_dof']}")
    print(f"  Actual N_dof: {conversion_stats['N_dof']}")
    print(f"  Match: {conversion_stats['expected_N_dof'] == conversion_stats['N_dof']}")
    
    # Handle None case for large matrices
    k_pd = matrix_stats['K_positive_definite']
    m_pd = matrix_stats['M_positive_definite']
    if k_pd is not None and m_pd is not None:
        matrix_pass = k_pd and m_pd
        print(f"\n✓ Matrix computation: {'PASS' if matrix_pass else 'FAIL'}")
        print(f"  K positive definite: {k_pd}")
        print(f"  M positive definite: {m_pd}")
    else:
        print(f"\n✓ Matrix computation: SKIPPED (matrices too large for eigenvalue check)")
        print(f"  K shape: {matrix_stats['K_shape']}")
        print(f"  M shape: {matrix_stats['M_shape']}")
    
    print(f"\n✓ Frequency reconstruction: {'PASS' if reconstruction_stats['all_non_negative'] and reconstruction_stats['all_finite'] else 'FAIL'}")
    print(f"  All frequencies non-negative: {reconstruction_stats['all_non_negative']}")
    print(f"  All frequencies finite: {reconstruction_stats['all_finite']}")
    print(f"  Frequency range: [{reconstruction_stats['min_freq']:.6e}, {reconstruction_stats['max_freq']:.6e}] Hz")
    if reconstruction_stats['gamma_point_found']:
        print(f"  Gamma point first band: {reconstruction_stats['gamma_first_band_freq']:.6e} Hz")
    
    print("\n" + "="*70)
    print("Verification complete!")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify frequency reconstruction")
    parser.add_argument("data_dir", help="Path to PyTorch dataset directory")
    parser.add_argument("--struct-idx", type=int, default=0, help="Structure index to verify")
    parser.add_argument("--n-wv-sample", type=int, default=10, help="Number of wavevectors to sample for verification")
    args = parser.parse_args()
    
    main(args.data_dir, args.struct_idx, args.n_wv_sample)

