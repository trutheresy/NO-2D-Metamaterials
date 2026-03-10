"""
Python translation of generate_dispersion_dataset_Han_Alex.m

This script generates dispersion datasets using the new Python dispersion library.
It creates multiple metamaterial designs and computes their dispersion relations,
eigenvectors, and system matrices.
"""

import argparse
import hashlib
import numpy as np
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pickle
import scipy.io as sio
import torch

# Add the new Python dispersion library path
dispersion_library_path = r'D:\Research\NO-2D-Metamaterials\2d-dispersion-py'
sys.path.insert(0, dispersion_library_path)  # Use insert(0, ...) to prioritize this path

    # Import functions from the new Python library
try:
    # Import the CORRECT function - dispersion_with_matrix_save_opt (not dispersion!)
    from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt
    from get_design import get_design
    from get_design2 import get_design2
    from wavevectors import get_IBZ_wavevectors
    from design_parameters import DesignParameters
    from design_conversion import convert_design, design_to_explicit, apply_steel_rubber_paradigm
    from kernels import generate_correlated_design
    from system_matrices import get_transformation_matrix
    from utils import validate_constants, check_contour_analysis
    
    print("SUCCESS: Successfully imported functions from the new Python dispersion library")
    print("SUCCESS: Using dispersion_with_matrix_save_opt (the correct function!)")
except ImportError as e:
    print(f"ERROR: Error importing from dispersion library: {e}")
    print("Make sure the 2d-dispersion-py directory is in the correct location")
    print(f"Current library path: {dispersion_library_path}")
    sys.exit(1)


_T_CACHE_BY_PATH = {}


def _wavevector_key(wv):
    return (round(float(wv[0]), 12), round(float(wv[1]), 12))


def _build_t_signature(const, wavevectors):
    wv64 = np.asarray(wavevectors, dtype=np.float64)
    digest = hashlib.sha1(wv64.tobytes()).hexdigest()
    return (
        f"Nele{int(const['N_ele'])}_"
        f"Npix{int(const['N_pix'])}_"
        f"a{float(const['a']):.12g}_"
        f"nwv{int(wv64.shape[0])}_"
        f"{digest}"
    )


def _load_t_cache_file(cache_path):
    if not os.path.exists(cache_path):
        return {"version": 1, "entries": {}}
    try:
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "entries" in obj:
            return obj
    except Exception as e:
        print(f"WARNING: Failed reading T cache at {cache_path}: {e}. Rebuilding.")
    return {"version": 1, "entries": {}}


def _save_t_cache_file(cache_path, cache_obj):
    with open(cache_path, "wb") as f:
        pickle.dump(cache_obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _get_or_create_precomputed_t_entry(const, wavevectors, cache_path):
    """
    Load or build precomputed T matrices for this lattice/wavevector signature.
    """
    global _T_CACHE_BY_PATH
    cache_path = os.path.abspath(cache_path)
    cache_obj = _T_CACHE_BY_PATH.get(cache_path)
    if cache_obj is None:
        cache_obj = _load_t_cache_file(cache_path)
        _T_CACHE_BY_PATH[cache_path] = cache_obj

    signature = _build_t_signature(const, wavevectors)
    entry = cache_obj["entries"].get(signature)
    if entry is None:
        wv_arr = np.asarray(wavevectors, dtype=np.float32)
        t_data = [get_transformation_matrix(wv, const) for wv in wv_arr]
        entry = {
            "signature": signature,
            "wavevectors": wv_arr.astype(np.float32),
            "T_data": t_data,
        }
        cache_obj["entries"][signature] = entry
        _save_t_cache_file(cache_path, cache_obj)
        print(f"SUCCESS: Built and cached precomputed T matrices ({len(t_data)} wavevectors) at {cache_path}")
    return entry


def _build_pt_dataset_outputs(
    designs,
    wavevector_data,
    eigenvalue_data,
    eigenvector_data,
    design_numbers,
    n_pix,
    n_eig,
):
    """
    Build .pt-friendly outputs using convert_mat_to_pytorch-style conventions.
    """
    # Convert design format (N_pix, N_pix, 3, N_struct) -> (N_struct, N_pix, N_pix)
    # Keep first pane only, matching existing .pt conventions in this repo.
    geometries = designs[:, :, 0, :].transpose(2, 0, 1)

    # Convert wavevector format (N_wv, 2, N_struct) -> (N_struct, N_wv, 2)
    wavevectors = wavevector_data.transpose(2, 0, 1)

    # Convert eigenvector format (N_dof, N_wv, N_eig, N_struct) -> split x/y panes
    eig = eigenvector_data.transpose(3, 1, 2, 0)  # (N_struct, N_wv, N_eig, N_dof)
    eig_x = eig[..., 0::2].reshape(eig.shape[0], eig.shape[1], eig.shape[2], n_pix, n_pix)
    eig_y = eig[..., 1::2].reshape(eig.shape[0], eig.shape[1], eig.shape[2], n_pix, n_pix)

    # Build full (non-reduced) sample index list.
    n_designs = eig.shape[0]
    n_wavevectors = eig.shape[1]
    n_bands = eig.shape[2]
    total_samples = n_designs * n_wavevectors * n_bands
    reduced_indices_reserved = [None] * total_samples
    cursor = 0
    for d_idx in range(n_designs):
        for w_idx in range(n_wavevectors):
            for b_idx in range(n_bands):
                reduced_indices_reserved[cursor] = (d_idx, w_idx, b_idx)
                cursor += 1

    # Flatten samples into TensorDataset-compatible arrays.
    d_idx = [idx[0] for idx in reduced_indices_reserved]
    w_idx = [idx[1] for idx in reduced_indices_reserved]
    b_idx = [idx[2] for idx in reduced_indices_reserved]
    eig_x_reduced = eig_x[d_idx, w_idx, b_idx]
    eig_y_reduced = eig_y[d_idx, w_idx, b_idx]

    # Optional wavelet embeddings to match convert_mat_to_pytorch.py.
    # If NO_utils_multiple is unavailable, placeholders are created.
    try:
        import NO_utils_multiple  # type: ignore

        waveforms = NO_utils_multiple.embed_2const_wavelet(
            wavevectors[0, :, 0],
            wavevectors[0, :, 1],
            size=n_pix,
        )
        bands_fft = NO_utils_multiple.embed_integer_wavelet(np.arange(1, n_eig + 1), size=n_pix)
    except Exception as e:
        print(f"WARNING: Could not generate wavelet embeddings: {e}")
        waveforms = np.zeros((wavevectors.shape[1], n_pix, n_pix), dtype=np.float16)
        bands_fft = np.zeros((n_eig, n_pix, n_pix), dtype=np.float16)

    # Build design_params tensor from generated design numbers (N_struct, 1).
    design_params_array = np.asarray(design_numbers, dtype=np.int64).reshape(-1, 1)

    # Convert to tensors (float16 to match existing reduced dataset files).
    outputs = {
        "displacements_dataset": torch.utils.data.TensorDataset(
            torch.from_numpy(eig_x_reduced.real).to(torch.float16),
            torch.from_numpy(eig_x_reduced.imag).to(torch.float16),
            torch.from_numpy(eig_y_reduced.real).to(torch.float16),
            torch.from_numpy(eig_y_reduced.imag).to(torch.float16),
        ),
        "reduced_indices": reduced_indices_reserved,
        "geometries_full": torch.from_numpy(geometries).to(torch.float16),
        "waveforms_full": torch.from_numpy(waveforms).to(torch.float16),
        "wavevectors_full": torch.from_numpy(wavevectors).to(torch.float16),
        "band_fft_full": torch.from_numpy(bands_fft).to(torch.float16),
        "design_params_full": torch.from_numpy(design_params_array).to(torch.float16),
        # Keep a direct frequency tensor for mat-free downstream parity/debug.
        "eigenvalue_data_full": torch.from_numpy(eigenvalue_data.transpose(2, 0, 1)).to(torch.float16),
    }
    return outputs


def _compute_single_structure(struct_idx, rng_seed_offset, binarize, const_template):
    """
    Compute one structure independently (safe for process-based parallel execution).
    """
    try:
        design_number = int(struct_idx + rng_seed_offset)

        # Local const copy so each worker/process is isolated.
        const_local = dict(const_template)
        t_cache_path = const_local.get('precomputed_T_cache_path')
        if t_cache_path:
            t_entry = _get_or_create_precomputed_t_entry(
                const_local,
                const_local['wavevectors'],
                t_cache_path,
            )
            const_local['precomputed_wavevectors'] = t_entry['wavevectors']
            const_local['precomputed_T_data'] = t_entry['T_data']

        # Build design params exactly as in the serial path.
        design_params_local = DesignParameters(1)
        design_params_local.property_coupling = 'coupled'
        design_params_local.design_style = 'kernel'
        design_params_local.design_options = {
            'kernel': 'periodic',
            'sigma_f': 1.0,
            'sigma_l': 1.0,
            'symmetry_type': 'p4mm',
            'N_value': np.inf
        }
        design_params_local.N_pix = [const_local['N_pix'], const_local['N_pix']]
        design_params_local.design_number = design_number
        design_params_local = design_params_local.prepare()

        # Generate design and map properties.
        design = get_design2(design_params_local)
        design = convert_design(
            design, 'linear', const_local['design_scale'],
            const_local['E_min'], const_local['E_max'],
            const_local['rho_min'], const_local['rho_max']
        )
        design = apply_steel_rubber_paradigm(design, const_local)
        if binarize:
            design = np.round(design)

        design_f16 = np.asarray(design, dtype=np.float16)
        const_local['design'] = design_f16

        # Compute dispersion and system matrices.
        wv, fr, ev, _, K, M, T = dispersion_with_matrix_save_opt(const_local, const_local['wavevectors'])

        # Material property expansion.
        design_f32 = design_f16.astype(np.float32, copy=False)
        explicit_props = design_to_explicit(
            design_f32, const_local['design_scale'],
            const_local['E_min'], const_local['E_max'],
            const_local['rho_min'], const_local['rho_max'],
            const_local['poisson_min'], const_local['poisson_max']
        )

        return {
            'ok': True,
            'struct_idx': struct_idx,
            'design_number': design_number,
            'design_f16': design_f16,
            'wv_f16': np.asarray(wv, dtype=np.float16),
            'fr_f16': np.asarray(np.real(fr), dtype=np.float16),
            'ev_c64': None if ev is None else np.asarray(ev, dtype=np.complex64),
            'E_f32': np.asarray(explicit_props['E'], dtype=np.float32),
            'rho_f32': np.asarray(explicit_props['rho'], dtype=np.float32),
            'nu_f32': np.asarray(explicit_props['nu'], dtype=np.float32),
            'K': K,
            'M': M,
            'T': T,
            'imag_max': float(np.max(np.abs(np.imag(fr)))) if fr is not None else 0.0,
        }
    except Exception as e:
        import traceback
        return {
            'ok': False,
            'struct_idx': struct_idx,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def generate_dispersion_dataset_with_matrices(
    n_struct: int = 5,
    rng_seed_offset_override: int = 0,
    binarize_override: bool = False,
    parallel_workers: int = 16,
    save_pkl: bool = False,
):
    """
    Generate dispersion dataset with system matrices (equivalent to MATLAB script).
    """
    
    print("="*80)
    print("GENERATING DISPERSION DATASET WITH PYTHON LIBRARY")
    print("="*80)
    
    # Get script information
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_path = os.path.dirname(os.path.abspath(__file__))
    
    # Output flags
    is_save_output = True
    is_save_eigenvectors = True
    is_profile = False
    is_save_mesh = False
    is_save_k_and_m = True
    
    # Discretization parameters - FIXED to match MATLAB exactly
    const = {
        'N_ele': 1,  # Number of elements along one pixel side length
        'N_pix': 32,  # Number of pixels along one unit cell side length (SCALAR, not array!)
        'N_wv': [25, 13],  # Number of wavevectors (25 x ceil(25/2) = 13)
        'N_eig': 6,  # Number of eigenvalue bands to compute
        'sigma_eig': 1e-2,  # Eigenvalue solver parameter
        'a': 1.0,  # Side length of square unit cell [m]
        'design_scale': 'linear'
    }
    
    # Flags for computational improvements
    use_parallel = int(parallel_workers) > 1
    const.update({
        'isUseGPU': False,
        'isUseImprovement': True,
        'isUseSecondImprovement': False,
        'isUseParallel': use_parallel,
        'isSaveEigenvectors': is_save_eigenvectors,
        'isComputeGroupVelocity': False,
        'isComputeFrequencyDesignSensitivity': False,
        'isComputeGroupVelocityDesignSensitivity': False
    })
    
    # Design parameters
    N_struct = int(n_struct)  # Number of designs to generate
    rng_seed_offset = int(rng_seed_offset_override)  # RNG seed offset
    binarize = bool(binarize_override)  # False=continuous, True=binarized
    
    # Material parameters (exact MATLAB translation)
    const.update({
        'E_min': 200e6,
        'E_max': 200e9,
        'rho_min': 8e2,
        'rho_max': 8e3,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 1.0
    })
    # Validate constants expects a design key; initialize placeholder before loop.
    const['design'] = np.zeros((const['N_pix'], const['N_pix'], 3), dtype=np.float16)
    
    # Imaginary tolerance
    imag_tol = 1e-3
    
    # Generate wavevectors
    print(f"Generating wavevectors for {const['N_wv']} grid...")
    const['wavevectors'] = np.asarray(
        get_IBZ_wavevectors(const['N_wv'], const['a'], 'none'), dtype=np.float16
    )
    print(f"Generated {len(const['wavevectors'])} wavevectors")
    t_cache_path = os.path.join(script_path, "precomputed_T_matrices.pkl")
    t_entry = _get_or_create_precomputed_t_entry(const, const['wavevectors'], t_cache_path)
    const['precomputed_T_cache_path'] = t_cache_path
    # Keep in-process references for serial mode and optional fallback.
    const['precomputed_wavevectors'] = t_entry['wavevectors']
    const['precomputed_T_data'] = t_entry['T_data']
    
    # Initialize design parameters (exact MATLAB translation)
    design_params = DesignParameters(N_struct)
    design_params.property_coupling = 'coupled'
    design_params.design_style = 'kernel'
    design_params.design_options = {
        'kernel': 'periodic',
        'sigma_f': 1.0,
        'sigma_l': 1.0,
        'symmetry_type': 'p4mm',
        'N_value': np.inf
    }
    design_params.N_pix = [const['N_pix'], const['N_pix']]  # MATLAB uses [N_pix, N_pix]
    design_params = design_params.prepare()
    
    # Initialize storage arrays - FIXED for scalar N_pix
    print("Initializing storage arrays...")
    designs = np.zeros((const['N_pix'], const['N_pix'], 3, N_struct), dtype=np.float16)
    wavevector_data = np.zeros((np.prod(const['N_wv']), 2, N_struct), dtype=np.float16)
    eigenvalue_data = np.zeros((np.prod(const['N_wv']), const['N_eig'], N_struct), dtype=np.float16)
    
    N_dof = 2 * (const['N_pix'] * const['N_ele'])**2
    eigenvector_data = np.zeros((N_dof, np.prod(const['N_wv']), const['N_eig'], N_struct), dtype=np.complex64)
    
    # Material property data
    # NOTE: E can reach 2e11 Pa, which cannot be represented in float16.
    # Keep constitutive arrays at float32 to avoid overflow/Inf.
    elastic_modulus_data = np.zeros((const['N_pix'], const['N_pix'], N_struct), dtype=np.float32)
    density_data = np.zeros((const['N_pix'], const['N_pix'], N_struct), dtype=np.float32)
    poisson_data = np.zeros((const['N_pix'], const['N_pix'], N_struct), dtype=np.float32)
    
    # System matrices (using lists for variable sizes)
    K_data = [None] * N_struct
    M_data = [None] * N_struct
    T_data = [None] * N_struct
    design_numbers = np.arange(N_struct, dtype=np.int64) + rng_seed_offset
    
    # Validate constants
    is_valid, missing_fields = validate_constants(const)
    if not is_valid:
        print(f"ERROR: Constants validation failed. Missing fields: {missing_fields}")
        return None
    
    print("SUCCESS: Constants validation passed")
    
    # Generate dataset
    print(f"\nGenerating {N_struct} structures...")
    print(f"Parallel workers: {parallel_workers} ({'enabled' if use_parallel else 'serial'})")
    progress_every = 100 if N_struct >= 100 else max(1, N_struct // 10)
    gen_start_time = time.perf_counter()
    processed = 0
    if use_parallel:
        const_template = dict(const)
        with ProcessPoolExecutor(max_workers=int(parallel_workers)) as executor:
            futures = {
                executor.submit(
                    _compute_single_structure,
                    struct_idx,
                    rng_seed_offset,
                    binarize,
                    const_template,
                ): struct_idx
                for struct_idx in range(N_struct)
            }
            for future in as_completed(futures):
                result = future.result()
                struct_idx = result['struct_idx']
                if not result['ok']:
                    print(f"  ERROR: Error processing structure {struct_idx + 1}: {result['error']}")
                    print(result.get('traceback', ''))
                    processed += 1
                    continue

                designs[:, :, :, struct_idx] = result['design_f16']
                wavevector_data[:, :, struct_idx] = result['wv_f16']
                eigenvalue_data[:, :, struct_idx] = result['fr_f16']
                if is_save_eigenvectors and result['ev_c64'] is not None:
                    eigenvector_data[:, :, :, struct_idx] = result['ev_c64']
                elastic_modulus_data[:, :, struct_idx] = result['E_f32']
                density_data[:, :, struct_idx] = result['rho_f32']
                poisson_data[:, :, struct_idx] = result['nu_f32']
                K_data[struct_idx] = result['K']
                M_data[struct_idx] = result['M']
                T_data[struct_idx] = result['T']
                if result['imag_max'] > imag_tol:
                    print(f"WARNING: Warning: Large imaginary component in frequency for structure {struct_idx + 1}")

                processed += 1
                if processed == 1 or processed == N_struct or processed % progress_every == 0:
                    elapsed = time.perf_counter() - gen_start_time
                    rate = processed / max(elapsed, 1e-9)
                    eta = (N_struct - processed) / max(rate, 1e-9)
                    print(
                        f"PROGRESS: {processed}/{N_struct} "
                        f"({100.0 * processed / N_struct:.1f}%) "
                        f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s"
                    )
    else:
        const_template = dict(const)
        for struct_idx in range(N_struct):
            result = _compute_single_structure(
                struct_idx=struct_idx,
                rng_seed_offset=rng_seed_offset,
                binarize=binarize,
                const_template=const_template,
            )
            if not result['ok']:
                print(f"  ERROR: Error processing structure {struct_idx + 1}: {result['error']}")
                print(result.get('traceback', ''))
                continue

            designs[:, :, :, struct_idx] = result['design_f16']
            wavevector_data[:, :, struct_idx] = result['wv_f16']
            eigenvalue_data[:, :, struct_idx] = result['fr_f16']
            if is_save_eigenvectors and result['ev_c64'] is not None:
                eigenvector_data[:, :, :, struct_idx] = result['ev_c64']
            elastic_modulus_data[:, :, struct_idx] = result['E_f32']
            density_data[:, :, struct_idx] = result['rho_f32']
            poisson_data[:, :, struct_idx] = result['nu_f32']
            K_data[struct_idx] = result['K']
            M_data[struct_idx] = result['M']
            T_data[struct_idx] = result['T']
            if result['imag_max'] > imag_tol:
                print(f"WARNING: Warning: Large imaginary component in frequency for structure {struct_idx + 1}")

            processed = struct_idx + 1
            if processed == 1 or processed == N_struct or processed % progress_every == 0:
                elapsed = time.perf_counter() - gen_start_time
                rate = processed / max(elapsed, 1e-9)
                eta = (N_struct - processed) / max(rate, 1e-9)
                print(
                    f"PROGRESS: {processed}/{N_struct} "
                    f"({100.0 * processed / N_struct:.1f}%) "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s"
                )
    
    # Collect constitutive data
    constitutive_data = {
        'modulus': elastic_modulus_data,
        'density': density_data,
        'poisson': poisson_data
    }
    
    # Set up save locations
    script_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if is_save_output:
        output_folder = os.path.join(script_path, f'OUTPUT/output_{script_start_time}')
        os.makedirs(output_folder, exist_ok=True)
        
        # Copy this script to output folder
        import shutil
        shutil.copy2(__file__, os.path.join(output_folder, f'{script_name}.py'))
    
    # Prepare data for optional Python object save/debug.
    dataset = {
        'WAVEVECTOR_DATA': wavevector_data,
        'EIGENVALUE_DATA': eigenvalue_data,
        'CONSTITUTIVE_DATA': constitutive_data,
        'designs': designs,
        'const': const,
        'design_params': design_params,
        'design_numbers': design_numbers,
        'N_struct': N_struct,
        'imag_tol': imag_tol,
        'rng_seed_offset': rng_seed_offset,
        'script_start_time': script_start_time
    }
    
    if is_save_eigenvectors:
        dataset['EIGENVECTOR_DATA'] = eigenvector_data
    
    if any(k is not None for k in K_data):
        dataset['K_DATA'] = K_data
    if any(m is not None for m in M_data):
        dataset['M_DATA'] = M_data
    if any(t is not None for t in T_data):
        dataset['T_DATA'] = T_data
    
    # Save results
    if is_save_output:
        design_type_label = 'binarized' if binarize else 'continuous'
        output_pt_path = os.path.join(
            output_folder,
            f'{design_type_label}_{script_start_time}_pt'
        )
        os.makedirs(output_pt_path, exist_ok=True)

        # The legacy .mat export is intentionally disabled:
        # output_file_path = os.path.join(output_folder, f'{design_type_label}_{script_start_time}.mat')
        # sio.savemat(output_file_path, dataset, oned_as='column')

        pt_outputs = _build_pt_dataset_outputs(
            designs=designs,
            wavevector_data=wavevector_data,
            eigenvalue_data=eigenvalue_data,
            eigenvector_data=eigenvector_data,
            design_numbers=design_numbers,
            n_pix=const['N_pix'],
            n_eig=const['N_eig'],
        )

        torch.save(pt_outputs['displacements_dataset'], os.path.join(output_pt_path, 'displacements_dataset.pt'))
        torch.save(pt_outputs['reduced_indices'], os.path.join(output_pt_path, 'reduced_indices.pt'))
        torch.save(pt_outputs['geometries_full'], os.path.join(output_pt_path, 'geometries_full.pt'))
        torch.save(pt_outputs['waveforms_full'], os.path.join(output_pt_path, 'waveforms_full.pt'))
        torch.save(pt_outputs['wavevectors_full'], os.path.join(output_pt_path, 'wavevectors_full.pt'))
        torch.save(pt_outputs['band_fft_full'], os.path.join(output_pt_path, 'band_fft_full.pt'))
        torch.save(pt_outputs['design_params_full'], os.path.join(output_pt_path, 'design_params_full.pt'))
        torch.save(pt_outputs['eigenvalue_data_full'], os.path.join(output_pt_path, 'eigenvalue_data_full.pt'))

        print(f"SUCCESS: PyTorch dataset bundle saved to: {output_pt_path}")
        if save_pkl:
            # Optional full Python object dump for debugging complex/sparse structures.
            pickle_file_path = os.path.join(output_folder, f'{design_type_label}_{script_start_time}.pkl')
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"SUCCESS: Python pickle file saved to: {pickle_file_path}")
    
    # Summary
    print("\n" + "="*80)
    print("DATASET GENERATION SUMMARY")
    print("="*80)
    print(f"SUCCESS: Generated {N_struct} structures")
    print(f"SUCCESS: Computed dispersion for {len(const['wavevectors'])} wavevectors")
    print(f"SUCCESS: Design size: {const['N_pix']}x{const['N_pix']} pixels")
    print(f"SUCCESS: Number of eigenvalues: {const['N_eig']}")
    print(f"SUCCESS: Frequency range: {np.min(eigenvalue_data):.3f} - {np.max(eigenvalue_data):.3f} Hz")
    print(f"SUCCESS: Design type: {design_type_label}")
    
    if is_save_eigenvectors:
        print(f"SUCCESS: Eigenvectors saved: {eigenvector_data.shape}")
    
    print("="*80)
    
    return dataset

def demonstrate_library_features():
    """
    Demonstrate additional features of the new library and generate visual checks.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING LIBRARY FEATURES")
    print("="*80)
    
    # Create quick_checks folder
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    
    quick_checks_folder = "quick_checks"
    if not os.path.exists(quick_checks_folder):
        os.makedirs(quick_checks_folder)
        print(f"SUCCESS: Created {quick_checks_folder} folder")
    
    # Generate different design types and visualize them
    print("1. Generating different design types...")
    design_types = ['homogeneous', 'dispersive-tetragonal', 'quasi-1D']
    designs = {}
    
    for design_type in design_types:
        try:
            design = get_design(design_type, 32)
            designs[design_type] = design
            print(f"  SUCCESS: Generated {design_type} design: {design.shape}")
        except Exception as e:
            print(f"  ERROR: Failed to generate {design_type}: {e}")
    
    # Create visualization of design patterns
    if designs:
        print("2. Creating design pattern visualizations...")
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Design Pattern Examples', fontsize=16, fontweight='bold')
            
            for idx, (design_type, design) in enumerate(designs.items()):
                # Plot the first material property (elastic modulus)
                im = axes[idx].imshow(design[:, :, 0], cmap='viridis', origin='lower')
                axes[idx].set_title(f'{design_type.replace("-", " ").title()}', fontweight='bold')
                axes[idx].set_xlabel('X (pixels)')
                axes[idx].set_ylabel('Y (pixels)')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[idx], label='Elastic Modulus (normalized)')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"design_patterns_{timestamp}.png"
            filepath = os.path.join(quick_checks_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  SUCCESS: Saved design patterns to {filepath}")
            
        except Exception as e:
            print(f"  ERROR: Failed to create design visualization: {e}")
    
    # Test different symmetry types
    print("3. Testing different symmetry types...")
    symmetry_types = ['none', 'p4mm', 'p2mm']
    wavevector_data = {}
    
    for sym_type in symmetry_types:
        try:
            wv = get_IBZ_wavevectors([10, 10], 1.0, sym_type)
            wavevector_data[sym_type] = wv
            print(f"  SUCCESS: Generated {sym_type} wavevectors: {len(wv)} points")
        except Exception as e:
            print(f"  ERROR: Failed to generate {sym_type} wavevectors: {e}")
    
    # Create visualization of wavevector patterns
    if wavevector_data:
        print("4. Creating wavevector pattern visualizations...")
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Irreducible Brillouin Zone (IBZ) Wavevector Patterns', fontsize=16, fontweight='bold')
            
            for idx, (sym_type, wv) in enumerate(wavevector_data.items()):
                # Plot wavevectors
                axes[idx].scatter(wv[:, 0], wv[:, 1], c='red', s=20, alpha=0.7)
                axes[idx].set_title(f'{sym_type.upper()} Symmetry\n({len(wv)} points)', fontweight='bold')
                axes[idx].set_xlabel('k_x')
                axes[idx].set_ylabel('k_y')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_aspect('equal')
                
                # Add some reference lines for context
                axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[idx].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"wavevector_patterns_{timestamp}.png"
            filepath = os.path.join(quick_checks_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  SUCCESS: Saved wavevector patterns to {filepath}")
            
        except Exception as e:
            print(f"  ERROR: Failed to create wavevector visualization: {e}")
    
    # Create visualization of the specific design used in MATLAB script
    print("5. Creating MATLAB-equivalent design visualization...")
    try:
        # Generate the exact design used in the MATLAB script: dispersive-tetragonal with p4mm symmetry
        from design_parameters import DesignParameters
        from get_design2 import get_design2
        
        # Create design parameters matching MATLAB script
        design_params_matlab = DesignParameters(1)
        design_params_matlab.property_coupling = 'coupled'
        design_params_matlab.design_style = 'kernel'
        design_params_matlab.design_options = {
            'kernel': 'periodic',
            'sigma_f': 1.0,
            'sigma_l': 1.0,
            'symmetry_type': 'p4mm',
            'N_value': np.inf
        }
        design_params_matlab.N_pix = 32
        design_params_matlab = design_params_matlab.prepare()
        
        # Generate the design using get_design2 (same as MATLAB)
        matlab_design = get_design2(design_params_matlab)
        
        # Create 3-panel visualization showing all material properties
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('MATLAB-Equivalent Design: Dispersive-Tetragonal with P4MM Symmetry', 
                     fontsize=16, fontweight='bold')
        
        # Material property names and colormaps
        properties = ['Elastic Modulus', 'Density', 'Poisson Ratio']
        colormaps = ['viridis', 'viridis', 'viridis']
        
        for idx in range(3):
            im = axes[idx].imshow(matlab_design[:, :, idx], cmap=colormaps[idx], origin='lower')
            axes[idx].set_title(f'{properties[idx]}', fontweight='bold')
            axes[idx].set_xlabel('X (pixels)')
            axes[idx].set_ylabel('Y (pixels)')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], label=f'{properties[idx]} (normalized)')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"matlab_equivalent_design_{timestamp}.png"
        filepath = os.path.join(quick_checks_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  SUCCESS: Saved MATLAB-equivalent design to {filepath}")
        print(f"  INFO: Design shape: {matlab_design.shape}")
        print(f"  INFO: Property ranges - E: [{np.min(matlab_design[:,:,0]):.3f}, {np.max(matlab_design[:,:,0]):.3f}]")
        print(f"  INFO: Property ranges - rho: [{np.min(matlab_design[:,:,1]):.3f}, {np.max(matlab_design[:,:,1]):.3f}]")
        print(f"  INFO: Property ranges - nu: [{np.min(matlab_design[:,:,2]):.3f}, {np.max(matlab_design[:,:,2]):.3f}]")
        
    except Exception as e:
        print(f"  ERROR: Failed to create MATLAB-equivalent design visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dispersion dataset with configurable sample count/seed offset.")
    parser.add_argument("--n-struct", type=int, default=5, help="Number of structures/designs to generate.")
    parser.add_argument("--rng-seed-offset", type=int, default=0, help="RNG seed offset applied to design index.")
    parser.add_argument("--binarize", action="store_true", help="Generate binarized designs (default: continuous).")
    parser.add_argument("--parallel-workers", type=int, default=16, help="Number of worker processes for geometry-level parallel generation (default: 16).")
    parser.add_argument("--save-pkl", action="store_true", help="Also save full Python dataset snapshot as .pkl (default: off).")
    parser.add_argument("--run-demo", action="store_true", help="Run post-generation demonstration plots (default: off).")
    # Backward-compatible no-op: generation now skips demo by default.
    parser.add_argument("--skip-demo", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Run the main dataset generation
    dataset = generate_dispersion_dataset_with_matrices(
        n_struct=args.n_struct,
        rng_seed_offset_override=args.rng_seed_offset,
        binarize_override=args.binarize,
        parallel_workers=args.parallel_workers,
        save_pkl=args.save_pkl,
    )

    # Demonstrate additional features
    if args.run_demo and not args.skip_demo:
        demonstrate_library_features()

    print("\nCOMPLETED: Script completed successfully!")
    print("The generated dataset is compatible with both MATLAB and Python workflows.")
