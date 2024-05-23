import os
import h5py
import numpy as np
import torch
import time

def load_mat_data(folder_path):
    start_time = time.time()
    # Find the .mat file in the specified folder
    mat_files = [file for file in os.listdir(folder_path) if file.endswith(".mat")]

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in the specified folder: {folder_path}")

    if len(mat_files) > 1:
        raise ValueError(f"Multiple .mat files found in the specified folder: {folder_path}. Please ensure there is only one .mat file.")

    data_path = os.path.join(folder_path, mat_files[0])

    # Load the .mat file using h5py
    with h5py.File(data_path, 'r') as file:
        # Load the data arrays
        EIGENVALUE_DATA = np.array(file['EIGENVALUE_DATA'], dtype=np.float16)
        EIGENVECTOR_DATA_real = np.array(file['EIGENVECTOR_DATA']['real'], dtype=np.float16)
        EIGENVECTOR_DATA_imag = np.array(file['EIGENVECTOR_DATA']['imag'], dtype=np.float16)
        EIGENVECTOR_DATA = EIGENVECTOR_DATA_real + 1j * EIGENVECTOR_DATA_imag
        WAVEVECTOR_DATA = np.array(file['WAVEVECTOR_DATA'], dtype=np.float16)

        # Convert to PyTorch tensors
        EIGENVALUE_DATA_tensor = torch.tensor(EIGENVALUE_DATA, dtype=torch.float16)
        EIGENVECTOR_DATA_tensor = torch.tensor(EIGENVECTOR_DATA, dtype=torch.complex32)
        WAVEVECTOR_DATA_tensor = torch.tensor(WAVEVECTOR_DATA, dtype=torch.float16)

        # Unpack the 'const' struct
        const = {key: np.array(file['const'][key]) for key in file['const']}

        # Assign numbers to variables
        N_struct = np.array(file['N_struct'])  # Adjust indexing if necessary
        design_params = np.array(file['design_params'])
        designs = np.array(file['designs'])
        imag_tol = np.array(file['imag_tol'])
        rng_seed_offset = np.array(file['rng_seed_offset'])

    end_time = time.time()
    print(f"Data loaded in {end_time - start_time:.2f} seconds.")

    return {
        'EIGENVALUE_DATA': EIGENVALUE_DATA,
        'EIGENVECTOR_DATA': EIGENVECTOR_DATA,
        'WAVEVECTOR_DATA': WAVEVECTOR_DATA,
        'EIGENVALUE_DATA_tensor': EIGENVALUE_DATA_tensor,
        'EIGENVECTOR_DATA_tensor': EIGENVECTOR_DATA_tensor,
        'WAVEVECTOR_DATA_tensor': WAVEVECTOR_DATA_tensor,
        'const': const,
        'N_struct': N_struct,
        'design_params': design_params,
        'designs': designs,
        'imag_tol': imag_tol,
        'rng_seed_offset': rng_seed_offset
    }