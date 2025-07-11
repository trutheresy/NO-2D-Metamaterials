import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import random
from scipy.optimize import minimize

def load_mat_data(folder_path, multiple_datafiles=False):
    start_time = time.time()
    mat_files = [file for file in os.listdir(folder_path) if file.endswith(".mat")]

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in the specified folder: {folder_path}")

    if len(mat_files) > 1 and not multiple_datafiles:
        raise ValueError(f"Multiple .mat files found in the specified folder: {folder_path}. Please ensure there is only one .mat file or set multiple_datafiles=True.")

    # Initialize containers for combined data if multiple datafiles is True
    combined_data = {
        'EIGENVALUE_DATA': [],
        'EIGENVECTOR_DATA': [],
        'WAVEVECTOR_DATA': [],
        'const': {},
        'N_struct': [],
        'design_params': [],
        'designs': [],
        'imag_tol': [],
        'rng_seed_offset': []
    }

    for mat_file in mat_files:
        data_path = os.path.join(folder_path, mat_file)

        with h5py.File(data_path, 'r') as file:
            EIGENVALUE_DATA = np.array(file['EIGENVALUE_DATA'], dtype=np.float16)
            EIGENVECTOR_DATA_real = np.array(file['EIGENVECTOR_DATA']['real'], dtype=np.float16)
            EIGENVECTOR_DATA_imag = np.array(file['EIGENVECTOR_DATA']['imag'], dtype=np.float16)
            EIGENVECTOR_DATA = EIGENVECTOR_DATA_real + 1j * EIGENVECTOR_DATA_imag
            WAVEVECTOR_DATA = np.array(file['WAVEVECTOR_DATA'], dtype=np.float16)

            const = {key: np.array(file['const'][key]) for key in file['const']}

            N_struct = np.array(file['N_struct'])
            design_params = np.array(file['design_params'])
            designs = np.array(file['designs'])
            imag_tol = np.array(file['imag_tol'])
            rng_seed_offset = np.array(file['rng_seed_offset'])

            # If multiple files, append data; otherwise return first one
            if multiple_datafiles:
                combined_data['EIGENVALUE_DATA'].append(EIGENVALUE_DATA)
                combined_data['EIGENVECTOR_DATA'].append(EIGENVECTOR_DATA)
                combined_data['WAVEVECTOR_DATA'].append(WAVEVECTOR_DATA)

                for key, value in const.items():
                    if key not in combined_data['const']:
                        combined_data['const'][key] = [value]
                    else:
                        combined_data['const'][key].append(value)

                combined_data['N_struct'].append(N_struct)
                combined_data['design_params'].append(design_params)
                combined_data['designs'].append(designs)
                combined_data['imag_tol'].append(imag_tol)
                combined_data['rng_seed_offset'].append(rng_seed_offset)
            else:
                end_time = time.time()
                print(f"Data loaded in {end_time - start_time:.2f} seconds.")
                return {
                    'EIGENVALUE_DATA': EIGENVALUE_DATA,
                    'EIGENVECTOR_DATA': EIGENVECTOR_DATA,
                    'WAVEVECTOR_DATA': WAVEVECTOR_DATA,
                    'const': const,
                    'N_struct': N_struct,
                    'design_params': design_params,
                    'designs': designs,
                    'imag_tol': imag_tol,
                    'rng_seed_offset': rng_seed_offset
                }

    # Combine data across multiple files
    if multiple_datafiles:
        combined_data['EIGENVALUE_DATA'] = np.concatenate(combined_data['EIGENVALUE_DATA'], axis=0)
        combined_data['EIGENVECTOR_DATA'] = np.concatenate(combined_data['EIGENVECTOR_DATA'], axis=0)
        combined_data['WAVEVECTOR_DATA'] = np.concatenate(combined_data['WAVEVECTOR_DATA'], axis=0)
        combined_data['N_struct'] = np.concatenate(combined_data['N_struct'], axis=0)
        combined_data['design_params'] = np.concatenate(combined_data['design_params'], axis=0)
        combined_data['designs'] = np.concatenate(combined_data['designs'], axis=0)
        combined_data['imag_tol'] = np.concatenate(combined_data['imag_tol'], axis=0)
        combined_data['rng_seed_offset'] = np.concatenate(combined_data['rng_seed_offset'], axis=0)

        # Combine dictionaries in `const`
        for key, value_list in combined_data['const'].items():
            combined_data['const'][key] = np.concatenate(value_list, axis=0)

        end_time = time.time()
        print(f"Data loaded and combined from {len(mat_files)} files in {end_time - start_time:.2f} seconds.")

        return combined_data

def split_array(arr, dim):
    # Get the shape of the input array
    shape = arr.shape

    # Check if the specified dimension is valid
    if dim < 0 or dim >= len(shape):
        raise ValueError(f"Invalid dimension: {dim}. Array has {len(shape)} dimensions.")

    # Check if the length of the specified dimension is even
    if shape[dim] % 2 != 0:
        raise ValueError(f"Length of dimension {dim} must be even.")

    # Create a list of slice objects for each dimension
    slices = [slice(None)] * len(shape)

    # Split the specified dimension into two halves
    half_length = shape[dim] // 2
    slices[dim] = slice(None, None, 2)
    arr1 = arr[tuple(slices)]
    slices[dim] = slice(1, None, 2)
    arr2 = arr[tuple(slices)]

    return arr1, arr2

def extract_data(data_path, multiple_datafiles=False):
    i_design = 0
    i_wavevector = 1
    i_band = 2
    i_dim = 3
    i_res = 4

    data = load_mat_data(data_path, multiple_datafiles)

    designs = np.array(data['designs'])
    design_params = np.array(data['design_params'])
    n_designs = designs.shape[0]
    n_panes = designs.shape[1]
    design_res = designs.shape[2]
    const = {key: np.array(data['const'][key]) for key in data['const']}

    WAVEVECTOR_DATA = data['WAVEVECTOR_DATA']
    n_dim = WAVEVECTOR_DATA.shape[1]
    WAVEVECTOR_DATA = WAVEVECTOR_DATA.transpose(0,2,1)
    n_wavevectors = WAVEVECTOR_DATA.shape[1]
    if np.any(np.iscomplex(WAVEVECTOR_DATA)):
        print("WAVEVECTOR_DATA array contains complex-valued elements.")
    WAVEFORM_DATA = wavevectors_to_spatial(WAVEVECTOR_DATA, design_res, length_scale=const['a'], plot_sample=False)

    EIGENVALUE_DATA = np.array(data['EIGENVALUE_DATA']).transpose(0,2,1)
    n_bands = EIGENVALUE_DATA.shape[2]
    EIGENVECTOR_DATA = np.array(data['EIGENVECTOR_DATA']).transpose(0,2,1,3)
    EIGENVECTOR_DATA_x, EIGENVECTOR_DATA_y = split_array(EIGENVECTOR_DATA, i_dim)

    N_struct = data['N_struct']
    imag_tol = data['imag_tol']
    rng_seed_offset = data['rng_seed_offset']

    EIGENVECTOR_DATA_x = EIGENVECTOR_DATA_x.reshape(n_designs, n_wavevectors, n_bands, design_res, design_res, order='C')
    EIGENVECTOR_DATA_y = EIGENVECTOR_DATA_y.reshape(n_designs, n_wavevectors, n_bands, design_res, design_res, order='C')

    #Print the shape of each variable
    print(f'n_designs: {n_designs}, n_panes: {n_panes}, design_res: {design_res}, d_design: {n_dim}, dispersion_bands: {n_bands}, rng_seed_offset: {rng_seed_offset}')
    print('EIGENVALUE_DATA shape:', EIGENVALUE_DATA.shape)
    print('EIGENVECTOR_DATA shape:', EIGENVECTOR_DATA.shape)
    print('EIGENVECTOR_DATA_x shape:', EIGENVECTOR_DATA_x.shape)
    print('EIGENVECTOR_DATA_y shape:', EIGENVECTOR_DATA_y.shape)
    print('WAVEVECTOR_DATA shape:', WAVEVECTOR_DATA.shape)
    print('WAVEFORM_DATA shape:', WAVEFORM_DATA.shape)
    print('designs shape:', designs.shape)
    print('design_params shape:', design_params.shape)
    print('const shape:', {key: const[key].shape for key in const})

    return designs, design_params, n_designs, n_panes, design_res, WAVEVECTOR_DATA, WAVEFORM_DATA, n_dim, n_wavevectors, EIGENVALUE_DATA, n_bands, EIGENVECTOR_DATA_x, EIGENVECTOR_DATA_y, const, N_struct, imag_tol, rng_seed_offset

def plot_geometry(sample_geometry, sample_index):
    fig, ax = plt.subplots()
    im = ax.imshow(sample_geometry)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Geometry #{sample_index}')
    plt.show()

def plot_eigenvectors(sample_eigenvector_x, sample_eigenvector_y, unify_scales=True):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    im00 = axs[0, 0].imshow(np.real(sample_eigenvector_x), cmap='viridis')
    axs[0, 0].set_title('x displacement field real', pad=1)
    axs[0, 0].axis('off')

    im10 = axs[1, 0].imshow(np.imag(sample_eigenvector_x), cmap='viridis')
    axs[1, 0].set_title('x displacement field imag', pad=1)
    axs[1, 0].axis('off')

    im01 = axs[0, 1].imshow(np.real(sample_eigenvector_y), cmap='viridis')
    axs[0, 1].set_title('y displacement field real', pad=1)
    axs[0, 1].axis('off')

    im11 = axs[1, 1].imshow(np.imag(sample_eigenvector_y), cmap='viridis')
    axs[1, 1].set_title('y displacement field imag', pad=1)
    axs[1, 1].axis('off')

    if unify_scales:
        vmin = min(im.get_array().min() for im in [im00, im10, im01, im11])
        vmax = max(im.get_array().max() for im in [im00, im10, im01, im11])

        for im in [im00, im10, im01, im11]:
            im.set_clim(vmin, vmax)

        cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
        fig.colorbar(im00, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=10)
    else:
        for ax in axs.flatten():
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(ax.images[0], cax=cax)
            cax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()

def wavevectors_to_spatial(wavevectors, design_res, length_scale, amplitude=1.0, phase=0.0, plot_sample=False):
    """
    Convert a 3D array of wavevectors into their spatial representations.

    Parameters:
        wavevectors (np.ndarray): 3D array of wavevectors with shape (N, 325, 2),
                                  where N is the number of samples,
                                  the first index indicates which of the 325 wavevectors it is,
                                  and the second index is the vector component in x and y.
        design_res (int): Resolution of the output grid.
        length_scale (float): Total length scale of the grid (meters).
        amplitude (float, optional): Amplitude of the wave. Default is 1.0.
        phase (float, optional): Phase shift of the wave. Default is 0.0.
        plot_sample (bool, optional): If True, plot a random sample of the output spatial waves.

    Returns:
        np.ndarray: 3D array of shape (N, design_res, design_res) with the wave values.
    """
    N = wavevectors.shape[0]
    W = wavevectors.shape[1]

    # Define the spatial grid
    x = np.linspace(-length_scale/2, length_scale/2, design_res)
    y = np.linspace(-length_scale/2, length_scale/2, design_res)
    X, Y = np.meshgrid(x, y)

    # Initialize the output array
    spatial_waves = np.zeros((N, W, design_res, design_res))

    # Generate the spatial representations
    for i in range(N):
        for j in range(W):
            k_x = wavevectors[i, j, 0]
            k_y = wavevectors[i, j, 1]
            spatial_waves[i, j] = amplitude * np.cos(k_x * X + k_y * Y + phase)

    # Plot a random sample if requested
    if plot_sample:
        sample_n = random.randint(0, N - 1)
        sample_w = random.randint(0, W - 1)
        plt.figure(figsize=(6, 6))
        plt.contourf(spatial_waves[sample_n], cmap='viridis')
        plt.colorbar(label='Wave Amplitude')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'Spatial Wave: Sample {sample_n}, Wavevector {sample_w} with $k_x$={wavevectors[sample_n, sample_w, 0]} $m^{{-1}}$, $k_y$={wavevectors[sample_n, sample_w, 1]} $m^{{-1}}$')
        plt.show()

    print('Spatial waves shape:', spatial_waves.shape)
    return spatial_waves

def const_to_spatial(test_band, design_res, plot_result=True, scaling_factor=1.0):
    x = np.linspace(-1/2, 1/2, design_res)
    y = np.linspace(-1/2, 1/2, design_res)
    X, Y = np.meshgrid(x, y)
    constant_array = scaling_factor * np.sin(test_band * np.pi * X) * np.sin(test_band * np.pi * Y)
    fft_result = np.fft.fft2(constant_array)
    fft_result_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.abs(fft_result_shifted)

    if plot_result:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(constant_array, cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title(f'Spatial Representation of {test_band}')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_spectrum, cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title(f'2D FFT Magnitude Spectrum of {test_band}')
        plt.xlabel('Frequency X')
        plt.ylabel('Frequency Y')

        plt.tight_layout()
        plt.show()

    return constant_array, magnitude_spectrum

def embed_integer_wavelet(c, size=32, freq_range=2.0):
    """
    Embed an array of integers into 2D patterns using Gabor wavelets.
    
    Args:
        c: Array of integers to embed
        size: Size of the square output array
        freq_range: Factor to control the frequency range (higher values = broader frequency range)
    
    Returns:
        3D numpy array with the Gabor wavelet embeddings, shape (N, size, size) where N is length of c
    """
    # Create coordinate grid (shared across all patterns)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Convert c to numpy array if not already
    c = np.asarray(c)
    
    # Calculate parameters for all values in c at once
    base_frequency = (1.0 + np.abs(c))[:, None, None] * (size/8)
    theta = (c % 8)[:, None, None] * (size/16)
    
    # Broadcasting the coordinates for rotation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Rotate coordinates for all patterns at once
    X_theta = X[None, :, :] * cos_theta + Y[None, :, :] * sin_theta
    Y_theta = -X[None, :, :] * sin_theta + Y[None, :, :] * cos_theta
    
    # Gaussian envelope parameters
    sigma_x = 0.3 / freq_range
    sigma_y = 0.3 / freq_range
    
    # Gaussian envelope (shared across all patterns)
    gaussian = np.exp(-(X_theta**2 / (2 * sigma_x**2) + Y_theta**2 / (2 * sigma_y**2)))
    
    # Gabor wavelets for all patterns at once
    gabor = gaussian * np.cos(base_frequency * X_theta)
    
    return gabor

def embed_2const_wavelet(c1, c2, size=32, freq_range=1.0):
    """
    Embed two arrays of integers into 2D patterns using Gabor wavelets with different frequencies in x and y.
    
    Args:
        c1: First array of integers to embed (controls x frequency and rotation)
        c2: Second array of integers to embed (controls y frequency and rotation)
        size: Size of the square output array
        freq_range: Factor to control the frequency range (higher values = broader frequency spectrum)
    
    Returns:
        3D numpy array with the 2D Gabor wavelet embeddings, shape (N, size, size) where N is length of c1/c2
    """
    # Convert inputs to numpy arrays if not already
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    print('c1 shape:', c1.shape, 'c2 shape:', c2.shape)
    
    # Create coordinate grid (shared across all patterns)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Set base frequencies based on input constants
    freq_x = (1.01 + c1)[:, None, None] * (size/2)
    freq_y = (1.01 + c2)[:, None, None] * (size/2)
    
    # Calculate rotation angles based on c1 and c2
    c1_cycles = 5  # Number of cycles in c1
    c2_cycles = 7  # Number of cycles in c2
    theta1 = (c1 % c1_cycles)[:, None, None] * (np.pi/c1_cycles)
    theta2 = (c2 % c2_cycles)[:, None, None] * (np.pi/c2_cycles)
    
    # Rotate coordinates for all patterns at once
    X_rot = X[None, :, :] * np.cos(theta1) + Y[None, :, :] * np.sin(theta2)
    Y_rot = -X[None, :, :] * np.sin(theta1) + Y[None, :, :] * np.cos(theta2)
    
    # Gaussian envelope parameters
    sigma_x = 0.4 / freq_range
    sigma_y = 0.4 / freq_range
    
    # Gaussian envelope with rotated coordinates
    gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
    
    # 2D Gabor wavelet with separate x and y frequencies on rotated coordinates
    gabor = gaussian * np.sin(freq_x * X_rot) * np.sin(freq_y * Y_rot)
    
    return gabor

def extract_2const_from_wavelet(pattern, size=32, freq_range=1.0):
    """
    Extract the c1 and c2 values from a Gabor wavelet pattern by analyzing its frequency components.
    
    Args:
        pattern: 2D numpy array containing the Gabor wavelet pattern
        size: Size of the input pattern (must match the size used in embed_2const_wavelet)
        freq_range: Factor controlling the frequency range (must match embed_2const_wavelet)
    
    Returns:
        tuple: (c1, c2) integers that generated the input pattern
    """
    # Create coordinate grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Take 2D FFT of the pattern
    fft_pattern = np.fft.fft2(pattern)
    fft_shifted = np.fft.fftshift(fft_pattern)
    
    # Find the dominant frequencies in x and y directions
    # Sum along each axis to get frequency profiles
    freq_profile_x = np.sum(np.abs(fft_shifted), axis=0)
    freq_profile_y = np.sum(np.abs(fft_shifted), axis=1)
    
    # Find the peak frequencies
    peak_freq_x = np.argmax(freq_profile_x)
    peak_freq_y = np.argmax(freq_profile_y)
    
    # Convert peak frequencies to normalized coordinates (-1 to 1)
    norm_freq_x = (peak_freq_x - size//2) / (size//2)
    norm_freq_y = (peak_freq_y - size//2) / (size//2)
    
    # Extract c1 and c2 from frequencies
    # From the forward function: freq_x = (1.01 + c1) * (size/2)
    c1 = int(round((norm_freq_x * (size/2) - 1.01)))
    c2 = int(round((norm_freq_y * (size/2) - 1.01)))
    
    # Ensure values are within valid ranges
    c1_cycles = 5
    c2_cycles = 7
    c1 = c1 % c1_cycles
    c2 = c2 % c2_cycles
    print('c1:', c1, 'c2:', c2)
    
    return c1, c2