import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import random


def load_mat_data(folder_path, multiple_datafiles=False):
    """
    Load MATLAB v7.3 .mat dataset(s) from a folder.

    - Default keeps single-file behavior.
    - If multiple_datafiles=True, concatenates data across all .mat files.
    """
    start_time = time.time()
    mat_files = [file for file in os.listdir(folder_path) if file.endswith(".mat")]

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in the specified folder: {folder_path}")

    if len(mat_files) > 1 and not multiple_datafiles:
        raise ValueError(
            f"Multiple .mat files found in the specified folder: {folder_path}. "
            "Please ensure there is only one .mat file or set multiple_datafiles=True."
        )

    if not multiple_datafiles:
        data_path = os.path.join(folder_path, mat_files[0])
        with h5py.File(data_path, "r") as file:
            EIGENVALUE_DATA = np.array(file["EIGENVALUE_DATA"], dtype=np.float16)
            EIGENVECTOR_DATA_real = np.array(file["EIGENVECTOR_DATA"]["real"], dtype=np.float16)
            EIGENVECTOR_DATA_imag = np.array(file["EIGENVECTOR_DATA"]["imag"], dtype=np.float16)
            EIGENVECTOR_DATA = EIGENVECTOR_DATA_real + 1j * EIGENVECTOR_DATA_imag
            WAVEVECTOR_DATA = np.array(file["WAVEVECTOR_DATA"], dtype=np.float16)
            const = {key: np.array(file["const"][key]) for key in file["const"]}
            N_struct = np.array(file["N_struct"])
            design_params = np.array(file["design_params"])
            designs = np.array(file["designs"])
            imag_tol = np.array(file["imag_tol"])
            rng_seed_offset = np.array(file["rng_seed_offset"])

        end_time = time.time()
        print(f"Data loaded in {end_time - start_time:.2f} seconds.")
        return {
            "EIGENVALUE_DATA": EIGENVALUE_DATA,
            "EIGENVECTOR_DATA": EIGENVECTOR_DATA,
            "WAVEVECTOR_DATA": WAVEVECTOR_DATA,
            "const": const,
            "N_struct": N_struct,
            "design_params": design_params,
            "designs": designs,
            "imag_tol": imag_tol,
            "rng_seed_offset": rng_seed_offset,
        }

    combined_data = {
        "EIGENVALUE_DATA": [],
        "EIGENVECTOR_DATA": [],
        "WAVEVECTOR_DATA": [],
        "const": {},
        "N_struct": [],
        "design_params": [],
        "designs": [],
        "imag_tol": [],
        "rng_seed_offset": [],
    }

    for mat_file in mat_files:
        data_path = os.path.join(folder_path, mat_file)
        with h5py.File(data_path, "r") as file:
            EIGENVALUE_DATA = np.array(file["EIGENVALUE_DATA"], dtype=np.float16)
            EIGENVECTOR_DATA_real = np.array(file["EIGENVECTOR_DATA"]["real"], dtype=np.float16)
            EIGENVECTOR_DATA_imag = np.array(file["EIGENVECTOR_DATA"]["imag"], dtype=np.float16)
            EIGENVECTOR_DATA = EIGENVECTOR_DATA_real + 1j * EIGENVECTOR_DATA_imag
            WAVEVECTOR_DATA = np.array(file["WAVEVECTOR_DATA"], dtype=np.float16)
            const = {key: np.array(file["const"][key]) for key in file["const"]}
            N_struct = np.array(file["N_struct"])
            design_params = np.array(file["design_params"])
            designs = np.array(file["designs"])
            imag_tol = np.array(file["imag_tol"])
            rng_seed_offset = np.array(file["rng_seed_offset"])

            combined_data["EIGENVALUE_DATA"].append(EIGENVALUE_DATA)
            combined_data["EIGENVECTOR_DATA"].append(EIGENVECTOR_DATA)
            combined_data["WAVEVECTOR_DATA"].append(WAVEVECTOR_DATA)
            for key, value in const.items():
                if key not in combined_data["const"]:
                    combined_data["const"][key] = [value]
                else:
                    combined_data["const"][key].append(value)
            combined_data["N_struct"].append(N_struct)
            combined_data["design_params"].append(design_params)
            combined_data["designs"].append(designs)
            combined_data["imag_tol"].append(imag_tol)
            combined_data["rng_seed_offset"].append(rng_seed_offset)

    combined_data["EIGENVALUE_DATA"] = np.concatenate(combined_data["EIGENVALUE_DATA"], axis=0)
    combined_data["EIGENVECTOR_DATA"] = np.concatenate(combined_data["EIGENVECTOR_DATA"], axis=0)
    combined_data["WAVEVECTOR_DATA"] = np.concatenate(combined_data["WAVEVECTOR_DATA"], axis=0)
    combined_data["N_struct"] = np.concatenate(combined_data["N_struct"], axis=0)
    combined_data["design_params"] = np.concatenate(combined_data["design_params"], axis=0)
    combined_data["designs"] = np.concatenate(combined_data["designs"], axis=0)
    combined_data["imag_tol"] = np.concatenate(combined_data["imag_tol"], axis=0)
    combined_data["rng_seed_offset"] = np.concatenate(combined_data["rng_seed_offset"], axis=0)
    for key, value_list in combined_data["const"].items():
        combined_data["const"][key] = np.concatenate(value_list, axis=0)

    end_time = time.time()
    print(f"Data loaded and combined from {len(mat_files)} files in {end_time - start_time:.2f} seconds.")
    return combined_data


def split_array(arr, dim):
    shape = arr.shape
    if dim < 0 or dim >= len(shape):
        raise ValueError(f"Invalid dimension: {dim}. Array has {len(shape)} dimensions.")
    if shape[dim] % 2 != 0:
        raise ValueError(f"Length of dimension {dim} must be even.")

    slices = [slice(None)] * len(shape)
    slices[dim] = slice(None, None, 2)
    arr1 = arr[tuple(slices)]
    slices[dim] = slice(1, None, 2)
    arr2 = arr[tuple(slices)]
    return arr1, arr2


def extract_data(data_path, multiple_datafiles=False):
    i_dim = 3
    data = load_mat_data(data_path, multiple_datafiles=multiple_datafiles)

    designs = np.array(data["designs"])
    design_params = np.array(data["design_params"])
    n_designs = designs.shape[0]
    n_panes = designs.shape[1]
    design_res = designs.shape[2]
    const = {key: np.array(data["const"][key]) for key in data["const"]}

    WAVEVECTOR_DATA = data["WAVEVECTOR_DATA"]
    n_dim = WAVEVECTOR_DATA.shape[1]
    WAVEVECTOR_DATA = WAVEVECTOR_DATA.transpose(0, 2, 1)
    n_wavevectors = WAVEVECTOR_DATA.shape[1]
    if np.any(np.iscomplex(WAVEVECTOR_DATA)):
        print("WAVEVECTOR_DATA array contains complex-valued elements.")
    WAVEFORM_DATA = wavevectors_to_spatial(WAVEVECTOR_DATA, design_res, length_scale=const["a"], plot_sample=False)

    EIGENVALUE_DATA = np.array(data["EIGENVALUE_DATA"]).transpose(0, 2, 1)
    n_bands = EIGENVALUE_DATA.shape[2]
    EIGENVECTOR_DATA = np.array(data["EIGENVECTOR_DATA"]).transpose(0, 2, 1, 3)
    EIGENVECTOR_DATA_x, EIGENVECTOR_DATA_y = split_array(EIGENVECTOR_DATA, i_dim)

    N_struct = data["N_struct"]
    imag_tol = data["imag_tol"]
    rng_seed_offset = data["rng_seed_offset"]

    EIGENVECTOR_DATA_x = EIGENVECTOR_DATA_x.reshape(
        n_designs, n_wavevectors, n_bands, design_res, design_res, order="C"
    )
    EIGENVECTOR_DATA_y = EIGENVECTOR_DATA_y.reshape(
        n_designs, n_wavevectors, n_bands, design_res, design_res, order="C"
    )

    print(
        f"n_designs: {n_designs}, n_panes: {n_panes}, design_res: {design_res}, "
        f"d_design: {n_dim}, dispersion_bands: {n_bands}, rng_seed_offset: {rng_seed_offset}"
    )
    print("EIGENVALUE_DATA shape:", EIGENVALUE_DATA.shape)
    print("EIGENVECTOR_DATA shape:", EIGENVECTOR_DATA.shape)
    print("EIGENVECTOR_DATA_x shape:", EIGENVECTOR_DATA_x.shape)
    print("EIGENVECTOR_DATA_y shape:", EIGENVECTOR_DATA_y.shape)
    print("WAVEVECTOR_DATA shape:", WAVEVECTOR_DATA.shape)
    print("WAVEFORM_DATA shape:", WAVEFORM_DATA.shape)
    print("designs shape:", designs.shape)
    print("design_params shape:", design_params.shape)
    print("const shape:", {key: const[key].shape for key in const})

    return (
        designs,
        design_params,
        n_designs,
        n_panes,
        design_res,
        WAVEVECTOR_DATA,
        WAVEFORM_DATA,
        n_dim,
        n_wavevectors,
        EIGENVALUE_DATA,
        n_bands,
        EIGENVECTOR_DATA_x,
        EIGENVECTOR_DATA_y,
        const,
        N_struct,
        imag_tol,
        rng_seed_offset,
    )


def plot_geometry(sample_geometry, sample_index):
    fig, ax = plt.subplots()
    ax.imshow(sample_geometry)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Geometry #{sample_index}")
    plt.show()


def visualize_sample(input_tensor, output_tensor, target_tensor, diffs=True):
    """
    Visualize input and output tensors from a single sample.

    Args:
        input_tensor: Tensor of shape (3, H, W) containing input components
        output_tensor: Tensor of shape (C, H, W) containing output components
        (optional) target_tensor: Tensor of shape (C, H, W) containing target components
    """
    input_titles = ["Geometry", "Wavevector (Encoded)", "Band (Encoded)"]

    # Create figure for input components
    fig1 = plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        tensor_data = input_tensor[i].abs()
        if tensor_data.dtype in [torch.float8_e4m3fn, torch.float16]:
            tensor_data = tensor_data.float()
        im = plt.imshow(tensor_data.numpy())
        plt.colorbar(im)
        plt.title(input_titles[i])
    plt.tight_layout()
    plt.show()

    channel_titles_default = [
        "Eigfreq (Encoded)",
        "Eigvec (x real)",
        "Eigvec (x imag)",
        "Eigvec (y real)",
        "Eigvec (y imag)",
    ]

    if target_tensor is not None:
        num_targets = target_tensor.shape[0]
        num_outputs = output_tensor.shape[0]
        if num_targets != num_outputs:
            raise ValueError("Target and output tensors must have the same number of channels for matched plotting.")

        n_rows = 3 if diffs else 2
        fig, axs = plt.subplots(n_rows, num_targets, figsize=(4 * num_targets, 4 * n_rows))
        if num_targets == 1:
            axs = np.expand_dims(axs, axis=1)
            if n_rows == 1:
                axs = np.expand_dims(axs, axis=0)

        channel_titles = channel_titles_default[:num_targets]
        if num_targets > len(channel_titles_default):
            channel_titles.extend([f"Channel {i + 1}" for i in range(len(channel_titles_default), num_targets)])

        # Shared min/max per channel between target and output rows.
        vmins = []
        vmaxs = []
        for i in range(num_targets):
            tgt_data = target_tensor[i]
            out_data = output_tensor[i]
            if tgt_data.dtype in [torch.float8_e4m3fn, torch.float16]:
                tgt_data = tgt_data.float()
            if out_data.dtype in [torch.float8_e4m3fn, torch.float16]:
                out_data = out_data.float()
            vmins.append(min(tgt_data.min().item(), out_data.min().item()))
            vmaxs.append(max(tgt_data.max().item(), out_data.max().item()))

        # Row 1: targets
        for i in range(num_targets):
            ax = axs[0, i]
            tensor_data = target_tensor[i]
            if tensor_data.dtype in [torch.float8_e4m3fn, torch.float16]:
                tensor_data = tensor_data.float()
            im = ax.imshow(tensor_data.numpy(), vmin=vmins[i], vmax=vmaxs[i])
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Target {channel_titles[i]}")

        # Row 2: outputs
        for i in range(num_outputs):
            ax = axs[1, i]
            tensor_data = output_tensor[i]
            if tensor_data.dtype in [torch.float8_e4m3fn, torch.float16]:
                tensor_data = tensor_data.float()
            im = ax.imshow(tensor_data.numpy(), vmin=vmins[i], vmax=vmaxs[i])
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Output {channel_titles[i]}")

        # Row 3: absolute differences (with same per-channel scales as rows 1-2)
        if diffs:
            for i in range(num_targets):
                ax = axs[2, i]
                tgt_data = target_tensor[i]
                out_data = output_tensor[i]
                if tgt_data.dtype in [torch.float8_e4m3fn, torch.float16]:
                    tgt_data = tgt_data.float()
                if out_data.dtype in [torch.float8_e4m3fn, torch.float16]:
                    out_data = out_data.float()
                diff_data = torch.abs(out_data - tgt_data)
                im = ax.imshow(diff_data.numpy())
                plt.colorbar(im, ax=ax)
                ax.set_title(f"Diff {channel_titles[i]}")

        plt.tight_layout()
        plt.show()
    else:
        # Fallback: outputs-only view
        num_outputs = output_tensor.shape[0]
        channel_titles = channel_titles_default[:num_outputs]
        if num_outputs > len(channel_titles_default):
            channel_titles.extend([f"Channel {i + 1}" for i in range(len(channel_titles_default), num_outputs)])
        fig2 = plt.figure(figsize=(4 * num_outputs, 4))
        for i in range(num_outputs):
            plt.subplot(1, num_outputs, i + 1)
            tensor_data = output_tensor[i].abs()
            if tensor_data.dtype in [torch.float8_e4m3fn, torch.float16]:
                tensor_data = tensor_data.float()
            im = plt.imshow(tensor_data.numpy())
            plt.colorbar(im)
            plt.title(f"Output {channel_titles[i]}")
        plt.tight_layout()
        plt.show()


def plot_eigenvectors(sample_eigenvector_x, sample_eigenvector_y, unify_scales=True):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    im00 = axs[0, 0].imshow(np.real(sample_eigenvector_x), cmap="viridis")
    axs[0, 0].set_title("x displacement field real", pad=1)
    axs[0, 0].axis("off")
    im10 = axs[1, 0].imshow(np.imag(sample_eigenvector_x), cmap="viridis")
    axs[1, 0].set_title("x displacement field imag", pad=1)
    axs[1, 0].axis("off")
    im01 = axs[0, 1].imshow(np.real(sample_eigenvector_y), cmap="viridis")
    axs[0, 1].set_title("y displacement field real", pad=1)
    axs[0, 1].axis("off")
    im11 = axs[1, 1].imshow(np.imag(sample_eigenvector_y), cmap="viridis")
    axs[1, 1].set_title("y displacement field imag", pad=1)
    axs[1, 1].axis("off")

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
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(ax.images[0], cax=cax)
            cax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()


def wavevectors_to_spatial(wavevectors, design_res, length_scale, amplitude=1.0, phase=0.0, plot_sample=False):
    """
    Vectorized implementation (from NO_utils.py) for performance.
    """
    N = wavevectors.shape[0]
    W = wavevectors.shape[1]
    x = np.linspace(-length_scale / 2, length_scale / 2, design_res)
    y = np.linspace(-length_scale / 2, length_scale / 2, design_res)
    X, Y = np.meshgrid(x, y)
    k_x = wavevectors[..., 0][..., None, None]
    k_y = wavevectors[..., 1][..., None, None]
    spatial_waves = amplitude * np.cos(k_x * X + k_y * Y + phase)

    if plot_sample:
        sample_n = random.randint(0, N - 1)
        sample_w = random.randint(0, W - 1)
        plt.figure(figsize=(6, 6))
        plt.contourf(spatial_waves[sample_n], cmap="viridis")
        plt.colorbar(label="Wave Amplitude")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(
            f"Spatial Wave: Sample {sample_n}, Wavevector {sample_w} with "
            f"$k_x$={wavevectors[sample_n, sample_w, 0]} $m^{{-1}}$, "
            f"$k_y$={wavevectors[sample_n, sample_w, 1]} $m^{{-1}}$"
        )
        plt.show()

    print("Spatial waves shape:", spatial_waves.shape)
    return spatial_waves


def const_to_spatial(test_band, design_res, plot_result=True, scaling_factor=1.0):
    x = np.linspace(-1 / 2, 1 / 2, design_res)
    y = np.linspace(-1 / 2, 1 / 2, design_res)
    X, Y = np.meshgrid(x, y)
    constant_array = scaling_factor * np.sin(test_band * np.pi * X) * np.sin(test_band * np.pi * Y)
    fft_result = np.fft.fft2(constant_array)
    fft_result_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.abs(fft_result_shifted)
    if plot_result:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(constant_array, cmap="viridis")
        plt.colorbar(label="Magnitude")
        plt.title(f"Spatial Representation of {test_band}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_spectrum, cmap="viridis")
        plt.colorbar(label="Magnitude")
        plt.title(f"2D FFT Magnitude Spectrum of {test_band}")
        plt.xlabel("Frequency X")
        plt.ylabel("Frequency Y")
        plt.tight_layout()
        plt.show()
    return constant_array, magnitude_spectrum


def embed_integer_wavelet(c, size=32, freq_range=2.0):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    c = np.asarray(c)
    base_frequency = (1.0 + np.abs(c))[:, None, None] * (size / 8)
    theta = (c % 8)[:, None, None] * (size / 16)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    X_theta = X[None, :, :] * cos_theta + Y[None, :, :] * sin_theta
    Y_theta = -X[None, :, :] * sin_theta + Y[None, :, :] * cos_theta
    sigma_x = 0.3 / freq_range
    sigma_y = 0.3 / freq_range
    gaussian = np.exp(-(X_theta**2 / (2 * sigma_x**2) + Y_theta**2 / (2 * sigma_y**2)))
    gabor = gaussian * np.cos(base_frequency * X_theta)
    return gabor


def embed_2const_wavelet(c1, c2, size=32, freq_range=1.0, verbose=True):
    """
    Keep NO_utils_multiple mapping (1.01 offset) to match most recent scripts.
    """
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    if verbose:
        print("c1 shape:", c1.shape, "c2 shape:", c2.shape)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    freq_x = (1.01 + c1)[:, None, None] * (size / 2)
    freq_y = (1.01 + c2)[:, None, None] * (size / 2)
    c1_cycles = 5
    c2_cycles = 7
    theta1 = (c1 % c1_cycles)[:, None, None] * (np.pi / c1_cycles)
    theta2 = (c2 % c2_cycles)[:, None, None] * (np.pi / c2_cycles)
    X_rot = X[None, :, :] * np.cos(theta1) + Y[None, :, :] * np.sin(theta2)
    Y_rot = -X[None, :, :] * np.sin(theta1) + Y[None, :, :] * np.cos(theta2)
    sigma_x = 0.4 / freq_range
    sigma_y = 0.4 / freq_range
    gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
    gabor = gaussian * np.sin(freq_x * X_rot) * np.sin(freq_y * Y_rot)
    return gabor


def extract_2const_from_wavelet(pattern, size=32, freq_range=1.0, verbose=True):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    fft_pattern = np.fft.fft2(pattern)
    fft_shifted = np.fft.fftshift(fft_pattern)
    freq_profile_x = np.sum(np.abs(fft_shifted), axis=0)
    freq_profile_y = np.sum(np.abs(fft_shifted), axis=1)
    peak_freq_x = np.argmax(freq_profile_x)
    peak_freq_y = np.argmax(freq_profile_y)
    norm_freq_x = (peak_freq_x - size // 2) / (size // 2)
    norm_freq_y = (peak_freq_y - size // 2) / (size // 2)
    c1 = int(round((norm_freq_x * (size / 2) - 1.01)))
    c2 = int(round((norm_freq_y * (size / 2) - 1.01)))
    c1_cycles = 5
    c2_cycles = 7
    c1 = c1 % c1_cycles
    c2 = c2 % c2_cycles
    if verbose:
        print("c1:", c1, "c2:", c2)
    return c1, c2


def embed_eigenfrequency_wavelet(
    s,
    size=32,
    ef_min=1.0,
    ef_max=8000.0,
    k_min=2.0,
    k_max=16.0,
    gamma=1,
    phi=0.0,
    sigma_factor=8,
    theta_min=np.pi / 30,
    theta_max=np.pi / 2 - np.pi / 30,
):
    if s <= 0:
        raise ValueError("s must be positive (log-domain mapping).")

    ln_s = np.log(s)
    ln_min = np.log(ef_min)
    ln_max = np.log(ef_max)
    log_range = ln_max - ln_min
    k_range = k_max - k_min
    log_per_unit_k = log_range / k_range if k_range > 0 else 0.0
    t = np.clip((ln_s - ln_min) / (ln_max - ln_min), 0.0, 1.0) if ln_max != ln_min else 0.0
    k = k_min + t * k_range

    if log_per_unit_k > 0:
        band_fraction = ((ln_s - ln_min) % log_per_unit_k) / log_per_unit_k
        theta = theta_min + band_fraction * (theta_max - theta_min)
    else:
        theta = theta_min

    coords = np.linspace(-size // 2, size // 2 - 1, size)
    X, Y = np.meshgrid(coords, coords)
    X_theta = X * np.cos(theta) + Y * np.sin(theta)
    Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
    sigma_x = sigma_factor
    sigma_y = sigma_x * gamma
    freq = 2.0 * np.pi * k / size
    gaussian = np.exp(-0.5 * ((X_theta**2) / sigma_x**2 + (Y_theta**2) / sigma_y**2))
    carrier = np.cos(freq * X_theta + phi)
    return gaussian * carrier, k, theta


def encode_eigenfrequency_uniform(s, size=32):
    """
    Encode eigenfrequency as a uniform float16 patch: every pixel is ln(s)/100.

    The input is cast to float16 first; ``log`` and division follow in NumPy's
    float16 arithmetic (wider internal promotion, if any, is left to NumPy).

    Parameters
    ----------
    s : float, or array_like of floats
        Eigenfrequency values; must be strictly positive (log domain).
    size : int, default 32
        Spatial extent of the square patch.

    Returns
    -------
    ndarray, dtype float16
        Shape ``(size, size)`` if ``s`` is scalar (0-d after casting), otherwise
        ``s.shape + (size, size)`` with one patch per element of ``s``.
    """
    s_f16 = np.asarray(s, dtype=np.float16)
    if np.any(s_f16 <= 0):
        raise ValueError("s must be positive (log domain).")

    pixel = (np.log(s_f16) / np.float16(100.0)).astype(np.float16)
    tail = (size, size)
    if pixel.ndim == 0:
        return np.full(tail, pixel, dtype=np.float16)
    out_shape = pixel.shape + tail
    return np.broadcast_to(pixel[..., np.newaxis, np.newaxis], out_shape).copy()


def encode_eigenfrequency_uniform_torch(s: torch.Tensor, size: int = 32) -> torch.Tensor:
    """
    Torch-only analogue of :func:`encode_eigenfrequency_uniform`: float16 patches of ``ln(s)/100``.

    Parameters
    ----------
    s : torch.Tensor
        Eigenfrequency values; must be strictly positive (after any clamping by the caller).
    size : int
        Square patch side length.

    Returns
    -------
    torch.Tensor
        ``dtype=float16``, shape ``s.shape + (size, size)`` on the same device as ``s``.
    """
    if not torch.is_tensor(s):
        raise TypeError("s must be a torch.Tensor")
    device = s.device
    s_f16 = s.to(device=device, dtype=torch.float16)
    if bool((s_f16 <= 0).any().item()):
        raise ValueError("s must be positive (log domain).")
    hundred = torch.tensor(100.0, device=device, dtype=torch.float16)
    pixel = (torch.log(s_f16) / hundred).to(torch.float16)
    h, w = size, size
    if pixel.ndim == 0:
        return torch.full((h, w), pixel, dtype=torch.float16, device=device)
    return pixel.unsqueeze(-1).unsqueeze(-1).expand(*pixel.shape, h, w).contiguous()


def decode_eigenfrequency_uniform(image):
    """
    Decode eigenfrequency from a patch produced by :func:`encode_eigenfrequency_uniform`.

    Reads the top-left pixel (all pixels are identical for valid encodings).
    Recovers ``s`` as ``exp(pixel * 100)`` in float16, then returns ``float64``
    scalars/arrays for downstream use.

    If the spatial mean of a patch differs from ``patch[0, 0]`` beyond a small
    tolerance, prints a warning to stdout (non-uniform or corrupted input).

    Parameters
    ----------
    image : array_like
        Last two dimensions are height and width (e.g. ``(size, size)`` or
        ``(..., size, size)``).

    Returns
    -------
    ndarray or scalar
        Shape ``()`` if ``image`` is 2-D, otherwise ``image.shape[:-2]``,
        dtype ``float64``.
    """
    arr = np.asarray(image, dtype=np.float16)
    if arr.ndim < 2:
        raise ValueError("image must be at least 2-D (..., height, width).")
    ref = arr[..., 0, 0]
    avg = np.mean(arr, axis=(-2, -1))
    if not np.allclose(avg, ref, rtol=1e-3, atol=1e-4):
        print(
            "Warning: decode_eigenfrequency_uniform: patch mean differs from pixel [0, 0]; "
            "input may not be a valid uniform encoding.",
            flush=True,
        )
    pixel = ref
    ln_s_f16 = (pixel * np.float16(100.0)).astype(np.float16)
    s_f16 = np.exp(ln_s_f16).astype(np.float16)
    return s_f16.astype(np.float64)


def extract_eigenfrequency_from_wavelet(
    image,
    size=32,
    ef_min=1.0,
    ef_max=8000.0,
    k_min=2.0,
    k_max=16.0,
    theta_min=np.pi / 30,
    theta_max=np.pi / 2 - np.pi / 30,
):
    CENTROID_RADIUS = 3
    CENTROID_POWER = 2
    ln_min = np.log(ef_min)
    ln_max = np.log(ef_max)
    log_range = ln_max - ln_min
    k_range = k_max - k_min
    log_per_unit_k = log_range / k_range if k_range > 0 else 0.0

    image_centered = image - np.mean(image)
    F = np.fft.fft2(image_centered)
    F_mag = np.abs(np.fft.fftshift(F))
    center = size // 2

    hp = np.zeros_like(F_mag)
    hp[:center, :] = F_mag[:center, :]
    hp[center, center + 1 :] = F_mag[center, center + 1 :]
    peak_idx = np.unravel_index(np.argmax(hp), hp.shape)
    peak_kx = peak_idx[1] - center
    peak_ky = peak_idx[0] - center
    peak_row, peak_col = peak_idx[0], peak_idx[1]

    sym_row = 2 * center - peak_row
    sym_col = 2 * center - peak_col
    sum_w = sum_kx = sum_ky = 0.0
    r_int = int(np.ceil(CENTROID_RADIUS))
    for dr in range(-r_int, r_int + 1):
        for dc in range(-r_int, r_int + 1):
            if dr * dr + dc * dc > CENTROID_RADIUS * CENTROID_RADIUS:
                continue
            r, c = peak_row + dr, peak_col + dc
            if 0 <= r < size and 0 <= c < size:
                d_main_sq = dr * dr + dc * dc
                d_sym_sq = (r - sym_row) ** 2 + (c - sym_col) ** 2
                if d_sym_sq < d_main_sq:
                    continue
                w = F_mag[r, c] ** CENTROID_POWER
                sum_w += w
                sum_kx += w * (c - center)
                sum_ky += w * (r - center)

    if sum_w > 0:
        kx_ref = sum_kx / sum_w
        ky_ref = sum_ky / sum_w
    else:
        kx_ref, ky_ref = float(peak_kx), float(peak_ky)

    k_extracted = np.sqrt(kx_ref**2 + ky_ref**2)
    theta_extracted = np.arctan2(ky_ref, kx_ref) % np.pi

    t = np.clip((k_extracted - k_min) / k_range, 0.0, 1.0) if k_range > 0 else 0.0
    ln_s_approx = ln_min + t * log_range
    if log_per_unit_k > 0:
        theta_range = theta_max - theta_min
        band_fraction = np.clip((theta_extracted - theta_min) / theta_range, 0.0, 1.0)
        log_position_in_band = band_fraction * log_per_unit_k
        band = int(np.floor((ln_s_approx - ln_min) / log_per_unit_k))
        best_ln_s = None
        best_dist = np.inf
        for b in [band - 1, band, band + 1]:
            cand = ln_min + b * log_per_unit_k + log_position_in_band
            dist = abs(cand - ln_s_approx)
            if dist < best_dist:
                best_dist = dist
                best_ln_s = cand
        ln_s = best_ln_s
    else:
        ln_s = ln_s_approx

    ln_s = np.clip(ln_s, ln_min, ln_max)
    s = np.exp(ln_s)
    return s, k_extracted, theta_extracted
