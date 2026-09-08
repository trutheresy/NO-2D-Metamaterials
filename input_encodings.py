"""Input wavevector/band encoding file map shared by training and inference.

Canonical names: ``wavelet``, ``sinusoidal``, ``constant``.
``uniform`` is an alias of ``constant`` (constant-field / uniform-field encoding).
"""

from __future__ import annotations

INPUT_ENCODING_FILES = {
    "wavelet": {
        "inputs": "inputs.pt",
        "waveforms": "waveforms_full.pt",
        "bands": "band_fft_full.pt",
    },
    "sinusoidal": {
        "inputs": "inputs_sinusoidal.pt",
        "waveforms": "waveforms_sinusoidal_full.pt",
        "bands": "band_sinusoidal_full.pt",
    },
    "constant": {
        "inputs": "inputs_constant.pt",
        "waveforms": "waveforms_constant_full.pt",
        "bands": "band_constant_full.pt",
    },
}
INPUT_ENCODING_ALIASES = {
    "uniform": "constant",  # user/manuscript synonym for constant-field inputs
}
INPUT_ENCODING_IN_CHANNELS = {
    "wavelet": 3,
    "sinusoidal": 3,
    "constant": 4,  # [geo, kx, ky, band]
}


def normalize_input_encoding(name: str) -> str:
    n = name.strip().lower()
    n = INPUT_ENCODING_ALIASES.get(n, n)
    if n not in INPUT_ENCODING_FILES:
        raise ValueError(
            f"Unknown input_encoding: {name!r}. "
            f"Supported: {', '.join(sorted(INPUT_ENCODING_FILES))} "
            f"(aliases: {', '.join(sorted(INPUT_ENCODING_ALIASES))})."
        )
    return n


def input_encoding_in_channels(encoding: str) -> int:
    return INPUT_ENCODING_IN_CHANNELS[normalize_input_encoding(encoding)]


def input_encoding_filenames(encoding: str) -> dict[str, str]:
    return dict(INPUT_ENCODING_FILES[normalize_input_encoding(encoding)])


def assemble_model_input(geometry, waveform, band, encoding: str = "wavelet"):
    """
    Stack one sample's model input channels.

    wavelet/sinusoidal -> (3, H, W) = [geo, Ik, Ib]
    constant/uniform   -> (4, H, W) = [geo, kx, ky, Ib]
    """
    import torch

    enc = normalize_input_encoding(encoding)
    geo = geometry if isinstance(geometry, torch.Tensor) else torch.as_tensor(geometry)
    wave = waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
    bnd = band if isinstance(band, torch.Tensor) else torch.as_tensor(band)
    if enc == "constant":
        if wave.ndim != 3 or int(wave.shape[0]) != 2:
            raise ValueError(
                f"constant encoding expects waveform shape (2, H, W); got {tuple(wave.shape)}"
            )
        return torch.stack([geo, wave[0], wave[1], bnd], dim=0)
    return torch.stack([geo, wave, bnd], dim=0)


INPUT_PANEL_TITLES = {
    "wavelet": ["Geometry", "Wavevector", "Band"],
    "sinusoidal": ["Geometry", "Wavevector", "Band"],
    "constant": ["Geometry", "kx", "ky", "Band"],
}


def input_panel_titles(encoding: str = "wavelet") -> list[str]:
    return list(INPUT_PANEL_TITLES[normalize_input_encoding(encoding)])
