"""
SISSI: Sign-Invariant Structural Similarity Index.

SSIM-like local similarity where only the luminance numerator is sign-invariant:

    SISSI = (2|mu_x||mu_y| + C1)(2 sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))

Contrast/covariance terms match standard SSIM so global sign flips yield ~-1 rather than ~+1.

Default C1/C2 are per-channel constants derived from b_test and c_test pixel scales
(I3O5 uniform eigenfrequency + four displacement channels), using Wang-style
k1=0.01, k2=0.03 with per-channel dynamic range L_c:

    C_i = (k_i * L_c)^2
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Per-channel dynamic range L_c from sampled b_test / c_test targets (uniform ch0,
# outputs.pt ch1-4). See module docstring and DEFAULT_CHANNEL_LABELS.
DEFAULT_CHANNEL_LABELS = (
    "eigenfrequency_uniform",
    "disp_x_real",
    "disp_x_imag",
    "disp_y_real",
    "disp_y_imag",
)
DEFAULT_DATA_RANGE_PER_CHANNEL: tuple[float, ...] = (0.15, 0.07, 0.07, 0.07, 0.07)
DEFAULT_K1 = 0.01
DEFAULT_K2 = 0.03


def stability_constants_from_data_range(
    data_range_per_channel: Tensor | tuple[float, ...] | list[float],
    *,
    k1: float = DEFAULT_K1,
    k2: float = DEFAULT_K2,
) -> tuple[Tensor, Tensor]:
    """Return (C1, C2) tensors shaped [C] with C_i = (k_i * L_c)^2."""
    L = torch.as_tensor(data_range_per_channel, dtype=torch.float32)
    c1 = (k1 * L).square()
    c2 = (k2 * L).square()
    return c1, c2


def default_stability_constants(
    out_channels: int = 5,
    *,
    k1: float = DEFAULT_K1,
    k2: float = DEFAULT_K2,
) -> tuple[Tensor, Tensor]:
    """Default C1/C2 for I3O5 (5 channels) or broadcast scalars for other widths."""
    if out_channels == len(DEFAULT_DATA_RANGE_PER_CHANNEL):
        L = DEFAULT_DATA_RANGE_PER_CHANNEL
    elif out_channels == 1:
        L = (2.0,)  # symmetric signed unit-scale fields in [-1, 1]
    else:
        raise ValueError(
            f"No baked-in data_range for out_channels={out_channels}. "
            "Pass data_range_per_channel explicitly."
        )
    c1, c2 = stability_constants_from_data_range(L, k1=k1, k2=k2)
    return c1, c2


def _gaussian_1d(
    kernel_size: int,
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """1D Gaussian kernel matching ``torchmetrics.functional.image.utils._gaussian``."""
    dist = torch.arange(
        (1 - kernel_size) / 2,
        (1 + kernel_size) / 2,
        dtype=dtype,
        device=device,
    )
    gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2.0)
    return gauss / gauss.sum()


def _gaussian_kernel_2d(
    channels: int,
    kernel_size: tuple[int, int],
    sigma: tuple[float, float],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Depthwise 2D Gaussian kernel, shape [C, 1, kh, kw] (TorchMetrics-compatible)."""
    kh, kw = kernel_size
    sig_h, sig_w = sigma
    g_h = _gaussian_1d(kh, sig_h, dtype, device).unsqueeze(0)
    g_w = _gaussian_1d(kw, sig_w, dtype, device).unsqueeze(0)
    kernel_2d = torch.matmul(g_h.t(), g_w)
    return kernel_2d.expand(channels, 1, kh, kw).contiguous()


def _ssim_window_size(sigma: float) -> int:
    return int(3.5 * sigma + 0.5) * 2 + 1


def kernel_size_from_window_radius(window_radius: int) -> int:
    """Odd kernel side length for a Gaussian window with the given pixel radius."""
    if window_radius <= 0:
        raise ValueError(f"window_radius must be positive, got {window_radius}")
    return 2 * window_radius + 1


def sigma_for_window_radius(window_radius: int) -> float:
    """Gaussian sigma matched to window_radius (int(3.5*sigma+0.5) == window_radius)."""
    return (window_radius - 0.5) / 3.5


def _resolve_window_params(
    *,
    sigma: float,
    gaussian_kernel: bool,
    kernel_size: int,
    window_radius: int | None,
) -> tuple[float, bool, int]:
    if window_radius is not None:
        if window_radius <= 0:
            raise ValueError(f"window_radius must be positive, got {window_radius}")
        return sigma_for_window_radius(window_radius), True, kernel_size_from_window_radius(window_radius)
    return sigma, gaussian_kernel, kernel_size


def _local_stats(
    pred: Tensor,
    target: Tensor,
    *,
    sigma: float,
    gaussian_kernel: bool,
    kernel_size: int,
    window_radius: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Gaussian-window means and variances; pred/target [B,C,H,W]."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(f"Expected pred/target [B,C,H,W] with same shape; got {pred.shape}, {target.shape}")

    b, c, _, _ = pred.shape
    dtype = pred.dtype
    device = pred.device

    sigma, gaussian_kernel, kernel_size = _resolve_window_params(
        sigma=sigma,
        gaussian_kernel=gaussian_kernel,
        kernel_size=kernel_size,
        window_radius=window_radius,
    )

    if gaussian_kernel:
        size = _ssim_window_size(sigma)
        sig = (sigma, sigma)
    else:
        if kernel_size % 2 == 0 or kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        size = kernel_size
        sig = (sigma, sigma)

    pad = size // 2
    if gaussian_kernel:
        kernel = _gaussian_kernel_2d(c, (size, size), sig, dtype, device)
    else:
        kernel = torch.ones((c, 1, size, size), dtype=dtype, device=device) / float(size * size)

    pred_p = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    target_p = F.pad(target, (pad, pad, pad, pad), mode="reflect")

    stacked = torch.cat(
        [pred_p, target_p, pred_p * pred_p, target_p * target_p, pred_p * target_p],
        dim=0,
    )
    conv = F.conv2d(stacked, kernel, groups=c)
    mu_x, mu_y, ex2, ey2, exy = conv.split(b, dim=0)

    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    sigma_x2 = (ex2 - mu_x2).clamp_min(0.0)
    sigma_y2 = (ey2 - mu_y2).clamp_min(0.0)
    sigma_xy = exy - mu_x * mu_y
    return mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy


def ssim_index_map(
    pred: Tensor,
    target: Tensor,
    *,
    c1: Tensor | float,
    c2: Tensor | float,
    sigma: float = 1.5,
    gaussian_kernel: bool = True,
    kernel_size: int = 11,
    window_radius: int | None = None,
) -> Tensor:
    """Standard SSIM index map, shape [B,C,H,W] (same as input; reflect-padded convolution)."""
    sigma, gaussian_kernel, kernel_size = _resolve_window_params(
        sigma=sigma,
        gaussian_kernel=gaussian_kernel,
        kernel_size=kernel_size,
        window_radius=window_radius,
    )
    mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy = _local_stats(
        pred,
        target,
        sigma=sigma,
        gaussian_kernel=gaussian_kernel,
        kernel_size=kernel_size,
        window_radius=window_radius,
    )

    c1_t = c1 if isinstance(c1, Tensor) else torch.as_tensor(c1, device=pred.device, dtype=pred.dtype)
    c2_t = c2 if isinstance(c2, Tensor) else torch.as_tensor(c2, device=pred.device, dtype=pred.dtype)
    if c1_t.ndim == 1:
        c1_t = c1_t.view(1, -1, 1, 1)
    if c2_t.ndim == 1:
        c2_t = c2_t.view(1, -1, 1, 1)

    lum = (2.0 * mu_x * mu_y + c1_t) / (mu_x.square() + mu_y.square() + c1_t)
    cs = (2.0 * sigma_xy + c2_t) / (sigma_x2 + sigma_y2 + c2_t)
    return lum * cs


def sissi_index_map(
    pred: Tensor,
    target: Tensor,
    *,
    c1: Tensor | float,
    c2: Tensor | float,
    sigma: float = 1.5,
    gaussian_kernel: bool = True,
    kernel_size: int = 11,
    window_radius: int | None = None,
) -> Tensor:
    """
    SISSI index map, shape [B,C,H,W] (same as input; reflect-padded convolution).

    Luminance numerator uses |mu_x| and |mu_y|; contrast terms are standard SSIM.
    """
    sigma, gaussian_kernel, kernel_size = _resolve_window_params(
        sigma=sigma,
        gaussian_kernel=gaussian_kernel,
        kernel_size=kernel_size,
        window_radius=window_radius,
    )
    mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy = _local_stats(
        pred,
        target,
        sigma=sigma,
        gaussian_kernel=gaussian_kernel,
        kernel_size=kernel_size,
        window_radius=window_radius,
    )

    c1_t = c1 if isinstance(c1, Tensor) else torch.as_tensor(c1, device=pred.device, dtype=pred.dtype)
    c2_t = c2 if isinstance(c2, Tensor) else torch.as_tensor(c2, device=pred.device, dtype=pred.dtype)
    if c1_t.ndim == 1:
        c1_t = c1_t.view(1, -1, 1, 1)
    if c2_t.ndim == 1:
        c2_t = c2_t.view(1, -1, 1, 1)

    lum = (2.0 * mu_x.abs() * mu_y.abs() + c1_t) / (mu_x.square() + mu_y.square() + c1_t)
    cs = (2.0 * sigma_xy + c2_t) / (sigma_x2 + sigma_y2 + c2_t)
    return lum * cs


def _reduce_index_map(
    index_map: Tensor,
    reduction: Literal["mean", "sum", "none"],
) -> Tensor:
    if reduction == "none":
        return index_map
    per_batch = index_map.flatten(1).mean(dim=1)
    if reduction == "sum":
        return per_batch.sum()
    return per_batch.mean()


def sissi_index(
    pred: Tensor,
    target: Tensor,
    *,
    c1: Tensor | float | None = None,
    c2: Tensor | float | None = None,
    data_range_per_channel: Tensor | tuple[float, ...] | list[float] | None = None,
    k1: float = DEFAULT_K1,
    k2: float = DEFAULT_K2,
    sigma: float = 1.5,
    gaussian_kernel: bool = True,
    kernel_size: int = 11,
    window_radius: int | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Scalar (or per-batch) SISSI index; higher is better."""
    if c1 is None or c2 is None:
        if data_range_per_channel is None:
            c1, c2 = default_stability_constants(pred.shape[1], k1=k1, k2=k2)
        else:
            c1, c2 = stability_constants_from_data_range(data_range_per_channel, k1=k1, k2=k2)
        c1 = c1.to(device=pred.device, dtype=pred.dtype)
        c2 = c2.to(device=pred.device, dtype=pred.dtype)

    index_map = sissi_index_map(
        pred,
        target,
        c1=c1,
        c2=c2,
        sigma=sigma,
        gaussian_kernel=gaussian_kernel,
        kernel_size=kernel_size,
        window_radius=window_radius,
    )
    return _reduce_index_map(index_map, reduction)


class SISSILoss(nn.Module):
    """
    Loss = 1 - SISSI (by default), averaged over batch/channels/spatial locations.

    For I3O5 models (5 output channels), default C1/C2 use per-channel L_c from
    b_test/c_test. Pass data_range_per_channel for other channel counts.
    """

    def __init__(
        self,
        *,
        data_range_per_channel: Tensor | tuple[float, ...] | list[float] | None = None,
        c1: Tensor | float | None = None,
        c2: Tensor | float | None = None,
        k1: float = DEFAULT_K1,
        k2: float = DEFAULT_K2,
        sigma: float = 1.5,
        gaussian_kernel: bool = True,
        kernel_size: int = 11,
        window_radius: int | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.gaussian_kernel = gaussian_kernel
        self.kernel_size = kernel_size
        self.window_radius = window_radius
        self.reduction = reduction

        if c1 is not None and c2 is not None:
            self.register_buffer("c1", torch.as_tensor(c1, dtype=torch.float32))
            self.register_buffer("c2", torch.as_tensor(c2, dtype=torch.float32))
        elif data_range_per_channel is not None:
            c1_buf, c2_buf = stability_constants_from_data_range(
                data_range_per_channel, k1=k1, k2=k2
            )
            self.register_buffer("c1", c1_buf)
            self.register_buffer("c2", c2_buf)
        else:
            # Placeholder; resolved on first forward from channel count.
            self.register_buffer("c1", torch.empty(0))
            self.register_buffer("c2", torch.empty(0))
        self._data_range_per_channel = data_range_per_channel

    def _resolve_constants(self, pred: Tensor) -> tuple[Tensor, Tensor]:
        if self.c1.numel() and self.c2.numel():
            c1 = self.c1.to(device=pred.device, dtype=pred.dtype)
            c2 = self.c2.to(device=pred.device, dtype=pred.dtype)
            if c1.ndim == 0:
                return c1, c2
            if c1.shape[0] != pred.shape[1]:
                raise ValueError(
                    f"SISSILoss configured for {int(c1.shape[0])} channels, got pred with {pred.shape[1]}"
                )
            return c1, c2

        if self._data_range_per_channel is not None:
            c1, c2 = stability_constants_from_data_range(
                self._data_range_per_channel, k1=self.k1, k2=self.k2
            )
        else:
            c1, c2 = default_stability_constants(pred.shape[1], k1=self.k1, k2=self.k2)
        return c1.to(device=pred.device, dtype=pred.dtype), c2.to(device=pred.device, dtype=pred.dtype)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must match; got {pred.shape} vs {target.shape}")
        c1, c2 = self._resolve_constants(pred)
        index = sissi_index(
            pred,
            target,
            c1=c1,
            c2=c2,
            sigma=self.sigma,
            gaussian_kernel=self.gaussian_kernel,
            kernel_size=self.kernel_size,
            window_radius=self.window_radius,
            reduction=self.reduction,
        )
        if self.reduction == "none":
            return 1.0 - index
        return 1.0 - index
