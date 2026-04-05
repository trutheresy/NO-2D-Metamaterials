"""
Random-index diagnostic figures for I3O1 / I3O4 / I3O5 surrogate models: inputs, ground truth, predictions.

I3O5: 3x5 grid, outputs in columns 0–4.
I3O4: same outer grid; displacement channels in columns 0–3; column 4 blank on output rows.
I3O1: same outer grid; single eigen channel in the center column (2); other output columns blank.
Each panel has its own colorbar.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


DEFAULT_INPUT_LABELS: tuple[str, ...] = (
    "Geometry",
    "Wavevector image",
    "Band FFT image",
)

DEFAULT_OUTPUT_LABELS: tuple[str, ...] = (
    "eigenfrequency",
    "disp_x_real",
    "disp_x_imag",
    "disp_y_real",
    "disp_y_imag",
)


def _amp_context(device: torch.device, amp_mode: str):
    if device.type != "cuda" or amp_mode == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    if amp_mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def _to_hw_float32(ch: torch.Tensor) -> np.ndarray:
    """Single channel [H, W] -> float32 numpy for imshow."""
    x = ch.detach().float().cpu().numpy()
    if x.ndim != 2:
        raise ValueError(f"expected 2D channel, got shape {tuple(x.shape)}")
    return x


def _imshow_panel(ax, data: np.ndarray, title: str) -> None:
    vmin, vmax = float(np.min(data)), float(np.max(data))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1e-6
    im = ax.imshow(data, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_i3o5_diagnostic_panel(
    inputs_hw3: np.ndarray,
    target_hw5: np.ndarray,
    pred_hw5: np.ndarray,
    *,
    input_labels: Sequence[str] = DEFAULT_INPUT_LABELS,
    output_labels: Sequence[str] = DEFAULT_OUTPUT_LABELS,
    suptitle: str | None = None,
    figsize: tuple[float, float] = (16.0, 9.6),
    dpi: int = 120,
) -> plt.Figure:
    """
    Build one 3x5 figure: inputs (centered), GT outputs, predicted outputs.

    Args:
        inputs_hw3: shape (3, H, W)
        target_hw5: shape (5, H, W)
        pred_hw5: shape (5, H, W)
    """
    if inputs_hw3.shape[0] != 3:
        raise ValueError(f"inputs must have 3 channels, got {inputs_hw3.shape}")
    if target_hw5.shape[0] != 5 or pred_hw5.shape[0] != 5:
        raise ValueError(f"targets/preds must have 5 channels, got {target_hw5.shape}, {pred_hw5.shape}")

    fig, axes = plt.subplots(3, 5, figsize=figsize, dpi=dpi)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)

    # Row 0: blank | in0 | in1 | in2 | blank
    axes[0, 0].axis("off")
    for j in range(3):
        _imshow_panel(axes[0, j + 1], inputs_hw3[j], input_labels[j])
    axes[0, 4].axis("off")

    for j in range(5):
        _imshow_panel(axes[1, j], target_hw5[j], output_labels[j])
        _imshow_panel(axes[2, j], pred_hw5[j], f"{output_labels[j]} (predicted)")

    fig.tight_layout()
    return fig


def plot_i3on_diagnostic_panel(
    inputs_hw3: np.ndarray,
    target_hwc: np.ndarray,
    pred_hwc: np.ndarray,
    *,
    output_labels: Sequence[str],
    suptitle: str | None = None,
    figsize: tuple[float, float] = (16.0, 9.6),
    dpi: int = 120,
) -> plt.Figure:
    """
    Same outer layout as I3O5 (3x5): row 0 centers three input channels; rows 1–2 show outputs.

    - 4 channels: displacement stack in columns 0–3; column 4 blank.
    - 1 channel: eigenfrequency in the center column (index 2); other output columns blank.
    """
    c = int(target_hwc.shape[0])
    if pred_hwc.shape[0] != c:
        raise ValueError(f"target/pred channel mismatch: {target_hwc.shape} vs {pred_hwc.shape}")
    if len(output_labels) != c:
        raise ValueError(f"expected {c} output_labels, got {len(output_labels)}")

    fig, axes = plt.subplots(3, 5, figsize=figsize, dpi=dpi)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)

    axes[0, 0].axis("off")
    for j in range(3):
        _imshow_panel(axes[0, j + 1], inputs_hw3[j], DEFAULT_INPUT_LABELS[j])
    axes[0, 4].axis("off")

    for j in range(5):
        axes[1, j].axis("off")
        axes[2, j].axis("off")

    if c == 1:
        jc = 2
        _imshow_panel(axes[1, jc], target_hwc[0], output_labels[0])
        _imshow_panel(axes[2, jc], pred_hwc[0], f"{output_labels[0]} (predicted)")
    elif c == 4:
        for j in range(4):
            _imshow_panel(axes[1, j], target_hwc[j], output_labels[j])
            _imshow_panel(axes[2, j], pred_hwc[j], f"{output_labels[j]} (predicted)")
    else:
        raise ValueError(f"plot_i3on_diagnostic_panel supports c=1 or c=4, got c={c}")

    fig.tight_layout()
    return fig


@torch.no_grad()
def save_random_test_diagnostic_panels(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    out_dir: Path,
    *,
    epoch: int,
    n_samples: int = 10,
    amp_mode: str = "none",
    seed: int | None = None,
    input_labels: Sequence[str] = DEFAULT_INPUT_LABELS,
    output_labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = (16.0, 9.6),
    dpi: int = 120,
) -> list[Path]:
    """
    Draw ``n_samples`` random panels from ``dataset`` using the current ``model`` weights.

    Saves one PNG per sample under ``out_dir`` (directory is created). Filenames include epoch
    and global dataset index.

    Supports target tensors with 1, 4, or 5 output channels (I3O1 / I3O4 / I3O5).

    Returns:
        Paths to written PNG files.
    """
    n = len(dataset)
    if n == 0:
        return []
    k = min(int(n_samples), n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    written: list[Path] = []

    for rank, idx in enumerate(indices):
        x, y = dataset[int(idx)]
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"bad input shape at idx={idx}: {tuple(x.shape)}")
        if y.ndim != 3 or y.shape[0] not in (1, 4, 5):
            raise ValueError(f"bad target shape at idx={idx}: {tuple(y.shape)}")

        c = int(y.shape[0])
        if output_labels is None:
            if c == 5:
                labels: Sequence[str] = DEFAULT_OUTPUT_LABELS
            elif c == 4:
                labels = DEFAULT_OUTPUT_LABELS[1:5]
            elif c == 1:
                labels = (DEFAULT_OUTPUT_LABELS[0],)
            else:
                labels = ()
        else:
            labels = output_labels
            if len(labels) != c:
                raise ValueError(f"output_labels length {len(labels)} != channels {c}")

        xb = x.to(device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
        with _amp_context(device, amp_mode):
            pred = model(xb)
        pred = pred.squeeze(0).float().cpu()
        y_cpu = y.float().cpu()

        inp_np = np.stack([_to_hw_float32(x[ch_i]) for ch_i in range(3)], axis=0)
        tgt_np = np.stack([_to_hw_float32(y_cpu[ch_i]) for ch_i in range(c)], axis=0)
        prd_np = np.stack([_to_hw_float32(pred[ch_i]) for ch_i in range(c)], axis=0)

        if c == 5:
            fig = plot_i3o5_diagnostic_panel(
                inp_np,
                tgt_np,
                prd_np,
                input_labels=input_labels,
                output_labels=labels,
                suptitle=f"epoch {epoch} | dataset index {int(idx)}",
                figsize=figsize,
                dpi=dpi,
            )
        else:
            fig = plot_i3on_diagnostic_panel(
                inp_np,
                tgt_np,
                prd_np,
                output_labels=labels,
                suptitle=f"epoch {epoch} | dataset index {int(idx)}",
                figsize=figsize,
                dpi=dpi,
            )
        path = out_dir / f"epoch_{epoch:03d}_sample_{rank:02d}_idx_{int(idx)}.png"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        written.append(path)

    return written
