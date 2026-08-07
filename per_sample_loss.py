"""
Per-sample raw-pixel loss between ground-truth fields and model-inference prediction
tensors.

A "sample" is one ``(geometry, wavevector, band)`` entry. By default the loss is
computed on **all five output channels** (prediction channels 0–4: eigenfrequency +
four displacement components), averaging each requested loss criterion over those
channels.

The loss is computed directly on the **raw (encoded) pixel fields** by averaging
per-pixel error over each 32x32 field, then averaging across the selected channels.
No decoding to physical units is performed.

Inputs
------
--dataset-pt-dir : folder with ``eigenfrequency_{uniform|fft}_full.pt`` (truth for channel 0;
                   select with ``--eigen-encoding``)
                   and ``displacements_dataset.pt`` (4 tensors, truth for channels 1–4).
--inference      : dense prediction tensor from ``run_model_inference.py`` with shape
                   ``(n_geom*n_wv*n_bands, out_channels, H, W)`` indexed as
                   ``combined = geom*(n_wv*n_bands) + wave*n_bands + band``.
--channel-groups : one or more of ``freq_ch`` (ch0), ``disp_ch`` (ch1–4 mean),
                   ``all_ch`` (ch0–4 mean). Each group writes files with a
                   ``_<group>`` suffix (e.g. ``..._mae_c_test_freq_ch.npy``).
                   When omitted, ``--channels`` selects a single output set with
                   no group suffix (legacy behavior).
--channels       : comma-separated prediction channel indices to score and average
                   when ``--channel-groups`` is not used (default ``0,1,2,3,4``).

For each requested loss criterion this writes one array file (``.npy``) with five
columns (all **float32**; index columns hold exact integers for the full grid):

    col 0 : combined index   (flat C-order index = geom*(n_wv*n_bands) + wave*n_bands + band)
    col 1 : geometry index
    col 2 : wavevector index
    col 3 : band index
    col 4 : loss (mean over selected channels of the per-channel field mean)

It also prints, per loss, samples at performance percentiles p0/p01/.../p99/p100
(performance = opposite of loss: p99 = best/lowest-loss, p01 = worst/highest-loss),
using nearest-rank order on the sorted per-sample losses so every reported case is a
real sample.

Supported losses (per channel, mean over the H x W field, then mean over channels):
    mae  : mean(|p - t|)
    mse  : mean((p - t)^2)
    rms  : sqrt(mse) = sqrt(mean((p - t)^2))   # per channel, then averaged
    nmae : mean(|p - t|) / (mean(|t|)   + eps_a)   # normalized by mean abs pixel value
    nmse : mean((p - t)^2) / (mean(t^2) + eps_s)   # normalized by mean square pixel value
    nrms : sqrt(nmse) = RMSE / sqrt(mean(t^2) + eps_s)   # per channel, then averaged
(l1 -> mae, l2 -> mse, rmse -> rms.) Default eps_a = 1e-5, eps_s = 1e-5.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from NO_utilities import EIGENFREQUENCY_ENCODING_FILES, eigenfrequency_full_filename, resolve_eigenfrequency_encoding
from output_layout import resolve_script_output_dir
from wave_mode_filters import degenerate_pivot_wave_indices, shear_mode_wave_indices


DEFAULT_SCORING_CHANNELS = [0, 1, 2, 3, 4]
# Backward-compatible alias used by plot_high_loss_samples.py
DEFAULT_DISPLACEMENT_CHANNELS = DEFAULT_SCORING_CHANNELS

CHANNEL_GROUPS: dict[str, list[int]] = {
    "freq_ch": [0],
    "disp_ch": [1, 2, 3, 4],
    "all_ch": [0, 1, 2, 3, 4],
}
SUPPORTED_LOSSES = ("mae", "mse", "rms", "nmae", "nmse", "nrms")
# Inference scoring dtypes (32-bit only): int32 index math; float32 loss accumulation/output.
INDEX_DTYPE = np.int32
LOSS_DTYPE = np.float32


def normalize_loss_name(name: str) -> str:
    n = name.strip().lower()
    if n in ("l1", "mae"):
        return "mae"
    if n in ("l2", "mse"):
        return "mse"
    if n in ("rms", "rmse"):
        return "rms"
    if n in ("nmae", "nl1"):
        return "nmae"
    if n in ("nmse", "nl2"):
        return "nmse"
    if n in ("nrms", "nrmse"):
        return "nrms"
    raise ValueError(
        f"Unsupported loss {name!r}. Supported: mae, mse, rms, nmae, nmse, nrms "
        f"(aliases l1, l2, rmse, nl1, nl2, nrmse)."
    )


def parse_channel_groups(names: list[str]) -> list[tuple[str, list[int]]]:
    """Return ordered (group_name, channel_indices) for each requested group."""
    if not names:
        raise ValueError("--channel-groups requires at least one of: freq_ch, disp_ch, all_ch.")
    seen: set[str] = set()
    out: list[tuple[str, list[int]]] = []
    for raw in names:
        key = raw.strip().lower()
        if key not in CHANNEL_GROUPS:
            raise ValueError(
                f"Unknown channel group {raw!r}. Supported: {', '.join(CHANNEL_GROUPS)}."
            )
        if key in seen:
            continue
        seen.add(key)
        out.append((key, list(CHANNEL_GROUPS[key])))
    return out


def parse_channels(spec: str) -> list[int]:
    """Parse a comma-separated list of prediction channel indices."""
    channels = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not channels:
        raise ValueError("--channels must list at least one prediction channel index.")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Duplicate channel indices in --channels {spec!r}.")
    return channels


def parse_index_list(spec: str, flag_name: str = "indices") -> list[int]:
    """Parse a comma-separated list of non-negative integer indices."""
    if not spec.strip():
        return []
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate values in {flag_name} {spec!r}.")
    return values


def normalize_channel_weighting(name: str) -> str:
    n = name.strip().lower()
    if n in ("uniform", "equal", "mean"):
        return "uniform"
    if n in ("group", "groups", "eig_disp", "50-50"):
        return "group"
    raise ValueError(
        f"Unsupported channel weighting {name!r}. Supported: uniform, group "
        f"(50% eigenfrequency ch0 + 50% mean of displacement ch1-4)."
    )


def combine_channel_losses(
    per_ch: torch.Tensor,
    channels: list[int],
    weighting: str,
) -> torch.Tensor:
    """Combine per-channel scalar losses (B, C) into one scalar per sample (B,)."""
    if weighting == "uniform":
        return per_ch.mean(dim=1)
    if weighting == "group":
        if channels != DEFAULT_SCORING_CHANNELS:
            raise ValueError(
                f"group weighting requires channels {DEFAULT_SCORING_CHANNELS}; got {channels}."
            )
        eig = per_ch[:, 0]
        disp = per_ch[:, 1:].mean(dim=1)
        return 0.5 * eig + 0.5 * disp
    raise ValueError(f"Unknown channel weighting: {weighting!r}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_cpu_threads(n: int | None) -> int | None:
    """Cap PyTorch/BLAS thread pools. Pass None or <=0 to leave defaults unchanged."""
    if n is None or n <= 0:
        return None
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    interop = max(1, n // 2)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass
    return n


@dataclass
class ScoringSources:
    """Memory-mapped eigenfrequency view plus optional displacement targets."""

    eigen_flat: torch.Tensor
    displacements: object | None
    total: int
    field_h: int
    field_w: int


def load_dataset_layout(
    dataset_pt_dir: Path,
    eigen_encoding: str = "uniform",
) -> tuple[int, int, int, int, int]:
    """Return (n_geom, n_wv, n_bands, field_h, field_w) from the eigenfrequency grid."""
    encoding = resolve_eigenfrequency_encoding(eigen_encoding)
    eigen_name = eigenfrequency_full_filename(encoding)
    eigen_path = dataset_pt_dir / eigen_name
    if not eigen_path.is_file():
        raise FileNotFoundError(f"Missing layout tensor: {eigen_path}")
    eigen = torch.load(eigen_path, map_location="cpu", mmap=True, weights_only=True)
    if eigen.ndim != 5:
        raise ValueError(
            f"Expected {eigen_name} shape (n_geom, n_wv, n_bands, H, W); "
            f"got {tuple(eigen.shape)}."
        )
    return tuple(int(s) for s in eigen.shape)  # type: ignore[return-value]


def load_displacements_dataset(dataset_pt_dir: Path):
    disp_path = dataset_pt_dir / "displacements_dataset.pt"
    if not disp_path.is_file():
        raise FileNotFoundError(f"Missing displacement targets: {disp_path}")
    displacements = torch.load(disp_path, map_location="cpu", weights_only=False)
    if not hasattr(displacements, "tensors") or len(displacements.tensors) != 4:
        raise ValueError(
            "displacements_dataset.pt must be a TensorDataset with exactly 4 tensors "
            f"(got {len(getattr(displacements, 'tensors', []))})."
        )
    return displacements


def load_truth_stack(
    dataset_pt_dir: Path,
    channels: list[int],
    total: int,
    field_hw: tuple[int, int],
    eigen_encoding: str = "uniform",
) -> torch.Tensor:
    """Stack truth fields for the requested prediction channels into (total, C, H, W).

    Channel 0 truth comes from ``eigenfrequency_{uniform|fft}_full.pt`` (see
    ``eigen_encoding``); channels 1–4 from displacements_dataset.pt tensors 0–3.
    """
    field_h, field_w = field_hw
    encoding = resolve_eigenfrequency_encoding(eigen_encoding)
    eigen_path = dataset_pt_dir / eigenfrequency_full_filename(encoding)
    eigen = torch.load(eigen_path, map_location="cpu", mmap=True, weights_only=True)
    eigen_flat = eigen.reshape(total, field_h, field_w)

    displacements = None
    chans: list[torch.Tensor] = []
    for ch in channels:
        if ch == 0:
            truth = eigen_flat
        elif 1 <= ch <= 4:
            if displacements is None:
                displacements = load_displacements_dataset(dataset_pt_dir)
            truth = displacements.tensors[ch - 1]
        else:
            raise ValueError(f"Unsupported prediction channel {ch}; expected 0–4 for I3O5.")
        if tuple(truth.shape) != (total, field_h, field_w):
            raise ValueError(
                f"Truth for channel {ch} shape {tuple(truth.shape)} != "
                f"expected ({total}, {field_h}, {field_w})."
            )
        chans.append(truth)
    return torch.stack(chans, dim=1)


def open_scoring_sources(
    dataset_pt_dir: Path,
    total: int,
    field_hw: tuple[int, int],
    need_displacements: bool,
    eigen_encoding: str = "uniform",
) -> ScoringSources:
    """Open mmap eigenfrequency and optionally load displacement targets once."""
    field_h, field_w = field_hw
    encoding = resolve_eigenfrequency_encoding(eigen_encoding)
    eigen_path = dataset_pt_dir / eigenfrequency_full_filename(encoding)
    eigen = torch.load(eigen_path, map_location="cpu", mmap=True, weights_only=True)
    eigen_flat = eigen.reshape(total, field_h, field_w)

    displacements = None
    if need_displacements:
        displacements = load_displacements_dataset(dataset_pt_dir)
        for ch_idx, tensor in enumerate(displacements.tensors, start=1):
            if tuple(tensor.shape) != (total, field_h, field_w):
                raise ValueError(
                    f"Truth for channel {ch_idx} shape {tuple(tensor.shape)} != "
                    f"expected ({total}, {field_h}, {field_w})."
                )

    return ScoringSources(
        eigen_flat=eigen_flat,
        displacements=displacements,
        total=total,
        field_h=field_h,
        field_w=field_w,
    )


def load_truth_batch(
    sources: ScoringSources,
    channels: list[int],
    start: int,
    end: int,
) -> torch.Tensor:
    """Materialize truth for one batch only: (batch, len(channels), H, W)."""
    chans: list[torch.Tensor] = []
    for ch in channels:
        if ch == 0:
            truth = sources.eigen_flat[start:end]
        elif 1 <= ch <= 4:
            if sources.displacements is None:
                raise RuntimeError(
                    f"Channel {ch} requested but displacements_dataset.pt was not opened."
                )
            truth = sources.displacements.tensors[ch - 1][start:end]
        else:
            raise ValueError(f"Unsupported prediction channel {ch}; expected 0–4 for I3O5.")
        chans.append(truth)
    return torch.stack(chans, dim=1)


def validate_channels(channels: list[int], out_channels: int) -> None:
    for ch in channels:
        if not (0 <= ch < out_channels):
            raise ValueError(f"Channel {ch} out of range for out_channels={out_channels}.")


def prepare_scoring_data(
    dataset_pt_dir: Path,
    predictions: torch.Tensor,
    channels: list[int] | None = None,
    eigen_encoding: str = "uniform",
) -> tuple[torch.Tensor, int, int, int, int, int, list[int]]:
    """Load truth fields and validate alignment with dense predictions."""
    if channels is None:
        channels = list(DEFAULT_SCORING_CHANNELS)

    dataset_pt_dir = Path(dataset_pt_dir)
    encoding = resolve_eigenfrequency_encoding(eigen_encoding)
    n_geom, n_wv, n_bands, field_h, field_w = load_dataset_layout(dataset_pt_dir, encoding)
    total = n_geom * n_wv * n_bands

    if predictions.ndim != 4:
        raise ValueError(f"Expected prediction shape (N, C, H, W); got {tuple(predictions.shape)}.")
    if predictions.shape[0] != total:
        raise ValueError(
            f"Prediction sample count {predictions.shape[0]} != n_geom*n_wv*n_bands ({total}). "
            f"Inference must be a dense full-dataset run for index alignment."
        )
    if predictions.shape[2:] != (field_h, field_w):
        raise ValueError(
            f"Field size mismatch: dataset {field_h}x{field_w} vs pred {tuple(predictions.shape[2:])}."
        )
    validate_channels(channels, int(predictions.shape[1]))

    truth_flat = load_truth_stack(
        dataset_pt_dir, channels, total, (field_h, field_w), eigen_encoding=encoding
    )
    return truth_flat, n_geom, n_wv, n_bands, field_h, field_w, channels


def compute_per_sample_losses(
    truth_flat: torch.Tensor | None,
    predictions: torch.Tensor,
    channels: list[int],
    losses: list[str],
    device: torch.device,
    batch_size: int,
    nmae_eps: float = 1e-5,
    nmse_eps: float = 1e-5,
    channel_weighting: str = "uniform",
    sources: ScoringSources | None = None,
) -> dict[str, np.ndarray]:
    """Return {loss_name: (n_samples,) per-sample loss}, combined over selected channels.

    For each channel, the loss is averaged over the H x W field. The sample score
    combines those per-channel values using ``channel_weighting``:

    - ``uniform``: arithmetic mean over all selected channels (1/5 each for I3O5).
    - ``group``: 50% channel 0 (eigenfrequency) + 50% mean(channels 1–4).

    Pass either a pre-stacked ``truth_flat`` (legacy) or ``sources`` for batched
    mmap/slice loading that avoids duplicating the full truth tensor in RAM.
    """
    if (truth_flat is None) == (sources is None):
        raise ValueError("Provide exactly one of truth_flat or sources.")

    if truth_flat is not None:
        if truth_flat.ndim != 4:
            raise ValueError(f"Expected truth shape (N, C, H, W); got {tuple(truth_flat.shape)}.")
        if truth_flat.shape[1] != len(channels):
            raise ValueError(
                f"Truth channel count {truth_flat.shape[1]} != len(channels)={len(channels)}."
            )
        n = truth_flat.shape[0]
    else:
        n = sources.total  # type: ignore[union-attr]

    channel_weighting = normalize_channel_weighting(channel_weighting)
    out = {loss: np.empty(n, dtype=LOSS_DTYPE) for loss in losses}
    need_mae = bool({"mae", "nmae"} & out.keys())
    need_mse = bool({"mse", "rms", "nmse", "nrms"} & out.keys())
    reduce_spatial = (2, 3)

    for start in tqdm(range(0, n, batch_size), desc="Scoring", unit="batch"):
        end = min(start + batch_size, n)
        if truth_flat is not None:
            truth_b = truth_flat[start:end].to(device, dtype=torch.float32)
        else:
            truth_b = load_truth_batch(sources, channels, start, end).to(device, dtype=torch.float32)
        pred_b = predictions[start:end, channels].to(device, dtype=torch.float32)
        err = pred_b - truth_b

        mae_per_ch = err.abs().mean(dim=reduce_spatial) if need_mae else None
        mse_per_ch = err.square().mean(dim=reduce_spatial) if need_mse else None

        if "mae" in out:
            out["mae"][start:end] = combine_channel_losses(
                mae_per_ch, channels, channel_weighting
            ).float().cpu().numpy()
        if "mse" in out:
            out["mse"][start:end] = combine_channel_losses(
                mse_per_ch, channels, channel_weighting
            ).float().cpu().numpy()
        if "rms" in out:
            rms_per_ch = torch.sqrt(mse_per_ch.clamp_min(0.0))
            out["rms"][start:end] = combine_channel_losses(
                rms_per_ch, channels, channel_weighting
            ).float().cpu().numpy()
        if "nmae" in out:
            denom_a = truth_b.abs().mean(dim=reduce_spatial)
            nmae_per_ch = mae_per_ch / (denom_a + nmae_eps)
            out["nmae"][start:end] = combine_channel_losses(
                nmae_per_ch, channels, channel_weighting
            ).float().cpu().numpy()
        if "nmse" in out or "nrms" in out:
            denom_s = truth_b.square().mean(dim=reduce_spatial)
            nmse_per_ch = mse_per_ch / (denom_s + nmse_eps)
            if "nmse" in out:
                out["nmse"][start:end] = combine_channel_losses(
                    nmse_per_ch, channels, channel_weighting
                ).float().cpu().numpy()
            if "nrms" in out:
                nrms_per_ch = torch.sqrt(nmse_per_ch.clamp_min(0.0))
                out["nrms"][start:end] = combine_channel_losses(
                    nrms_per_ch, channels, channel_weighting
                ).float().cpu().numpy()
    return out


def nrms_array_from_nmse_array(nmse_arr: np.ndarray) -> np.ndarray:
    """Return a copy of a five-column NMSE array with column 4 replaced by sqrt(NMSE).

    Note: this is only exact when NRMS was derived from a single-channel NMSE. After
    multi-channel averaging, prefer computing ``nrms`` directly via
    ``compute_per_sample_losses``.
    """
    out = nmse_arr.copy()
    out[:, 4] = np.sqrt(np.maximum(out[:, 4], 0.0))
    return out


# Percentiles are in PERFORMANCE (opposite of loss): higher p = better performance =
# lower loss. The sample for performance percentile p is taken at loss rank
# round((1 - p/100) * (n-1)) on the ascending-sorted losses (rank 0 = lowest loss =
# best performance, rank n-1 = highest loss = worst performance).
PERFORMANCE_PERCENTILES = [0, 1, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99, 100]


def performance_case_label(p: int) -> str:
    if p == 0:
        return "p0"
    if p == 100:
        return "p100"
    return f"p{p:02d}"


def rank_report(losses_flat: np.ndarray, geom: np.ndarray, wave: np.ndarray, band: np.ndarray):
    """Return ordered (label, combined_idx, g, w, b, loss) for the performance percentiles.

    p100 -> best performance (lowest loss); p0 -> worst performance (highest loss).
    Nearest-rank on the ascending-sorted per-sample losses, so every reported case is a
    real sample.
    """
    order = np.argsort(losses_flat, kind="stable")
    n = order.shape[0]
    rows = []
    for p in PERFORMANCE_PERCENTILES:
        rank = int(round((1.0 - p / 100.0) * (n - 1)))
        idx = int(order[rank])  # combined index == flat row index
        rows.append((performance_case_label(p), idx, int(geom[idx]), int(wave[idx]), int(band[idx]), float(losses_flat[idx])))
    return rows


def save_loss_arrays(
    *,
    per_sample: dict[str, np.ndarray],
    losses: list[str],
    loss_desc: dict[str, str],
    out_dir: Path,
    out_prefix: str,
    tag_suffix: str,
    group_suffix: str,
    combined: np.ndarray,
    geom: np.ndarray,
    wave: np.ndarray,
    band: np.ndarray,
    channels: list[int],
) -> None:
    """Write .npy arrays and print percentile reports for one channel group."""
    group_label = f" [{group_suffix}]" if group_suffix else ""
    for loss in losses:
        lg = per_sample[loss]
        # Homogeneous float32 (N, 5): cols 0–3 hold exact integers up to ~2e6; col 4 is loss.
        arr = np.empty((lg.shape[0], 5), dtype=LOSS_DTYPE)
        arr[:, 0] = combined.astype(LOSS_DTYPE)
        arr[:, 1] = geom.astype(LOSS_DTYPE)
        arr[:, 2] = wave.astype(LOSS_DTYPE)
        arr[:, 3] = band.astype(LOSS_DTYPE)
        arr[:, 4] = lg.astype(LOSS_DTYPE, copy=False)
        name_parts = [out_prefix, loss, tag_suffix, group_suffix]
        stem = "_".join(part for part in name_parts if part)
        out_path = out_dir / f"{stem}.npy"
        np.save(out_path, arr)

        rows = rank_report(lg, geom, wave, band)
        ch_label = ",".join(str(c) for c in channels)
        print(f"\n=== loss = {loss.upper()}{group_label} (channels {ch_label}; {loss_desc.get(loss, '')}) ===")
        print(f"saved array : {out_path}  shape={arr.shape}  "
              f"(cols: combined_idx, geom_idx, wave_idx, band_idx, {loss})")
        print(f"mean={lg.mean():.6e}")
        print(f"{'scenario':<14}{'combined':>12}{'geom':>8}{'wave':>8}{'band':>6}{'loss':>16}")
        for label, idx, g, w, b, val in rows:
            print(f"{label:<14}{idx:>12}{g:>8}{w:>8}{b:>6}{val:>16.6e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset-pt-dir",
        required=True,
        help=(
            "Dataset *_pt folder with displacements_dataset.pt and "
            "eigenfrequency_{uniform|fft}_full.pt (see --eigen-encoding)."
        ),
    )
    p.add_argument(
        "--eigen-encoding",
        choices=tuple(EIGENFREQUENCY_ENCODING_FILES),
        default="uniform",
        help=(
            "Channel-0 truth encoding: uniform or fft (wavelet). "
            "Must match the model / eigenfrequency_*_full.pt file. Default: uniform."
        ),
    )
    p.add_argument("--inference", required=True, help="Dense prediction tensor (.pt), shape (n_geom*n_wv*n_bands, C, H, W).")
    p.add_argument(
        "--losses",
        nargs="+",
        required=True,
        help=(
            "Loss criteria (one or more): mae, mse, rms, nmae, nmse, nrms "
            "(aliases l1, l2, rmse, nl1, nl2, nrmse)."
        ),
    )
    p.add_argument(
        "--channel-groups",
        nargs="+",
        choices=tuple(CHANNEL_GROUPS),
        metavar="GROUP",
        help=(
            "Channel groups to score (repeatable). freq_ch=ch0, disp_ch=mean(ch1-4), "
            "all_ch=mean(ch0-4). Each writes files with a _<group> suffix."
        ),
    )
    p.add_argument(
        "--channels",
        default=",".join(str(c) for c in DEFAULT_SCORING_CHANNELS),
        help="Comma-separated channels when --channel-groups is omitted (default: 0,1,2,3,4).",
    )
    p.add_argument("--output-dir", default="", help="Explicit output folder (overrides the model/dataset layout below).")
    p.add_argument("--model-name", default="", help="Model name for the PLOTS/INFERENCE/<model>/<dataset>/<subdir> layout.")
    p.add_argument("--dataset", default="", help="Dataset folder for the layout (default: --tag).")
    p.add_argument("--output-subdir", default="", help="Script output folder name under <model>/<dataset> (e.g. MAE_sample_case_plots).")
    p.add_argument("--category", default="inference", choices=("plots", "inference"),
                   help="Top-level root for the layout: 'inference' (data, default) or 'plots'.")
    p.add_argument("--out-prefix", default="per_sample_loss", help="Output filename prefix.")
    p.add_argument("--tag", default="", help="Optional tag appended to filenames (e.g. dataset name).")
    p.add_argument("--nmae-eps", type=float, default=1e-5, help="Epsilon added to mean(|t|) denominator for nmae (default 1e-5).")
    p.add_argument("--nmse-eps", type=float, default=1e-5, help="Epsilon added to mean(t^2) denominator for nmse (default 1e-5).")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "Max CPU threads for PyTorch/BLAS (default: 4). Use 0 to leave defaults "
            "(all cores). Lower values reduce interference with concurrent GPU training."
        ),
    )
    p.add_argument("--device", default="cpu", choices=("auto", "cuda", "cpu"),
                   help="Compute device (default: cpu). Use 'cuda' or 'auto' to opt into GPU.")
    p.add_argument(
        "--exclude-wave-indices",
        default="",
        help="Comma-separated wavevector indices to omit before saving arrays / percentile reports.",
    )
    p.add_argument(
        "--exclude-shear-modes",
        action="store_true",
        help="Omit ky=0 and kx=0 wavevectors (dead phase-pivot lines; see wave_mode_filters.py).",
    )
    p.add_argument(
        "--exclude-degenerate-pivot-cases",
        action="store_true",
        help=(
            "Omit ky=0, kx=0, and TRIM (k≡-k) wavevectors including M corners "
            "(see degenerate_pivot_wave_indices in wave_mode_filters.py)."
        ),
    )
    args = p.parse_args()

    thread_cap = configure_cpu_threads(args.threads)
    if thread_cap is not None:
        print(f"CPU threads: capped at {thread_cap} (torch.get_num_threads()={torch.get_num_threads()})")

    losses = []
    for name in args.losses:
        ln = normalize_loss_name(name)
        if ln not in losses:
            losses.append(ln)

    if args.channel_groups:
        scoring_groups = parse_channel_groups(args.channel_groups)
    else:
        scoring_groups = [("", parse_channels(args.channels))]

    device = resolve_device(args.device)
    dataset_pt_dir = Path(args.dataset_pt_dir)
    eigen_encoding = resolve_eigenfrequency_encoding(args.eigen_encoding)
    infer_path = Path(args.inference)
    out_dir = resolve_script_output_dir(
        explicit=args.output_dir or None,
        category=args.category,
        model_name=args.model_name or None,
        dataset=args.dataset or args.tag,
        subdir=args.output_subdir,
        fallback=infer_path.parent,
    )

    predictions = torch.load(infer_path, map_location="cpu", mmap=True, weights_only=True)
    n_geom, n_wv, n_bands, field_h, field_w = load_dataset_layout(dataset_pt_dir, eigen_encoding)
    total = n_geom * n_wv * n_bands

    if predictions.shape[0] != total:
        raise ValueError(
            f"Prediction sample count {predictions.shape[0]} != n_geom*n_wv*n_bands ({total}). "
            f"Inference must be a dense full-dataset run for index alignment."
        )

    print(f"Dataset   : {dataset_pt_dir}")
    print(f"Eigen enc : {eigen_encoding} ({eigenfrequency_full_filename(eigen_encoding)})")
    print(f"Inference : {infer_path}  shape={tuple(predictions.shape)} dtype={predictions.dtype}")
    print(f"Losses    : {', '.join(losses)}")
    print(
        "Supported : "
        + ", ".join(SUPPORTED_LOSSES)
        + " (aliases l1->mae, l2->mse, rmse->rms, nl1->nmae, nl2->nmse, nrmse->nrms)"
    )
    print(f"Field     : {field_h}x{field_w}   Device: {device}")
    exclude_waves = parse_index_list(args.exclude_wave_indices, "--exclude-wave-indices")
    if args.exclude_shear_modes:
        exclude_waves = sorted(set(exclude_waves) | set(shear_mode_wave_indices()))
    if args.exclude_degenerate_pivot_cases:
        exclude_waves = sorted(set(exclude_waves) | set(degenerate_pivot_wave_indices()))
    if exclude_waves:
        print(f"Excluding wave indices: {len(exclude_waves)} wavevectors")

    print(f"Samples   : {total}  (n_geom={n_geom}, n_waveforms={n_wv}, n_bands={n_bands})")
    print(f"Output dir: {out_dir}")

    combined = np.arange(total, dtype=INDEX_DTYPE)
    geom = (combined // (n_wv * n_bands)).astype(INDEX_DTYPE)
    wave = ((combined % (n_wv * n_bands)) // n_bands).astype(INDEX_DTYPE)
    band = (combined % n_bands).astype(INDEX_DTYPE)

    loss_desc = {
        "mae": f"mean over channels of mean(|p-t|) over {field_h}x{field_w}",
        "mse": f"mean over channels of mean((p-t)^2) over {field_h}x{field_w}",
        "rms": f"mean over channels of sqrt(mean((p-t)^2)) over {field_h}x{field_w}",
        "nmae": f"mean over channels of mean(|p-t|) / (mean(|t|) + {args.nmae_eps:g})",
        "nmse": f"mean over channels of mean((p-t)^2) / (mean(t^2) + {args.nmse_eps:g})",
        "nrms": f"mean over channels of sqrt(mean((p-t)^2) / (mean(t^2) + {args.nmse_eps:g}))",
    }
    tag_suffix = args.tag.strip("_") if args.tag else ""

    all_channels = sorted({ch for _, chs in scoring_groups for ch in chs})
    validate_channels(all_channels, int(predictions.shape[1]))
    need_displacements = any(ch >= 1 for ch in all_channels)
    sources = open_scoring_sources(
        dataset_pt_dir,
        total,
        (field_h, field_w),
        need_displacements,
        eigen_encoding=eigen_encoding,
    )
    truth_mode = "batched mmap/slice" if sources else "full stack"
    print(
        f"Truth mode: {truth_mode}  "
        f"(eigen mmap; displacements={'loaded' if need_displacements else 'skipped'})"
    )

    for group_suffix, channels in scoring_groups:
        validate_channels(channels, int(predictions.shape[1]))
        group_note = f" ({group_suffix})" if group_suffix else ""
        print(f"\n--- channel group{group_note}: {channels} ---")

        per_sample = compute_per_sample_losses(
            truth_flat=None,
            predictions=predictions,
            channels=channels,
            losses=losses,
            device=device,
            batch_size=args.batch_size,
            nmae_eps=args.nmae_eps,
            nmse_eps=args.nmse_eps,
            sources=sources,
        )
        if exclude_waves:
            keep = ~np.isin(wave, np.array(exclude_waves, dtype=INDEX_DTYPE))
            n_kept = int(keep.sum())
            print(f"Sample count after wave exclusion: {n_kept} / {total}")
            combined = combined[keep]
            geom = geom[keep]
            wave = wave[keep]
            band = band[keep]
            per_sample = {loss: arr[keep] for loss, arr in per_sample.items()}
        save_loss_arrays(
            per_sample=per_sample,
            losses=losses,
            loss_desc=loss_desc,
            out_dir=out_dir,
            out_prefix=args.out_prefix,
            tag_suffix=tag_suffix,
            group_suffix=group_suffix,
            combined=combined,
            geom=geom,
            wave=wave,
            band=band,
            channels=channels,
        )


if __name__ == "__main__":
    main()
