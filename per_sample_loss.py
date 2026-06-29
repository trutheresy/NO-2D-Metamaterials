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
--dataset-pt-dir : folder with ``eigenfrequency_uniform_full.pt`` (truth for channel 0)
                   and ``displacements_dataset.pt`` (4 tensors, truth for channels 1–4).
--inference      : dense prediction tensor from ``run_model_inference.py`` with shape
                   ``(n_geom*n_wv*n_bands, out_channels, H, W)`` indexed as
                   ``combined = geom*(n_wv*n_bands) + wave*n_bands + band``.
--channels       : comma-separated prediction channel indices to score and average
                   (default ``0,1,2,3,4`` = all output channels).

For each requested loss criterion this writes one array file (``.npy``) with five
columns:

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
    nmae : mean(|p - t|) / (mean(|t|)   + eps_a)   # normalized by mean abs pixel value
    nmse : mean((p - t)^2) / (mean(t^2) + eps_s)   # normalized by mean square pixel value
    nrms : sqrt(nmse) = RMSE / sqrt(mean(t^2) + eps_s)   # per channel, then averaged
(l1 -> mae, l2 -> mse.) Default eps_a = 1e-5, eps_s = 1e-5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from output_layout import resolve_script_output_dir


DEFAULT_SCORING_CHANNELS = [0, 1, 2, 3, 4]
# Backward-compatible alias used by plot_high_loss_samples.py
DEFAULT_DISPLACEMENT_CHANNELS = DEFAULT_SCORING_CHANNELS


def normalize_loss_name(name: str) -> str:
    n = name.strip().lower()
    if n in ("l1", "mae"):
        return "mae"
    if n in ("l2", "mse"):
        return "mse"
    if n in ("nmae", "nl1"):
        return "nmae"
    if n in ("nmse", "nl2"):
        return "nmse"
    if n in ("nrms", "nrmse"):
        return "nrms"
    raise ValueError(
        f"Unsupported loss {name!r}. Supported: mae, mse, nmae, nmse, nrms "
        f"(aliases l1, l2, nl1, nl2, nrmse)."
    )


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


def load_dataset_layout(dataset_pt_dir: Path) -> tuple[int, int, int, int, int]:
    """Return (n_geom, n_wv, n_bands, field_h, field_w) from the eigenfrequency grid."""
    eigen_path = dataset_pt_dir / "eigenfrequency_uniform_full.pt"
    if not eigen_path.is_file():
        raise FileNotFoundError(f"Missing layout tensor: {eigen_path}")
    eigen = torch.load(eigen_path, map_location="cpu", mmap=True, weights_only=True)
    if eigen.ndim != 5:
        raise ValueError(
            f"Expected eigenfrequency_uniform_full shape (n_geom, n_wv, n_bands, H, W); "
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
) -> torch.Tensor:
    """Stack truth fields for the requested prediction channels into (total, C, H, W).

    Channel 0 truth comes from eigenfrequency_uniform_full.pt; channels 1–4 from
    displacements_dataset.pt tensors 0–3.
    """
    field_h, field_w = field_hw
    eigen_path = dataset_pt_dir / "eigenfrequency_uniform_full.pt"
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


def validate_channels(channels: list[int], out_channels: int) -> None:
    for ch in channels:
        if not (0 <= ch < out_channels):
            raise ValueError(f"Channel {ch} out of range for out_channels={out_channels}.")


def prepare_scoring_data(
    dataset_pt_dir: Path,
    predictions: torch.Tensor,
    channels: list[int] | None = None,
) -> tuple[torch.Tensor, int, int, int, int, int, list[int]]:
    """Load truth fields and validate alignment with dense predictions."""
    if channels is None:
        channels = list(DEFAULT_SCORING_CHANNELS)

    dataset_pt_dir = Path(dataset_pt_dir)
    n_geom, n_wv, n_bands, field_h, field_w = load_dataset_layout(dataset_pt_dir)
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

    truth_flat = load_truth_stack(dataset_pt_dir, channels, total, (field_h, field_w))
    return truth_flat, n_geom, n_wv, n_bands, field_h, field_w, channels


def compute_per_sample_losses(
    truth_flat: torch.Tensor,
    predictions: torch.Tensor,
    channels: list[int],
    losses: list[str],
    device: torch.device,
    batch_size: int,
    nmae_eps: float = 1e-5,
    nmse_eps: float = 1e-5,
    channel_weighting: str = "uniform",
) -> dict[str, np.ndarray]:
    """Return {loss_name: (n_samples,) per-sample loss}, combined over selected channels.

    For each channel, the loss is averaged over the H x W field. The sample score
    combines those per-channel values using ``channel_weighting``:

    - ``uniform``: arithmetic mean over all selected channels (1/5 each for I3O5).
    - ``group``: 50% channel 0 (eigenfrequency) + 50% mean(channels 1–4).
    """
    if truth_flat.ndim != 4:
        raise ValueError(f"Expected truth shape (N, C, H, W); got {tuple(truth_flat.shape)}.")
    if truth_flat.shape[1] != len(channels):
        raise ValueError(
            f"Truth channel count {truth_flat.shape[1]} != len(channels)={len(channels)}."
        )

    channel_weighting = normalize_channel_weighting(channel_weighting)
    n = truth_flat.shape[0]
    out = {loss: np.empty(n, dtype=np.float64) for loss in losses}
    need_mae = bool({"mae", "nmae"} & out.keys())
    need_mse = bool({"mse", "nmse", "nrms"} & out.keys())
    reduce_spatial = (2, 3)

    for start in tqdm(range(0, n, batch_size), desc="Scoring", unit="batch"):
        end = min(start + batch_size, n)
        truth_b = truth_flat[start:end].to(device, dtype=torch.float32)
        pred_b = predictions[start:end, channels].to(device, dtype=torch.float32)
        err = pred_b - truth_b

        mae_per_ch = err.abs().mean(dim=reduce_spatial) if need_mae else None
        mse_per_ch = err.square().mean(dim=reduce_spatial) if need_mse else None

        if "mae" in out:
            out["mae"][start:end] = combine_channel_losses(
                mae_per_ch, channels, channel_weighting
            ).double().cpu().numpy()
        if "mse" in out:
            out["mse"][start:end] = combine_channel_losses(
                mse_per_ch, channels, channel_weighting
            ).double().cpu().numpy()
        if "nmae" in out:
            denom_a = truth_b.abs().mean(dim=reduce_spatial)
            nmae_per_ch = mae_per_ch / (denom_a + nmae_eps)
            out["nmae"][start:end] = combine_channel_losses(
                nmae_per_ch, channels, channel_weighting
            ).double().cpu().numpy()
        if "nmse" in out or "nrms" in out:
            denom_s = truth_b.square().mean(dim=reduce_spatial)
            nmse_per_ch = mse_per_ch / (denom_s + nmse_eps)
            if "nmse" in out:
                out["nmse"][start:end] = combine_channel_losses(
                    nmse_per_ch, channels, channel_weighting
                ).double().cpu().numpy()
            if "nrms" in out:
                nrms_per_ch = torch.sqrt(nmse_per_ch.clamp_min(0.0))
                out["nrms"][start:end] = combine_channel_losses(
                    nrms_per_ch, channels, channel_weighting
                ).double().cpu().numpy()
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset-pt-dir",
        required=True,
        help="Dataset *_pt folder with displacements_dataset.pt and eigenfrequency_uniform_full.pt.",
    )
    p.add_argument("--inference", required=True, help="Dense prediction tensor (.pt), shape (n_geom*n_wv*n_bands, C, H, W).")
    p.add_argument("--losses", nargs="+", required=True, help="Loss criteria: mae mse nmae nmse nrms (aliases l1, l2, nl1, nl2, nrmse).")
    p.add_argument(
        "--channels",
        default=",".join(str(c) for c in DEFAULT_SCORING_CHANNELS),
        help="Comma-separated prediction channels to score and average (default: 0,1,2,3,4).",
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
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = p.parse_args()

    losses = []
    for name in args.losses:
        ln = normalize_loss_name(name)
        if ln not in losses:
            losses.append(ln)

    channels = parse_channels(args.channels)
    device = resolve_device(args.device)
    dataset_pt_dir = Path(args.dataset_pt_dir)
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
    truth_flat, n_geom, n_wv, n_bands, field_h, field_w, channels = prepare_scoring_data(
        dataset_pt_dir, predictions, channels
    )
    total = n_geom * n_wv * n_bands

    print(f"Dataset   : {dataset_pt_dir}")
    print(f"Truth     : eigenfrequency_uniform + displacements  shape={tuple(truth_flat.shape)} dtype={truth_flat.dtype}")
    print(f"Inference : {infer_path}  shape={tuple(predictions.shape)} dtype={predictions.dtype}")
    print(f"Channels  : {channels} (mean loss over these prediction channels)")
    print(f"Field     : {field_h}x{field_w}   Device: {device}")
    print(f"Samples   : {total}  (n_geom={n_geom}, n_waveforms={n_wv}, n_bands={n_bands})")
    print(f"Output dir: {out_dir}")

    combined = np.arange(total, dtype=np.int64)
    geom = combined // (n_wv * n_bands)
    wave = (combined % (n_wv * n_bands)) // n_bands
    band = combined % n_bands

    per_sample = compute_per_sample_losses(
        truth_flat=truth_flat,
        predictions=predictions,
        channels=channels,
        losses=losses,
        device=device,
        batch_size=args.batch_size,
        nmae_eps=args.nmae_eps,
        nmse_eps=args.nmse_eps,
    )

    loss_desc = {
        "mae": f"mean over channels of mean(|p-t|) over {field_h}x{field_w}",
        "mse": f"mean over channels of mean((p-t)^2) over {field_h}x{field_w}",
        "nmae": f"mean over channels of mean(|p-t|) / (mean(|t|) + {args.nmae_eps:g})",
        "nmse": f"mean over channels of mean((p-t)^2) / (mean(t^2) + {args.nmse_eps:g})",
        "nrms": f"mean over channels of sqrt(mean((p-t)^2) / (mean(t^2) + {args.nmse_eps:g}))",
    }
    tag = f"_{args.tag}" if args.tag else ""
    for loss in losses:
        lg = per_sample[loss]
        arr = np.column_stack([
            combined.astype(np.float64),
            geom.astype(np.float64),
            wave.astype(np.float64),
            band.astype(np.float64),
            lg.astype(np.float64),
        ])
        out_path = out_dir / f"{args.out_prefix}_{loss}{tag}.npy"
        np.save(out_path, arr)

        rows = rank_report(lg, geom, wave, band)
        print(f"\n=== loss = {loss.upper()} ({loss_desc.get(loss, '')}) ===")
        print(f"saved array : {out_path}  shape={arr.shape}  "
              f"(cols: combined_idx, geom_idx, wave_idx, band_idx, {loss})")
        print(f"mean={lg.mean():.6e}")
        print(f"{'scenario':<14}{'combined':>12}{'geom':>8}{'wave':>8}{'band':>6}{'loss':>16}")
        for label, idx, g, w, b, val in rows:
            print(f"{label:<14}{idx:>12}{g:>8}{w:>8}{b:>6}{val:>16.6e}")


if __name__ == "__main__":
    main()
