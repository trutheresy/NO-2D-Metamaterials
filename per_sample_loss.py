"""
Per-sample raw-pixel loss between a ground-truth field tensor and a model-inference
prediction tensor.

A "sample" is one ``(geometry, wavevector, band)`` entry. The loss between the
inference sample and the true sample is computed directly on the **raw (encoded)
pixel fields** -- the same quantity scored by ``compare_inference_to_truth.py`` -- by
averaging the per-pixel error over the 32x32 field. No decoding to physical units
(rad/s) is performed, so the magnitudes match the other loss scripts.

Inputs
------
--truth      : ground-truth field tensor, e.g. ``eigenfrequency_uniform_full.pt``
               with shape ``(n_geom, n_wv, n_bands, H, W)``.
--inference  : dense prediction tensor from ``run_model_inference.py`` with shape
               ``(n_geom*n_wv*n_bands, out_channels, H, W)`` indexed as
               ``combined = geom*(n_wv*n_bands) + wave*n_bands + band``.
--channel    : prediction channel to compare against the truth field (default 0,
               the eigenfrequency channel).

For each requested loss criterion this writes one array file (``.npy``) with five
columns:

    col 0 : combined index   (flat C-order index = geom*(n_wv*n_bands) + wave*n_bands + band)
    col 1 : geometry index
    col 2 : wavevector index
    col 3 : band index
    col 4 : loss (mean over the H x W field for that sample)

It also prints, per loss, samples at performance percentiles p01/p10/.../p90/p99
(performance = opposite of loss: p99 = best/lowest-loss, p01 = worst/highest-loss),
using nearest-rank order on the sorted per-sample losses so every reported case is a
real sample.

Supported losses (per sample, mean over the H x W field):
    mae  : mean(|p - t|)
    mse  : mean((p - t)^2)
    nmae : mean(|p - t|) / (mean(|t|)   + eps_a)   # normalized by mean abs pixel value
    nmse : mean((p - t)^2) / (mean(t^2) + eps_s)   # normalized by mean square pixel value
    nrms : sqrt(nmse) = RMSE / sqrt(mean(t^2) + eps_s)   # normalized RMS error
(l1 -> mae, l2 -> mse.) Default eps_a = 1e-5, eps_s = 1e-7.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def compute_per_sample_losses(
    truth_flat: torch.Tensor,
    predictions: torch.Tensor,
    channel: int,
    losses: list[str],
    device: torch.device,
    batch_size: int,
    nmae_eps: float = 1e-5,
    nmse_eps: float = 1e-7,
) -> dict[str, np.ndarray]:
    """Return {loss_name: (n_samples,) per-sample loss}, averaging error over the field.

    Per sample (mean over the H x W field):
        mae  = mean(|p - t|)
        mse  = mean((p - t)^2)
        nmae = mean(|p - t|)  / (mean(|t|)  + nmae_eps)
        nmse = mean((p - t)^2) / (mean(t^2) + nmse_eps)
        nrms = sqrt(nmse)
    """
    n = truth_flat.shape[0]
    out = {loss: np.empty(n, dtype=np.float64) for loss in losses}
    need_mae = bool({"mae", "nmae"} & out.keys())
    need_mse = bool({"mse", "nmse", "nrms"} & out.keys())
    for start in tqdm(range(0, n, batch_size), desc="Scoring", unit="batch"):
        end = min(start + batch_size, n)
        truth_b = truth_flat[start:end].to(device, dtype=torch.float32)
        pred_b = predictions[start:end, channel].to(device, dtype=torch.float32)
        err = pred_b - truth_b
        reduce_dims = tuple(range(1, err.ndim))

        mae_b = err.abs().mean(dim=reduce_dims) if need_mae else None
        mse_b = err.square().mean(dim=reduce_dims) if need_mse else None

        if "mae" in out:
            out["mae"][start:end] = mae_b.double().cpu().numpy()
        if "mse" in out:
            out["mse"][start:end] = mse_b.double().cpu().numpy()
        if "nmae" in out:
            denom_a = truth_b.abs().mean(dim=reduce_dims)  # mean abs pixel value of the true field
            out["nmae"][start:end] = (mae_b / (denom_a + nmae_eps)).double().cpu().numpy()
        if "nmse" in out or "nrms" in out:
            denom_s = truth_b.square().mean(dim=reduce_dims)  # mean square pixel value of the true field
            nmse_b = mse_b / (denom_s + nmse_eps)
            if "nmse" in out:
                out["nmse"][start:end] = nmse_b.double().cpu().numpy()
            if "nrms" in out:
                out["nrms"][start:end] = torch.sqrt(nmse_b.clamp_min(0.0)).double().cpu().numpy()
    return out


def nrms_array_from_nmse_array(nmse_arr: np.ndarray) -> np.ndarray:
    """Return a copy of a five-column NMSE array with column 4 replaced by sqrt(NMSE)."""
    out = nmse_arr.copy()
    out[:, 4] = np.sqrt(np.maximum(out[:, 4], 0.0))
    return out


# Percentiles are in PERFORMANCE (opposite of loss): higher p = better performance =
# lower loss. The sample for performance percentile p is taken at loss rank
# round((1 - p/100) * (n-1)) on the ascending-sorted losses (rank 0 = lowest loss =
# best performance, rank n-1 = highest loss = worst performance).
PERFORMANCE_PERCENTILES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


def rank_report(losses_flat: np.ndarray, geom: np.ndarray, wave: np.ndarray, band: np.ndarray):
    """Return ordered (label, combined_idx, g, w, b, loss) for the performance percentiles.

    p99 -> best performance (near-lowest loss); p01 -> worst performance (near-highest
    loss). Nearest-rank on the ascending-sorted per-sample losses, so every reported
    case is a real sample.
    """
    order = np.argsort(losses_flat, kind="stable")
    n = order.shape[0]
    rows = []
    for p in PERFORMANCE_PERCENTILES:
        rank = int(round((1.0 - p / 100.0) * (n - 1)))
        idx = int(order[rank])  # combined index == flat row index
        rows.append((f"p{p:02d}", idx, int(geom[idx]), int(wave[idx]), int(band[idx]), float(losses_flat[idx])))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", required=True, help="Ground-truth field tensor (.pt), shape (n_geom, n_wv, n_bands, H, W).")
    p.add_argument("--inference", required=True, help="Dense prediction tensor (.pt), shape (n_geom*n_wv*n_bands, C, H, W).")
    p.add_argument("--losses", nargs="+", required=True, help="Loss criteria: mae mse nmae nmse nrms (aliases l1, l2, nl1, nl2, nrmse).")
    p.add_argument("--channel", type=int, default=0, help="Prediction channel to compare (default 0 = eigenfrequency).")
    p.add_argument("--output-dir", default="", help="Folder for the per-loss .npy files (default: inference file's folder).")
    p.add_argument("--out-prefix", default="per_sample_loss", help="Output filename prefix.")
    p.add_argument("--tag", default="", help="Optional tag appended to filenames (e.g. dataset name).")
    p.add_argument("--nmae-eps", type=float, default=1e-5, help="Epsilon added to mean(|t|) denominator for nmae (default 1e-5).")
    p.add_argument("--nmse-eps", type=float, default=1e-7, help="Epsilon added to mean(t^2) denominator for nmse (default 1e-7).")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = p.parse_args()

    losses = []
    for name in args.losses:
        ln = normalize_loss_name(name)
        if ln not in losses:
            losses.append(ln)

    device = resolve_device(args.device)
    truth_path = Path(args.truth)
    infer_path = Path(args.inference)
    out_dir = Path(args.output_dir) if args.output_dir else infer_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = torch.load(truth_path, map_location="cpu", mmap=True, weights_only=True)
    predictions = torch.load(infer_path, map_location="cpu", mmap=True, weights_only=True)

    if truth.ndim != 5:
        raise ValueError(f"Expected truth shape (n_geom, n_wv, n_bands, H, W); got {tuple(truth.shape)}.")
    n_geom, n_wv, n_bands, field_h, field_w = (int(s) for s in truth.shape)
    total = n_geom * n_wv * n_bands

    if predictions.ndim != 4:
        raise ValueError(f"Expected prediction shape (N, C, H, W); got {tuple(predictions.shape)}.")
    if predictions.shape[0] != total:
        raise ValueError(
            f"Prediction sample count {predictions.shape[0]} != n_geom*n_wv*n_bands ({total}). "
            f"Inference must be a dense full-dataset run for index alignment."
        )
    if predictions.shape[2:] != truth.shape[3:]:
        raise ValueError(f"Field size mismatch: truth {tuple(truth.shape[3:])} vs pred {tuple(predictions.shape[2:])}.")
    if not (0 <= args.channel < predictions.shape[1]):
        raise ValueError(f"--channel {args.channel} out of range for out_channels={predictions.shape[1]}.")

    # C-order flatten of (n_geom, n_wv, n_bands, H, W) over the first three dims aligns
    # row-for-row with the dense prediction index, so combined index == row index.
    truth_flat = truth.reshape(total, field_h, field_w)

    print(f"Truth     : {truth_path}  shape={tuple(truth.shape)} dtype={truth.dtype}")
    print(f"Inference : {infer_path}  shape={tuple(predictions.shape)} dtype={predictions.dtype}")
    print(f"Channel   : {args.channel}   Field: {field_h}x{field_w}   Device: {device}")
    print(f"Samples   : {total}  (n_geom={n_geom}, n_waveforms={n_wv}, n_bands={n_bands})")
    print(f"Output dir: {out_dir}")

    combined = np.arange(total, dtype=np.int64)
    geom = combined // (n_wv * n_bands)
    wave = (combined % (n_wv * n_bands)) // n_bands
    band = combined % n_bands

    per_sample = compute_per_sample_losses(
        truth_flat=truth_flat,
        predictions=predictions,
        channel=args.channel,
        losses=losses,
        device=device,
        batch_size=args.batch_size,
        nmae_eps=args.nmae_eps,
        nmse_eps=args.nmse_eps,
    )

    loss_desc = {
        "mae": f"mean(|p-t|) over {field_h}x{field_w} field",
        "mse": f"mean((p-t)^2) over {field_h}x{field_w} field",
        "nmae": f"mean(|p-t|) / (mean(|t|) + {args.nmae_eps:g})",
        "nmse": f"mean((p-t)^2) / (mean(t^2) + {args.nmse_eps:g})",
        "nrms": f"sqrt(mean((p-t)^2) / (mean(t^2) + {args.nmse_eps:g}))",
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
