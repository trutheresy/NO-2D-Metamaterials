"""
Decode a model's predicted eigenfrequency channel (uniform encoding) back into
scalar eigenvalues laid out like ``eigenvalue_data_full.pt``.

Input  : an INFERENCE run folder containing ``predictions_*.pt`` of shape
         ``(n_geom * n_waveforms * n_bands, out_channels, 32, 32)`` (the dense
         output of ``run_model_inference.py``; flat index
         ``idx = geom*(n_waveforms*n_bands) + wave*n_bands + band``).
         Channel 0 holds the eigenfrequency in *uniform encoding* (a 32x32 patch
         whose value is ``ln(s)/100``).

Output : ``eigenvalues_predictions_full.pt`` written to the output folder, a tensor
         of shape ``(n_geom, n_waveforms, n_bands)`` of scalar predicted
         eigenvalues -- the same layout (N_struct, N_wv, N_eig) that
         ``eigenvalue_data_full.pt`` uses, so it can drive the dispersion plotters.

Decoding reuses ``NO_utilities.decode_eigenfrequency_uniform`` (``--reduce pixel``,
the canonical corner-pixel decoder) or, for the model's only-approximately-uniform
output, the patch-mean variant ``s = exp(100 * mean(patch))`` (``--reduce mean``).

The (n_geom, n_waveforms) grid is read from a reference dataset's
``eigenvalue_data_full.pt`` (auto-resolved under the output folder, or via
``--reference-pt-dir``); ``n_bands`` is then derived from the prediction count.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import NO_utilities as NU
from output_layout import resolve_script_output_dir


def find_predictions_file(inference_dir: Path) -> Path:
    cands = sorted(inference_dir.glob("predictions_*.pt"))
    if not cands:
        raise FileNotFoundError(f"No predictions_*.pt found in {inference_dir}")
    if len(cands) > 1:
        print(f"Warning: multiple prediction tensors in {inference_dir}; using {cands[0].name}")
    return cands[0]


def resolve_reference_pt_dir(folder: Path) -> Path:
    """Return a folder that contains eigenvalue_data_full.pt (for the n_geom x n_wv grid)."""
    if (folder / "eigenvalue_data_full.pt").exists():
        return folder
    pt_dirs = [p for p in folder.iterdir() if p.is_dir() and p.name.endswith("_pt")]
    pt_dirs = [p for p in pt_dirs if (p / "eigenvalue_data_full.pt").exists()]
    if not pt_dirs:
        raise FileNotFoundError(
            f"No eigenvalue_data_full.pt found in {folder} or a *_pt subfolder. "
            f"Pass --reference-pt-dir explicitly."
        )
    return max(pt_dirs, key=lambda p: p.stat().st_mtime)


def decode_channel(
    predictions: torch.Tensor,
    channel: int,
    reduce: str,
    batch_size: int,
) -> np.ndarray:
    """Decode the encoded eigenfrequency channel into a flat (N,) float64 array."""
    n = predictions.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in tqdm(range(0, n, batch_size), desc="Decoding", unit="batch"):
        end = min(start + batch_size, n)
        patch = predictions[start:end, channel]  # (B, 32, 32) float16
        if reduce == "pixel":
            # Canonical decoder (reads corner pixel); suppress its per-call warning.
            arr = patch.to(torch.float16).numpy()
            with contextlib.redirect_stdout(io.StringIO()):
                vals = NU.decode_eigenfrequency_uniform(arr)
            out[start:end] = np.asarray(vals, dtype=np.float64).reshape(-1)
        else:  # mean: same formula as the decoder, averaged over the patch (denoises)
            pixel_mean = patch.to(torch.float32).mean(dim=(1, 2)).numpy().astype(np.float64)
            out[start:end] = np.exp(100.0 * pixel_mean)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="INFERENCE run folder containing predictions_*.pt.")
    p.add_argument("--output-dir", default="", help="Explicit folder to write the output into (overrides the model/dataset layout below).")
    p.add_argument("--model-name", default="", help="Model name for the INFERENCE/<model>/<dataset>/<subdir> layout.")
    p.add_argument("--dataset", default="", help="Dataset folder for the layout.")
    p.add_argument("--output-subdir", default="", help="Optional script output folder name under INFERENCE/<model>/<dataset> (default: none).")
    p.add_argument("--predictions", default="", help="Explicit prediction tensor path (overrides --input-dir lookup).")
    p.add_argument("--reference-pt-dir", default="", help="Folder with eigenvalue_data_full.pt for the (n_geom, n_wv) grid. Default: auto-resolve under --output-dir.")
    p.add_argument("--out-name", default="eigenvalues_predictions_full.pt", help="Output filename.")
    p.add_argument("--channel", type=int, default=0, help="Prediction channel holding the encoded eigenfrequency (default: 0).")
    p.add_argument("--reduce", choices=("pixel", "mean"), default="pixel", help="Decode via NO_utilities corner-pixel decoder (pixel) or patch-mean (mean).")
    p.add_argument("--save-dtype", choices=("float32", "float16", "float64"), default="float32")
    p.add_argument("--batch-size", type=int, default=65536)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = resolve_script_output_dir(
        explicit=args.output_dir or None,
        category="inference",
        model_name=args.model_name or None,
        dataset=args.dataset,
        subdir=args.output_subdir,
        fallback=input_dir,
    )

    pred_path = Path(args.predictions) if args.predictions else find_predictions_file(input_dir)
    print(f"Predictions      : {pred_path}")

    ref_dir = Path(args.reference_pt_dir) if args.reference_pt_dir else resolve_reference_pt_dir(output_dir)
    print(f"Reference grid   : {ref_dir / 'eigenvalue_data_full.pt'}")

    predictions = torch.load(pred_path, map_location="cpu", mmap=True, weights_only=True)
    eigen_ref = torch.load(ref_dir / "eigenvalue_data_full.pt", map_location="cpu", mmap=True, weights_only=True)

    if predictions.ndim != 4 or predictions.shape[-2:] != (32, 32):
        raise ValueError(f"Unexpected predictions shape {tuple(predictions.shape)}; expected (N, C, 32, 32).")
    if args.channel >= predictions.shape[1]:
        raise ValueError(f"--channel {args.channel} out of range for out_channels={predictions.shape[1]}.")

    n_geom, n_wv, n_eig_ref = int(eigen_ref.shape[0]), int(eigen_ref.shape[1]), int(eigen_ref.shape[2])
    total = int(predictions.shape[0])
    combos = n_geom * n_wv
    if total % combos != 0:
        raise ValueError(
            f"Prediction count {total} is not divisible by n_geom*n_waveforms ({combos}). "
            f"The prediction tensor must be a dense full-dataset run for this grid."
        )
    n_bands = total // combos

    print(f"  predictions shape : {tuple(predictions.shape)} dtype={predictions.dtype}")
    print(f"  reference eigen   : {tuple(eigen_ref.shape)} (n_geom={n_geom}, n_waveforms={n_wv}, n_eig={n_eig_ref})")
    print(f"  derived n_bands   : {n_bands}  (== predictions / (n_geom*n_waveforms))")
    if n_bands != n_eig_ref:
        print(f"  note: predicted bands ({n_bands}) != reference eigenvalue bands ({n_eig_ref}).")
    print(f"  decode reduce     : {args.reduce} (channel {args.channel})")

    decoded_flat = decode_channel(predictions, args.channel, args.reduce, args.batch_size)
    decoded = torch.from_numpy(decoded_flat).reshape(n_geom, n_wv, n_bands)

    dtype = {"float32": torch.float32, "float16": torch.float16, "float64": torch.float64}[args.save_dtype]
    decoded = decoded.to(dtype)

    out_path = output_dir / args.out_name
    torch.save(decoded, out_path)

    finite = decoded_flat[np.isfinite(decoded_flat)]
    print("\nDecoded predicted eigenvalues:")
    print(f"  shape : {tuple(decoded.shape)}  dtype={decoded.dtype}")
    if finite.size:
        print(f"  range : min={finite.min():.6g}  median={np.median(finite):.6g}  max={finite.max():.6g}")
    print(f"  saved : {out_path}")


if __name__ == "__main__":
    main()
