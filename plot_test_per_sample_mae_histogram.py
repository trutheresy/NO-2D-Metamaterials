"""
Run evaluate_from_disk.run_evaluation on one test shard (c_test or b_test) and plot per-sample MAE.

MAE matches evaluate_from_disk loss name "l1" (mean |pred - target| over all channels and pixels).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, Subset

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import evaluate_from_disk as efd


DEFAULT_MODEL = Path(
    r"D:\Research\NO-2D-Metamaterials\MODELS\training_runs"
    r"\NO_I3O5_BCF16_L1&L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401"
    r"\NO_I3O5_BCF16_L1&L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_E30.pth"
)


def _hparams_from_resolved_config(run_dir: Path) -> dict:
    rc = run_dir / "resolved_config.json"
    if not rc.is_file():
        return {}
    try:
        data = json.loads(rc.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    p = data.get("params") or {}
    a = data.get("args") or {}
    out: dict = {}
    for key in (
        "out_channels",
        "hidden_channels",
        "layers",
        "modes_height",
        "modes_width",
    ):
        if key in p and p[key] is not None:
            out[key] = p[key]
    enc = a.get("eigen_ch0_encoding") or p.get("eigen_ch0_encoding")
    if isinstance(enc, str) and enc:
        out["eigen_ch0_encoding"] = enc
    amp = a.get("amp") or p.get("amp")
    if isinstance(amp, str) and amp:
        out["amp"] = amp
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=("c_test", "b_test"),
        default="c_test",
        help="Which test split under --output-root to evaluate (default: c_test).",
    )
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\Research\NO-2D-Metamaterials\DATASETS"),
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory with resolved_config.json (defaults to parent of --model-path).",
    )
    p.add_argument("--batch-size", type=int, default=520)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default=None, help="Override AMP; default from resolved_config or none.")
    p.add_argument("--max-samples", type=int, default=0, help="If >0, evaluate only the first N samples (for quick tests).")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output PNG path (default: {_REPO / 'PLOTS'}/{{dataset}}_per_sample_mae_histogram.png).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    run_dir = (args.run_dir or model_path.parent).resolve()
    hp = _hparams_from_resolved_config(run_dir)

    out_channels = int(hp.get("out_channels", 5))
    hidden_channels = int(hp.get("hidden_channels", 128))
    layers = int(hp.get("layers", 4))
    modes_height = int(hp.get("modes_height", 32))
    modes_width = int(hp.get("modes_width", 32))
    eigen_ch0_encoding = str(hp.get("eigen_ch0_encoding", "uniform"))
    amp = args.amp if args.amp is not None else str(hp.get("amp", "none"))

    all_shards = efd.discover_test_shards(args.output_root, eigen_ch0_encoding)
    picked = [s for s in all_shards if s.name == args.dataset]
    if not picked:
        raise SystemExit(f"No {args.dataset!r} shard under {args.output_root} (found: {[s.name for s in all_shards]})")

    base_ds = efd.ShardedTensorPairDataset(picked, out_channels=out_channels)
    if args.max_samples and args.max_samples > 0:
        n = min(int(args.max_samples), len(base_ds))
        dataset: Dataset = Subset(base_ds, range(n))
    else:
        dataset = base_ds

    device = efd.resolve_device(args.allow_cpu)
    model = efd.FourierNeuralOperator(
        modes_height=modes_height,
        modes_width=modes_width,
        hidden_channels=hidden_channels,
        n_layers=layers,
        out_channels=out_channels,
    ).to(device)
    efd.load_model_state(model, model_path)

    losses = [efd.normalize_loss_name("l1")]
    ev = efd.run_evaluation(
        dataset=dataset,
        model=model,
        model_path=model_path,
        device=device,
        output_root=args.output_root.resolve(),
        losses=losses,
        out_channels=out_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=bool(args.pin_memory),
        amp=amp,
        unified_loss=False,
        per_sample_loss="l1",
        per_sample_loss_for_each=False,
        per_sample_criterion=None,
    )

    if ev.per_sample_losses is None:
        raise SystemExit("Expected per_sample_losses on EvaluationResult (unified_loss must be False).")

    vals = ev.per_sample_losses.float().numpy()
    out_path = args.out if args.out is not None else _REPO / "PLOTS" / f"{args.dataset}_per_sample_mae_histogram.png"
    print(
        f"{args.dataset} samples={len(vals)} mean_mae={float(vals.mean()):.6e} "
        f"median={float(np.median(vals)):.6e} "
        f"model={model_path.name}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(vals, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Per-sample MAE (mean |pred − target| over channels and pixels)")
    ax.set_ylabel("Count")
    ax.set_title(f"{args.dataset} — {model_path.name}")
    ax.axvline(float(vals.mean()), color="darkorange", linestyle="--", linewidth=1.5, label=f"mean = {float(vals.mean()):.4g}")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
