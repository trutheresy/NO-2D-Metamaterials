#!/usr/bin/env python3
"""Evaluate Huber run checkpoints for full-val L1/MSE without touching metrics.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate_from_disk import (
    FourierNeuralOperator,
    build_test_dataset,
    load_model_state,
    resolve_device,
    run_evaluation,
)

RUNS = ROOT / "MODELS" / "training_runs"
HUBER_DIR = RUNS / "NO_I3O5_BCF16_SL1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260626"
OUT_PATH = HUBER_DIR / "val_l1_mse_eval.json"


def main() -> None:
    run_name = HUBER_DIR.name
    rc = json.loads((HUBER_DIR / "resolved_config.json").read_text(encoding="utf-8"))
    rc_args = rc["args"]
    out_ch = int(rc["params"]["out_channels"])
    eigen = str(rc_args.get("eigen_ch0_encoding", "uniform"))
    output_root = Path(rc_args.get("output_root", str(ROOT / "DATASETS")))
    device = resolve_device(False)
    dataset = build_test_dataset(output_root, eigen, out_ch, val_full_test=True)
    model = FourierNeuralOperator(
        modes_height=int(rc_args.get("modes_height", 32)),
        modes_width=int(rc_args.get("modes_width", 32)),
        hidden_channels=int(rc_args.get("hidden_channels", 128)),
        n_layers=int(rc_args.get("layers", 4)),
        out_channels=out_ch,
    ).to(device)

    results: dict[str, dict[str, float]] = {}
    if OUT_PATH.is_file():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))

    for ep in range(1, 32):
        ckpt = HUBER_DIR / f"{run_name}_E{ep}.pth"
        if not ckpt.is_file():
            break
        if str(ep) in results:
            continue
        print(f"Evaluating epoch {ep}...", flush=True)
        load_model_state(model, ckpt, run_dir=HUBER_DIR)
        ev = run_evaluation(
            dataset=dataset,
            model=model,
            model_path=ckpt,
            device=device,
            output_root=output_root,
            losses=["l1", "mse"],
            out_channels=out_ch,
            batch_size=520,
            num_workers=2,
            prefetch_factor=3,
            pin_memory=True,
            amp="none",
            unified_loss=True,
        )
        results[str(ep)] = {
            "val_l1": float(ev.losses["l1"]["avg_pixel_loss"]),
            "val_mse": float(ev.losses["mse"]["avg_pixel_loss"]),
        }
        OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"epoch {ep} val_l1={results[str(ep)]['val_l1']:.6e} "
            f"val_mse={results[str(ep)]['val_mse']:.6e}",
            flush=True,
        )
    print(f"Done. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
