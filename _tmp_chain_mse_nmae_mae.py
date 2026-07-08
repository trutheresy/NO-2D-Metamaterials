"""Temporary orchestrator: 260701 MSE x8, then fresh NMAE x12 + MAE x8."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = Path(r"C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe")
TRAIN = REPO / "train_from_disk.py"
SAVE_DIR = REPO / "MODELS" / "training_runs"
RUN_260701 = SAVE_DIR / "NO_I3O5_BCF16_NMSE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260701"
LOG = SAVE_DIR / "chain_mse_nmae_mae.log"

COMMON = [
    "--amp",
    "none",
    "--hidden-channels",
    "128",
    "--learning-rate",
    "2e-3",
    "--weight-decay",
    "0",
    "--scheduler",
    "steplr",
    "--step-size",
    "1",
    "--gamma",
    "0.9",
    "--eigen-ch0-encoding",
    "uniform",
    "--batch-size",
    "520",
    "--num-workers",
    "2",
    "--prefetch-factor",
    "3",
    "--seed",
    "0",
    "--progress-mode",
    "plain",
]


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} | {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_status(run_dir: Path) -> str:
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    return str(meta.get("status", "unknown"))


def wait_for_run(run_dir: Path, poll_sec: float = 60.0) -> None:
    while True:
        st = run_status(run_dir)
        if st == "completed":
            log(f"completed: {run_dir.name}")
            return
        if st == "failed":
            raise RuntimeError(f"run failed: {run_dir}")
        time.sleep(poll_sec)


def run_train(args: list[str], phase_log: Path) -> int:
    cmd = [str(PYTHON), str(TRAIN), *COMMON, *args]
    log(f"launch: {' '.join(cmd)}")
    with phase_log.open("w", encoding="utf-8") as out:
        out.write(f"=== {datetime.now(timezone.utc).isoformat()} ===\n")
        out.write(" ".join(cmd) + "\n\n")
        out.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    return proc.wait()


def latest_nmae_run(after_utc: str) -> Path:
    best: Path | None = None
    best_start = after_utc
    for meta_path in SAVE_DIR.glob("NO_I3O5_BCF16_NMAE_*/run_metadata.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        started = str(meta.get("started_at_utc", ""))
        if started >= after_utc and meta.get("status") == "completed":
            if started >= best_start:
                best_start = started
                best = meta_path.parent
    if best is None:
        raise FileNotFoundError(f"no completed NMAE run found after {after_utc}")
    return best


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    log("=== phase 1: 260701 resume + 8 epochs MSE ===")
    rc = run_train(
        [
            "--resume-run-dir",
            str(RUN_260701),
            "--extend-epochs",
            "8",
            "--loss",
            "mse",
            "--reset-optimizer-scheduler",
        ],
        SAVE_DIR / "chain_phase1_mse.log",
    )
    if rc != 0:
        log(f"phase 1 exited rc={rc}")
        return rc
    wait_for_run(RUN_260701)

    t0 = datetime.now(timezone.utc).isoformat()
    log("=== phase 2: fresh init + 12 epochs NMAE ===")
    rc = run_train(
        ["--epochs", "12", "--loss", "nmae"],
        SAVE_DIR / "chain_phase2_nmae.log",
    )
    if rc != 0:
        log(f"phase 2 exited rc={rc}")
        return rc

    nmae_dir = latest_nmae_run(t0)
    wait_for_run(nmae_dir)

    log(f"=== phase 3: {nmae_dir.name} resume + 8 epochs MAE ===")
    rc = run_train(
        [
            "--resume-run-dir",
            str(nmae_dir),
            "--extend-epochs",
            "8",
            "--loss",
            "mae",
            "--reset-optimizer-scheduler",
        ],
        SAVE_DIR / "chain_phase3_mae.log",
    )
    if rc != 0:
        log(f"phase 3 exited rc={rc}")
        return rc
    wait_for_run(nmae_dir)

    log("=== all phases complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
