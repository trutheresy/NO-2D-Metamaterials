from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run_one_batch(
    repo_root: Path,
    n_struct: int,
    seed_offset: int,
    binarize: bool,
    log_path: Path,
) -> dict:
    print(
        f"[START] n_struct={n_struct} seed_offset={seed_offset} "
        f"binarize={binarize} log={log_path}"
    )
    cmd = [
        sys.executable,
        str(repo_root / "generate_dispersion_dataset_Han_Alex.py"),
        "--n-struct",
        str(n_struct),
        "--rng-seed-offset",
        str(seed_offset),
        "--skip-demo",
    ]
    if binarize:
        cmd.append("--binarize")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_file.write(proc.stdout)

    output_pt_path = ""
    output_pkl_path = ""
    for line in proc.stdout.splitlines():
        if "SUCCESS: PyTorch dataset bundle saved to:" in line:
            output_pt_path = line.split("SUCCESS: PyTorch dataset bundle saved to:", 1)[1].strip()
        elif "SUCCESS: Python pickle file saved to:" in line:
            output_pkl_path = line.split("SUCCESS: Python pickle file saved to:", 1)[1].strip()

    return {
        "exit_code": int(proc.returncode),
        "seed_offset": int(seed_offset),
        "n_struct": int(n_struct),
        "output_pt_path": output_pt_path,
        "output_pkl_path": output_pkl_path,
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generate_dispersion_dataset_Han_Alex.py in seed-shifted batches.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--total-samples", type=int, default=24000, help="Total training samples to generate.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Structures per batch.")
    parser.add_argument("--start-seed-offset", type=int, default=0, help="Seed offset for first training batch.")
    parser.add_argument("--run-validation", action="store_true", help="Generate a validation batch after training batches.")
    parser.add_argument("--validation-size", type=int, default=1000, help="Validation structures to generate.")
    parser.add_argument("--validation-seed-offset", type=int, default=24000, help="Seed offset for validation batch.")
    parser.add_argument("--binarize", action="store_true", help="Generate binarized designs.")
    args = parser.parse_args()

    if args.total_samples % args.batch_size != 0:
        raise ValueError("--total-samples must be divisible by --batch-size.")

    repo_root = Path(args.repo_root).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = repo_root / "OUTPUT" / f"batched_generation_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "batch_size": int(args.batch_size),
        "total_samples": int(args.total_samples),
        "start_seed_offset": int(args.start_seed_offset),
        "binarize": bool(args.binarize),
        "train_batches": [],
        "validation_batch": None,
    }

    n_batches = args.total_samples // args.batch_size
    for batch_idx in range(n_batches):
        seed_offset = args.start_seed_offset + batch_idx * args.batch_size
        log_path = run_root / "logs" / f"train_batch_{batch_idx:03d}.log"
        result = _run_one_batch(
            repo_root=repo_root,
            n_struct=args.batch_size,
            seed_offset=seed_offset,
            binarize=args.binarize,
            log_path=log_path,
        )
        result["batch_idx"] = int(batch_idx)
        manifest["train_batches"].append(result)
        print(
            f"[DONE] train batch {batch_idx + 1}/{n_batches} "
            f"(seed_offset={seed_offset}) exit_code={result['exit_code']} "
            f"pt_out={result.get('output_pt_path') or 'N/A'}"
        )
        if result["exit_code"] != 0:
            print(f"[STOP] train batch {batch_idx + 1} failed. See log: {result['log_path']}")
            break

    if args.run_validation and all(b["exit_code"] == 0 for b in manifest["train_batches"]):
        v_log = run_root / "logs" / "validation_batch.log"
        v_result = _run_one_batch(
            repo_root=repo_root,
            n_struct=args.validation_size,
            seed_offset=args.validation_seed_offset,
            binarize=args.binarize,
            log_path=v_log,
        )
        v_result["batch_idx"] = "validation"
        manifest["validation_batch"] = v_result
        print(
            f"[DONE] validation batch (seed_offset={args.validation_seed_offset}) "
            f"exit_code={v_result['exit_code']} "
            f"pt_out={v_result.get('output_pt_path') or 'N/A'}"
        )

    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    successful_train = sum(1 for b in manifest["train_batches"] if b["exit_code"] == 0)
    print(
        f"[SUMMARY] successful_train_batches={successful_train}/{n_batches} "
        f"run_root={run_root}"
    )
    print(json.dumps({"manifest": str(manifest_path), "run_root": str(run_root)}, indent=2))


if __name__ == "__main__":
    main()

