from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch


def _choose_checkpoint(run_dir: Path) -> Tuple[Path | None, str]:
    best_files = sorted(p for p in run_dir.glob("*best.pth") if p.is_file())
    if not best_files:
        return None, "best_file_missing"
    if len(best_files) == 1:
        return best_files[0], "glob.best"
    # If multiple best checkpoints exist, pick the most recently modified one.
    best_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return best_files[0], "glob.best_latest_mtime"


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    # Most runs save a raw state_dict, but handle common wrappers too.
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")


def _to_unwrapped_fno2d_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Drop metadata-like keys and remove optional wrapper prefix "fno.".
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("_"):
            continue
        nk = k[4:] if k.startswith("fno.") else k
        out[nk] = v
    return out


def _output_path(src_path: Path, suffix: str) -> Path:
    return src_path.with_name(f"{src_path.stem}{suffix}{src_path.suffix}")


def convert_runs(training_runs_dir: Path, suffix: str) -> dict:
    report = {
        "training_runs_dir": str(training_runs_dir),
        "suffix": suffix,
        "runs": [],
    }
    for run_dir in sorted(p for p in training_runs_dir.iterdir() if p.is_dir()):
        src_ckpt, select_reason = _choose_checkpoint(run_dir)
        entry = {
            "run_dir": str(run_dir),
            "selected_by": select_reason,
            "source_checkpoint": str(src_ckpt) if src_ckpt else None,
            "status": "skipped",
        }
        if src_ckpt is None:
            report["runs"].append(entry)
            continue

        dst_ckpt = _output_path(src_ckpt, suffix)
        entry["output_checkpoint"] = str(dst_ckpt)
        if dst_ckpt.exists():
            entry["status"] = "exists_skip"
            report["runs"].append(entry)
            continue

        try:
            obj = torch.load(src_ckpt, map_location="cpu", weights_only=False)
            sd = _extract_state_dict(obj)
            sd_new = _to_unwrapped_fno2d_state_dict(sd)
            torch.save(sd_new, dst_ckpt)
            entry["status"] = "converted"
            entry["n_keys_in"] = int(len(sd))
            entry["n_keys_out"] = int(len(sd_new))
            entry["prefixed_keys_in"] = int(sum(1 for k in sd.keys() if k.startswith("fno.")))
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)

        report["runs"].append(entry)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert wrapper-saved checkpoints (fno.<...> keys) into direct FNO2d-compatible "
            "state_dict files without overwriting originals."
        )
    )
    parser.add_argument(
        "--training-runs-dir",
        default=str(Path(__file__).resolve().parent / "MODELS" / "training_runs"),
    )
    parser.add_argument(
        "--suffix",
        default="_fno2d_compat",
        help="Suffix appended to converted checkpoint filenames before .pth",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional path to write JSON conversion report",
    )
    args = parser.parse_args()

    training_runs_dir = Path(args.training_runs_dir).resolve()
    if not training_runs_dir.is_dir():
        raise FileNotFoundError(f"Training runs directory not found: {training_runs_dir}")

    report = convert_runs(training_runs_dir, args.suffix)
    print(json.dumps(report, indent=2))

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
