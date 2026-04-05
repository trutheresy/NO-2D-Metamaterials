from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import torch


EPOCH_RE = re.compile(r"_E(\d+)\.pth$", re.IGNORECASE)


def _choose_checkpoint(run_dir: Path) -> Tuple[Path | None, str]:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best = summary.get("checkpoints", {}).get("best")
            if best:
                best_path = Path(best)
                if not best_path.is_absolute():
                    best_path = run_dir / best_path
                if best_path.is_file():
                    return best_path, "summary.best"
        except Exception:
            pass

    epoch_files = []
    for p in run_dir.glob("*.pth"):
        m = EPOCH_RE.search(p.name)
        if m:
            epoch_files.append((int(m.group(1)), p))
    if epoch_files:
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        return epoch_files[0][1], f"highest_epoch_E{epoch_files[0][0]}"

    return None, "none_found"


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
