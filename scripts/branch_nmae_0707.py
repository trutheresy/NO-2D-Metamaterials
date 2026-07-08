"""Branch NMAE 0705 run to 0707: rollback to ep 12, relabel artifacts."""
from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "MODELS" / "training_runs"
SRC_NAME = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705"
DST_NAME = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260707"
SRC = RUNS / SRC_NAME
DST = RUNS / DST_NAME
MAX_EPOCH = 12
MAE_EPOCH_RE = re.compile(r"epoch=(1[3-6])/20\b|epoch_01[3-6]\b|loss=mae\b|active=mae\b")


def replace_run_id(text: str) -> str:
    return text.replace(SRC_NAME, DST_NAME)


def truncate_metrics_csv(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as rf:
        rows = list(csv.DictReader(rf))
        fieldnames = rows[0].keys() if rows else []
    kept = [r for r in rows if int(float(r["epoch"])) <= MAX_EPOCH]
    with path.open("w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return len(kept)


def truncate_metrics_jsonl(path: Path) -> int:
    kept: list[str] = []
    with path.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("epoch", 0)) <= MAX_EPOCH:
                kept.append(replace_run_id(line))
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept)


def filter_train_log(path: Path) -> tuple[int, int]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    kept: list[str] = []
    for line in lines:
        if MAE_EPOCH_RE.search(line):
            continue
        if "epoch_range=13.." in line:
            continue
        if "reset optimizer/scheduler" in line.lower():
            continue
        if "Loss logging: active=mae" in line:
            continue
        if "Fault diagnostics" in line and "04:09:50" in line:
            continue
        kept.append(replace_run_id(line))
    path.write_text("".join(kept), encoding="utf-8")
    return len(lines), len(kept)


def rename_checkpoints(run_dir: Path) -> None:
    for p in sorted(run_dir.glob(f"{SRC_NAME}_*")):
        p.rename(p.with_name(p.name.replace(SRC_NAME, DST_NAME, 1)))


def delete_epoch_artifacts(run_dir: Path) -> None:
    for ep in range(MAX_EPOCH + 1, 17):
        ckpt = run_dir / f"{DST_NAME}_E{ep}.pth"
        ckpt.unlink(missing_ok=True)
    diag = run_dir / "diagnostics"
    if diag.is_dir():
        for ep in range(MAX_EPOCH + 1, 17):
            d = diag / f"epoch_{ep:03d}"
            if d.is_dir():
                shutil.rmtree(d)
    for stale in (
        run_dir / "training_state_latest.pt",
        run_dir / f"{DST_NAME}_final.pth",
    ):
        stale.unlink(missing_ok=True)


def patch_json_file(path: Path, *, branched_from: str) -> None:
    text = replace_run_id(path.read_text(encoding="utf-8"))
    data = json.loads(text)
    if path.name == "run_metadata.json":
        data["run_name"] = DST_NAME
        data["run_id"] = DST_NAME
        data["status"] = "completed"
        data["best_epoch"] = MAX_EPOCH
        data["best_val_loss"] = 0.2396924654463927
        data["branched_from_run_dir"] = branched_from
        data["branched_at_utc"] = datetime.now(timezone.utc).isoformat()
        for key in (
            "resumed_at_utc",
            "resume_extend_epochs",
            "stopped_at_utc",
            "stop_reason",
            "completed_epochs",
            "pinned_best_from_epoch",
        ):
            data.pop(key, None)
    if path.name == "summary.json":
        data["run_name"] = DST_NAME
        data["run_id"] = DST_NAME
        data["status"] = "completed"
        data["best_epoch"] = MAX_EPOCH
        data["best_val_loss"] = 0.2396924654463927
        data["branched_from_run_dir"] = branched_from
        data.pop("stopped_at_utc", None)
        data.pop("stop_reason", None)
        data.pop("completed_epochs", None)
        data.pop("pinned_best_from_epoch", None)
        if "checkpoints" in data:
            data["checkpoints"].pop("best_pinned_copy", None)
    if path.name == "resolved_config.json":
        if "args" in data:
            data["args"]["loss"] = "nmae"
            data["args"]["resume_run_dir"] = ""
            data["args"]["extend_epochs"] = 0
            data["args"]["epochs"] = 12
            data["args"]["reset_optimizer_scheduler"] = False
            data["args"]["resume_from_epoch"] = 0
        if "params" in data:
            data["params"]["loss"] = "nmae"
            data["params"]["epochs"] = 12
            data["params"].pop("start_epoch", None)
        data["resume_mode"] = False
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def replace_in_text_files(run_dir: Path) -> None:
    for rel in ("main_process_fault.log",):
        p = run_dir / rel
        if p.is_file() and p.stat().st_size:
            p.write_text(replace_run_id(p.read_text(encoding="utf-8")), encoding="utf-8")


def update_registry() -> None:
    reg_path = RUNS / "comparison_registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    reg["runs"]["NMAE_0707"] = {
        "run_dir": DST_NAME,
        "val_index_source": "indices_full",
        "branched_from": "NMAE_0705",
    }
    reg_path.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"Source run missing: {SRC}")
    if DST.exists():
        raise SystemExit(f"Destination already exists: {DST}")

    shutil.copytree(SRC, DST)
    rename_checkpoints(DST)
    delete_epoch_artifacts(DST)

    n_csv = truncate_metrics_csv(DST / "metrics.csv")
    n_jsonl = truncate_metrics_jsonl(DST / "metrics.jsonl")
    n_in, n_out = filter_train_log(DST / "train.log")

    for name in ("run_metadata.json", "summary.json", "resolved_config.json"):
        patch_json_file(DST / name, branched_from=str(SRC))
    replace_in_text_files(DST)
    update_registry()

    print(f"Created branch: {DST}")
    print(f"  metrics rows kept: {n_csv}")
    print(f"  metrics.jsonl rows kept: {n_jsonl}")
    print(f"  train.log lines: {n_in} -> {n_out}")
    print(f"  registry: added NMAE_0707")


if __name__ == "__main__":
    main()
