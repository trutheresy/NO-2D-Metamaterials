# Project Rules

Conventions for researchers and Cursor agents working in this repo.

**Related:** [ML_TRAINING.md](ML_TRAINING.md) · [DATA_GENERATION.md](DATA_GENERATION.md) · [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md) · [.cursor/rules/](.cursor/rules/)

---

## 1. Style guide and scripting

### Python style and linting

Match patterns already used in trainers and tooling (`train_from_disk.py`, `model_inference_common.py`, etc.):

- Use `from __future__ import annotations` and type hints on public functions.
- Prefer `pathlib.Path` over string paths; use `argparse` for CLI entry points.
- Follow PEP 8 naming; use descriptive names (`ShardInfo`, `build_run_name`, …).
- **Preallocate** lists, NumPy arrays, and Torch tensors in hot loops — do not grow with `.append()` inside tight loops ([`.cursor/rules/preallocate-lists.mdc`](.cursor/rules/preallocate-lists.mdc)).
- Use PyTorch as the primary DL framework; load disk tensors with `mmap=True` where the repo already does.
- For headless figure export, set `matplotlib.use("Agg")` before importing `pyplot`.
- Prefer **minimal diffs**: extend an existing module rather than copying logic into a new file.
- **Commits:** only create git commits when explicitly requested.
- **Do not commit binary artifacts:** `.gitignore` excludes `*.pt`, `*.pth`, `*.npy`, `*.zip`, etc.

### When to create a new script

| Situation | Action |
|-----------|--------|
| One-off check that will be discarded | Temporary script — naming below; **delete when done** |
| Same logic needed repeatedly | Add to or wrap an existing canonical script |
| New reusable pipeline step | New root-level `.py` with `argparse`, docstring, shared imports |
| Exploratory analysis | Notebook under `NOTEBOOKS/` (`*sandbox*`) — not a production script |

**Do not** add parallel copies of trainers or inference drivers at repo root. Canonical entry points: [ML_TRAINING.md](ML_TRAINING.md), [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md).

**Do not assume scripts exist** because they appear in old logs.

### Naming temporary / diagnostic scripts

| Prefix / pattern | Purpose | Lifecycle |
|------------------|---------|-----------|
| `_tmp_*`, `_eval_*`, `_debug_*`, `_plot_*` | Scratch / one-off at repo root | Delete when the question is answered |
| `*_debug.py` | Targeted debugging | Delete after debugging |
| `run_*_diagnostics*.py` | Named diagnostic tools | Keep if reused; document what they check |
| `diagnostic_panels.py` | Training-time test-set panels | Prefer over ad-hoc plotting during training |

All notebooks live under [`NOTEBOOKS/`](NOTEBOOKS/). Exploratory notebooks use `*sandbox*` in the name. Superseded notebooks use `obsolete` / `_old` in the name or live under `NOTEBOOKS/OBSOLETE/`.

---

## 2. Training and inference

This machine has a **single GPU**. There is no lock file — coordination is manual. Training state is in `MODELS/training_runs/<run_name>/run_metadata.json` (`"status": "running" | "completed" | "failed"`).

### GPU: do not start CUDA workloads during active training

If any training run is active, **do not start** CUDA scripts. Wait, use a CPU path, or ask the user.

**Before launching GPU work:**

1. Scan `MODELS/training_runs/*/run_metadata.json` for `"status": "running"`.
2. If found, check that run's `train.log` — if epochs are advancing, treat the GPU as occupied.
3. Optionally corroborate: live `train_from_disk*.py` process or `nvidia-smi`.

| If metadata scan… | Then… |
|-------------------|--------|
| no `"running"` | Proceed |
| `"running"` and log/process active | Stop — use CPU path or wait |
| `"running"` but log frozen / no process | Ask user before reclaiming GPU; optionally mark `"failed"` |

**CUDA-default scripts blocked while training:** `train_from_disk.py` (+ variants), `train_disk_mlflow.py`, `run_model_inference_gpu.py`, `evaluate_from_disk.py`, `backfill_val_dual_loss.py`, and wrappers that invoke them.

**Allowed while training:** `run_model_inference_cpu.py`, read-only log/metrics inspection, notebooks that do not load models on GPU.

### Training runs

- **One job at a time:** never launch a second trainer while another has `status: running`.
- **Resume, don't overwrite:** `--resume-run-dir` + `--extend-epochs`; `--output-run-dir` only when intentionally copying.
- **Output location:** `MODELS/training_runs/<run_name>/` via `build_run_name` ([ML_TRAINING.md](ML_TRAINING.md)).
- **Windows DataLoader defaults:** `--batch-size 520`, `--num-workers 2`, `--prefetch-factor 3` unless measured otherwise.
- **Environment:** conda env `NO_2D_Metamaterials`.
- **Do not edit active run artifacts** while `status: running`.

### Inference and evaluation

- **GPU vs CPU:** `run_model_inference_gpu.py` when free; `run_model_inference_cpu.py` when training holds the GPU.
- **Config fidelity:** load architecture from the run's `resolved_config.json`.
- **Layout:** `INFERENCE/<model>/<dataset>/…` and `PLOTS/<model>/…` — see [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md).
- Long jobs: redirect stdout/stderr to a log under `INFERENCE/` (keep logs; do not treat them as disposable scratch).

### Metric comparisons

Edit `MODELS/training_runs/comparison_registry.json` (one label per run, format `LOSS_MMDD`). Generate tables with `python report_metrics_table.py`.

---

## 3. Notebooks, data, and plot labelling

### Notebooks

Store every `.ipynb` under [`NOTEBOOKS/`](NOTEBOOKS/); not at repo root.

| Pattern | Use |
|---------|-----|
| `figures_{continuous\|binary}_I3O5_*.ipynb` | Publication / results figures |
| `figures_methodology.ipynb` | Methods figures |
| `figures_*obsolete*`, `figures_old_*` | Superseded — do not extend |
| `NO_trainer*.ipynb` | Legacy — prefer `train_from_disk.py` |
| `*sandbox*.ipynb` | Exploratory — not pipeline source of truth |

Prefer saving figures to `PLOTS/` rather than scattering PNGs at repo root.

### Data labelling and layout

See [DATA_GENERATION.md](DATA_GENERATION.md). Training discovers `DATASETS/` by prefix:

- **Train:** `c_train_*`, `b_train_*`
- **Test:** `c_test`, `b_test`

Each latest `*_pt` folder needs `inputs.pt`, `outputs.pt`, `reduced_indices.pt`, and `eigenfrequency_{uniform,fft}_full.pt`.

**In-place replacements:** rename the old file with an `_old` suffix before writing the new file.

**Do not run** `run_generate_dispersion_batched.py` (16 workers by default) concurrently with active training.

### Plot and figure output labelling

| Context | Location / pattern |
|---------|-------------------|
| Training diagnostics | `<run_dir>/diagnostics/epoch_<NNN>/` |
| Dataset histograms | Colocated in the `*_pt` folder |
| Exploratory / sweeps | `PLOTS/<study>/<variant>/` |
| Inference artifacts | `INFERENCE/<model>/<dataset>/` |
| Downstream CSVs | Beside inference output |

Figure export defaults: `dpi=150–160`, `bbox_inches="tight"` where used. Include epoch, sample rank, model name, and timestamp in filenames so outputs are traceable without opening files.
