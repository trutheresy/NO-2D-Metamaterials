# ML Training

Neural-operator training workflow. Canonical trainer:
[`train_from_disk.py`](train_from_disk.py). MLflow-integrated variant:
[`train_disk_mlflow.py`](train_disk_mlflow.py).

Data production: [DATA_GENERATION.md](DATA_GENERATION.md).
Inference after training: [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md).
GPU / run conventions: [RULES.md](RULES.md).

---

## 1) Data prep (prerequisite)

Training reads per-dataset bundles under `--output-root` (default `DATASETS/`):

- **Train:** `c_train_*`, `b_train_*`
- **Test:** `c_test`, `b_test`

Each dataset's latest `*_pt` folder needs `inputs.pt`, `outputs.pt`,
`reduced_indices.pt`, and the selected `eigenfrequency_{uniform,fft}_full.pt`.

Relevant prep scripts (full detail in [DATA_GENERATION.md](DATA_GENERATION.md)):

- `build_inputs_outputs_from_reduced_indices.py` — `inputs.pt` and optional stacked `outputs_w_*.pt`.
- `rename_and_downselect_indices.py` — `indices_full.pt` + downselected `reduced_indices.pt`.

---

## 2) Main trainer: `train_from_disk.py`

Disk-backed I3O5 training (3 input → 5 output channels), streaming shards so memory stays bounded.

### 2.1 Model contract

- **Input:** `(B, 3, 32, 32)` — geometry, wavevector embedding, band embedding.
- **Output:** `(B, 5, 32, 32)` — ch0 eigenfrequency, ch1–4 displacements (x_real, x_imag, y_real, y_imag).
- **Model:** `FourierNeuralOperator` (FNO2d), configurable `hidden_channels`, `n_layers`, `n_modes_*`.

### 2.2 On-the-fly target assembly

`ShardedTensorPairDataset` builds each target as:

- ch0 from `eigenfrequency_*_full.pt[d, w, b]` via `reduced_indices.pt`,
- ch1–4 from `outputs.pt[local_idx, 1:5]`.

The trainer does **not** depend on `outputs_w_*.pt`. Shards use `mmap=True`;
`ShardAwareBatchSampler` keeps batches shard-local.

### 2.3 Key CLI arguments (defaults)

| Argument | Default | Meaning |
|----------|---------|---------|
| `--output-root` | `DATASETS/` | Dataset root. |
| `--save-dir` | `MODELS/training_runs` | Run folders / checkpoints. |
| `--eigen-ch0-encoding` | uniform/fft | Which eigenfrequency tensor fills output ch0. |
| `--epochs` | 12 | Training epochs. |
| `--batch-size` | 520 | Samples per batch. |
| `--num-workers` / `--prefetch-factor` | 2 / 3 | DataLoader streaming. |
| `--hidden-channels` | 128 | FNO width. |
| `--layers` | 4 | FNO layers. |
| `--modes-height` / `--modes-width` | 32 / 32 | Fourier modes. |
| `--learning-rate` | 2e-3 | Optimizer LR. |
| `--weight-decay` | 0.0 | Weight decay. |
| `--loss` | `l1` | `mse`, `l1`, `smoothl1`, `nmae`, … (see script). |
| `--scheduler` | `steplr` | `steplr`, `cosine`, or `none`. |
| `--step-size` / `--gamma` | 1 / 0.9 | StepLR schedule. |
| `--amp` | `none` | `none`, `fp16`, `bf16`. |
| `--seed` | 0 | RNG seed. |

Manuscript primary runs use **NMAE**, batch **520**, step size **1**, gamma **0.9**, weight decay **0**.

### 2.4 Training flow

1. Discover `c_train`/`b_train` + `c_test`/`b_test` shards.
2. Build `ShardedTensorPairDataset` (mmap inputs / outputs / eigen ch0).
3. `DataLoader` + `ShardAwareBatchSampler`.
4. Train `FourierNeuralOperator` (in3 / out5); per-epoch train/val + per-channel losses.
5. Write checkpoints, `resolved_config.json`, `metrics.csv`; step scheduler; track best by val loss.

### 2.5 Outputs per run

Under `MODELS/training_runs/<run_name>/`:

- Epoch checkpoints `<run_name>_E{epoch}.pth`, `_best.pth`, `_final.pth`.
- `resolved_config.json` (consumed by evaluation tooling).
- `metrics.csv` (`train_loss`, `val_loss`, `lr`, per-channel losses).
- Optional diagnostic panels (`diagnostic_panels.py`).

### 2.6 Run naming

Example:

```
NO_I3O5_BCF16_{L1|L2|SL1|NMAE}_HC{hidden}_LR{lr}_WD{wd}_SS{step}_G{gamma}_{ch0u|ch0fft}_{YYMMDD}
```

`I3O5` = 3-in/5-out; `BCF16` = binarized+continuous at float16.

---

## 3) MLflow variant: `train_disk_mlflow.py`

Same dataset/model contract as `train_from_disk.py`, with MLflow tracking (params,
per-epoch metrics, checkpoint artifacts). Prefer `train_from_disk.py` for light local
runs.

---

## 4) Task-specific variants

| Script | Contract | Targets |
|--------|----------|---------|
| `train_from_disk.py` | **I3O5** (main) | eigenfrequency + 4 displacements |
| `train_from_disk_eigenfrequency.py` | I3O1 | single eigenfrequency channel |
| `train_from_disk_displacement.py` | I3O4 | displacement channels 1–4 |
| `train_from_disk_lambda_weighted.py` | I3O5 | loss-weighted objective |

All share the disk-backed streaming flow; they differ in output-channel count and targets.
