# ML Training

Train a Fourier Neural Operator (FNO) that maps metamaterial geometry plus modal
encodings to eigenfrequency and Bloch displacement fields.

**Canonical trainer:** [`train_from_disk.py`](train_from_disk.py)

**Related:** [DATA_GENERATION.md](DATA_GENERATION.md) · [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md)

---

## 1) Prerequisites

Datasets under `--output-root` (default `DATASETS/`):

| Role | Folder prefix |
|------|----------------|
| Train | `c_train_*`, `b_train_*` |
| Test / validation | `c_test`, `b_test` |

Each shard’s latest `*_pt` directory must contain:

- `inputs.pt`
- `outputs.pt` (displacement channels)
- `reduced_indices.pt`
- `eigenfrequency_uniform_full.pt` and/or `eigenfrequency_fft_full.pt`

See [DATA_GENERATION.md](DATA_GENERATION.md) for how those files are produced.

Conda environment used on this project: `NO_2D_Metamaterials`.

---

## 2) Model contract (I3O5)

| | Shape / meaning |
|--|-----------------|
| Input | `(B, 3, 32, 32)` — geometry, wavevector embedding, band embedding |
| Output | `(B, 5, 32, 32)` — ch0 eigenfrequency patch; ch1–4 = `u_x,u_y` real/imag |
| Architecture | `FourierNeuralOperator` (FNO2d) from NeuralOp |

Default working configuration:

| Setting | Value |
|---------|-------|
| Hidden channels | `128` |
| Fourier layers | `4` |
| Modes (H×W) | `32 × 32` |
| Loss | **NMAE** (`--loss nmae`) |
| Optimizer | AdamW |
| Learning rate | `2e-3` |
| Weight decay | `0` |
| Batch size | `520` |
| Epochs | `12` |
| Scheduler | StepLR, `--step-size 1`, `--gamma 0.9` |
| Eigenfrequency ch0 | `--eigen-ch0-encoding uniform` (or `fft`) |

Wavelet input embeddings are the default path that performs best for mode selection.
Constant or sinusoidal input stacks are supported for comparisons
(`build_constant_input_tensors.py` / `build_sinusoidal_input_tensors.py` + matching
`--input-encoding` at inference time).

---

## 3) How `train_from_disk.py` works

### On-the-fly targets

`ShardedTensorPairDataset` builds each sample as:

1. Input row from `inputs.pt`
2. Output ch0 from `eigenfrequency_*_full.pt[design, wv, band]` via `reduced_indices`
3. Output ch1–4 from `outputs.pt`

Stacked `outputs_w_*.pt` files are optional and **not** required by this trainer.

Shards are memory-mapped (`mmap=True`). `ShardAwareBatchSampler` keeps batches inside
a single shard to reduce disk thrashing.

### Training loop

1. Discover train (`c_train`/`b_train`) and test (`c_test`/`b_test`) shards.
2. Build datasets and DataLoaders (`--num-workers 2`, `--prefetch-factor 3` are solid
   Windows defaults).
3. Train; log per-epoch train/val loss and per-channel losses.
4. Save checkpoints and configs; track best validation loss.

### Outputs

Under `MODELS/training_runs/<run_name>/`:

| Artifact | Purpose |
|----------|---------|
| `<run_name>_E{k}.pth`, `_best.pth`, `_final.pth` | Checkpoints |
| `resolved_config.json` | Architecture + CLI resolved for inference |
| `metrics.csv` | Train/val curves |
| `run_metadata.json` | Run status (`running` / `completed` / `failed`) |
| optional `diagnostics/` | Panels from `DIAGNOSTICS/diagnostic_panels.py` if enabled (local-only folder) |

### Run name pattern

```
NO_I3O5_BCF16_{LOSS}_HC{hidden}_LR{lr}_WD{wd}_SS{step}_G{gamma}_{ch0u|ch0fft}_{YYMMDD}
```

`I3O5` = 3 inputs / 5 outputs; `BCF16` = binary+continuous float16 data.

---

## 4) Recommended command shape

```bash
python train_from_disk.py \
  --output-root DATASETS \
  --save-dir MODELS/training_runs \
  --loss nmae \
  --epochs 12 \
  --batch-size 520 \
  --hidden-channels 128 \
  --layers 4 \
  --learning-rate 2e-3 \
  --weight-decay 0 \
  --step-size 1 \
  --gamma 0.9 \
  --eigen-ch0-encoding uniform
```

Resume or extend an existing run with `--resume-run-dir` / `--extend-epochs` rather than
overwriting a live folder. Only one GPU training job at a time on a single-GPU machine;
check `MODELS/training_runs/*/run_metadata.json` for `"status": "running"` before
launching another CUDA workload.

---

## 5) Other trainers (optional)

| Script | Use |
|--------|-----|
| `train_disk_mlflow.py` | Same I3O5 flow with MLflow logging |
| `train_from_disk_eigenfrequency.py` | I3O1 — eigenfrequency only |
| `train_from_disk_displacement.py` | I3O4 — displacements only |
| `train_from_disk_lambda_weighted.py` | I3O5 with channel-weighted loss |
| `train_from_disk_fast.py` | Faster experimental variant |

Prefer `train_from_disk.py` unless you specifically need one of the above.

---

## 6) Next step

Evaluate checkpoints on `c_test` / `b_test` with [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md).
Metric tables across runs: edit `MODELS/training_runs/comparison_registry.json`, then
`python report_metrics_table.py`.
