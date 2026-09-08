# Neural Operators for 2D Acoustic Metamaterials

Python pipeline to generate Bloch FEA dispersion datasets, train a Fourier Neural
Operator (FNO) on geometry + modal encodings, and evaluate predictions
(displacement fields, eigenfrequencies, dispersion overlays).

## Quick start

```bash
# 1) Create the environment (conda)
conda env create -f environment.yml
conda activate NO_2D_Metamaterials

# or: pip install -r requirements.txt   (into your own venv / conda env)

# 2) Generate data  → see DATA_GENERATION.md
python run_generate_dispersion_batched.py --help

# 3) Train          → see ML_TRAINING.md
python train_from_disk.py --help

# 4) Evaluate       → see INFERENCE_PIPELINE.md
python run_model_inference_gpu.py --help
```

## Documentation

| Doc | Contents |
|-----|----------|
| [DATA_GENERATION.md](DATA_GENERATION.md) | Design synthesis, FEA solve (`2d-dispersion-py/`), tensor layouts |
| [ML_TRAINING.md](ML_TRAINING.md) | I3O5 FNO training with `train_from_disk.py` |
| [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md) | Prediction, metrics, and standard result figures |

## Repository layout

| Path | Role |
|------|------|
| `2d-dispersion-py/` | Bloch FEA solver and dispersion plotting utilities |
| Root `*.py` | Dataset build, training, inference, figure scripts |
| `DATASETS/`, `MODELS/`, `INFERENCE/`, `PLOTS/` | Local runtime trees (gitignored) |

Runtime folders are **not** shipped in git. Populate `DATASETS/` by running the
generation pipeline (or by placing an existing dataset tree that matches the
docs). Checkpoints land under `MODELS/training_runs/`.

Optional local-only tooling may live under `DIAGNOSTICS/` (also gitignored).

## Model contract (default)

- **Input** `(B, 3, 32, 32)`: geometry, wavevector embedding, band embedding  
- **Output** `(B, 5, 32, 32)`: eigenfrequency channel + 4 displacement channels  
- **Solver grid**: `32×32` pixels, `25×13` IBZ wavevectors, 6 bands  

Default training settings that work well: NMAE loss, 128 hidden channels, 4 Fourier
layers, batch size 520, 12 epochs (see `ML_TRAINING.md`).

## License / citation

Add your preferred license and citation text here before publishing.
