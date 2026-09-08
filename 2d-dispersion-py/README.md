# 2D dispersion (Bloch FEA) package

Python FEA utilities used by the root dataset generator
(`generate_dispersion_dataset.py` / `run_generate_dispersion_batched.py`).

## Install

From the repo root (recommended):

```bash
pip install -r requirements.txt
```

Package-local pins (FEA-focused subset):

```bash
pip install -r 2d-dispersion-py/requirements.txt
```

Add this directory to `PYTHONPATH`, or rely on the generator script, which inserts
`2d-dispersion-py/` on `sys.path` automatically.

## Main entry used by dataset generation

| Module | Role |
|--------|------|
| `dispersion_with_matrix_save_opt.py` | Assemble `K`/`M`, reduced eigensolve per wavevector |
| `design_parameters.py` / `get_design2.py` / `kernels.py` | Correlated unit-cell designs |
| `design_conversion.py` | Steel–polymer material mapping, binarize |
| `wavevectors.py` | IBZ sampling (`25×13` in the default pipeline) |
| `system_matrices*.py` / `elements*.py` | FEM assembly |
| `plot_dispersions*.py` / `plot_dispersion_infer_eigenfrequencies.py` | Evaluation plots |

## Tests

```bash
cd 2d-dispersion-py
python -m pytest tests/ -q
# or: python tests/run_tests.py
```

Plotting smoke tests may write figures under `test_plots/` (gitignored).

## Notes

- Demo PNG trees (`plots/`, `png/`) are gitignored; regenerate locally if needed.
- `fileshare_licenses/` retains the license text for `linspaceNDim` used in `utils.py`.
