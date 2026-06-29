"""
Shared output-layout helper for the inference / analysis pipeline.

All downstream results are organized by *model* and *dataset* under two top-level roots:

    PLOTS/<model>/<dataset>/<script-output-folder>/      # primarily-plot outputs
    INFERENCE/<model>/<dataset>/<script-output-folder>/   # primarily-data outputs

- ``<model>``   : the model checkpoint name with the dataset prefix (``B_``/``C_``) and the
                  trailing ``_<YYMMDD-HHMMSS>`` run timestamp stripped off
                  (e.g. ``NO_..._260401_best_fno2d_compat``).
- ``<dataset>`` : the evaluation set tag (e.g. ``c_test`` / ``b_test``).
- ``<script-output-folder>`` : a stable folder name per script
                  (e.g. ``MAE_sample_case_plots``, ``histograms``, ``boundary_length_vs_loss``).

Scripts call :func:`resolve_script_output_dir`, which creates the model folder (and the
dataset / script-output subfolders) if they do not yet exist and returns the final path.
An explicit ``--output-dir`` always wins, so existing call sites keep working unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLOTS_ROOT = ROOT / "PLOTS"
INFERENCE_ROOT = ROOT / "INFERENCE"

CATEGORY_ROOTS = {
    "plots": PLOTS_ROOT,
    "inference": INFERENCE_ROOT,
}

# Trailing run timestamp, e.g. "_260530-023704".
_TIMESTAMP_RE = re.compile(r"_\d{6}-\d{6}$")
# Leading single-letter dataset prefix on a run folder, e.g. "B_NO_...", "C_NO_...".
_DATASET_PREFIX_RE = re.compile(r"^[A-Za-z]_(?=NO_)")


def model_name_from_run_name(run_name: str) -> str:
    """Strip the dataset prefix (``B_``/``C_``) and trailing ``_<timestamp>`` from a run folder.

    ``B_NO_..._260401_best_fno2d_compat_260530-024713`` -> ``NO_..._260401_best_fno2d_compat``
    ``NO_..._260401_best_fno2d_compat_260530-023704``    -> ``NO_..._260401_best_fno2d_compat``
    """
    name = _DATASET_PREFIX_RE.sub("", run_name.strip())
    name = _TIMESTAMP_RE.sub("", name)
    return name


def category_root(category: str) -> Path:
    try:
        return CATEGORY_ROOTS[category]
    except KeyError:
        raise ValueError(
            f"category must be one of {sorted(CATEGORY_ROOTS)}; got {category!r}."
        ) from None


def model_dir(category: str, model_name: str, *, create: bool = False) -> Path:
    """Return ``<root>/<model_name>`` for the given category, optionally creating it."""
    if not model_name:
        raise ValueError("model_name must be a non-empty string.")
    d = category_root(category) / model_name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_output_dir(
    category: str,
    model_name: str,
    dataset: str = "",
    subdir: str = "",
    *,
    create: bool = True,
) -> Path:
    """Build ``<root>/<model_name>[/<dataset>][/<subdir>]`` and create it by default."""
    d = model_dir(category, model_name)
    if dataset:
        d = d / dataset
    if subdir:
        d = d / subdir
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_script_output_dir(
    *,
    explicit: str | Path | None,
    category: str,
    model_name: str | None,
    dataset: str = "",
    subdir: str = "",
    fallback: str | Path | None = None,
    create: bool = True,
) -> Path:
    """Resolve a script's output directory under the shared PLOTS/INFERENCE layout.

    Priority:
      1. ``explicit`` (e.g. a ``--output-dir`` the caller passed) -- used verbatim.
      2. ``<root>/<model_name>/<dataset>/<subdir>`` when ``model_name`` is given.
      3. ``fallback`` (legacy behavior, e.g. the prediction file's folder).

    The chosen directory (and any missing parents, including the ``<model_name>`` folder)
    is created when ``create`` is True.
    """
    if explicit:
        out = Path(explicit)
    elif model_name:
        out = resolve_output_dir(category, model_name, dataset, subdir, create=False)
    elif fallback is not None:
        out = Path(fallback)
    else:
        raise ValueError(
            "Provide --output-dir, or --model-name (with --output-subdir), or a fallback."
        )
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out
