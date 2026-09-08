"""Fast launcher for train_from_disk.py with GPU throughput optimizations on by default.

This is a thin wrapper around ``train_from_disk.main`` that enables three
low-risk speedups for RTX 40-series (Ada) / Ampere GPUs:

  1. TF32 matmul + cuDNN (``--tf32``)         -> faster fp32 tensor-core math
  2. cuDNN autotuner (``--cudnn-benchmark``)  -> best conv kernels for fixed shapes
  4. bf16 autocast (``--amp bf16``)           -> mixed precision forward/backward

(Item 3, torch.compile / channels_last, is intentionally not enabled here.)

bf16 + FFT
----------
The model is a neuralop FNO whose spectral convolutions call
``torch.fft.rfftn`` / ``irfftn``. Those ops do NOT support bf16/fp16 on CUDA
("Unsupported dtype BFloat16"), and torch.autocast does not automatically cast
their inputs back to fp32. Under ``--amp bf16`` this launcher therefore installs
a small shim that runs every FFT in fp32 (input cast to float, complex result
flows on unchanged). This keeps the accuracy-sensitive spectral transform in
fp32 while still letting the pointwise/skip convolutions and channel mixing run
in bf16. The dominant complex mode-multiply is accelerated by TF32 regardless.

Every optimization is an ordinary flag on ``train_from_disk.py``; this launcher
only supplies sensible defaults and the FFT shim. Any flag can be overridden on
the command line, and all other train_from_disk flags are forwarded unchanged.

Examples
--------
Quick single-shard perf test (one c_train dataset, small val):
    python train_from_disk_fast.py \
        --train-prefixes c_train_01 --test-prefixes c_test --max-test-samples 50000 \
        --hidden-channels 128 --learning-rate 2e-3 --loss nmae \
        --step-size 1 --gamma 0.9 --batch-size 520 --num-workers 2 \
        --prefetch-factor 3 --seed 0 --progress-mode plain --epochs 4

TF32 + cudnn.benchmark only (no bf16):
    python train_from_disk_fast.py --amp none ...  # everything else unchanged
"""
from __future__ import annotations

import sys

import torch

import train_from_disk as tfd

_LOW_PRECISION = (torch.bfloat16, torch.float16)


def _patch_fft_fp32() -> None:
    """Force torch.fft transforms to run in fp32 even under bf16/fp16 autocast.

    neuralop calls ``torch.fft.rfftn``/``irfftn`` as module attributes, so
    replacing those attributes intercepts every spectral-conv transform.
    """
    fft_names = ("rfftn", "irfftn", "rfft", "irfft", "rfft2", "irfft2", "fftn", "ifftn")
    for name in fft_names:
        orig = getattr(torch.fft, name, None)
        if orig is None:
            continue

        def make_wrapper(fn):
            def wrapper(input, *args, **kwargs):  # noqa: A002 - mirror torch signature
                if torch.is_tensor(input) and input.dtype in _LOW_PRECISION:
                    with torch.autocast(device_type="cuda", enabled=False):
                        return fn(input.float(), *args, **kwargs)
                return fn(input, *args, **kwargs)

            return wrapper

        setattr(torch.fft, name, make_wrapper(orig))


def _inject_default(argv: list[str], flag: str, value: str | None, *, aliases: tuple[str, ...] = ()) -> None:
    """Append ``flag`` (optionally with ``value``) if the user did not supply it.

    ``aliases`` lists mutually-exclusive forms (e.g. --no-tf32) that also count
    as the user having made a choice, so we never override an explicit decision.
    """
    present = any(a == flag or a.startswith(flag + "=") for a in argv)
    for alias in aliases:
        present = present or any(a == alias or a.startswith(alias + "=") for a in argv)
    if present:
        return
    argv.append(flag)
    if value is not None:
        argv.append(value)


def _amp_requested(argv: list[str]) -> str:
    for i, a in enumerate(argv):
        if a == "--amp" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--amp="):
            return a.split("=", 1)[1]
    return ""


def main() -> None:
    argv = sys.argv[1:]

    _inject_default(argv, "--tf32", None, aliases=("--no-tf32",))
    _inject_default(argv, "--cudnn-benchmark", None, aliases=("--no-cudnn-benchmark",))
    _inject_default(argv, "--amp", "bf16")

    if _amp_requested(argv) in ("bf16", "fp16"):
        _patch_fft_fp32()

    sys.argv = [sys.argv[0]] + argv
    tfd.main()


if __name__ == "__main__":
    main()
