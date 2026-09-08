"""Derive per-sample NRMS arrays by square-rooting column 4 of existing NMSE .npy files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from per_sample_loss import nrms_array_from_nmse_array


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nmse-array", required=True, help="Input five-column per_sample_loss_nmse_*.npy path.")
    p.add_argument("--output", required=True, help="Output per_sample_loss_nrms_*.npy path.")
    args = p.parse_args()

    nmse_path = Path(args.nmse_array)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nmse_arr = np.load(nmse_path)
    nrms_arr = nrms_array_from_nmse_array(nmse_arr)
    np.save(out_path, nrms_arr)

    v = nrms_arr[:, 4]
    print(f"NMSE input : {nmse_path}  shape={nmse_arr.shape}")
    print(f"NRMS output: {out_path}  shape={nrms_arr.shape}")
    print(f"  mean={v.mean():.6e}  median={np.median(v):.6e}  min={v.min():.6e}  max={v.max():.6e}")


if __name__ == "__main__":
    main()
