from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one .pt tensor/object to .mat")
    parser.add_argument("--pt-path", required=True, help="Path to input .pt file")
    parser.add_argument("--mat-out", required=True, help="Path to output .mat file")
    parser.add_argument("--var-name", default=None, help="MATLAB variable name (default: stem)")
    args = parser.parse_args()

    pt_path = Path(args.pt_path)
    mat_out = Path(args.mat_out)
    mat_out.parent.mkdir(parents=True, exist_ok=True)
    var_name = args.var_name or pt_path.stem

    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    else:
        # Best effort for non-tensor objects.
        arr = np.asarray(obj)

    sio.savemat(str(mat_out), {var_name: arr})
    print(f"Saved {pt_path} -> {mat_out} as variable '{var_name}', shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
