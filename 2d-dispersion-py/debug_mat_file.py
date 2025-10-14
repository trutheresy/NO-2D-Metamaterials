"""
Debug script to diagnose MATLAB v7.3 file loading issues.

This script helps identify data structure and type issues when loading
MATLAB files with h5py.

Usage:
    python debug_mat_file.py path/to/your/file.mat
"""

import sys
import numpy as np
import h5py
from pathlib import Path

def inspect_hdf5_item(item, name, indent=0):
    """Print detailed information about an HDF5 item."""
    prefix = "  " * indent
    
    if isinstance(item, h5py.Dataset):
        data = item[()]
        print(f"{prefix}├─ {name}: Dataset")
        print(f"{prefix}│  ├─ Shape: {data.shape}")
        print(f"{prefix}│  ├─ Dtype: {data.dtype}")
        print(f"{prefix}│  ├─ Dtype kind: {data.dtype.kind}")
        print(f"{prefix}│  ├─ Dtype itemsize: {data.dtype.itemsize} bytes")
        
        if data.dtype.names:
            print(f"{prefix}│  ├─ Structured array with fields: {data.dtype.names}")
        
        if data.dtype == h5py.ref_dtype:
            print(f"{prefix}│  └─ Type: Cell array (HDF5 references)")
        elif data.dtype.kind == 'V':
            print(f"{prefix}│  └─ Type: Void type (likely complex data as binary)")
        else:
            print(f"{prefix}│  └─ Type: Regular array")
            if data.size <= 5:
                print(f"{prefix}│     └─ Values: {data.flatten()}")
    
    elif isinstance(item, h5py.Group):
        keys = list(item.keys())
        print(f"{prefix}├─ {name}: Group")
        print(f"{prefix}│  ├─ Keys: {keys}")
        
        # Check if sparse matrix
        if 'data' in keys and 'ir' in keys and 'jc' in keys:
            print(f"{prefix}│  └─ Type: Sparse matrix")
            data_dtype = item['data'][()].dtype
            print(f"{prefix}│     ├─ data dtype: {data_dtype}")
            print(f"{prefix}│     ├─ data dtype kind: {data_dtype.kind}")
            print(f"{prefix}│     └─ data itemsize: {data_dtype.itemsize} bytes")
        else:
            print(f"{prefix}│  └─ Type: Struct")


def main():
    """Main diagnostic function."""
    if len(sys.argv) < 2:
        print("Usage: python debug_mat_file.py <filepath.mat>")
        print("\nThis script inspects MATLAB v7.3 files and shows their HDF5 structure.")
        return
    
    filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return
    
    print("=" * 70)
    print(f"Inspecting MATLAB v7.3 file: {filepath.name}")
    print("=" * 70)
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"\nTop-level variables: {len(f.keys())}")
            print()
            
            for key in f.keys():
                if key.startswith('#'):
                    continue
                
                item = f[key]
                inspect_hdf5_item(item, key, indent=0)
                print()
                
                # For K_DATA, M_DATA, T_DATA - inspect first element
                if key in ['K_DATA', 'M_DATA', 'T_DATA']:
                    data = item[()]
                    if data.dtype == h5py.ref_dtype and data.size > 0:
                        ref = data.flat[0]
                        if ref:
                            print(f"  First element of {key}:")
                            inspect_hdf5_item(f[ref], f"{key}[0]", indent=1)
                            print()
    
    except Exception as e:
        print(f"\nERROR during inspection: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 70)
    print("Inspection complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

