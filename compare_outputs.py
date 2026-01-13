#!/usr/bin/env python3
"""
Compare two output directories to verify they contain identical data.
"""

import torch
import numpy as np
from pathlib import Path
import sys


def compare_tensors(tensor1, tensor2, name, tolerance=1e-5):
    """Compare two tensors and return True if they match."""
    if tensor1.shape != tensor2.shape:
        print(f"  ❌ {name}: Shape mismatch - {tensor1.shape} vs {tensor2.shape}")
        return False
    
    if tensor1.dtype != tensor2.dtype:
        print(f"  ⚠️  {name}: Dtype mismatch - {tensor1.dtype} vs {tensor2.dtype}")
        # Continue comparison anyway
    
    # Convert to float32 for comparison if needed
    t1 = tensor1.float() if tensor1.dtype in [torch.float16, torch.float8_e4m3fn] else tensor1
    t2 = tensor2.float() if tensor2.dtype in [torch.float16, torch.float8_e4m3fn] else tensor2
    
    if torch.allclose(t1, t2, rtol=tolerance, atol=tolerance):
        print(f"  ✅ {name}: Matches (shape: {tensor1.shape}, dtype: {tensor1.dtype})")
        return True
    else:
        max_diff = torch.max(torch.abs(t1 - t2)).item()
        mean_diff = torch.mean(torch.abs(t1 - t2)).item()
        print(f"  ❌ {name}: Values differ - max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        return False


def compare_datasets(dir1, dir2):
    """Compare all .pt files in two directories."""
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)
    
    if not dir1_path.exists():
        print(f"Error: Directory 1 does not exist: {dir1_path}")
        return False
    
    if not dir2_path.exists():
        print(f"Error: Directory 2 does not exist: {dir2_path}")
        return False
    
    print(f"Comparing directories:")
    print(f"  Original: {dir1_path}")
    print(f"  New:      {dir2_path}")
    print("=" * 80)
    
    # Get all subdirectories
    subdirs1 = {d.name: d for d in dir1_path.iterdir() if d.is_dir()}
    subdirs2 = {d.name: d for d in dir2_path.iterdir() if d.is_dir()}
    
    if set(subdirs1.keys()) != set(subdirs2.keys()):
        print(f"❌ Subdirectory mismatch!")
        print(f"  Original: {set(subdirs1.keys())}")
        print(f"  New:      {set(subdirs2.keys())}")
        return False
    
    all_match = True
    
    for subdir_name in sorted(subdirs1.keys()):
        print(f"\n📁 Comparing subdirectory: {subdir_name}")
        print("-" * 80)
        
        subdir1 = subdirs1[subdir_name]
        subdir2 = subdirs2[subdir_name]
        
        # Get all .pt files
        files1 = {f.name: f for f in subdir1.glob("*.pt")}
        files2 = {f.name: f for f in subdir2.glob("*.pt")}
        
        if set(files1.keys()) != set(files2.keys()):
            print(f"  ❌ File mismatch in {subdir_name}!")
            print(f"    Original: {set(files1.keys())}")
            print(f"    New:      {set(files2.keys())}")
            all_match = False
            continue
        
        for filename in sorted(files1.keys()):
            file1 = files1[filename]
            file2 = files2[filename]
            
            try:
                # Load files
                data1 = torch.load(file1, map_location='cpu', weights_only=False)
                data2 = torch.load(file2, map_location='cpu', weights_only=False)
                
                # Handle different data types
                if filename == "displacements_dataset.pt":
                    # TensorDataset - compare each tensor
                    if len(data1.tensors) != len(data2.tensors):
                        print(f"  ❌ {filename}: Tensor count mismatch")
                        all_match = False
                        continue
                    
                    for i, (t1, t2) in enumerate(zip(data1.tensors, data2.tensors)):
                        if not compare_tensors(t1, t2, f"{filename}[{i}]"):
                            all_match = False
                
                elif filename == "reduced_indices.pt":
                    # List of tuples
                    if data1 != data2:
                        print(f"  ❌ {filename}: Indices mismatch")
                        print(f"    Length: {len(data1)} vs {len(data2)}")
                        if len(data1) == len(data2):
                            mismatches = sum(1 for i, (a, b) in enumerate(zip(data1, data2)) if a != b)
                            print(f"    Mismatched entries: {mismatches}/{len(data1)}")
                        all_match = False
                    else:
                        print(f"  ✅ {filename}: Matches ({len(data1)} indices)")
                
                else:
                    # Regular tensor
                    if not compare_tensors(data1, data2, filename):
                        all_match = False
                        
            except Exception as e:
                print(f"  ❌ {filename}: Error loading/comparing - {e}")
                all_match = False
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ ALL FILES MATCH!")
    else:
        print("❌ SOME FILES DO NOT MATCH!")
    print("=" * 80)
    
    return all_match


if __name__ == "__main__":
    dir1 = r"D:\Research\NO-2D-Metamaterials\data\out_test_10"
    dir2 = r"D:\Research\NO-2D-Metamaterials\data\out_test_10_verify"
    
    if len(sys.argv) > 1:
        dir1 = sys.argv[1]
    if len(sys.argv) > 2:
        dir2 = sys.argv[2]
    
    success = compare_datasets(dir1, dir2)
    sys.exit(0 if success else 1)

