#!/usr/bin/env python3
"""
Helper script to create full_indices.pt for predictions that cover all combinations.
This creates indices for all geometry × waveform × band combinations.
"""

import torch
import argparse
import os

def create_full_indices(n_geometries, n_waveforms, n_bands, output_path):
    """
    Create indices for all combinations of geometries, waveforms, and bands.
    
    Args:
        n_geometries: Number of geometries
        n_waveforms: Number of waveforms/wavevectors
        n_bands: Number of bands
        output_path: Path to save the indices file
    """
    indices = []
    
    for geom_idx in range(n_geometries):
        for wave_idx in range(n_waveforms):
            for band_idx in range(n_bands):
                indices.append([geom_idx, wave_idx, band_idx])
    
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    torch.save(indices_tensor, output_path)
    
    print(f"Created full_indices.pt with {len(indices)} indices")
    print(f"  - {n_geometries} geometries")
    print(f"  - {n_waveforms} waveforms")
    print(f"  - {n_bands} bands")
    print(f"  - Total: {len(indices)} combinations")
    print(f"  - Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create full_indices.pt for predictions covering all combinations'
    )
    parser.add_argument(
        '--n_geometries',
        type=int,
        required=True,
        help='Number of geometries'
    )
    parser.add_argument(
        '--n_waveforms',
        type=int,
        default=91,
        help='Number of waveforms/wavevectors (default: 91)'
    )
    parser.add_argument(
        '--n_bands',
        type=int,
        default=6,
        help='Number of bands (default: 6)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save full_indices.pt'
    )
    
    args = parser.parse_args()
    create_full_indices(args.n_geometries, args.n_waveforms, args.n_bands, args.output_path)

