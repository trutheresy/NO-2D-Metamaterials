"""
Script to generate dispersion diagrams from unreduced cr and br datasets.

This script loads samples from the unreduced datasets (which already have
computed eigenvalues/frequencies) and plots the dispersion diagrams along
high-symmetry paths in the Irreducible Brillouin Zone (IBZ).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys

# Add the dispersion library to path to use IBZ contour functions
sys.path.insert(0, '2d-dispersion-py')
from wavevectors import get_IBZ_contour_wavevectors


def load_unreduced_dataset(dataset_name):
    """Load unreduced dataset with eigenvalue data."""
    dataset_path = os.path.join('data', dataset_name)
    
    geometries = np.load(os.path.join(dataset_path, 'designs.npy'))
    wavevectors = np.load(os.path.join(dataset_path, 'wavevectors.npy'))
    eigenvalues = np.load(os.path.join(dataset_path, 'eigenvalue_data.npy'))
    
    return {
        'geometries': geometries,
        'wavevectors': wavevectors,
        'eigenvalues': eigenvalues
    }


def find_ibz_contour_indices(full_wavevectors, contour_wavevectors):
    """
    Find indices in full wavevector array that correspond to IBZ contour.
    
    Parameters
    ----------
    full_wavevectors : array, shape (n_full, 2)
        Full set of wavevectors from dataset
    contour_wavevectors : array, shape (n_contour, 2)
        Wavevectors along IBZ contour path
        
    Returns
    -------
    indices : array
        Indices into full_wavevectors that best match contour_wavevectors
    """
    indices = []
    for wv_contour in contour_wavevectors:
        # Find closest match in full wavevector set
        distances = np.sqrt(np.sum((full_wavevectors - wv_contour)**2, axis=1))
        closest_idx = np.argmin(distances)
        indices.append(closest_idx)
    return np.array(indices)


def plot_dispersion_diagram(wavevectors, frequencies, geometry, title, output_path, a=1.0):
    """
    Plot dispersion diagram along IBZ high-symmetry path.
    
    Parameters
    ----------
    wavevectors : array, shape (n_wavevectors, 2)
        Wavevectors for ONE sample  
    frequencies : array, shape (n_wavevectors, n_bands)
        Frequencies for ONE sample
    geometry : array, shape (32, 32, 3)
        Material property maps
    title : str
        Plot title
    output_path : str
        Where to save the plot
    a : float
        Lattice parameter (default 1.0 m)
    """
    n_bands = frequencies.shape[1]
    
    # Get IBZ contour wavevectors (high-symmetry path: Γ→X→M→Γ→Y→O→Γ for 'none' symmetry)
    # For 'none' symmetry, the path is: Γ(0,0) → X(π/a,0) → M(π/a,π/a) → Γ(0,0) → Y(0,π/a) → O(-π/a,π/a) → Γ(0,0)
    N_k_contour = 50  # Number of points along each segment
    contour_wv = get_IBZ_contour_wavevectors(N_k_contour, a, 'none')
    
    # Create contour info manually (matching MATLAB implementation)
    vertex_labels = ['Γ', 'X', 'M', 'Γ', 'Y', 'O', 'Γ']
    N_segments = len(vertex_labels) - 1  # 6 segments
    wv_param = np.linspace(0, N_segments, len(contour_wv))
    
    # Find which wavevectors in our dataset correspond to the contour
    contour_indices = find_ibz_contour_indices(wavevectors, contour_wv)
    
    # Extract frequencies along contour
    freq_contour = frequencies[contour_indices, :]
    
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Material properties
    for i, (prop_name, channel) in enumerate([('Elastic Modulus', 0), ('Density', 1), ("Poisson's Ratio", 2)]):
        ax = plt.subplot(3, 3, i+1)
        im = ax.imshow(geometry[:, :, channel], cmap='viridis', origin='lower')
        ax.set_title(f'{prop_name}', fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax)
    
    # Row 2: Complete dispersion diagram along IBZ contour
    ax = plt.subplot(3, 3, 4)
    for band_idx in range(n_bands):
        ax.plot(wv_param, freq_contour[:, band_idx], 'k.-', markersize=2, linewidth=1)
    
    # Add vertical lines at high-symmetry points
    for i in range(1, N_segments):
        ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Add high-symmetry point labels
    ax.set_xticks(range(N_segments + 1))
    ax.set_xticklabels(vertex_labels)
    
    ax.set_title(f'Dispersion Along IBZ High-Symmetry Path ({n_bands} bands)', fontweight='bold')
    ax.set_xlabel('High-Symmetry Path')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True, alpha=0.3, which='both')
    ax.minorticks_on()
    
    # Row 2-3: Individual band plots along IBZ contour
    for band_idx in range(min(5, n_bands)):
        ax = plt.subplot(3, 3, 5 + band_idx)
        ax.plot(wv_param, freq_contour[:, band_idx], 'k.-', markersize=3, linewidth=1.5)
        
        # Add vertical lines at high-symmetry points
        for i in range(1, N_segments):
            ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Add high-symmetry point labels
        ax.set_xticks(range(N_segments + 1))
        ax.set_xticklabels(vertex_labels)
        
        ax.set_title(f'Band {band_idx + 1}', fontweight='bold')
        ax.set_xlabel('High-Symmetry Path')
        ax.set_ylabel('Frequency (Hz)')
        ax.grid(True, alpha=0.3, which='both')
        ax.minorticks_on()
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved dispersion diagram to: {output_path}")


def main():
    """Main function."""
    print("="*80)
    print("DISPERSION DIAGRAM GENERATION FROM UNREDUCED DATASETS")
    print("="*80)
    
    # Create output directory
    output_dir = 'dispersion_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process continuous dataset
    print("\n1. Processing CONTINUOUS dataset (cr)...")
    print("-" *80)
    cr_data = load_unreduced_dataset('set_cr_1200n')
    sample_idx = 0
    
    print(f"  Geometries shape: {cr_data['geometries'].shape}")
    print(f"  Wavevectors shape: {cr_data['wavevectors'].shape}")
    print(f"  Eigenvalues shape: {cr_data['eigenvalues'].shape}")
    
    # Extract data for first sample
    # Geometries are stored as (n_samples, 32, 32) - single channel normalized design (0-1)
    # The properties are coupled, so all vary with the same spatial pattern
    # Reconstruct the 3 material property channels from the single design parameter
    design_param = cr_data['geometries'][sample_idx, :, :]  # (32, 32), values in [0, 1]
    
    # Material parameters from the generation script (set_cr uses these ranges)
    # Note: These should match the values in data/set_cr_1200n/generate_dispersion_dataset_Han.m
    E_min, E_max = 20e6, 200e9
    rho_min, rho_max = 400, 8000
    nu_min, nu_max = 0.05, 0.3
    
    # Compute actual material properties (coupled - all follow same design pattern)
    elastic_modulus = E_min + (E_max - E_min) * design_param
    density = rho_min + (rho_max - rho_min) * design_param
    poisson_ratio = nu_min + (nu_max - nu_min) * design_param
    
    geometry_cr = np.stack([elastic_modulus, density, poisson_ratio], axis=-1)  # (32, 32, 3)
    
    wavevectors_cr = cr_data['wavevectors'][sample_idx, :, :]  # (n_wv, 2)
    frequencies_cr = cr_data['eigenvalues'][sample_idx, :, :]  # (n_wv, n_bands)
    
    print(f"  Sample {sample_idx} - geometry: {geometry_cr.shape}, wv: {wavevectors_cr.shape}, freq: {frequencies_cr.shape}")
    
    output_path_cr = os.path.join(output_dir, f'dispersion_cr_sample{sample_idx}_{timestamp}.png')
    plot_dispersion_diagram(
        wavevectors_cr, frequencies_cr, geometry_cr,
        f'Continuous Dataset - Sample {sample_idx}',
        output_path_cr
    )
    
    # Process binary dataset
    print("\n2. Processing BINARY dataset (br)...")
    print("-" *80)
    br_data = load_unreduced_dataset('set_br_1200n')
    
    print(f"  Geometries shape: {br_data['geometries'].shape}")
    print(f"  Wavevectors shape: {br_data['wavevectors'].shape}")
    print(f"  Eigenvalues shape: {br_data['eigenvalues'].shape}")
    
    # Extract data for first sample
    # Geometries are stored as (n_samples, 32, 32) - single channel normalized design (0-1)
    # The properties are coupled, so all vary with the same spatial pattern
    # Reconstruct the 3 material property channels from the single design parameter
    design_param = br_data['geometries'][sample_idx, :, :]  # (32, 32), values in [0, 1]
    
    # Material parameters from the generation script (set_br uses these ranges)
    # Note: For binary dataset, check data/set_br_1200n/generate_dispersion_dataset_Han.m
    # Using same ranges as continuous for now (should verify)
    E_min, E_max = 20e6, 200e9
    rho_min, rho_max = 400, 8000
    nu_min, nu_max = 0.05, 0.3
    
    # Compute actual material properties (coupled - all follow same design pattern)
    elastic_modulus = E_min + (E_max - E_min) * design_param
    density = rho_min + (rho_max - rho_min) * design_param
    poisson_ratio = nu_min + (nu_max - nu_min) * design_param
    
    geometry_br = np.stack([elastic_modulus, density, poisson_ratio], axis=-1)  # (32, 32, 3)
    
    wavevectors_br = br_data['wavevectors'][sample_idx, :, :]  # (n_wv, 2)
    frequencies_br = br_data['eigenvalues'][sample_idx, :, :]  # (n_wv, n_bands)
    
    print(f"  Sample {sample_idx} - geometry: {geometry_br.shape}, wv: {wavevectors_br.shape}, freq: {frequencies_br.shape}")
    
    output_path_br = os.path.join(output_dir, f'dispersion_br_sample{sample_idx}_{timestamp}.png')
    plot_dispersion_diagram(
        wavevectors_br, frequencies_br, geometry_br,
        f'Binary Dataset - Sample {sample_idx}',
        output_path_br
    )
    
    print("\n" + "="*80)
    print("DISPERSION DIAGRAM GENERATION COMPLETED")
    print("="*80)
    print(f"\nDispersion diagrams saved to '{output_dir}/' folder:")
    print(f"  - {os.path.basename(output_path_cr)}")
    print(f"  - {os.path.basename(output_path_br)}")


if __name__ == '__main__':
    main()

