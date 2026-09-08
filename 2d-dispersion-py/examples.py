"""
Example scripts for 2D dispersion analysis.

This module provides example scripts demonstrating how to use the
2D dispersion analysis package.
"""

import numpy as np
import matplotlib.pyplot as plt
from .dispersion import dispersion, dispersion2
from .get_design import get_design
from .wavevectors import get_IBZ_wavevectors
from .plotting import plot_dispersion, plot_design, plot_dispersion_surface
from .dataset_generation import generate_dispersion_dataset


def example_basic_dispersion():
    """
    Basic example: compute dispersion for a simple design.
    """
    print("Running basic dispersion example...")
    
    # Set up constants
    const = {
        'a': 1.0,  # lattice parameter [m]
        'N_ele': 2,  # elements per pixel
        'N_pix': [5, 5],  # pixels in each direction
        'N_eig': 6,  # number of eigenvalues
        'isUseGPU': False,
        'isUseImprovement': True,
        'isUseParallel': False,
        'isSaveEigenvectors': False,
        'isComputeGroupVelocity': False,
        'isComputeFrequencyDesignSensitivity': False,
        'isComputeGroupVelocityDesignSensitivity': False,
        'E_min': 2e9,
        'E_max': 200e9,
        'rho_min': 1e3,
        'rho_max': 8e3,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 1.0,
        'sigma_eig': 1.0,
        'design_scale': 'linear'
    }
    
    # Generate wavevectors
    const['wavevectors'] = get_IBZ_wavevectors([11, 6], const['a'], 'none')
    
    # Create a simple design
    design = get_design('dispersive-tetragonal', 5)
    const['design'] = design
    
    # Plot design
    fig, axes = plot_design(design)
    plt.suptitle('Example Design: Dispersive Tetragonal')
    plt.show()
    
    # Compute dispersion
    wv, fr, ev = dispersion(const, const['wavevectors'])
    
    # Plot dispersion
    fig, ax, _ = plot_dispersion(np.arange(len(wv)), fr[:, 0], 5)
    plt.title('Dispersion Relations - First Band')
    plt.show()
    
    print(f"Computed dispersion for {len(wv)} wavevectors")
    print(f"Frequency range: {np.min(fr):.3f} - {np.max(fr):.3f} Hz")
    
    return wv, fr, ev


def example_dispersion_surface():
    """
    Example: compute and visualize 3D dispersion surface.
    """
    print("Running dispersion surface example...")
    
    # Set up constants
    const = {
        'a': 1.0,
        'N_ele': 2,
        'N_pix': [3, 3],
        'N_eig': 4,
        'isUseGPU': False,
        'isUseImprovement': True,
        'isUseParallel': False,
        'isSaveEigenvectors': False,
        'isComputeGroupVelocity': False,
        'isComputeFrequencyDesignSensitivity': False,
        'isComputeGroupVelocityDesignSensitivity': False,
        'E_min': 2e9,
        'E_max': 200e9,
        'rho_min': 1e3,
        'rho_max': 8e3,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 1.0,
        'sigma_eig': 1.0,
        'design_scale': 'linear'
    }
    
    # Generate wavevectors for surface plot
    N_k = 15
    const['wavevectors'] = get_IBZ_wavevectors([N_k, N_k], const['a'], 'p4mm')
    
    # Create design
    design = get_design('homogeneous', 3)
    const['design'] = design
    
    # Compute dispersion
    wv, fr, ev = dispersion(const, const['wavevectors'])
    
    # Plot 3D surface
    fig, ax = plot_dispersion_surface(wv, fr[:, 0], N_k, N_k)
    plt.title('3D Dispersion Surface - First Band')
    plt.show()
    
    return wv, fr, ev


def example_multiple_designs():
    """
    Example: compare dispersion for multiple designs.
    """
    print("Running multiple designs example...")
    
    # Set up constants
    const = {
        'a': 1.0,
        'N_ele': 2,
        'N_pix': [4, 4],
        'N_eig': 4,
        'isUseGPU': False,
        'isUseImprovement': True,
        'isUseParallel': False,
        'isSaveEigenvectors': False,
        'isComputeGroupVelocity': False,
        'isComputeFrequencyDesignSensitivity': False,
        'isComputeGroupVelocityDesignSensitivity': False,
        'E_min': 2e9,
        'E_max': 200e9,
        'rho_min': 1e3,
        'rho_max': 8e3,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 1.0,
        'sigma_eig': 1.0,
        'design_scale': 'linear'
    }
    
    # Generate wavevectors
    const['wavevectors'] = get_IBZ_wavevectors([9, 5], const['a'], 'none')
    
    # Test different designs
    design_names = ['homogeneous', 'dispersive-tetragonal', 'quasi-1D']
    colors = ['blue', 'red', 'green']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (design_name, color) in enumerate(zip(design_names, colors)):
        # Create design
        design = get_design(design_name, 4)
        const['design'] = design
        
        # Compute dispersion
        wv, fr, ev = dispersion(const, const['wavevectors'])
        
        # Plot first band
        ax.plot(np.arange(len(wv)), fr[:, 0], color=color, 
               label=design_name, linewidth=2)
    
    ax.set_xlabel('Wavevector Index')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('Dispersion Comparison - Different Designs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
    
    return design_names, colors


def example_dataset_generation():
    """
    Example: generate a small dataset of dispersion relations.
    """
    print("Running dataset generation example...")
    
    # Generate small dataset
    dataset = generate_dispersion_dataset(
        N_struct=50,  # Small dataset for example
        N_pix=3,
        N_ele=2,
        N_eig=4,
        N_wv=7,
        symmetry_type='none',
        isSaveEigenvectors=False,
        isIncludeHomogeneous=True
    )
    
    print(f"Generated dataset with {dataset['N_struct']} structures")
    print(f"Design shape: {dataset['designs'].shape}")
    print(f"Frequency shape: {dataset['frequencies'].shape}")
    
    # Analyze dataset
    from .dataset_generation import analyze_dataset_statistics
    stats = analyze_dataset_statistics(dataset)
    
    print("\nDataset Statistics:")
    print(f"Mean modulus: {stats['design']['mean_modulus']:.3f}")
    print(f"Mean density: {stats['design']['mean_density']:.3f}")
    print(f"Frequency range: {np.min(dataset['frequencies']):.3f} - {np.max(dataset['frequencies']):.3f} Hz")
    
    return dataset, stats


def example_group_velocity():
    """
    Example: compute group velocities.
    """
    print("Running group velocity example...")
    
    # Set up constants with group velocity computation
    const = {
        'a': 1.0,
        'N_ele': 2,
        'N_pix': [3, 3],
        'N_eig': 3,
        'isUseGPU': False,
        'isUseImprovement': True,
        'isUseParallel': False,
        'isSaveEigenvectors': False,
        'isComputeGroupVelocity': True,  # Enable group velocity computation
        'isComputeFrequencyDesignSensitivity': False,
        'isComputeGroupVelocityDesignSensitivity': False,
        'E_min': 2e9,
        'E_max': 200e9,
        'rho_min': 1e3,
        'rho_max': 8e3,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 1.0,
        'sigma_eig': 1.0,
        'design_scale': 'linear'
    }
    
    # Generate wavevectors
    const['wavevectors'] = get_IBZ_wavevectors([7, 4], const['a'], 'none')
    
    # Create design
    design = get_design('dispersive-tetragonal', 3)
    const['design'] = design
    
    # Compute dispersion with group velocities
    wv, fr, ev, cg, _, _ = dispersion2(const, const['wavevectors'])
    
    # Plot group velocities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot frequencies
    ax1.plot(np.arange(len(wv)), fr[:, 0], 'b-', label='Band 1')
    ax1.plot(np.arange(len(wv)), fr[:, 1], 'r-', label='Band 2')
    ax1.set_xlabel('Wavevector Index')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_title('Dispersion Relations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot group velocities
    ax2.plot(np.arange(len(wv)), cg[:, 0, 0], 'b-', label='Band 1, x-component')
    ax2.plot(np.arange(len(wv)), cg[:, 1, 0], 'b--', label='Band 1, y-component')
    ax2.plot(np.arange(len(wv)), cg[:, 0, 1], 'r-', label='Band 2, x-component')
    ax2.plot(np.arange(len(wv)), cg[:, 1, 1], 'r--', label='Band 2, y-component')
    ax2.set_xlabel('Wavevector Index')
    ax2.set_ylabel('Group Velocity')
    ax2.set_title('Group Velocities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Computed group velocities for {len(wv)} wavevectors")
    print(f"Group velocity range: {np.min(cg):.3f} - {np.max(cg):.3f}")
    
    return wv, fr, cg


def run_all_examples():
    """
    Run all examples.
    """
    print("Running all 2D dispersion analysis examples...\n")
    
    try:
        # Basic dispersion
        example_basic_dispersion()
        print("✓ Basic dispersion example completed\n")
        
        # Dispersion surface
        example_dispersion_surface()
        print("✓ Dispersion surface example completed\n")
        
        # Multiple designs
        example_multiple_designs()
        print("✓ Multiple designs example completed\n")
        
        # Dataset generation
        example_dataset_generation()
        print("✓ Dataset generation example completed\n")
        
        # Group velocity
        example_group_velocity()
        print("✓ Group velocity example completed\n")
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    run_all_examples()

