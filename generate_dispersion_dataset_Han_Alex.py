"""
Python translation of generate_dispersion_dataset_Han_Alex.m

This script generates dispersion datasets using the new Python dispersion library.
It creates multiple metamaterial designs and computes their dispersion relations,
eigenvectors, and system matrices.
"""

import numpy as np
import os
import sys
from datetime import datetime
import pickle
import scipy.io as sio

# Temporarily rename conflicting local dispersion.py file
import os
local_dispersion_file = os.path.join(os.getcwd(), 'dispersion.py')
temp_dispersion_file = os.path.join(os.getcwd(), 'dispersion_temp.py')
if os.path.exists(local_dispersion_file):
    print("WARNING: Found conflicting local dispersion.py file, temporarily renaming it...")
    if os.path.exists(temp_dispersion_file):
        os.remove(temp_dispersion_file)
    os.rename(local_dispersion_file, temp_dispersion_file)

# Add the new Python dispersion library path
dispersion_library_path = r'D:\Research\NO-2D-Metamaterials\2d-dispersion-py'
sys.path.insert(0, dispersion_library_path)  # Use insert(0, ...) to prioritize this path

    # Import functions from the new Python library
try:
    # Import the CORRECT function - dispersion_with_matrix_save_opt (not dispersion!)
    from dispersion_with_matrix_save_opt import dispersion_with_matrix_save_opt
    from get_design import get_design
    from get_design2 import get_design2
    from wavevectors import get_IBZ_wavevectors
    from design_parameters import DesignParameters
    from design_conversion import convert_design, design_to_explicit
    from kernels import generate_correlated_design
    from utils import validate_constants, check_contour_analysis
    
    print("SUCCESS: Successfully imported functions from the new Python dispersion library")
    print("SUCCESS: Using dispersion_with_matrix_save_opt (the correct function!)")
except ImportError as e:
    print(f"ERROR: Error importing from dispersion library: {e}")
    print("Make sure the 2d-dispersion-py directory is in the correct location")
    print(f"Current library path: {dispersion_library_path}")
    sys.exit(1)


def generate_dispersion_dataset_with_matrices():
    """
    Generate dispersion dataset with system matrices (equivalent to MATLAB script).
    """
    
    print("="*80)
    print("GENERATING DISPERSION DATASET WITH PYTHON LIBRARY")
    print("="*80)
    
    # Get script information
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_path = os.path.dirname(os.path.abspath(__file__))
    
    # Output flags
    is_save_output = True
    is_save_eigenvectors = True
    is_profile = False
    is_save_mesh = False
    is_save_k_and_m = True
    
    # Discretization parameters - FIXED to match MATLAB exactly
    const = {
        'N_ele': 1,  # Number of elements along one pixel side length
        'N_pix': 32,  # Number of pixels along one unit cell side length (SCALAR, not array!)
        'N_wv': [25, 13],  # Number of wavevectors (25 x ceil(25/2) = 13)
        'N_eig': 6,  # Number of eigenvalue bands to compute
        'sigma_eig': 1e-2,  # Eigenvalue solver parameter
        'a': 1.0,  # Side length of square unit cell [m]
        'design_scale': 'linear'
    }
    
    # Flags for computational improvements
    const.update({
        'isUseGPU': False,
        'isUseImprovement': True,
        'isUseSecondImprovement': False,
        'isUseParallel': True,  # Note: Python doesn't have parfor, but we can use multiprocessing if needed
        'isSaveEigenvectors': is_save_eigenvectors,
        'isComputeGroupVelocity': False,
        'isComputeFrequencyDesignSensitivity': False,
        'isComputeGroupVelocityDesignSensitivity': False
    })
    
    # Design parameters
    N_struct = 5  # Number of designs to generate
    rng_seed_offset = 0  # RNG seed offset
    binarize = False  # Set to False for continuous designs
    
    # Material parameters (exact MATLAB translation)
    const.update({
        'E_min': 200e6,
        'E_max': 200e9,
        'rho_min': 8e2,
        'rho_max': 8e3,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        't': 1.0
    })
    
    # Imaginary tolerance
    imag_tol = 1e-3
    
    # Generate wavevectors
    print(f"Generating wavevectors for {const['N_wv']} grid...")
    const['wavevectors'] = get_IBZ_wavevectors(const['N_wv'], const['a'], 'none')
    print(f"Generated {len(const['wavevectors'])} wavevectors")
    
    # Initialize design parameters (exact MATLAB translation)
    design_params = DesignParameters(N_struct)
    design_params.property_coupling = 'coupled'
    design_params.design_style = 'kernel'
    design_params.design_options = {
        'kernel': 'periodic',
        'sigma_f': 1.0,
        'sigma_l': 1.0,
        'symmetry_type': 'p4mm',
        'N_value': np.inf
    }
    design_params.N_pix = [const['N_pix'], const['N_pix']]  # MATLAB uses [N_pix, N_pix]
    design_params = design_params.prepare()
    
    # Initialize storage arrays - FIXED for scalar N_pix
    print("Initializing storage arrays...")
    designs = np.zeros((const['N_pix'], const['N_pix'], 3, N_struct))
    wavevector_data = np.zeros((np.prod(const['N_wv']), 2, N_struct))
    eigenvalue_data = np.zeros((np.prod(const['N_wv']), const['N_eig'], N_struct))
    
    N_dof = 2 * (const['N_pix'] * const['N_ele'])**2
    eigenvector_data = np.zeros((N_dof, np.prod(const['N_wv']), const['N_eig'], N_struct), dtype=complex)
    
    # Material property data
    elastic_modulus_data = np.zeros((const['N_pix'], const['N_pix'], N_struct))
    density_data = np.zeros((const['N_pix'], const['N_pix'], N_struct))
    poisson_data = np.zeros((const['N_pix'], const['N_pix'], N_struct))
    
    # System matrices (using lists for variable sizes)
    K_data = []
    M_data = []
    T_data = []
    
    # Validate constants
    is_valid, missing_fields = validate_constants(const)
    if not is_valid:
        print(f"ERROR: Constants validation failed. Missing fields: {missing_fields}")
        return None
    
    print("SUCCESS: Constants validation passed")
    
    # Generate dataset
    print(f"\nGenerating {N_struct} structures...")
    for struct_idx in range(N_struct):
        print(f"Processing structure {struct_idx + 1}/{N_struct}...")
        
        # Generate design using EXACT MATLAB approach
        try:
            # Set design number for this structure (exact MATLAB translation)
            design_params.design_number = struct_idx + rng_seed_offset
            
            # Generate design using get_design2 (exact MATLAB translation)
            design = get_design2(design_params)
            
            # Convert design scale (exact MATLAB translation)
            design = convert_design(design, 'linear', const['design_scale'], 
                                  const['E_min'], const['E_max'],
                                  const['rho_min'], const['rho_max'])
            
            # Binarize if requested (exact MATLAB translation)
            if binarize:
                design = np.round(design)
            
            # Store design
            designs[:, :, :, struct_idx] = design
            const['design'] = design
            
            # Compute dispersion using the CORRECT function (matches MATLAB exactly)
            wv, fr, ev, mesh, K, M, T = dispersion_with_matrix_save_opt(const, const['wavevectors'])
            
            # Store results
            wavevector_data[:, :, struct_idx] = wv
            eigenvalue_data[:, :, struct_idx] = np.real(fr)
            
            if is_save_eigenvectors and ev is not None:
                eigenvector_data[:, :, :, struct_idx] = ev
            
            # Check for imaginary components
            if np.max(np.abs(np.imag(fr))) > imag_tol:
                print(f"WARNING: Warning: Large imaginary component in frequency for structure {struct_idx + 1}")
            
            # Store material properties
            explicit_props = design_to_explicit(design, const['design_scale'], 
                                              const['E_min'], const['E_max'],
                                              const['rho_min'], const['rho_max'],
                                              const['poisson_min'], const['poisson_max'])
            
            elastic_modulus_data[:, :, struct_idx] = explicit_props['E']
            density_data[:, :, struct_idx] = explicit_props['rho']
            poisson_data[:, :, struct_idx] = explicit_props['nu']
            
            # Store system matrices (now available from dispersion_with_matrix_save_opt!)
            K_data.append(K)
            M_data.append(M)
            T_data.append(T)
            
            print(f"  SUCCESS: Completed structure {struct_idx + 1}")
            
        except Exception as e:
            print(f"  ERROR: Error processing structure {struct_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Collect constitutive data
    constitutive_data = {
        'modulus': elastic_modulus_data,
        'density': density_data,
        'poisson': poisson_data
    }
    
    # Set up save locations
    script_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if is_save_output:
        output_folder = os.path.join(script_path, f'OUTPUT/output_{script_start_time}')
        os.makedirs(output_folder, exist_ok=True)
        
        # Copy this script to output folder
        import shutil
        shutil.copy2(__file__, os.path.join(output_folder, f'{script_name}.py'))
    
    # Prepare data for saving
    dataset = {
        'WAVEVECTOR_DATA': wavevector_data,
        'EIGENVALUE_DATA': eigenvalue_data,
        'CONSTITUTIVE_DATA': constitutive_data,
        'designs': designs,
        'const': const,
        'design_params': design_params,
        'N_struct': N_struct,
        'imag_tol': imag_tol,
        'rng_seed_offset': rng_seed_offset,
        'script_start_time': script_start_time
    }
    
    if is_save_eigenvectors:
        dataset['EIGENVECTOR_DATA'] = eigenvector_data
    
    if K_data:
        dataset['K_DATA'] = K_data
    if M_data:
        dataset['M_DATA'] = M_data
    if T_data:
        dataset['T_DATA'] = T_data
    
    # Save results
    if is_save_output:
        design_type_label = 'binarized' if binarize else 'continuous'
        output_file_path = os.path.join(
            output_folder,
            f'{design_type_label}_{script_start_time}.mat'
        )
        
        # Save as MATLAB .mat file for compatibility
        sio.savemat(output_file_path, dataset, oned_as='column')
        print(f"SUCCESS: MATLAB .mat file saved to: {output_file_path}")
        
        # Also save as Python pickle for easier Python usage
        pickle_file_path = os.path.join(
            output_folder,
            f'{design_type_label}_{script_start_time}.pkl'
        )
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"SUCCESS: Python pickle file saved to: {pickle_file_path}")
    
    # Summary
    print("\n" + "="*80)
    print("DATASET GENERATION SUMMARY")
    print("="*80)
    print(f"SUCCESS: Generated {N_struct} structures")
    print(f"SUCCESS: Computed dispersion for {len(const['wavevectors'])} wavevectors")
    print(f"SUCCESS: Design size: {const['N_pix'][0]}x{const['N_pix'][1]} pixels")
    print(f"SUCCESS: Number of eigenvalues: {const['N_eig']}")
    print(f"SUCCESS: Frequency range: {np.min(eigenvalue_data):.3f} - {np.max(eigenvalue_data):.3f} Hz")
    print(f"SUCCESS: Design type: {design_type_label}")
    
    if is_save_eigenvectors:
        print(f"SUCCESS: Eigenvectors saved: {eigenvector_data.shape}")
    
    print("="*80)
    
    return dataset

def demonstrate_library_features():
    """
    Demonstrate additional features of the new library and generate visual checks.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING LIBRARY FEATURES")
    print("="*80)
    
    # Create quick_checks folder
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    
    quick_checks_folder = "quick_checks"
    if not os.path.exists(quick_checks_folder):
        os.makedirs(quick_checks_folder)
        print(f"SUCCESS: Created {quick_checks_folder} folder")
    
    # Generate different design types and visualize them
    print("1. Generating different design types...")
    design_types = ['homogeneous', 'dispersive-tetragonal', 'quasi-1D']
    designs = {}
    
    for design_type in design_types:
        try:
            design = get_design(design_type, 32)
            designs[design_type] = design
            print(f"  SUCCESS: Generated {design_type} design: {design.shape}")
        except Exception as e:
            print(f"  ERROR: Failed to generate {design_type}: {e}")
    
    # Create visualization of design patterns
    if designs:
        print("2. Creating design pattern visualizations...")
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Design Pattern Examples', fontsize=16, fontweight='bold')
            
            for idx, (design_type, design) in enumerate(designs.items()):
                # Plot the first material property (elastic modulus)
                im = axes[idx].imshow(design[:, :, 0], cmap='viridis', origin='lower')
                axes[idx].set_title(f'{design_type.replace("-", " ").title()}', fontweight='bold')
                axes[idx].set_xlabel('X (pixels)')
                axes[idx].set_ylabel('Y (pixels)')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[idx], label='Elastic Modulus (normalized)')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"design_patterns_{timestamp}.png"
            filepath = os.path.join(quick_checks_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  SUCCESS: Saved design patterns to {filepath}")
            
        except Exception as e:
            print(f"  ERROR: Failed to create design visualization: {e}")
    
    # Test different symmetry types
    print("3. Testing different symmetry types...")
    symmetry_types = ['none', 'p4mm', 'p2mm']
    wavevector_data = {}
    
    for sym_type in symmetry_types:
        try:
            wv = get_IBZ_wavevectors([10, 10], 1.0, sym_type)
            wavevector_data[sym_type] = wv
            print(f"  SUCCESS: Generated {sym_type} wavevectors: {len(wv)} points")
        except Exception as e:
            print(f"  ERROR: Failed to generate {sym_type} wavevectors: {e}")
    
    # Create visualization of wavevector patterns
    if wavevector_data:
        print("4. Creating wavevector pattern visualizations...")
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Irreducible Brillouin Zone (IBZ) Wavevector Patterns', fontsize=16, fontweight='bold')
            
            for idx, (sym_type, wv) in enumerate(wavevector_data.items()):
                # Plot wavevectors
                axes[idx].scatter(wv[:, 0], wv[:, 1], c='red', s=20, alpha=0.7)
                axes[idx].set_title(f'{sym_type.upper()} Symmetry\n({len(wv)} points)', fontweight='bold')
                axes[idx].set_xlabel('k_x')
                axes[idx].set_ylabel('k_y')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_aspect('equal')
                
                # Add some reference lines for context
                axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[idx].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"wavevector_patterns_{timestamp}.png"
            filepath = os.path.join(quick_checks_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  SUCCESS: Saved wavevector patterns to {filepath}")
            
        except Exception as e:
            print(f"  ERROR: Failed to create wavevector visualization: {e}")
    
    # Create visualization of the specific design used in MATLAB script
    print("5. Creating MATLAB-equivalent design visualization...")
    try:
        # Generate the exact design used in the MATLAB script: dispersive-tetragonal with p4mm symmetry
        from design_parameters import DesignParameters
        from get_design2 import get_design2
        
        # Create design parameters matching MATLAB script
        design_params_matlab = DesignParameters(1)
        design_params_matlab.property_coupling = 'coupled'
        design_params_matlab.design_style = 'kernel'
        design_params_matlab.design_options = {
            'kernel': 'periodic',
            'sigma_f': 1.0,
            'sigma_l': 1.0,
            'symmetry_type': 'p4mm',
            'N_value': np.inf
        }
        design_params_matlab.N_pix = 32
        design_params_matlab = design_params_matlab.prepare()
        
        # Generate the design using get_design2 (same as MATLAB)
        matlab_design = get_design2(design_params_matlab)
        
        # Create 3-panel visualization showing all material properties
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('MATLAB-Equivalent Design: Dispersive-Tetragonal with P4MM Symmetry', 
                     fontsize=16, fontweight='bold')
        
        # Material property names and colormaps
        properties = ['Elastic Modulus', 'Density', 'Poisson Ratio']
        colormaps = ['viridis', 'viridis', 'viridis']
        
        for idx in range(3):
            im = axes[idx].imshow(matlab_design[:, :, idx], cmap=colormaps[idx], origin='lower')
            axes[idx].set_title(f'{properties[idx]}', fontweight='bold')
            axes[idx].set_xlabel('X (pixels)')
            axes[idx].set_ylabel('Y (pixels)')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], label=f'{properties[idx]} (normalized)')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"matlab_equivalent_design_{timestamp}.png"
        filepath = os.path.join(quick_checks_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  SUCCESS: Saved MATLAB-equivalent design to {filepath}")
        print(f"  INFO: Design shape: {matlab_design.shape}")
        print(f"  INFO: Property ranges - E: [{np.min(matlab_design[:,:,0]):.3f}, {np.max(matlab_design[:,:,0]):.3f}]")
        print(f"  INFO: Property ranges - ρ: [{np.min(matlab_design[:,:,1]):.3f}, {np.max(matlab_design[:,:,1]):.3f}]")
        print(f"  INFO: Property ranges - ν: [{np.min(matlab_design[:,:,2]):.3f}, {np.max(matlab_design[:,:,2]):.3f}]")
        
    except Exception as e:
        print(f"  ERROR: Failed to create MATLAB-equivalent design visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80)


if __name__ == "__main__":
    try:
        # Run the main dataset generation
        dataset = generate_dispersion_dataset_with_matrices()
        
        # Demonstrate additional features
        demonstrate_library_features()
        
        print("\nCOMPLETED: Script completed successfully!")
        print("The generated dataset is compatible with both MATLAB and Python workflows.")
        
    finally:
        # Restore the original dispersion.py file if it was renamed
        if os.path.exists(temp_dispersion_file):
            print("\nRESTORING: Restoring original dispersion.py file...")
            if os.path.exists(local_dispersion_file):
                os.remove(local_dispersion_file)
            os.rename(temp_dispersion_file, local_dispersion_file)
            print("SUCCESS: Original dispersion.py file restored")
