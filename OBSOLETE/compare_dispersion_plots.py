"""
Compare Dispersion Plots Between MATLAB and Python Libraries

This script:
1. Generates test geometries in both libraries
2. Runs dispersion calculations
3. Creates comparison plots
4. If designs differ, ports them between libraries and re-runs
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import subprocess
import json
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))

# Import Python library functions
from get_design import get_design
from wavevectors import get_IBZ_wavevectors, get_IBZ_contour_wavevectors
from dispersion import dispersion
from plotting import plot_design

# Import MATLAB loader for saving/loading designs
try:
    from mat73_loader import load_matlab_v73
    import scipy.io as sio
except ImportError:
    print("Warning: Could not import mat73_loader or scipy.io")


def generate_test_designs():
    """Generate test geometries for comparison."""
    N_pix = 16  # Use 16x16 for reasonable computation time
    
    designs = {
        'homogeneous': get_design('homogeneous', N_pix),
        'quasi-1D': get_design('quasi-1D', N_pix),
        'dispersive-tetragonal': get_design('dispersive-tetragonal', N_pix),
        'dispersive-orthotropic': get_design('dispersive-orthotropic', N_pix),
    }
    
    return designs, N_pix


def run_python_dispersion(design, const):
    """Run dispersion calculation in Python."""
    # Add design to constants
    const_with_design = const.copy()
    const_with_design['design'] = design
    
    # Get wavevectors - note: get_IBZ_wavevectors takes (N_wv, a, symmetry_type)
    wavevectors = get_IBZ_wavevectors(const['N_k'], const['a'], const['symmetry_type'])
    
    # Run dispersion - returns (wv, fr, ev, mesh)
    result = dispersion(const_with_design, wavevectors)
    if len(result) == 4:
        wv, fr, ev, mesh = result
    else:
        wv, fr, ev = result
        mesh = None
    
    return wv, fr, ev


def save_design_for_matlab(design, design_name, output_dir):
    """Save design in MATLAB-compatible format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as .mat file (MATLAB uses (N_pix, N_pix, 3) format)
    # Python uses (N_pix, N_pix, 3) so we can save directly
    # Use default format (v5) which is more compatible
    mat_file = output_dir / f'{design_name}_design.mat'
    sio.savemat(str(mat_file), {design_name: design})
    
    return mat_file


def load_design_from_matlab(mat_file, design_name):
    """Load design from MATLAB .mat file."""
    data = load_matlab_v73(str(mat_file), verbose=False)
    design = data[design_name]
    
    # MATLAB might save as (3, N_pix, N_pix), transpose if needed
    if design.shape[0] == 3 and len(design.shape) == 3:
        design = np.transpose(design, (1, 2, 0))
    
    return design


def run_matlab_dispersion(design_name, const_dict, output_dir):
    """Run dispersion calculation in MATLAB via script."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create MATLAB script
    matlab_script = output_dir / f'run_matlab_dispersion_{design_name}.m'
    
    script_content = f"""
% MATLAB script to run dispersion calculation
addpath(genpath('{Path(__file__).parent / "2D-dispersion-han"}'));
addpath(genpath('{Path(__file__).parent}'));

% Load design
design_file = '{output_dir / f"{design_name}_design.mat"}';
data = load(design_file);
design = data.{design_name};

% Set up constants
const = struct();
const.N_pix = {const_dict['N_pix']};
const.N_ele = {const_dict['N_ele']};
const.N_k = {const_dict['N_k']};
const.symmetry_type = '{const_dict['symmetry_type']}';
const.a = {const_dict['a']};
const.t = {const_dict['t']};
const.E_min = {const_dict['E_min']};
const.E_max = {const_dict['E_max']};
const.rho_min = {const_dict['rho_min']};
const.rho_max = {const_dict['rho_max']};
const.poisson_min = {const_dict['poisson_min']};
const.poisson_max = {const_dict['poisson_max']};
const.design_scale = '{const_dict['design_scale']}';
const.N_eig = {const_dict['N_eig']};
const.sigma_eig = {const_dict['sigma_eig']};
const.isSaveEigenvectors = false;
const.isSaveMesh = false;
const.isUseSecondImprovement = true;
const.design = design;

% Get wavevectors
wavevectors = get_IBZ_wavevectors(const.N_k, const.symmetry_type, const.a);

% Run dispersion
[wv, fr, ev] = dispersion(const, wavevectors);

% Save results
results_file = '{output_dir / f"{design_name}_matlab_results.mat"}';
save(results_file, 'wv', 'fr', 'ev', 'wavevectors', 'const', '-v7.3');
fprintf('Saved MATLAB results to: %s\\n', results_file);
"""
    
    with open(matlab_script, 'w') as f:
        f.write(script_content)
    
    # Run MATLAB script
    try:
        result = subprocess.run(
            ['matlab', '-batch', f"run('{matlab_script}')"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            print(f"MATLAB error for {design_name}:")
            print(result.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"MATLAB timeout for {design_name}")
        return None
    except FileNotFoundError:
        print("MATLAB not found in PATH. Skipping MATLAB calculations.")
        return None
    
    # Load results
    results_file = output_dir / f'{design_name}_matlab_results.mat'
    if results_file.exists():
        return load_matlab_v73(str(results_file), verbose=False)
    else:
        return None


def create_comparison_plots(design_name, design, python_results, matlab_results, output_dir):
    """Create comparison plots for a design."""
    from scipy.spatial.distance import cdist
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Row 1: Design visualization
    property_names = ['Elastic Modulus', 'Density', "Poisson's Ratio"]
    for i in range(3):
        ax = plt.subplot(3, 3, i + 1)
        im = ax.imshow(design[:, :, i], cmap='viridis', origin='lower')
        ax.set_title(f'{property_names[i]}')
        plt.colorbar(im, ax=ax)
    
    # Get contour for plotting (use Python const if available, otherwise defaults)
    if python_results is not None:
        const = python_results.get('const', {})
    elif matlab_results is not None:
        const = matlab_results.get('const', {})
    else:
        const = {'a': 1.0, 'symmetry_type': 'none'}
    
    N_k_contour = 50
    contour_wv, contour_info = get_IBZ_contour_wavevectors(N_k_contour, const.get('a', 1.0), const.get('symmetry_type', 'none'))
    
    # Row 2: Python dispersion
    if python_results is not None:
        wv_py = python_results['wv']
        fr_py = python_results['fr']
        
        # Find matching wavevectors
        distances = cdist(contour_wv, wv_py)
        closest_indices = np.argmin(distances, axis=1)
        contour_fr = fr_py[closest_indices, :]
        
        ax = plt.subplot(3, 3, 4)
        for band_idx in range(min(6, fr_py.shape[1])):
            ax.plot(contour_info['wavevector_parameter'], contour_fr[:, band_idx], 
                   label=f'Band {band_idx+1}', linewidth=2)
        ax.set_xlabel('Wavevector Parameter')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Python Dispersion')
        ax.grid(True, alpha=0.3)
        if fr_py.shape[1] <= 6:
            ax.legend()
    
    # Row 2: MATLAB dispersion (if available)
    if matlab_results is not None:
        wv_mat = matlab_results.get('wv', None)
        fr_mat = matlab_results.get('fr', None)
        
        if wv_mat is not None and fr_mat is not None:
            # Handle MATLAB shape differences
            if len(wv_mat.shape) == 3:
                wv_mat = wv_mat[:, :, 0]  # Take first structure
            if len(fr_mat.shape) == 3:
                fr_mat = fr_mat[:, :, 0]
            
            # Find matching wavevectors
            distances = cdist(contour_wv, wv_mat)
            closest_indices = np.argmin(distances, axis=1)
            contour_fr = fr_mat[closest_indices, :]
            
            ax = plt.subplot(3, 3, 5)
            for band_idx in range(min(6, fr_mat.shape[1])):
                ax.plot(contour_info['wavevector_parameter'], contour_fr[:, band_idx], 
                       label=f'Band {band_idx+1}', linewidth=2)
            ax.set_xlabel('Wavevector Parameter')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title('MATLAB Dispersion')
            ax.grid(True, alpha=0.3)
            if fr_mat.shape[1] <= 6:
                ax.legend()
    
    # Row 2: Overlay comparison
    if python_results is not None and matlab_results is not None:
        wv_py = python_results['wv']
        fr_py = python_results['fr']
        wv_mat = matlab_results.get('wv', None)
        fr_mat = matlab_results.get('fr', None)
        
        if wv_mat is not None and fr_mat is not None:
            # Handle MATLAB shape differences
            if len(wv_mat.shape) == 3:
                wv_mat = wv_mat[:, :, 0]
            if len(fr_mat.shape) == 3:
                fr_mat = fr_mat[:, :, 0]
            
            ax = plt.subplot(3, 3, 6)
            # Use same contour data from above
            for band_idx in range(min(6, min(fr_py.shape[1], fr_mat.shape[1]))):
                # Python
                distances_py = cdist(contour_wv, wv_py)
                closest_py = np.argmin(distances_py, axis=1)
                fr_py_contour = fr_py[closest_py, band_idx]
                
                # MATLAB
                distances_mat = cdist(contour_wv, wv_mat)
                closest_mat = np.argmin(distances_mat, axis=1)
                fr_mat_contour = fr_mat[closest_mat, band_idx]
                
                ax.plot(contour_info['wavevector_parameter'], fr_py_contour, 
                       'b-', label='Python' if band_idx == 0 else '', linewidth=2, alpha=0.7)
                ax.plot(contour_info['wavevector_parameter'], fr_mat_contour, 
                       'r--', label='MATLAB' if band_idx == 0 else '', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Wavevector Parameter')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title('Overlay Comparison')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Row 3: Difference plots
    if python_results is not None and matlab_results is not None:
        wv_py = python_results['wv']
        fr_py = python_results['fr']
        wv_mat = matlab_results.get('wv', None)
        fr_mat = matlab_results.get('fr', None)
        
        if wv_mat is not None and fr_mat is not None:
            # Handle MATLAB shape differences
            if len(wv_mat.shape) == 3:
                wv_mat = wv_mat[:, :, 0]
            if len(fr_mat.shape) == 3:
                fr_mat = fr_mat[:, :, 0]
            
            # Calculate differences
            distances_py = cdist(contour_wv, wv_py)
            closest_py = np.argmin(distances_py, axis=1)
            distances_mat = cdist(contour_wv, wv_mat)
            closest_mat = np.argmin(distances_mat, axis=1)
            
            n_bands = min(fr_py.shape[1], fr_mat.shape[1])
            for band_idx in range(min(3, n_bands)):
                ax = plt.subplot(3, 3, 7 + band_idx)
                fr_py_contour = fr_py[closest_py, band_idx]
                fr_mat_contour = fr_mat[closest_mat, band_idx]
                diff = fr_py_contour - fr_mat_contour
                rel_diff = diff / (np.abs(fr_mat_contour) + 1e-10) * 100
                
                ax.plot(contour_info['wavevector_parameter'], rel_diff, 'g-', linewidth=2)
                ax.axhline(0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('Wavevector Parameter')
                ax.set_ylabel('Relative Difference [%]')
                ax.set_title(f'Band {band_idx+1} Difference')
                ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Dispersion Comparison: {design_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'{design_name}_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison plot: {output_file}")
    return output_file


def main():
    """Main execution function."""
    print("="*60)
    print("Dispersion Plot Comparison: MATLAB vs Python")
    print("="*60)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'dispersion_comparison_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test designs
    print("\n1. Generating test geometries...")
    designs, N_pix = generate_test_designs()
    print(f"   Generated {len(designs)} test designs with N_pix={N_pix}")
    
    # Constants for dispersion calculation
    const_base = {
        'N_pix': N_pix,
        'N_ele': 2,  # Elements per pixel
        'N_k': 15,  # Reduced for faster computation
        'symmetry_type': 'none',
        'a': 1.0,
        't': 1.0,  # Thickness
        'E_min': 2e9,
        'E_max': 200e9,
        'rho_min': 1e3,
        'rho_max': 8e3,
        'poisson_min': 0.0,
        'poisson_max': 0.5,
        'design_scale': 'linear',
        'N_eig': 6,
        'sigma_eig': 1.0,
        'isSaveEigenvectors': False,
        'isSaveMesh': False,
        'isUseSecondImprovement': True,
        'isUseGPU': False,
        'isUseParallel': False,
    }
    
    # Process each design
    results_summary = {}
    
    for design_name, design in designs.items():
        print(f"\n2. Processing design: {design_name}")
        
        # Create constants for this design
        const = const_base.copy()
        
        # Save design for MATLAB
        matlab_design_file = save_design_for_matlab(design, design_name, output_dir)
        print(f"   Saved design for MATLAB: {matlab_design_file}")
        
        # Run Python dispersion
        print(f"   Running Python dispersion calculation...")
        try:
            wv_py, fr_py, ev_py = run_python_dispersion(design, const)
            python_results = {
                'wv': wv_py,
                'fr': fr_py,
                'ev': ev_py,
                'const': const
            }
            print(f"   Python: {wv_py.shape[0]} wavevectors, {fr_py.shape[1]} bands")
        except Exception as e:
            print(f"   ERROR in Python calculation: {e}")
            import traceback
            traceback.print_exc()
            python_results = None
        
        # Run MATLAB dispersion
        print(f"   Running MATLAB dispersion calculation...")
        matlab_results = run_matlab_dispersion(design_name, const, output_dir)
        if matlab_results is not None:
            print(f"   MATLAB results loaded successfully")
        else:
            print(f"   MATLAB calculation skipped or failed")
        
        # Create comparison plots
        print(f"   Creating comparison plots...")
        plot_file = create_comparison_plots(design_name, design, python_results, matlab_results, output_dir)
        
        # Store results
        results_summary[design_name] = {
            'python_success': python_results is not None,
            'matlab_success': matlab_results is not None,
            'plot_file': str(plot_file) if plot_file else None
        }
    
    # Save summary
    summary_file = output_dir / 'comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    
    return output_dir


if __name__ == '__main__':
    main()

