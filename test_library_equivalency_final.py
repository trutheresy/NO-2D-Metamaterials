"""
Final Comprehensive Test for Library Equivalency

This script performs a 4-step cross-validation test to ensure the MATLAB
(2D-dispersion-han) and Python (2d-dispersion-py) libraries are equivalent:

1. Generate geometries in MATLAB → run through MATLAB library → get eigenmodes/dispersion plots
2. Generate geometries in Python → run through Python library → get eigenmodes/dispersion plots
3. Extract geometries from MATLAB dataset → run through Python library → get eigenmodes/dispersion plots
4. Extract geometries from Python dataset → run through MATLAB library → get eigenmodes/dispersion plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import subprocess
import json
from datetime import datetime
import scipy.io as sio
from scipy.spatial.distance import cdist

# Add Python library path
sys.path.insert(0, str(Path(__file__).parent / '2d-dispersion-py'))

# Import Python library functions
from get_design import get_design
from wavevectors import get_IBZ_wavevectors, get_IBZ_contour_wavevectors
from dispersion import dispersion
from plotting import plot_design, plot_dispersion_contour

# Import MATLAB loader
try:
    from mat73_loader import load_matlab_v73
except ImportError:
    print("Warning: Could not import mat73_loader. Some features may not work.")


class LibraryEquivalencyTest:
    """Comprehensive test suite for library equivalency."""
    
    def __init__(self, output_dir=None):
        """Initialize test suite."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f'equivalency_test_{timestamp}')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test parameters
        self.N_pix = 16  # Use 16x16 for reasonable computation time
        self.N_ele = 1
        self.N_wv = [15, 8]  # [N_wv_x, N_wv_y]
        self.N_eig = 6
        self.symmetry_type = 'none'
        self.a = 1.0
        
        # Material properties
        self.E_min = 20e6
        self.E_max = 200e9
        self.rho_min = 1200
        self.rho_max = 8e3
        self.poisson_min = 0.0
        self.poisson_max = 0.5
        self.t = 1.0
        
        # Test design names
        self.test_designs = ['homogeneous', 'quasi-1D', 'dispersive-tetragonal', 'dispersive-orthotropic']
        
        # Results storage
        self.results = {}
        
        print(f"Test output directory: {self.output_dir}")
    
    def get_const_dict(self, design=None):
        """Get constants dictionary for dispersion calculation."""
        const = {
            'N_pix': self.N_pix,
            'N_ele': self.N_ele,
            'N_eig': self.N_eig,
            'symmetry_type': self.symmetry_type,
            'a': self.a,
            't': self.t,
            'E_min': self.E_min,
            'E_max': self.E_max,
            'rho_min': self.rho_min,
            'rho_max': self.rho_max,
            'poisson_min': self.poisson_min,
            'poisson_max': self.poisson_max,
            'design_scale': 'linear',
            'sigma_eig': 1e-2,
            'isSaveEigenvectors': True,
            'isSaveMesh': False,
            'isUseSecondImprovement': False,
            'isUseImprovement': True,
            'isUseGPU': False,
            'isUseParallel': False,
        }
        if design is not None:
            const['design'] = design
        return const
    
    def step1_matlab_to_matlab(self, design_name):
        """
        Step 1: Generate geometry in MATLAB → run through MATLAB library → get eigenmodes/dispersion.
        """
        print(f"\n{'='*70}")
        print(f"STEP 1: MATLAB Geometry -> MATLAB Library")
        print(f"Design: {design_name}")
        print(f"{'='*70}")
        
        # Create MATLAB script
        matlab_script = self.output_dir / f'step1_{design_name}.m'
        design_file = self.output_dir / f'{design_name}_design.mat'
        results_file = self.output_dir / f'step1_{design_name}_results.mat'
        
        # First, generate design in MATLAB
        matlab_gen_script = self.output_dir / f'step1_{design_name}_generate.m'
        # Format paths for MATLAB (use forward slashes)
        matlab_lib_path = str(Path(__file__).parent / "2D-dispersion-han").replace('\\', '/')
        matlab_base_path = str(Path(__file__).parent).replace('\\', '/')
        design_file_path = str(design_file.resolve()).replace('\\', '/')
        
        matlab_gen_content = f"""
% Generate design in MATLAB
addpath(genpath('{matlab_lib_path}'));
addpath(genpath('{matlab_base_path}'));

% Generate design
N_pix = {self.N_pix};
design = get_design('{design_name}', N_pix);

% Convert to material properties
E_min = {self.E_min};
E_max = {self.E_max};
rho_min = {self.rho_min};
rho_max = {self.rho_max};
poisson_min = {self.poisson_min};
poisson_max = {self.poisson_max};

% Apply material conversion
design = convert_design(design, 'linear', 'linear', E_min, E_max, rho_min, rho_max);
design = apply_steel_rubber_paradigm(design, struct('E_min', E_min, 'E_max', E_max, ...
    'rho_min', rho_min, 'rho_max', rho_max, 'poisson_min', poisson_min, 'poisson_max', poisson_max));

% Save design
save('{design_file_path}', 'design', '-v7.3');
fprintf('Generated and saved design: %s\\n', '{design_file_path}');
"""
        
        with open(matlab_gen_script, 'w') as f:
            f.write(matlab_gen_content)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run MATLAB to generate design
        print(f"  Running MATLAB to generate design...")
        try:
            # Use absolute path for MATLAB script and change to script directory
            matlab_script_abs = str(matlab_gen_script.resolve())
            matlab_script_dir = str(matlab_gen_script.parent)
            result = subprocess.run(
                ['matlab', '-batch', f"cd('{matlab_script_dir}'); run('{matlab_gen_script.name}')"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(Path(__file__).parent)
            )
            if result.returncode != 0:
                print(f"  ERROR: MATLAB design generation failed")
                print(f"  {result.stderr}")
                return None
        except Exception as e:
            print(f"  ERROR: Could not run MATLAB: {e}")
            return None
        
        # Now run dispersion in MATLAB
        # Format paths for MATLAB
        matlab_lib_path = str(Path(__file__).parent / "2D-dispersion-han").replace('\\', '/')
        matlab_base_path = str(Path(__file__).parent).replace('\\', '/')
        design_file_path = str(design_file.resolve()).replace('\\', '/')
        results_file_path = str(results_file.resolve()).replace('\\', '/')
        
        matlab_disp_content = f"""
% Run dispersion calculation in MATLAB
addpath(genpath('{matlab_lib_path}'));
addpath(genpath('{matlab_base_path}'));

% Load design
load('{design_file_path}');

% Set up constants
const = struct();
const.N_pix = {self.N_pix};
const.N_ele = {self.N_ele};
const.N_eig = {self.N_eig};
const.symmetry_type = '{self.symmetry_type}';
const.a = {self.a};
const.t = {self.t};
const.E_min = {self.E_min};
const.E_max = {self.E_max};
const.rho_min = {self.rho_min};
const.rho_max = {self.rho_max};
const.poisson_min = {self.poisson_min};
const.poisson_max = {self.poisson_max};
const.design_scale = 'linear';
const.sigma_eig = {1e-2};
const.isSaveEigenvectors = true;
const.isSaveMesh = false;
const.isUseSecondImprovement = false;
const.isUseImprovement = true;
const.isUseGPU = false;
const.isUseParallel = false;
const.design = design;

% Get wavevectors
wavevectors = get_IBZ_wavevectors([{self.N_wv[0]}, {self.N_wv[1]}], const.a, const.symmetry_type);

% Run dispersion
[wv, fr, ev] = dispersion(const, wavevectors);

% Save results
save('{results_file_path}', 'wv', 'fr', 'ev', 'wavevectors', 'design', 'const', '-v7.3');
fprintf('Saved MATLAB results to: %s\\n', '{results_file_path}');
"""
        
        with open(matlab_script, 'w') as f:
            f.write(matlab_disp_content)
        
        # Run MATLAB dispersion
        print(f"  Running MATLAB dispersion calculation...")
        try:
            # Use absolute path and change to script directory
            matlab_script_dir = str(matlab_script.parent)
            result = subprocess.run(
                ['matlab', '-batch', f"cd('{matlab_script_dir}'); run('{matlab_script.name}')"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent)
            )
            if result.returncode != 0:
                print(f"  ERROR: MATLAB dispersion calculation failed")
                print(f"  {result.stderr}")
                return None
        except Exception as e:
            print(f"  ERROR: Could not run MATLAB: {e}")
            return None
        
        # Load results
        if results_file.exists():
            try:
                data = load_matlab_v73(str(results_file), verbose=False)
                print(f"  Successfully loaded MATLAB results")
                return {
                    'wv': data.get('wv', None),
                    'fr': data.get('fr', None),
                    'ev': data.get('ev', None),
                    'design': data.get('design', None),
                    'wavevectors': data.get('wavevectors', None)
                }
            except Exception as e:
                print(f"  ERROR loading results: {e}")
                return None
        else:
            print(f"  ERROR: Results file not found: {results_file}")
            return None
    
    def step2_python_to_python(self, design_name):
        """
        Step 2: Generate geometry in Python → run through Python library → get eigenmodes/dispersion.
        """
        print(f"\n{'='*70}")
        print(f"STEP 2: Python Geometry -> Python Library")
        print(f"Design: {design_name}")
        print(f"{'='*70}")
        
        # Generate design in Python
        print(f"  Generating design in Python...")
        design = get_design(design_name, self.N_pix)
        
        # Convert to material properties (matching MATLAB conversion)
        from design_conversion import convert_design, apply_steel_rubber_paradigm
        
        const = self.get_const_dict(design)
        design_converted = convert_design(
            design, 'linear', 'linear',
            self.E_min, self.E_max, self.rho_min, self.rho_max,
            self.poisson_min, self.poisson_max
        )
        design_converted = apply_steel_rubber_paradigm(design_converted, const)
        
        # Update const with converted design
        const['design'] = design_converted
        
        # Get wavevectors
        print(f"  Getting wavevectors...")
        wavevectors = get_IBZ_wavevectors(self.N_wv, self.a, self.symmetry_type)
        
        # Run dispersion
        print(f"  Running Python dispersion calculation...")
        try:
            wv, fr, ev, mesh = dispersion(const, wavevectors)
            print(f"  Successfully computed dispersion")
            print(f"    Wavevectors: {wv.shape}")
            print(f"    Frequencies: {fr.shape}")
            print(f"    Eigenvectors: {ev.shape if ev is not None else None}")
            
            return {
                'wv': wv,
                'fr': fr,
                'ev': ev,
                'design': design_converted,
                'wavevectors': wavevectors
            }
        except Exception as e:
            print(f"  ERROR in Python dispersion calculation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step3_matlab_dataset_to_python(self, matlab_dataset_path, struct_idx=0):
        """
        Step 3: Extract geometry from MATLAB dataset → run through Python library → get eigenmodes/dispersion.
        """
        print(f"\n{'='*70}")
        print(f"STEP 3: MATLAB Dataset Geometry -> Python Library")
        print(f"Dataset: {matlab_dataset_path}")
        print(f"Structure index: {struct_idx}")
        print(f"{'='*70}")
        
        # Load MATLAB dataset
        print(f"  Loading MATLAB dataset...")
        try:
            data = load_matlab_v73(str(matlab_dataset_path), verbose=False)
        except Exception as e:
            print(f"  ERROR loading MATLAB dataset: {e}")
            return None
        
        # Extract design (geometry)
        if 'designs' in data:
            designs = data['designs']
            if len(designs.shape) == 4:  # (N_pix, N_pix, 3, N_struct)
                design = designs[:, :, :, struct_idx]
            elif len(designs.shape) == 3:  # (N_pix, N_pix, 3)
                design = designs
            else:
                print(f"  ERROR: Unexpected designs shape: {designs.shape}")
                return None
        elif 'CONSTITUTIVE_DATA' in data:
            # Extract from constitutive data
            constitutive = data['CONSTITUTIVE_DATA']
            if isinstance(constitutive, dict):
                E = constitutive.get('modulus', None)
                rho = constitutive.get('density', None)
                nu = constitutive.get('poisson', None)
                if E is not None and rho is not None and nu is not None:
                    # Convert back to normalized design (0-1)
                    E_norm = (E[:, :, struct_idx] - self.E_min) / (self.E_max - self.E_min)
                    rho_norm = (rho[:, :, struct_idx] - self.rho_min) / (self.rho_max - self.rho_min)
                    nu_norm = (nu[:, :, struct_idx] - self.poisson_min) / (self.poisson_max - self.poisson_min)
                    design = np.stack([E_norm, rho_norm, nu_norm], axis=-1)
                else:
                    print(f"  ERROR: Could not extract constitutive data")
                    return None
            else:
                print(f"  ERROR: CONSTITUTIVE_DATA is not a dict")
                return None
        else:
            print(f"  ERROR: Could not find designs or CONSTITUTIVE_DATA in dataset")
            return None
        
        print(f"  Extracted design shape: {design.shape}")
        
        # Ensure design is in correct format (N_pix, N_pix, 3)
        if design.shape[0] == 3 and len(design.shape) == 3:
            design = np.transpose(design, (1, 2, 0))
        
        # Convert to material properties (if needed)
        const = self.get_const_dict(design)
        
        # Get wavevectors
        print(f"  Getting wavevectors...")
        wavevectors = get_IBZ_wavevectors(self.N_wv, self.a, self.symmetry_type)
        
        # Run dispersion in Python
        print(f"  Running Python dispersion calculation...")
        try:
            wv, fr, ev, mesh = dispersion(const, wavevectors)
            print(f"  Successfully computed dispersion")
            print(f"    Wavevectors: {wv.shape}")
            print(f"    Frequencies: {fr.shape}")
            print(f"    Eigenvectors: {ev.shape if ev is not None else None}")
            
            return {
                'wv': wv,
                'fr': fr,
                'ev': ev,
                'design': design,
                'wavevectors': wavevectors,
                'source': 'matlab_dataset'
            }
        except Exception as e:
            print(f"  ERROR in Python dispersion calculation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step4_python_dataset_to_matlab(self, python_dataset_path, struct_idx=0):
        """
        Step 4: Extract geometry from Python dataset → run through MATLAB library → get eigenmodes/dispersion.
        """
        print(f"\n{'='*70}")
        print(f"STEP 4: Python Dataset Geometry -> MATLAB Library")
        print(f"Dataset: {python_dataset_path}")
        print(f"Structure index: {struct_idx}")
        print(f"{'='*70}")
        
        # Load Python dataset (could be .npz, .mat, or PyTorch format)
        print(f"  Loading Python dataset...")
        dataset_path = Path(python_dataset_path)
        
        if dataset_path.suffix == '.npz':
            data = np.load(dataset_path, allow_pickle=True)
            if 'designs' in data:
                design = data['designs'][struct_idx]
            elif 'geometries' in data:
                design = data['geometries'][struct_idx]
            else:
                print(f"  ERROR: Could not find designs or geometries in dataset")
                return None
        elif dataset_path.suffix == '.mat':
            try:
                data = load_matlab_v73(str(dataset_path), verbose=False)
                if 'designs' in data:
                    designs = data['designs']
                    if len(designs.shape) == 4:
                        design = designs[:, :, :, struct_idx]
                    else:
                        design = designs
                else:
                    print(f"  ERROR: Could not find designs in dataset")
                    return None
            except Exception as e:
                print(f"  ERROR loading .mat file: {e}")
                return None
        else:
            # Try PyTorch format
            try:
                import torch
                if (dataset_path / 'geometries_full.pt').exists():
                    geometries = torch.load(dataset_path / 'geometries_full.pt', map_location='cpu')
                    design = geometries[struct_idx].numpy()
                    # PyTorch format might be (N_pix, N_pix) - single channel
                    if len(design.shape) == 2:
                        # Convert to 3-channel by replicating
                        design = np.stack([design, design, 0.6 * np.ones_like(design)], axis=-1)
                else:
                    print(f"  ERROR: Could not find geometries_full.pt in dataset directory")
                    return None
            except Exception as e:
                print(f"  ERROR loading PyTorch format: {e}")
                return None
        
        print(f"  Extracted design shape: {design.shape}")
        
        # Ensure design is in correct format (N_pix, N_pix, 3)
        if len(design.shape) == 2:
            design = np.stack([design, design, 0.6 * np.ones_like(design)], axis=-1)
        elif design.shape[0] == 3 and len(design.shape) == 3:
            design = np.transpose(design, (1, 2, 0))
        
        # Save design for MATLAB
        design_file = self.output_dir / f'step4_struct{struct_idx}_design.mat'
        sio.savemat(str(design_file), {'design': design}, format='7.3')
        print(f"  Saved design for MATLAB: {design_file}")
        
        # Create MATLAB script
        matlab_script = self.output_dir / f'step4_struct{struct_idx}.m'
        results_file = self.output_dir / f'step4_struct{struct_idx}_results.mat'
        
        # Format paths for MATLAB
        matlab_lib_path = str(Path(__file__).parent / "2D-dispersion-han").replace('\\', '/')
        matlab_base_path = str(Path(__file__).parent).replace('\\', '/')
        design_file_path = str(design_file.resolve()).replace('\\', '/')
        results_file_path = str(results_file.resolve()).replace('\\', '/')
        
        matlab_content = f"""
% Run dispersion calculation in MATLAB with Python dataset geometry
addpath(genpath('{matlab_lib_path}'));
addpath(genpath('{matlab_base_path}'));

% Load design from Python dataset
load('{design_file_path}');

% Set up constants
const = struct();
const.N_pix = {self.N_pix};
const.N_ele = {self.N_ele};
const.N_eig = {self.N_eig};
const.symmetry_type = '{self.symmetry_type}';
const.a = {self.a};
const.t = {self.t};
const.E_min = {self.E_min};
const.E_max = {self.E_max};
const.rho_min = {self.rho_min};
const.rho_max = {self.rho_max};
const.poisson_min = {self.poisson_min};
const.poisson_max = {self.poisson_max};
const.design_scale = 'linear';
const.sigma_eig = {1e-2};
const.isSaveEigenvectors = true;
const.isSaveMesh = false;
const.isUseSecondImprovement = false;
const.isUseImprovement = true;
const.isUseGPU = false;
const.isUseParallel = false;
const.design = design;

% Get wavevectors
wavevectors = get_IBZ_wavevectors([{self.N_wv[0]}, {self.N_wv[1]}], const.a, const.symmetry_type);

% Run dispersion
[wv, fr, ev] = dispersion(const, wavevectors);

% Save results
save('{results_file_path}', 'wv', 'fr', 'ev', 'wavevectors', 'design', 'const', '-v7.3');
fprintf('Saved MATLAB results to: %s\\n', '{results_file_path}');
"""
        
        with open(matlab_script, 'w') as f:
            f.write(matlab_content)
        
        # Run MATLAB dispersion
        print(f"  Running MATLAB dispersion calculation...")
        try:
            # Use absolute path and change to script directory
            matlab_script_dir = str(matlab_script.parent)
            result = subprocess.run(
                ['matlab', '-batch', f"cd('{matlab_script_dir}'); run('{matlab_script.name}')"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent)
            )
            if result.returncode != 0:
                print(f"  ERROR: MATLAB dispersion calculation failed")
                print(f"  {result.stderr}")
                return None
        except Exception as e:
            print(f"  ERROR: Could not run MATLAB: {e}")
            return None
        
        # Load results
        if results_file.exists():
            try:
                data = load_matlab_v73(str(results_file), verbose=False)
                print(f"  Successfully loaded MATLAB results")
                return {
                    'wv': data.get('wv', None),
                    'fr': data.get('fr', None),
                    'ev': data.get('ev', None),
                    'design': data.get('design', None),
                    'wavevectors': data.get('wavevectors', None),
                    'source': 'python_dataset'
                }
            except Exception as e:
                print(f"  ERROR loading results: {e}")
                return None
        else:
            print(f"  ERROR: Results file not found: {results_file}")
            return None
    
    def compare_results(self, result1, result2, label1, label2, design_name):
        """Compare two results and create visualization."""
        if result1 is None or result2 is None:
            print(f"  Cannot compare: one or both results are None")
            return None
        
        print(f"\n  Comparing {label1} vs {label2}...")
        
        # Extract data
        wv1 = result1.get('wv', None)
        fr1 = result1.get('fr', None)
        wv2 = result2.get('wv', None)
        fr2 = result2.get('fr', None)
        
        if wv1 is None or fr1 is None or wv2 is None or fr2 is None:
            print(f"  ERROR: Missing wavevector or frequency data")
            return None
        
        # Handle MATLAB shape differences and ensure correct format
        # Wavevectors should be (N_wv, 2) - rows are wavevectors, columns are x,y components
        # Frequencies should be (N_wv, N_eig) - rows are wavevectors, columns are bands
        
        # Handle wavevectors
        if len(wv1.shape) == 3:
            wv1 = wv1[:, :, 0]
        elif len(wv1.shape) == 2 and wv1.shape[1] != 2:
            # If transposed (2, N_wv), transpose it
            if wv1.shape[0] == 2:
                wv1 = wv1.T
        # Ensure it's 2D
        if len(wv1.shape) == 1:
            wv1 = wv1.reshape(-1, 2)
            
        if len(wv2.shape) == 3:
            wv2 = wv2[:, :, 0]
        elif len(wv2.shape) == 2 and wv2.shape[1] != 2:
            # If transposed (2, N_wv), transpose it
            if wv2.shape[0] == 2:
                wv2 = wv2.T
        # Ensure it's 2D
        if len(wv2.shape) == 1:
            wv2 = wv2.reshape(-1, 2)
        
        # Handle frequencies
        # Frequencies should be (N_wv, N_eig) - rows are wavevectors, columns are bands
        if len(fr1.shape) == 3:
            fr1 = fr1[:, :, 0]
        elif len(fr1.shape) == 2:
            # Check if transposed (N_eig, N_wv) -> transpose to (N_wv, N_eig)
            if fr1.shape[0] < fr1.shape[1] and fr1.shape[0] <= 10:  # Likely (N_eig, N_wv)
                fr1 = fr1.T
        elif len(fr1.shape) == 1:
            fr1 = fr1.reshape(-1, 1)
            
        if len(fr2.shape) == 3:
            fr2 = fr2[:, :, 0]
        elif len(fr2.shape) == 2:
            # Check if transposed (N_eig, N_wv) -> transpose to (N_wv, N_eig)
            if fr2.shape[0] < fr2.shape[1] and fr2.shape[0] <= 10:  # Likely (N_eig, N_wv)
                fr2 = fr2.T
        elif len(fr2.shape) == 1:
            fr2 = fr2.reshape(-1, 1)
        
        # Handle complex numbers (MATLAB may return structured arrays)
        if fr1.dtype.names is not None:  # Structured array (complex from MATLAB)
            if 'real' in fr1.dtype.names and 'imag' in fr1.dtype.names:
                fr1 = fr1['real'] + 1j * fr1['imag']
            else:
                fr1 = fr1.view(np.complex128)
        if isinstance(fr1, np.ndarray) and np.iscomplexobj(fr1):
            fr1 = np.real(fr1)
            
        if fr2.dtype.names is not None:  # Structured array (complex from MATLAB)
            if 'real' in fr2.dtype.names and 'imag' in fr2.dtype.names:
                fr2 = fr2['real'] + 1j * fr2['imag']
            else:
                fr2 = fr2.view(np.complex128)
        if isinstance(fr2, np.ndarray) and np.iscomplexobj(fr2):
            fr2 = np.real(fr2)
        
        # Ensure frequencies match wavevector count
        if fr1.shape[0] != wv1.shape[0]:
            print(f"  WARNING: fr1 shape {fr1.shape} doesn't match wv1 shape {wv1.shape}, attempting transpose")
            if fr1.shape[1] == wv1.shape[0]:
                fr1 = fr1.T
        if fr2.shape[0] != wv2.shape[0]:
            print(f"  WARNING: fr2 shape {fr2.shape} doesn't match wv2 shape {wv2.shape}, attempting transpose")
            if fr2.shape[1] == wv2.shape[0]:
                fr2 = fr2.T
        
        # Verify shapes
        if wv1.shape[1] != 2:
            print(f"  ERROR: wv1 has wrong shape: {wv1.shape}, expected (N, 2)")
            return None
        if wv2.shape[1] != 2:
            print(f"  ERROR: wv2 has wrong shape: {wv2.shape}, expected (N, 2)")
            return None
        
        # Sort wavevectors for comparison
        if wv1.shape[0] > 0 and wv2.shape[0] > 0:
            # Sort by first component, then second
            idx1 = np.lexsort((wv1[:, 1], wv1[:, 0]))
            idx2 = np.lexsort((wv2[:, 1], wv2[:, 0]))
            wv1_sorted = wv1[idx1]
            fr1_sorted = fr1[idx1]
            wv2_sorted = wv2[idx2]
            fr2_sorted = fr2[idx2]
            
            # Match wavevectors (find closest matches)
            distances = cdist(wv1_sorted, wv2_sorted)
            matches = np.argmin(distances, axis=1)
            
            # Calculate differences
            n_bands = min(fr1_sorted.shape[1], fr2_sorted.shape[1])
            max_diff = 0
            mean_diff = 0
            max_rel_diff = 0
            mean_rel_diff = 0
            
            for band_idx in range(n_bands):
                fr1_band = fr1_sorted[:, band_idx]
                fr2_band = fr2_sorted[matches, band_idx]
                
                diff = np.abs(fr1_band - fr2_band)
                rel_diff = diff / (np.abs(fr2_band) + 1e-10) * 100
                
                max_diff = max(max_diff, np.max(diff))
                mean_diff += np.mean(diff)
                max_rel_diff = max(max_rel_diff, np.max(rel_diff))
                mean_rel_diff += np.mean(rel_diff)
            
            mean_diff /= n_bands
            mean_rel_diff /= n_bands
            
            print(f"    Max absolute difference: {max_diff:.6f} Hz")
            print(f"    Mean absolute difference: {mean_diff:.6f} Hz")
            print(f"    Max relative difference: {max_rel_diff:.2f}%")
            print(f"    Mean relative difference: {mean_rel_diff:.2f}%")
            
            # Create comparison plot
            self.create_comparison_plot(
                wv1_sorted, fr1_sorted, wv2_sorted, fr2_sorted,
                label1, label2, design_name
            )
            
            return {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_rel_diff': max_rel_diff,
                'mean_rel_diff': mean_rel_diff
            }
        else:
            print(f"  ERROR: Empty wavevector arrays")
            return None
    
    def save_geometry_plot(self, design, design_name, library_name):
        """Save geometry visualization for a design."""
        from plotting import plot_design
        
        # Create design-specific folder
        design_dir = self.output_dir / design_name
        design_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot and save
        fig, axes = plot_design(design)
        output_file = design_dir / f'geometry_{library_name}.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved geometry plot: {output_file}")
        return output_file
    
    def save_dispersion_plot(self, wv, fr, design_name, library_name):
        """Save individual dispersion plot."""
        # Create design-specific folder
        design_dir = self.output_dir / design_name
        design_dir.mkdir(parents=True, exist_ok=True)
        
        # Get contour for plotting
        contour_wv, contour_info = get_IBZ_contour_wavevectors(50, self.a, self.symmetry_type)
        
        # Find matching wavevectors
        distances = cdist(contour_wv, wv)
        closest = np.argmin(distances, axis=1)
        contour_fr = fr[closest, :]
        
        # Create plot using simple line plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_bands = min(6, fr.shape[1])
        colors = plt.cm.tab10(np.linspace(0, 1, n_bands))
        
        for band_idx in range(n_bands):
            ax.plot(contour_info['wavevector_parameter'], contour_fr[:, band_idx],
                   color=colors[band_idx], linewidth=2, label=f'Band {band_idx+1}')
        
        ax.set_xlabel('Wavevector Parameter', fontsize=11)
        ax.set_ylabel('Frequency [Hz]', fontsize=11)
        ax.set_title(f'Dispersion: {design_name} ({library_name})', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        output_file = design_dir / f'dispersion_{library_name}.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved dispersion plot: {output_file}")
        return output_file
    
    def create_comparison_plot(self, wv1, fr1, wv2, fr2, label1, label2, design_name):
        """Create comparison plot for two results."""
        # Sanitize labels for filename (Windows doesn't allow -> in filenames)
        label1_safe = label1.replace('->', '_to_').replace(' ', '_')
        label2_safe = label2.replace('->', '_to_').replace(' ', '_')
        
        # Get contour for plotting
        contour_wv, contour_info = get_IBZ_contour_wavevectors(50, self.a, self.symmetry_type)
        
        # Find matching wavevectors
        distances1 = cdist(contour_wv, wv1)
        closest1 = np.argmin(distances1, axis=1)
        distances2 = cdist(contour_wv, wv2)
        closest2 = np.argmin(distances2, axis=1)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Overlay comparison
        ax = axes[0, 0]
        n_bands = min(fr1.shape[1], fr2.shape[1])
        for band_idx in range(min(6, n_bands)):
            fr1_contour = fr1[closest1, band_idx]
            fr2_contour = fr2[closest2, band_idx]
            ax.plot(contour_info['wavevector_parameter'], fr1_contour,
                   'b-', label=label1 if band_idx == 0 else '', linewidth=2, alpha=0.7)
            ax.plot(contour_info['wavevector_parameter'], fr2_contour,
                   'r--', label=label2 if band_idx == 0 else '', linewidth=2, alpha=0.7)
        ax.set_xlabel('Wavevector Parameter')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(f'Overlay Comparison: {design_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Difference plot
        ax = axes[0, 1]
        for band_idx in range(min(3, n_bands)):
            fr1_contour = fr1[closest1, band_idx]
            fr2_contour = fr2[closest2, band_idx]
            diff = fr1_contour - fr2_contour
            rel_diff = diff / (np.abs(fr2_contour) + 1e-10) * 100
            ax.plot(contour_info['wavevector_parameter'], rel_diff,
                   label=f'Band {band_idx+1}', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Wavevector Parameter')
        ax.set_ylabel('Relative Difference [%]')
        ax.set_title('Relative Difference')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Band 1 detailed
        ax = axes[1, 0]
        fr1_contour = fr1[closest1, 0]
        fr2_contour = fr2[closest2, 0]
        ax.plot(contour_info['wavevector_parameter'], fr1_contour,
               'b-', label=label1, linewidth=2)
        ax.plot(contour_info['wavevector_parameter'], fr2_contour,
               'r--', label=label2, linewidth=2)
        ax.set_xlabel('Wavevector Parameter')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Band 1 Detailed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Band 2 detailed
        if n_bands > 1:
            ax = axes[1, 1]
            fr1_contour = fr1[closest1, 1]
            fr2_contour = fr2[closest2, 1]
            ax.plot(contour_info['wavevector_parameter'], fr1_contour,
                   'b-', label=label1, linewidth=2)
            ax.plot(contour_info['wavevector_parameter'], fr2_contour,
                   'r--', label=label2, linewidth=2)
            ax.set_xlabel('Wavevector Parameter')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title('Band 2 Detailed')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.suptitle(f'Comparison: {label1} vs {label2} - {design_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure in design-specific folder (use sanitized labels)
        design_dir = self.output_dir / design_name
        design_dir.mkdir(parents=True, exist_ok=True)
        output_file = design_dir / f'comparison_{label1_safe}_vs_{label2_safe}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved comparison plot: {output_file}")
    
    def run_all_tests(self, matlab_dataset_path=None, python_dataset_path=None):
        """Run all four test steps."""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE LIBRARY EQUIVALENCY TEST")
        print(f"{'='*70}")
        
        summary = {}
        
        # Test with predefined designs
        for design_name in self.test_designs:
            print(f"\n{'#'*70}")
            print(f"Testing design: {design_name}")
            print(f"{'#'*70}")
            
            design_summary = {}
            
            # Step 1: MATLAB → MATLAB
            result1 = self.step1_matlab_to_matlab(design_name)
            design_summary['step1_matlab_to_matlab'] = result1 is not None
            
            # Save MATLAB geometry and dispersion plots
            if result1 is not None and result1.get('design') is not None:
                self.save_geometry_plot(result1['design'], design_name, 'MATLAB')
                if result1.get('wv') is not None and result1.get('fr') is not None:
                    # Handle MATLAB shape differences
                    wv1 = result1['wv']
                    fr1 = result1['fr']
                    if len(wv1.shape) == 3:
                        wv1 = wv1[:, :, 0]
                    if len(fr1.shape) == 3:
                        fr1 = fr1[:, :, 0]
                    # Handle complex numbers
                    if fr1.dtype.names is not None:
                        if 'real' in fr1.dtype.names and 'imag' in fr1.dtype.names:
                            fr1 = fr1['real'] + 1j * fr1['imag']
                    if isinstance(fr1, np.ndarray) and np.iscomplexobj(fr1):
                        fr1 = np.real(fr1)
                    # Ensure correct shape
                    if len(wv1.shape) == 2 and wv1.shape[1] != 2:
                        if wv1.shape[0] == 2:
                            wv1 = wv1.T
                    if len(fr1.shape) == 2 and fr1.shape[0] < fr1.shape[1] and fr1.shape[0] <= 10:
                        fr1 = fr1.T
                    self.save_dispersion_plot(wv1, fr1, design_name, 'MATLAB')
            
            # Step 2: Python → Python
            result2 = self.step2_python_to_python(design_name)
            design_summary['step2_python_to_python'] = result2 is not None
            
            # Save Python geometry and dispersion plots
            if result2 is not None and result2.get('design') is not None:
                self.save_geometry_plot(result2['design'], design_name, 'Python')
                if result2.get('wv') is not None and result2.get('fr') is not None:
                    self.save_dispersion_plot(result2['wv'], result2['fr'], design_name, 'Python')
            
            # Compare Step 1 and Step 2 (should be similar if libraries are equivalent)
            if result1 is not None and result2 is not None:
                comparison = self.compare_results(
                    result1, result2,
                    'MATLAB->MATLAB', 'Python->Python',
                    design_name
                )
                design_summary['comparison_1_vs_2'] = comparison
            
            # Step 3: MATLAB dataset → Python (if dataset provided)
            if matlab_dataset_path is not None:
                result3 = self.step3_matlab_dataset_to_python(matlab_dataset_path, struct_idx=0)
                design_summary['step3_matlab_to_python'] = result3 is not None
                
                # Compare with Step 1 (should match if same geometry)
                if result1 is not None and result3 is not None:
                    comparison = self.compare_results(
                        result1, result3,
                        'MATLAB->MATLAB', 'MATLAB->Python',
                        design_name
                    )
                    design_summary['comparison_1_vs_3'] = comparison
            
            # Step 4: Python dataset → MATLAB (if dataset provided)
            if python_dataset_path is not None:
                result4 = self.step4_python_dataset_to_matlab(python_dataset_path, struct_idx=0)
                design_summary['step4_python_to_matlab'] = result4 is not None
                
                # Compare with Step 2 (should match if same geometry)
                if result2 is not None and result4 is not None:
                    comparison = self.compare_results(
                        result2, result4,
                        'Python->Python', 'Python->MATLAB',
                        design_name
                    )
                    design_summary['comparison_2_vs_4'] = comparison
            
            summary[design_name] = design_summary
        
        # Save summary
        summary_file = self.output_dir / 'test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print("TEST COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {self.output_dir}")
        print(f"Summary saved to: {summary_file}")
        
        return summary


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test library equivalency')
    parser.add_argument('--matlab-dataset', type=str, help='Path to MATLAB dataset (.mat file)')
    parser.add_argument('--python-dataset', type=str, help='Path to Python dataset (.npz, .mat, or directory with .pt files)')
    parser.add_argument('--output-dir', type=str, help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Create test suite
    test = LibraryEquivalencyTest(output_dir=args.output_dir)
    
    # Run all tests
    summary = test.run_all_tests(
        matlab_dataset_path=args.matlab_dataset,
        python_dataset_path=args.python_dataset
    )
    
    return summary


if __name__ == '__main__':
    main()

