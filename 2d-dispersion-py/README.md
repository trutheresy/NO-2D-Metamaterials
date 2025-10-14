# 2D Dispersion Analysis Package

A Python translation of the MATLAB 2D dispersion analysis code for metamaterials. This package provides functionality for calculating dispersion relations, group velocities, and design sensitivities for 2D periodic structures.

## Features

- **Dispersion Calculation**: Compute dispersion relations for 2D periodic metamaterials
- **Group Velocity Analysis**: Calculate group velocities and their design sensitivities
- **Design Generation**: Generate various types of metamaterial designs
- **Visualization**: Plot dispersion relations, design patterns, and 3D surfaces
- **Dataset Generation**: Create datasets of dispersion relations for machine learning
- **Gaussian Process Kernels**: Generate correlated designs using various kernel functions

## Installation

1. Clone or download this package
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from dispersion import dispersion
from get_design import get_design
from wavevectors import get_IBZ_wavevectors
from plotting import plot_dispersion, plot_design

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

# Create a design
design = get_design('dispersive-tetragonal', 5)
const['design'] = design

# Plot design
plot_design(design)

# Compute dispersion
wv, fr, ev = dispersion(const, const['wavevectors'])

# Plot dispersion
plot_dispersion(np.arange(len(wv)), fr[:, 0], 5)
```

## Examples

Run the example scripts to see the package in action:

```python
from examples import run_all_examples
run_all_examples()
```

Or run individual examples:

```python
from examples import example_basic_dispersion, example_dispersion_surface
example_basic_dispersion()
example_dispersion_surface()
```

### Demo: Creating Reduced Matrices

To understand how reduced stiffness (Kr) and mass (Mr) matrices are created from the global K, M matrices and transformation matrix T:

```bash
python demo_create_Kr_and_Mr.py
```

This demo shows:
- Loading a dataset with K_DATA, M_DATA, and T_DATA
- Extracting matrices for a specific unit cell and wavevector
- Creating reduced matrices: Kr = T' * K * T and Mr = T' * M * T
- Visualizing sparsity patterns of all matrices

### Script: Comprehensive Dispersion Plotting

The `plot_dispersion_script.py` is a comprehensive visualization script equivalent to MATLAB's `plot_dispersion.m`:

```bash
python plot_dispersion_script.py
```

This script:
- Loads datasets with eigenvalues and eigenvectors
- Plots unit cell designs
- Creates IBZ contour plots
- Interpolates dispersion relations onto contours
- Reconstructs frequencies from eigenvectors for verification
- Exports all plots to PNG files organized by type

## Main Modules

### Core Functions
- `dispersion.py`: Main dispersion calculation functions
- `dispersion2.py`: Enhanced dispersion with group velocity and sensitivity analysis

### Design and Parameters
- `design_parameters.py`: Design parameter management class
- `get_design.py`: Design generation functions
- `kernels.py`: Gaussian process kernel functions

### System Assembly
- `system_matrices.py`: System matrix assembly and transformation
- `elements.py`: Element-level calculations
- `get_global_idxs.py`: Global DOF indexing

### Visualization
- `plotting.py`: Plotting and visualization functions
- `wavevectors.py`: Wavevector generation for different symmetries

### Utilities
- `utils.py`: Utility functions for data processing and validation
  - `linspaceNDim`: N-dimensional linspace for creating paths between points
  - `validate_constants`: Validate required fields in constants structure
  - `compute_band_gap`: Calculate band gap information from frequencies
- `dataset_generation.py`: Dataset generation for machine learning
- `examples.py`: Example scripts and tutorials

### New Features (Recently Added)
- **linspaceNDim**: Create linearly spaced points between N-dimensional vectors
  ```python
  from utils import linspaceNDim
  path = linspaceNDim([0, 0], [1, 2], 50)  # 50 points from (0,0) to (1,2)
  ```
- **Enhanced IBZ contours**: Improved `get_IBZ_contour_wavevectors` with support for:
  - Multiple symmetry types (p4mm, c1m1, p6mm, none)
  - Returns contour_info dict with vertex labels and parameters
  - Proper handling of high-symmetry points
- **Demo script**: `demo_create_Kr_and_Mr.py` for visualizing matrix reduction process

## Design Types

The package supports various predefined design patterns:

- `'homogeneous'`: Uniform material properties
- `'dispersive-tetragonal'`: Tetragonal symmetry with dispersion
- `'dispersive-orthotropic'`: Orthotropic symmetry
- `'quasi-1D'`: One-dimensional variation
- `'rotationally-symmetric'`: Rotationally symmetric design
- Random designs using seed numbers

## Symmetry Types

Wavevector generation supports different symmetry types:

- `'none'`: No symmetry (asymmetric IBZ)
- `'omit'`: Square IBZ centered at origin
- `'p4mm'`: P4mm symmetry (triangular IBZ)
- `'c1m1'`: C1m1 symmetry
- `'p2mm'`: P2mm symmetry

## Dataset Generation

Generate datasets for machine learning applications:

```python
from dataset_generation import generate_dispersion_dataset

dataset = generate_dispersion_dataset(
    N_struct=1000,
    N_pix=5,
    N_ele=2,
    N_eig=10,
    N_wv=11,
    symmetry_type='none',
    isSaveEigenvectors=False
)
```

## Dataset Conversion to NumPy/PyTorch Format

Convert MATLAB datasets to NumPy format following `dataset_conversion_reduction.ipynb` conventions, ready for PyTorch neural network training:

### Convert a Dataset

```bash
python convert_mat_to_pytorch.py <path_to_mat_file>
```

This creates a new folder with `_py` suffix containing NumPy arrays matching `data/set_cr_1200n/` format:

- `designs.npy` - Elastic modulus only (N_struct, N_pix, N_pix) [float16]
- `wavevectors.npy` - Wavevector grid (N_struct, N_wv, 2) [float16]
- `waveforms.npy` - Wavelet-embedded wavevectors (N_wv, N_pix, N_pix) [float16]
- `eigenvalue_data.npy` - Eigenfrequencies (N_struct, N_wv, N_eig) [original dtype]
- `eigenvector_data_x.npy`, `eigenvector_data_y.npy` - Spatial eigenvectors [complex]
- `bands_fft.npy` - Wavelet-embedded band indices (N_eig, N_pix, N_pix) [float16]
- `design_params.npy` - Parameters [float64]

### Use with PyTorch

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from example_pytorch_dataset import DispersionDataset

# Load and use dataset
dataset = DispersionDataset('data/my_dataset_py')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    designs = batch['design']  # (batch, N_pix, N_pix) - elastic modulus
    eigenvals = batch['eigenvalues']  # (batch, N_wv, N_eig)
    # ... your training code
```

### Example Usage

Run the example script:

```bash
python example_pytorch_dataset.py data/my_dataset_py
```

### Key Features

- **Exactly matches `data/set_cr_1200n/` format**
- **50% smaller** than MATLAB .mat files (float16 precision)
- **NumPy format** for flexibility (easy conversion to PyTorch)
- **Wavelet embeddings** for spatial neural network input
- **Only elastic modulus** in designs (following notebook convention)

### Documentation

- **`PYTORCH_CONVERSION.md`** - Complete conversion guide
- **`WAVELET_EMBEDDING_NOTE.md`** - About wavelet encoding (important!)
- **`convert_mat_to_pytorch.py`** - Conversion script with inline docs
- **`example_pytorch_dataset.py`** - Usage examples

## Requirements

- Python 3.7+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- h5py >= 3.0.0 (required for MATLAB v7.3 files)

## Troubleshooting

If you encounter errors when running scripts, see:
- **`TROUBLESHOOTING.md`** - Common issues and solutions
- **`MATLAB_V73_LOADING_GUIDE.md`** - Detailed guide for loading MATLAB files

Quick diagnostic:
```bash
python mat73_loader.py path/to/your/file.mat
```

## Citation

If you use this package in your research, please cite the original MATLAB implementation and any relevant papers.

## License

This package is a translation of the original MATLAB code. Please refer to the original code for licensing information.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Original MATLAB Code

This Python package is a translation of the MATLAB code found in the `2D-dispersion_alex` directory. The original MATLAB implementation provides the foundation for this Python version.

