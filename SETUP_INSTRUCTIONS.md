# Setup Instructions: 2D-Dispersion-Han Python Library

## Quick Setup (Recommended)

### Option 1: NumPy 2.0 + Compatible Packages (2025+ Recommended)

```bash
# Create environment with NumPy 2.0
conda env create -f environment-numpy2.yml

# Activate environment
conda activate dispersion_han_numpy2

# Verify installation
python -c "import numpy, scipy, matplotlib; print(f'NumPy {numpy.__version__}, SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}')"
```

### Option 2: NumPy 1.x (Conservative, Maximum Compatibility)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate dispersion_han

# Verify installation
python -c "import numpy, scipy, matplotlib; print('✅ All packages installed')"
```

### Option 2: Conda Manual

```bash
# Create environment
conda create -n dispersion_han python=3.9

# Activate
conda activate dispersion_han

# Install packages
conda install numpy=1.24 scipy matplotlib h5py -c conda-forge
```

### Option 3: pip with Virtualenv (NumPy 2.0)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install NumPy 2.0 compatible requirements
pip install -r requirements-numpy2.txt
```

### Option 4: pip with Virtualenv (NumPy 1.x)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install NumPy 1.x requirements
pip install -r requirements.txt
```

## Verify Installation

```bash
# Test imports
python -c "from dispersion import dispersion; print('✅ Library loaded')"

# Run simple test
python test_han_simple.py
```

## Troubleshooting

### NumPy 2.0 Compatibility Issue

If you see:
```
ImportError: numpy.core.multiarray failed to import
```

**Solution A: Upgrade to NumPy 2.0 Compatible Packages (Recommended)**
```bash
# Upgrade SciPy and Matplotlib to NumPy 2.0 compatible versions
pip install --upgrade "numpy>=2.0" "scipy>=1.13.0" "matplotlib>=3.9.0"

# Or recreate environment with NumPy 2.0
conda env remove -n dispersion_han
conda env create -f environment-numpy2.yml
```

**Solution B: Downgrade to NumPy 1.x (Conservative)**
```bash
# Downgrade NumPy
pip install "numpy<2.0"

# Or recreate environment
conda env remove -n dispersion_han
conda env create -f environment.yml
```

### SciPy/Matplotlib Won't Import

**Solution:**
```bash
# Reinstall with conda (handles binaries better)
conda install scipy matplotlib -c conda-forge --force-reinstall
```

## Development Setup

For development, also install:

```bash
# Testing
pip install pytest pytest-cov

# Code quality
pip install black flake8 mypy

# Documentation
pip install sphinx sphinx-rtd-theme
```

## Updating Dependencies

```bash
# Update within constraints
pip install --upgrade "numpy>=1.24,<2.0" scipy matplotlib

# Check for security updates
pip list --outdated
```

## Environment Management

### List environments
```bash
conda env list
```

### Remove environment
```bash
conda env remove -n dispersion_han
```

### Export environment
```bash
conda env export > environment.yml
```

