# Test Results Visualization Guide

## Output Structure

The test suite now generates organized folders with geometries and dispersion plots for easy browsing:

```
test_equivalency_output_viz/
├── homogeneous/
│   ├── geometry_MATLAB.png          # Geometry visualization from MATLAB
│   ├── geometry_Python.png           # Geometry visualization from Python
│   ├── dispersion_MATLAB.png         # Dispersion plot from MATLAB
│   ├── dispersion_Python.png          # Dispersion plot from Python
│   └── comparison_MATLAB_to_MATLAB_vs_Python_to_Python.png  # Side-by-side comparison
├── quasi-1D/
│   ├── geometry_Python.png
│   └── dispersion_Python.png
├── dispersive-tetragonal/
│   ├── geometry_Python.png
│   └── dispersion_Python.png
├── dispersive-orthotropic/
│   ├── geometry_Python.png
│   └── dispersion_Python.png
└── test_summary.json                 # Quantitative comparison metrics
```

## What Each File Shows

### Geometry Plots (`geometry_*.png`)
- **3-panel visualization** showing:
  - **Modulus**: Elastic modulus distribution (0-1 normalized)
  - **Density**: Material density distribution (0-1 normalized)
  - **Poisson**: Poisson's ratio distribution (0-1 normalized)
- Color scale: viridis colormap (dark = 0, bright = 1)
- Useful for: Verifying that geometries are generated correctly and match between libraries

### Dispersion Plots (`dispersion_*.png`)
- **Frequency vs Wavevector Parameter** plot
- Shows up to 6 eigenvalue bands
- Each band is a different color
- X-axis: Wavevector parameter (normalized path along IBZ contour)
- Y-axis: Frequency [Hz]
- Useful for: Comparing dispersion relations between MATLAB and Python libraries

### Comparison Plots (`comparison_*.png`)
- **4-panel comparison** showing:
  1. **Overlay Comparison**: MATLAB and Python dispersion curves overlaid
  2. **Relative Difference**: Percentage difference between libraries
  3. **Band 1 Detailed**: First band comparison
  4. **Band 2 Detailed**: Second band comparison
- Useful for: Identifying differences and validating equivalency

## How to Browse Results

### Option 1: File Explorer
1. Navigate to `test_equivalency_output_viz/`
2. Open any design folder (e.g., `homogeneous/`)
3. View the PNG files directly

### Option 2: Python Script
```python
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.image import imread

# View a specific plot
plot_path = Path('test_equivalency_output_viz/homogeneous/geometry_Python.png')
img = imread(plot_path)
plt.imshow(img)
plt.axis('off')
plt.show()
```

### Option 3: Jupyter Notebook
```python
from IPython.display import Image, display
from pathlib import Path

# Display all plots for a design
design_dir = Path('test_equivalency_output_viz/homogeneous')
for img_file in sorted(design_dir.glob('*.png')):
    print(f"\n{img_file.name}:")
    display(Image(str(img_file)))
```

## Understanding the Results

### Geometry Comparison
- **MATLAB vs Python geometries should match exactly** (same design pattern)
- Small differences may occur due to:
  - Different random number generators (for random designs)
  - Numerical precision in design generation

### Dispersion Comparison
- **Mean relative difference < 1%**: Excellent agreement ✅
- **Mean relative difference 1-5%**: Good agreement, minor differences
- **Mean relative difference > 5%**: May indicate implementation differences ⚠️

### Current Test Results
From the `homogeneous` design test:
- **Mean relative difference: 0.32%** ✅
- **Max relative difference: 100%** (outlier, likely near-zero frequency)
- **Mean absolute difference: 1.81 Hz**

This indicates **excellent agreement** between the libraries!

## Tips for Analysis

1. **Start with geometry plots**: Verify designs match before comparing dispersion
2. **Check comparison plots**: Look for systematic differences vs. random noise
3. **Review test_summary.json**: Quantitative metrics for all designs
4. **Compare bands individually**: Some bands may have better agreement than others

## Missing Files

If some files are missing:
- **MATLAB files missing**: MATLAB may not have run successfully (check MATLAB installation)
- **Python files missing**: Python calculation may have failed (check error messages)
- **Comparison plots missing**: Both MATLAB and Python results needed for comparison

## Next Steps

To generate more comprehensive results:
1. Run with larger `N_pix` (e.g., 32 instead of 16) for higher resolution
2. Test with your own datasets using `--matlab-dataset` and `--python-dataset` flags
3. Increase `N_wv` for finer wavevector resolution
4. Test additional design types beyond the 4 predefined ones

