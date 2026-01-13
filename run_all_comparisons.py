#!/usr/bin/env python3
"""
Run all comparison steps:
1. Scan discrepancies
2. Generate dispersion plots
3. Create summary report
"""

import subprocess
import sys
from pathlib import Path

def main():
    original = Path("data/out_test_10_mat_original/out_binarized_1.mat")
    regenerated = Path("data/out_test_10_mat_regenerated/out_binarized_1.mat")
    
    if not regenerated.exists():
        print("ERROR: Regenerated file not found!")
        print(f"Expected: {regenerated}")
        print("Please wait for regenerate_eigenvalue_data.py to complete.")
        return 1
    
    print("=" * 80)
    print("Running All Comparisons")
    print("=" * 80)
    
    # Step 1: Scan discrepancies
    print("\n1. Scanning eigenvalue discrepancies...")
    try:
        subprocess.run([sys.executable, "scan_eigenvalue_discrepancies.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"   Error running scan: {e}")
        return 1
    
    # Step 2: Generate plots
    print("\n2. Generating dispersion plots...")
    try:
        subprocess.run([
            sys.executable, "plot_regenerated_dispersion.py",
            "--original", str(original),
            "--regenerated", str(regenerated),
            "--struct", "0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"   Error generating plots: {e}")
        return 1
    
    print("\n" + "=" * 80)
    print("All comparisons complete!")
    print("=" * 80)
    print("\nOutput locations:")
    print("  - Discrepancy analysis: data/discrepancy_analysis/discrepancy_summary.png")
    print("  - Dispersion plots: data/dispersion_plots_regenerated/")
    print("    * comparison_struct_0.png - Side by side comparison")
    print("    * overlay_struct_0.png - Overlay plot")
    print("    * difference_struct_0.png - Difference plot")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

