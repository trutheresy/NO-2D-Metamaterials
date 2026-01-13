"""Check which datasets have real eigenvalue data (not all NaN)."""
import sys
from pathlib import Path
sys.path.insert(0, '2d-dispersion-py')
from mat73_loader import load_matlab_v73
import numpy as np

# Check multiple potential datasets
datasets_to_check = [
    '2D-dispersion-han/OUTPUT/out_test_10/out_binarized_1.mat',
    '2D-dispersion-han/OUTPUT/out_test_10/out_continuous_1.mat',
    'data/out_test_10_matlab/out_binarized_1.mat',
    'data/out_test_10/out_binarized_1/out_binarized_1_predictions.mat',
]

print("Checking datasets for real eigenvalue data:\n")
print("="*70)

for mat_path_str in datasets_to_check:
    mat_file = Path(mat_path_str)
    if not mat_file.exists():
        print(f"\n{mat_file}")
        print(f"  ✗ File does not exist")
        continue
    
    try:
        data = load_matlab_v73(str(mat_file), verbose=False)
        eigval = data.get('EIGENVALUE_DATA')
        
        if eigval is None:
            print(f"\n{mat_file}")
            print(f"  ✗ No EIGENVALUE_DATA field")
            continue
        
        nan_count = np.isnan(eigval).sum()
        total_count = eigval.size
        nan_pct = 100 * nan_count / total_count if total_count > 0 else 0
        
        print(f"\n{mat_file}")
        print(f"  EIGENVALUE_DATA shape: {eigval.shape}")
        print(f"  NaN count: {nan_count}/{total_count} ({nan_pct:.1f}%)")
        
        if nan_count == total_count:
            print(f"  ✗ All NaN - needs reconstruction")
        elif nan_count > total_count * 0.9:
            print(f"  ⚠ Mostly NaN ({nan_pct:.1f}%) - likely needs reconstruction")
        else:
            print(f"  ✓ Has real eigenvalue data!")
            print(f"  Min value: {np.nanmin(eigval):.6e}")
            print(f"  Max value: {np.nanmax(eigval):.6e}")
            print(f"  First few values (struct 0, eig 0): {eigval[0, 0, :5]}")
            
    except Exception as e:
        print(f"\n{mat_file}")
        print(f"  ✗ Error loading: {e}")

print("\n" + "="*70)

