import h5py
import numpy as np

# Check original file
print("ORIGINAL FILE:")
f1 = h5py.File(r'D:\Research\NO-2D-Metamaterials\2D-dispersion-han\OUTPUT\out_test_10\out_binarized_1.mat', 'r')
print(f"Root attributes: {dict(f1.attrs)}")
print(f"Keys: {list(f1.keys())[:10]}")
for key in ['WAVEVECTOR_DATA', 'designs', 'design_params', 'N_struct']:
    if key in f1:
        print(f"{key} attributes: {dict(f1[key].attrs)}")
        print(f"{key} dtype: {f1[key].dtype}")
f1.close()

print("\n" + "="*80)
print("PREDICTIONS FILE:")
f2 = h5py.File(r'D:\Research\NO-2D-Metamaterials\data\out_test_10\out_binarized_1\out_binarized_1_predictions.mat', 'r')
print(f"Root attributes: {dict(f2.attrs)}")
print(f"Keys: {list(f2.keys())}")
for key in ['WAVEVECTOR_DATA', 'designs', 'design_params', 'N_struct']:
    if key in f2:
        print(f"{key} attributes: {dict(f2[key].attrs)}")
        print(f"{key} dtype: {f2[key].dtype}")
f2.close()

