#!/usr/bin/env python3
import h5py
import numpy as np

f1 = h5py.File(r'D:\Research\NO-2D-Metamaterials\data\out_test_10_sanity_check_reconstruction\out_binarized_1.mat','r')
f2 = h5py.File(r'D:\Research\NO-2D-Metamaterials\data\out_test_10_mat_reconstructed\out_binarized_1.mat','r')

print('=== Keys in Earlier but NOT in Sanity Check ===')
for key in f2.keys():
    if key not in f1:
        print(f'  {key}')

print('\n=== Keys in Sanity Check but NOT in Earlier ===')
for key in f1.keys():
    if key not in f2:
        print(f'  {key}')

print('\n=== Large Datasets in Earlier File ===')
for key in ['K_DATA', 'M_DATA', 'T_DATA', 'CONSTITUTIVE_DATA']:
    if key in f2:
        dset = f2[key]
        if hasattr(dset, 'shape'):
            size_mb = np.prod(dset.shape) * dset.dtype.itemsize / 1024 / 1024
            print(f'{key}: {size_mb:.2f} MB, shape={dset.shape}, dtype={dset.dtype}')
        else:
            print(f'{key}: present but size unknown')

f1.close()
f2.close()

