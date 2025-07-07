# Fix for the integers variable error in encoding_figures.ipynb
# The issue is in Cell 6 where the code uses 'integers' instead of 'bands'

# In Cell 6, replace these lines:
# for i, val in enumerate(integers):
#     max_val = np.max(np.abs(wavelet_embeddings[i]))
#     print(f"n={val}: {max_val:.4f}")

# for i, val in enumerate(integers):
#     max_val = np.max(wavelet_fourier[i]) 
#     print(f"n={val}: {max_val:.4f}")

# With these corrected lines:
# for i, val in enumerate(bands):
#     max_val = np.max(np.abs(wavelet_embeddings[i]))
#     print(f"n={val}: {max_val:.4f}")

# for i, val in enumerate(bands):
#     max_val = np.max(wavelet_fourier[i]) 
#     print(f"n={val}: {max_val:.4f}")

# The variable 'bands' is defined at the beginning of the notebook as:
# bands = list(range(1, 7))

# This fix ensures that the code uses the correct variable name that was defined earlier in the notebook. 