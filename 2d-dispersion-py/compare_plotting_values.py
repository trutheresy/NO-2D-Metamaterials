"""Compare values saved from test vs values computed in plotting function."""
import scipy.io as sio
import numpy as np
from pathlib import Path

test_plots_dir = Path('test_plots')

# Load test data
test_data_path = test_plots_dir / 'plot_dispersion_contour_data.mat'
plotting_data_path = test_plots_dir / 'plot_dispersion_contour_values_from_plotting_func.mat'

if not test_data_path.exists():
    print(f"❌ Test data not found: {test_data_path}")
    exit(1)

if not plotting_data_path.exists():
    print(f"❌ Plotting function data not found: {plotting_data_path}")
    print("   Run the plotting test first to generate this file.")
    exit(1)

test_data = sio.loadmat(str(test_data_path))
plotting_data = sio.loadmat(str(plotting_data_path))

print("=" * 80)
print("COMPARING TEST DATA vs PLOTTING FUNCTION DATA")
print("=" * 80)

# Compare X matrices
print("\n" + "=" * 80)
print("X MATRIX COMPARISON")
print("=" * 80)
X_test = test_data['X']
X_plot = plotting_data['X_plot']
print(f"Test X shape: {X_test.shape}")
print(f"Plot X shape: {X_plot.shape}")

if X_test.shape == X_plot.shape:
    x_diff = np.abs(X_test - X_plot)
    print(f"Max difference: {np.max(x_diff):.6e}")
    print(f"Mean difference: {np.mean(x_diff):.6e}")
    if np.max(x_diff) < 1e-10:
        print("✅ X matrices match exactly")
    else:
        print("❌ X matrices differ")
        max_idx = np.unravel_index(np.argmax(x_diff), x_diff.shape)
        print(f"   Max diff at index {max_idx}")
        print(f"   Test value: {X_test[max_idx]:.6f}")
        print(f"   Plot value: {X_plot[max_idx]:.6f}")
        print(f"\n   Test X (first 5x5):")
        print(X_test[:5, :5])
        print(f"   Plot X (first 5x5):")
        print(X_plot[:5, :5])
else:
    print("❌ X shapes don't match!")

# Compare Y matrices
print("\n" + "=" * 80)
print("Y MATRIX COMPARISON")
print("=" * 80)
Y_test = test_data['Y']
Y_plot = plotting_data['Y_plot']
print(f"Test Y shape: {Y_test.shape}")
print(f"Plot Y shape: {Y_plot.shape}")

if Y_test.shape == Y_plot.shape:
    y_diff = np.abs(Y_test - Y_plot)
    print(f"Max difference: {np.max(y_diff):.6e}")
    print(f"Mean difference: {np.mean(y_diff):.6e}")
    if np.max(y_diff) < 1e-10:
        print("✅ Y matrices match exactly")
    else:
        print("❌ Y matrices differ")
        max_idx = np.unravel_index(np.argmax(y_diff), y_diff.shape)
        print(f"   Max diff at index {max_idx}")
        print(f"   Test value: {Y_test[max_idx]:.6f}")
        print(f"   Plot value: {Y_plot[max_idx]:.6f}")
else:
    print("❌ Y shapes don't match!")

# Compare Z matrices
print("\n" + "=" * 80)
print("Z MATRIX COMPARISON")
print("=" * 80)
Z_test = test_data['Z']
Z_plot = plotting_data['Z_plot']
print(f"Test Z shape: {Z_test.shape}")
print(f"Plot Z shape: {Z_plot.shape}")

if Z_test.shape == Z_plot.shape:
    z_diff = np.abs(Z_test - Z_plot)
    print(f"Max difference: {np.max(z_diff):.6e}")
    print(f"Mean difference: {np.mean(z_diff):.6e}")
    print(f"Relative error: {np.max(z_diff) / np.max(np.abs(Z_test)):.6e}")
    if np.max(z_diff) < 1e-6:
        print("✅ Z matrices match (within tolerance)")
    else:
        print("❌ Z matrices differ")
        max_idx = np.unravel_index(np.argmax(z_diff), z_diff.shape)
        print(f"   Max diff at index {max_idx}")
        print(f"   Test value: {Z_test[max_idx]:.6f}")
        print(f"   Plot value: {Z_plot[max_idx]:.6f}")
        print(f"\n   Test Z (first 5x5):")
        print(Z_test[:5, :5])
        print(f"   Plot Z (first 5x5):")
        print(Z_plot[:5, :5])
        print(f"\n   Difference (first 5x5):")
        print(z_diff[:5, :5])
else:
    print("❌ Z shapes don't match!")

# Compare input wavevectors
print("\n" + "=" * 80)
print("INPUT WAVEVECTOR COMPARISON")
print("=" * 80)
wv_test = test_data['wv_grid']
wv_plot = plotting_data['wv_input']
print(f"Test wv shape: {wv_test.shape}")
print(f"Plot wv shape: {wv_plot.shape}")

if wv_test.shape == wv_plot.shape:
    wv_diff = np.abs(wv_test - wv_plot)
    print(f"Max difference: {np.max(wv_diff):.6e}")
    if np.max(wv_diff) < 1e-10:
        print("✅ Input wavevectors match exactly")
    else:
        print("❌ Input wavevectors differ")
        max_idx = np.unravel_index(np.argmax(wv_diff), wv_diff.shape)
        print(f"   Max diff at index {max_idx}")
        print(f"   Test value: {wv_test[max_idx]}")
        print(f"   Plot value: {wv_plot[max_idx]}")
else:
    print("❌ Wavevector shapes don't match!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
all_match = True
if X_test.shape == X_plot.shape and np.max(np.abs(X_test - X_plot)) > 1e-10:
    all_match = False
if Y_test.shape == Y_plot.shape and np.max(np.abs(Y_test - Y_plot)) > 1e-10:
    all_match = False
if Z_test.shape == Z_plot.shape and np.max(np.abs(Z_test - Z_plot)) > 1e-6:
    all_match = False

if all_match:
    print("✅ All values match! The plotting function is using the correct data.")
else:
    print("❌ Values differ. There may be an issue with how data is passed or reshaped.")

