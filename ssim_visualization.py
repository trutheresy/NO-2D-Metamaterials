import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter


def original_ssim(image1, image2, L=255, K1=0.01, K2=0.03, kernel=11, sigma=1.5):
    # Constants
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2
    # Convert to float
    image1 = image1.astype(float)
    image2 = image2.astype(float)
    # Compute means using Gaussian kernel
    mu1 = gaussian_filter(image1, sigma=sigma, radius=int((kernel-1)/2))
    mu2 = gaussian_filter(image2, sigma=sigma, radius=int((kernel-1)/2))
    # Compute standard deviations
    sigma1 = np.sqrt(gaussian_filter(image1**2, sigma=sigma, radius=int((kernel-1)/2)) - mu1**2)
    sigma2 = np.sqrt(gaussian_filter(image2**2, sigma=sigma, radius=int((kernel-1)/2)) - mu2**2)
    # Compute covariance
    sigma12 = gaussian_filter(image1 * image2, sigma=sigma, radius=int((kernel-1)/2)) - mu1 * mu2
    # Luminance term
    luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    # Contrast term
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)
    # Original structural term
    structural = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    structural[np.isnan(structural)] = 0
    # Combine terms
    ssim = luminance * contrast * structural
    return np.mean(ssim)

def signed_ssim(image1, image2, L=255, K1=0.01, K2=0.03, kernel=11, sigma=1.5):
    # Constants
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    # Convert to float
    image1 = image1.astype(float)
    image2 = image2.astype(float)
    # Compute means using Gaussian kernel
    mu1 = gaussian_filter(image1, sigma=sigma, radius=int((kernel-1)/2))
    mu2 = gaussian_filter(image2, sigma=sigma, radius=int((kernel-1)/2))
    # Compute standard deviations
    sigma1 = np.sqrt(gaussian_filter(image1**2, sigma=sigma, radius=int((kernel-1)/2)) - mu1**2)
    sigma2 = np.sqrt(gaussian_filter(image2**2, sigma=sigma, radius=int((kernel-1)/2)) - mu2**2)
    # Compute covariance
    sigma12 = gaussian_filter(image1 * image2, sigma=sigma, radius=int((kernel-1)/2)) - mu1 * mu2
    # Luminance term
    luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    # Contrast term
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)
    # THIS IS THE IMPORTANT PART: Modified structural term
    correlation = sigma12 / (sigma1 * sigma2)
    correlation[np.isnan(correlation)] = 0
    structural = (correlation + 1) / 2
    # Combine terms
    ssim = luminance * contrast * structural
    return np.mean(ssim)

def create_sine_wave(x, amplitude=1, phase=0):
    return amplitude * np.sin(x + phase)

def plot_comparison(reference, test, title, original_ssim, signed_ssim):
    plt.plot(x, reference, 'b-', label='Reference', linewidth=2)
    plt.plot(x, test, 'r--', label='Test', linewidth=2)
    plt.title(f'{title}\nOriginal SSIM = {original_ssim:.3f}\nSigned SSIM = {signed_ssim:.3f}')
    plt.legend()
    plt.grid(True)

# Create a range of x values
x = np.linspace(0, 4*np.pi, 100)

# Create different test cases
reference = create_sine_wave(x)
test_identical = reference.copy()
test_negated = -reference
test_shifted = create_sine_wave(x, phase=np.pi/2)
test_amplitude = create_sine_wave(x, amplitude=0.5)
test_combined = create_sine_wave(x, amplitude=0.5, phase=np.pi/4)

# Calculate SSIM values
original_ssim_identical = original_ssim(reference, test_identical)
signed_ssim_identical = signed_ssim(reference, test_identical)

original_ssim_negated = original_ssim(reference, test_negated)
signed_ssim_negated = signed_ssim(reference, test_negated)

original_ssim_shifted = original_ssim(reference, test_shifted)
signed_ssim_shifted = signed_ssim(reference, test_shifted)

original_ssim_amplitude = original_ssim(reference, test_amplitude)
signed_ssim_amplitude = signed_ssim(reference, test_amplitude)

original_ssim_combined = original_ssim(reference, test_combined)
signed_ssim_combined = signed_ssim(reference, test_combined)

# Create the visualization
plt.figure(figsize=(15, 15))
gs = GridSpec(3, 2)

# Plot 1: Identical signals
plt.subplot(gs[0, 0])
plot_comparison(reference, test_identical, 'Identical Signals', 
                original_ssim_identical, signed_ssim_identical)

# Plot 2: Negated signals
plt.subplot(gs[0, 1])
plot_comparison(reference, test_negated, 'Negated Signals', 
                original_ssim_negated, signed_ssim_negated)

# Plot 3: Phase-shifted signals
plt.subplot(gs[1, 0])
plot_comparison(reference, test_shifted, 'Phase-shifted Signals', 
                original_ssim_shifted, signed_ssim_shifted)

# Plot 4: Different amplitude signals
plt.subplot(gs[1, 1])
plot_comparison(reference, test_amplitude, 'Different Amplitude Signals', 
                original_ssim_amplitude, signed_ssim_amplitude)

# Plot 5: Combined changes (amplitude and phase)
plt.subplot(gs[2, 0])
plot_comparison(reference, test_combined, 'Combined Changes (Amplitude + Phase)', 
                original_ssim_combined, signed_ssim_combined)

plt.tight_layout()
plt.savefig('ssim_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create 2D examples
def create_2d_sine(x, y, amplitude=1, phase=0):
    return amplitude * np.sin(x + phase) * np.sin(y + phase)

# Create 2D grid
x_2d = np.linspace(0, 4*np.pi, 50)
y_2d = np.linspace(0, 4*np.pi, 50)
X, Y = np.meshgrid(x_2d, y_2d)

# Create different 2D test cases
reference_2d = create_2d_sine(X, Y)
test_2d_identical = reference_2d.copy()
test_2d_negated = -reference_2d
test_2d_shifted = create_2d_sine(X, Y, phase=np.pi/2)
test_2d_combined = create_2d_sine(X, Y, amplitude=0.5, phase=np.pi/4)

# Calculate SSIM values for 2D cases
original_ssim_2d_identical = original_ssim(reference_2d, test_2d_identical)
signed_ssim_2d_identical = signed_ssim(reference_2d, test_2d_identical)

original_ssim_2d_negated = original_ssim(reference_2d, test_2d_negated)
signed_ssim_2d_negated = signed_ssim(reference_2d, test_2d_negated)

original_ssim_2d_shifted = original_ssim(reference_2d, test_2d_shifted)
signed_ssim_2d_shifted = signed_ssim(reference_2d, test_2d_shifted)

original_ssim_2d_combined = original_ssim(reference_2d, test_2d_combined)
signed_ssim_2d_combined = signed_ssim(reference_2d, test_2d_combined)

# Create 2D visualization
plt.figure(figsize=(6, 9))
gs = GridSpec(3, 2)

# Plot 2D examples
plt.subplot(gs[0, 0])
plt.imshow(reference_2d, cmap='grey', vmin=-1, vmax=1)
plt.title('Reference Pattern')
plt.text(-0.09, 1.1, 'a', transform=plt.gca().transAxes, fontsize=14, fontweight='bold')

plt.subplot(gs[0, 1])
plt.imshow(test_2d_identical, cmap='grey', vmin=-1, vmax=1)
plt.title(f'Identical Pattern\nOriginal SSIM = {original_ssim_2d_identical:.3f}\nSigned SSIM = {signed_ssim_2d_identical:.3f}')
plt.text(-0.09, 1.1, 'b', transform=plt.gca().transAxes, fontsize=14, fontweight='bold')

plt.subplot(gs[1, 0])
plt.imshow(test_2d_negated, cmap='grey', vmin=-1, vmax=1)
plt.title(f'Negated Pattern\nOriginal SSIM = {original_ssim_2d_negated:.3f}\nSigned SSIM = {signed_ssim_2d_negated:.3f}')
plt.text(-0.09, 1.1, 'c', transform=plt.gca().transAxes, fontsize=14, fontweight='bold')

plt.subplot(gs[1, 1])
plt.imshow(test_2d_shifted, cmap='grey', vmin=-1, vmax=1)
plt.title(f'Phase-shifted Pattern\nOriginal SSIM = {original_ssim_2d_shifted:.3f}\nSigned SSIM = {signed_ssim_2d_shifted:.3f}')
plt.text(-0.09, 1.1, 'd', transform=plt.gca().transAxes, fontsize=14, fontweight='bold')

plt.subplot(gs[2, 0])
plt.imshow(test_2d_combined, cmap='grey', vmin=-1, vmax=1)
plt.title(f'Combined Changes Pattern\nOriginal SSIM = {original_ssim_2d_combined:.3f}\nSigned SSIM = {signed_ssim_2d_combined:.3f}')
plt.text(-0.09, 1.1, 'e', transform=plt.gca().transAxes, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('ssim_2d_comparison.png', dpi=300, bbox_inches='tight')
plt.close()