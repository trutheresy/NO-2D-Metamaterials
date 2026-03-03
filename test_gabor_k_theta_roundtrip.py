"""
Test Gabor wavelet k/theta encoding and decoding roundtrip accuracy.

Directly generates Gabor wavelets for a grid of (k, theta) values,
then extracts k and theta using the centroid method (r=3, p=2, Voronoi exclusion).
Reports per-sample and aggregate statistics.

k range:     [2, 16]  (spatial frequency in cycles per image)
theta range: [0, 180) degrees  (orientation, pi-periodic)
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np


# ============================================================
# Gabor wavelet generation (from eigenfrequency_encoding_tests.ipynb Cell 1)
# ============================================================
def generate_gabor(k, theta_rad, size=32, sigma_factor=8, gamma=1.0, phi=0.0):
    """Generate a 2D Gabor wavelet with specified k and theta."""
    coords = np.linspace(-size // 2, size // 2 - 1, size)
    X, Y = np.meshgrid(coords, coords)

    X_theta =  X * np.cos(theta_rad) + Y * np.sin(theta_rad)
    Y_theta = -X * np.sin(theta_rad) + Y * np.cos(theta_rad)

    sigma_x = sigma_factor
    sigma_y = sigma_x * gamma
    freq = 2.0 * np.pi * k / size  # radians per pixel

    gaussian = np.exp(-0.5 * ((X_theta**2) / sigma_x**2 + (Y_theta**2) / sigma_y**2))
    carrier = np.cos(freq * X_theta + phi)

    return gaussian * carrier


# ============================================================
# k/theta extraction via centroid (from eigenfrequency_encoding_tests.ipynb Cell 2)
# ============================================================
def extract_k_theta_centroid(image, size=32, centroid_radius=3, centroid_power=2):
    """
    Extract spatial frequency k and orientation theta from a Gabor wavelet image
    using 2D FFT + weighted centroid with Voronoi exclusion.

    Returns: k_extracted, theta_extracted (radians in [0, pi))
    """
    # Remove DC and compute centered FFT magnitude
    image_centered = image - np.mean(image)
    F = np.fft.fft2(image_centered)
    F_mag = np.abs(np.fft.fftshift(F))

    center = size // 2

    # Step 1: Find brightest pixel in the half-plane (resolves pi-ambiguity)
    hp = np.zeros_like(F_mag)
    hp[:center, :] = F_mag[:center, :]                    # rows with ky < 0
    hp[center, center + 1:] = F_mag[center, center + 1:]  # ky=0, kx>0

    peak_idx = np.unravel_index(np.argmax(hp), hp.shape)
    peak_kx = peak_idx[1] - center
    peak_ky = peak_idx[0] - center
    peak_row = peak_idx[0]
    peak_col = peak_idx[1]

    # Symmetric peak location (conjugate)
    sym_row = 2 * center - peak_row
    sym_col = 2 * center - peak_col

    # Step 2: Weighted centroid within circle, Voronoi exclusion
    sum_w = 0.0
    sum_kx = 0.0
    sum_ky = 0.0
    r_int = int(np.ceil(centroid_radius))

    for dr in range(-r_int, r_int + 1):
        for dc in range(-r_int, r_int + 1):
            if dr * dr + dc * dc > centroid_radius * centroid_radius:
                continue
            r = peak_row + dr
            c = peak_col + dc
            if 0 <= r < size and 0 <= c < size:
                d_main_sq = dr * dr + dc * dc
                d_sym_sq = (r - sym_row)**2 + (c - sym_col)**2
                if d_sym_sq < d_main_sq:
                    continue
                w = F_mag[r, c] ** centroid_power
                sum_w += w
                sum_kx += w * (c - center)
                sum_ky += w * (r - center)

    if sum_w > 0:
        kx_refined = sum_kx / sum_w
        ky_refined = sum_ky / sum_w
    else:
        kx_refined = float(peak_kx)
        ky_refined = float(peak_ky)

    # Step 3: Convert to polar
    k_extracted = np.sqrt(kx_refined**2 + ky_refined**2)
    theta_extracted = np.arctan2(ky_refined, kx_refined) % np.pi  # [0, pi)

    return k_extracted, theta_extracted


# ============================================================
# Signed theta error respecting pi-periodicity
# ============================================================
def signed_theta_error_deg(theta_true_rad, theta_ext_rad):
    """Signed angular error in degrees, wrapped to [-90, 90)."""
    diff = theta_ext_rad - theta_true_rad
    # Wrap to [-pi/2, pi/2)
    diff = (diff + np.pi / 2) % np.pi - np.pi / 2
    return np.degrees(diff)


# ============================================================
# Main test
# ============================================================
def main():
    size = 32

    # Test grid
    k_vals = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
    theta_degs = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

    n_total = len(k_vals) * len(theta_degs)

    # Table header
    hdr = (f"{'k_enc':>6s}  {'th_enc':>7s}  "
           f"{'max_sp':>7s}  {'max_fft':>9s}  "
           f"{'k_dec':>7s}  {'k_err%':>8s}  "
           f"{'th_dec':>7s}  {'th_err':>8s}")
    sep = "-" * len(hdr)

    print("Gabor wavelet k/theta roundtrip test")
    print(f"  size={size}, sigma_factor=8, gamma=1.0, phi=0.0")
    print(f"  centroid: radius=3, power=2, Voronoi exclusion")
    print(f"  k in {k_vals}")
    print(f"  theta in {theta_degs} deg")
    print(f"  {n_total} test cases\n")
    print(hdr)
    print(sep)

    all_k_err = []
    all_th_err = []
    all_k_err_signed = []
    all_th_err_signed = []

    for k_true in k_vals:
        for th_deg in theta_degs:
            th_rad = np.radians(th_deg)

            # Encode
            gabor = generate_gabor(k_true, th_rad, size=size)
            max_spatial = np.max(np.abs(gabor))

            # FFT magnitude
            F = np.fft.fft2(gabor - np.mean(gabor))
            F_mag = np.abs(np.fft.fftshift(F))
            max_fft = np.max(F_mag)

            # Decode
            k_dec, th_dec_rad = extract_k_theta_centroid(gabor, size=size)
            th_dec_deg = np.degrees(th_dec_rad)

            # Signed relative k error (%)
            k_err_signed = (k_dec - k_true) / k_true * 100.0

            # Signed theta error (degrees)
            th_err_signed = signed_theta_error_deg(th_rad, th_dec_rad)

            all_k_err.append(abs(k_err_signed))
            all_th_err.append(abs(th_err_signed))
            all_k_err_signed.append(k_err_signed)
            all_th_err_signed.append(th_err_signed)

            print(f"{k_true:6.1f}  {th_deg:7.1f}  "
                  f"{max_spatial:7.4f}  {max_fft:9.1f}  "
                  f"{k_dec:7.3f}  {k_err_signed:+7.3f}%  "
                  f"{th_dec_deg:7.2f}  {th_err_signed:+7.3f} deg")

        # Blank line between k groups
        print()

    all_k_err = np.array(all_k_err)
    all_th_err = np.array(all_th_err)
    all_k_err_signed = np.array(all_k_err_signed)
    all_th_err_signed = np.array(all_th_err_signed)

    # ---- Aggregate statistics ----
    print(sep)
    print("AGGREGATE STATISTICS")
    print(sep)
    print(f"  k error (absolute):")
    print(f"    mean:   {np.mean(all_k_err):.4f} %")
    print(f"    median: {np.median(all_k_err):.4f} %")
    print(f"    max:    {np.max(all_k_err):.4f} %")
    print(f"    std:    {np.std(all_k_err):.4f} %")
    print(f"  k error (signed):")
    print(f"    mean:   {np.mean(all_k_err_signed):+.4f} %")
    print(f"    std:    {np.std(all_k_err_signed):.4f} %")
    print()
    print(f"  theta error (absolute):")
    print(f"    mean:   {np.mean(all_th_err):.4f} deg")
    print(f"    median: {np.median(all_th_err):.4f} deg")
    print(f"    max:    {np.max(all_th_err):.4f} deg")
    print(f"    std:    {np.std(all_th_err):.4f} deg")
    print(f"  theta error (signed):")
    print(f"    mean:   {np.mean(all_th_err_signed):+.4f} deg")
    print(f"    std:    {np.std(all_th_err_signed):.4f} deg")

    # ---- Per-k summary ----
    print(f"\n{'k':>4s}  {'mean|k_err|':>11s}  {'max|k_err|':>11s}  "
          f"{'mean|th_err|':>12s}  {'max|th_err|':>12s}")
    print("-" * 60)
    idx = 0
    for k_true in k_vals:
        n_th = len(theta_degs)
        ke = all_k_err[idx:idx+n_th]
        te = all_th_err[idx:idx+n_th]
        print(f"{k_true:4d}  {np.mean(ke):10.4f}%  {np.max(ke):10.4f}%  "
              f"{np.mean(te):11.4f} deg  {np.max(te):10.4f} deg")
        idx += n_th


if __name__ == "__main__":
    main()
