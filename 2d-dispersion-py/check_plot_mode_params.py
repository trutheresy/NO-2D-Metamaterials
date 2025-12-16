"""Check geometry and wavevector used in plot_mode_basic test."""
import scipy.io as sio
import numpy as np

# Load the saved data
data = sio.loadmat('test_plots/plot_mode_basic_data.mat')

print("=" * 80)
print("GEOMETRY AND WAVEVECTOR USED FOR plot_mode_basic")
print("=" * 80)

# Extract geometry information
const_struct = data['const'][0, 0]
# Handle MATLAB struct format
def get_field(struct, field_name):
    if field_name in struct.dtype.names:
        field_val = struct[field_name][0, 0]
        # If it's a scalar array, extract the value
        if field_val.size == 1:
            return field_val.item() if hasattr(field_val, 'item') else field_val[0]
        return field_val
    return None

N_pix = get_field(const_struct, 'N_pix')
N_ele = get_field(const_struct, 'N_ele')
a = get_field(const_struct, 'a')
design_scale = get_field(const_struct, 'design_scale')

print("\n📐 GEOMETRY:")
print(f"  Design type: homogeneous")
print(f"  N_pix (pixels per side): {N_pix}")
print(f"  N_ele (elements per pixel): {N_ele}")
print(f"  a (unit cell size): {a}")
print(f"  Design scale: {design_scale}")

# Material properties
E_min = get_field(const_struct, 'E_min')
E_max = get_field(const_struct, 'E_max')
rho_min = get_field(const_struct, 'rho_min')
rho_max = get_field(const_struct, 'rho_max')
poisson_min = get_field(const_struct, 'poisson_min')
poisson_max = get_field(const_struct, 'poisson_max')
    
    print(f"\n📊 MATERIAL PROPERTIES:")
    print(f"  E (modulus): {E_min:.2e} to {E_max:.2e} Pa")
    print(f"  rho (density): {rho_min:.0f} to {rho_max:.0f} kg/m³")
    print(f"  nu (Poisson's ratio): {poisson_min:.2f} to {poisson_max:.2f}")

# Wavevector
wv = data['wv']
print(f"\n🌊 WAVEVECTOR:")
print(f"  k_x: {wv[0, 0]:.6f}")
print(f"  k_y: {wv[0, 1]:.6f}")
print(f"  Location: Gamma point (k = 0, 0)")

# Frequency information
fr = data['fr']
print(f"\n📈 FREQUENCY:")
print(f"  Frequency: {fr[0, 0]:.2f} Hz")
print(f"  Mode index: 0 (first/lowest frequency mode)")

print("\n" + "=" * 80)

