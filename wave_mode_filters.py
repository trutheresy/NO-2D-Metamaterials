"""Wavevector index filters for inference analysis (IBZ grid layout)."""

from __future__ import annotations

# Conventional half-plane IBZ meshgrid: 25 kx x 13 ky, row-major wave index.
DEFAULT_N_KX = 25
DEFAULT_N_KY = 13
DEFAULT_KX0_INDEX = 12  # kx = 0 column in the meshgrid
# |kx|=pi columns on the same mesh (left/right edges of the IBZ strip).
DEFAULT_KX_MPI_INDEX = 0
DEFAULT_KX_PPI_INDEX = 24
# ky=pi is the top row (i_ky = n_ky - 1).
DEFAULT_KY_PI_INDEX = DEFAULT_N_KY - 1


def wave_to_grid(wave: int, n_kx: int = DEFAULT_N_KX) -> tuple[int, int]:
    """Map flat wave index to (i_kx, i_ky)."""
    return wave % n_kx, wave // n_kx


def shear_mode_wave_indices(
    n_kx: int = DEFAULT_N_KX,
    n_ky: int = DEFAULT_N_KY,
    kx0_index: int = DEFAULT_KX0_INDEX,
) -> list[int]:
    """
    Wave indices on symmetry lines with dead phase-pivot modes.

    Returns the union of the ky=0 row (Gamma–X, 25 points) and the kx=0
    column (13 points), deduplicated (Gamma at wave 12 appears once).
    """
    ky0_row = list(range(n_kx))
    kx0_col = [kx0_index + j * n_kx for j in range(n_ky)]
    return sorted(set(ky0_row) | set(kx0_col))


def trim_wave_indices(
    n_kx: int = DEFAULT_N_KX,
    n_ky: int = DEFAULT_N_KY,
    kx0_index: int = DEFAULT_KX0_INDEX,
    kx_mpi_index: int = DEFAULT_KX_MPI_INDEX,
    kx_ppi_index: int = DEFAULT_KX_PPI_INDEX,
    ky_pi_index: int = DEFAULT_KY_PI_INDEX,
) -> list[int]:
    """
    Time-reversal invariant momenta (k ≡ -k) on the half-plane IBZ.

    On the square lattice these are combinations of kx, ky in {0, ±π}:
      Gamma (0,0), X (±π,0), (0,π), M (±π,π).
    """
    ky0 = 0
    points = [
        (kx0_index, ky0),  # Gamma
        (kx_mpi_index, ky0),  # X-
        (kx_ppi_index, ky0),  # X+
        (kx0_index, ky_pi_index),  # (0, π)
        (kx_mpi_index, ky_pi_index),  # M-
        (kx_ppi_index, ky_pi_index),  # M+
    ]
    return sorted({i_kx + i_ky * n_kx for i_kx, i_ky in points})


def degenerate_pivot_wave_indices(
    n_kx: int = DEFAULT_N_KX,
    n_ky: int = DEFAULT_N_KY,
    kx0_index: int = DEFAULT_KX0_INDEX,
) -> list[int]:
    """
    Wavevectors to omit for degenerate / ill-defined phase-pivot analysis.

    Union of:
      - ky=0 row and kx=0 column (dead-pivot shear / longitudinal lines)
      - TRIM points with k ≡ -k (adds M corners (±π, π) not already on those axes)
    """
    return sorted(
        set(shear_mode_wave_indices(n_kx=n_kx, n_ky=n_ky, kx0_index=kx0_index))
        | set(trim_wave_indices(n_kx=n_kx, n_ky=n_ky, kx0_index=kx0_index))
    )


def format_wave_index_list(waves: list[int]) -> str:
    return ",".join(str(w) for w in waves)
