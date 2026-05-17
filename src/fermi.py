"""Fermi-surface utilities for the 2D NFE model.

For ``V_0 = 0`` the Fermi surface in the reduced-zone scheme is obtained by
folding the free-electron Fermi circle from the extended-zone scheme into
BZ1. Each BZ2 piece is translated by the appropriate reciprocal vector G
that maps its BZ2 triangle back to BZ1. See :func:`fold_fermi_circle_to_bz1`.

For ``V_0 > 0`` the Fermi surface is computed from the eigenvalues on a
dense k-grid by contouring at ``E = E_F`` (introduced in notebook 06).
"""

import numpy as np

from .hamiltonian import VFunc, build_H_batch
from .lattice import Lattice


def compute_bands_on_grid(
    N_kx: int,
    N_ky: int,
    V_func: VFunc,
    N_cut: int = 3,
    lattice: Lattice = Lattice(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize ``H(k)`` on a uniform grid covering BZ1.

    Grid uses ``linspace(-b/2, b/2, N, endpoint=False)`` so the periodic
    image at ``+b/2`` is not double-counted (important for filling counts).
    ``meshgrid(..., indexing="ij")`` so axis 0 is k_x and axis 1 is k_y.

    Returns
    -------
    KX, KY : ndarray of shape ``(N_kx, N_ky)``
    E_grid : ndarray of shape ``(N_kx, N_ky, N_pw)``
        Eigenvalues sorted ascending along axis -1.
    """
    bx2 = lattice.b_x / 2.0
    by2 = lattice.b_y / 2.0
    kx = np.linspace(-bx2, bx2, N_kx, endpoint=False)
    ky = np.linspace(-by2, by2, N_ky, endpoint=False)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k_pts = np.stack([KX.ravel(), KY.ravel()], axis=1)

    H_batch = build_H_batch(k_pts, V_func, N_cut, lattice)
    E_flat = np.linalg.eigvalsh(H_batch)  # (N_k, N_pw)
    E_grid = E_flat.reshape(N_kx, N_ky, -1)
    return KX, KY, E_grid


def divalent_k_F(lattice: Lattice = Lattice()) -> float:
    """Free-electron Fermi wavevector for divalent filling (``n = 2`` per cell).

    From the 2D density-of-states / particle-count relation with spin
    degeneracy ``g_s = 2``:

        ``k_F = sqrt(4 * pi / cell_area)``.

    For the unit square this returns ``2 * sqrt(pi) ~= 3.545``.
    """
    return float(np.sqrt(4.0 * np.pi / lattice.cell_area))


def find_E_F(
    energies: np.ndarray, n_per_cell: int = 2, g_s: int = 2
) -> float:
    """Find the Fermi energy by filling the lowest band-states.

    With ``n_per_cell`` electrons per primitive cell and spin degeneracy
    ``g_s``, the number of band-states (each holding ``g_s`` electrons)
    to fill across the entire k-grid is ``N_k * n_per_cell // g_s``.
    For divalent (``n_per_cell = 2``, ``g_s = 2``) this is exactly ``N_k``.
    """
    energies = np.asarray(energies)
    N_k = int(np.prod(energies.shape[:-1]))
    N_filled = (n_per_cell * N_k) // g_s
    sorted_E = np.sort(energies.ravel())
    return float(sorted_E[N_filled - 1])


def chemical_potential(
    energies: np.ndarray, n_per_cell: int = 2, g_s: int = 2
) -> float:
    """T = 0 chemical potential: midpoint of highest filled and lowest unfilled state.

    Equals :func:`find_E_F` in a metal (the two flanking states sit on the
    Fermi surface and are degenerate to within discretization noise). In an
    insulator they straddle the gap, so this returns the gap midpoint — the
    physically meaningful energy for evaluating DOS at the chemical potential.
    """
    energies = np.asarray(energies)
    N_k = int(np.prod(energies.shape[:-1]))
    N_filled = (n_per_cell * N_k) // g_s
    sorted_E = np.sort(energies.ravel())
    return float(0.5 * (sorted_E[N_filled - 1] + sorted_E[N_filled]))


def band_overlap(E_grid: np.ndarray, bands: tuple[int, int] = (0, 1)) -> float:
    """Signed band overlap ``min(E_upper) - max(E_lower)``.

    Negative -> bands overlap (metallic).
    Positive -> gap opens (insulator).
    """
    n_lower, n_upper = bands
    return float(np.min(E_grid[..., n_upper]) - np.max(E_grid[..., n_lower]))


def dos_at(E: float, energies: np.ndarray, sigma: float = 0.05) -> float:
    """Gaussian-broadened density of states at energy ``E``, per unit cell."""
    energies = np.asarray(energies)
    N_k = int(np.prod(energies.shape[:-1]))
    delta = E - energies.ravel()
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma * N_k)
    return float(norm * np.sum(np.exp(-0.5 * (delta / sigma) ** 2)))


def wrap_for_plot(
    KX: np.ndarray,
    KY: np.ndarray,
    E_grid: np.ndarray,
    lattice: Lattice = Lattice(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extend an ``endpoint=False`` BZ1 grid to the closed rectangle.

    Appends a periodic image row/column so contour plots close cleanly at
    the ``+b_x/2`` / ``+b_y/2`` walls (where pockets centered at X / Y
    points live). Use only for plotting — for E_F counting this would
    double-count one row and one column.
    """
    bx2 = lattice.b_x / 2.0
    by2 = lattice.b_y / 2.0
    kx_plot = np.append(KX[:, 0], bx2)
    ky_plot = np.append(KY[0, :], by2)
    KX_plot, KY_plot = np.meshgrid(kx_plot, ky_plot, indexing="ij")
    E_plot = np.pad(E_grid, ((0, 1), (0, 1), (0, 0)), mode="wrap")
    return KX_plot, KY_plot, E_plot


def fold_fermi_circle_to_bz1(
    k_F: float, n_samples: int = 2000
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Fold the extended-zone Fermi circle back into BZ1.

    Parametrize the circle by ``theta in [0, 2*pi)``. Each sample is
    classified as lying in BZ1 or one of the four BZ2 triangles; BZ2
    samples are translated by the appropriate G so the folded coordinates
    sit inside BZ1. The result is split into connected arcs at every
    BZ-id transition, with wrap-around (theta = 0 / 2*pi) stitched.

    Parameters
    ----------
    k_F : float
        Fermi wavevector. Must satisfy ``k_F < pi*sqrt(2)`` so the circle
        stays within BZ1 + BZ2 (does not reach BZ3).
    n_samples : int
        Number of samples around the circle. Larger means smoother arcs.

    Returns
    -------
    band1_arcs : list of ndarray of shape ``(M_i, 2)``
        Connected arcs of the band-1 Fermi surface in BZ1 (parts of the
        circle originally inside BZ1, near the M-diagonal directions).
    band2_arcs : list of ndarray of shape ``(M_i, 2)``
        Connected arcs of the band-2 Fermi surface in BZ1 (parts of the
        circle originally in BZ2, folded back). For the free-electron
        divalent square these form pockets centered on the X points.
    """
    pi = np.pi
    if k_F >= pi * np.sqrt(2):
        raise ValueError(
            f"k_F = {k_F:.4f} exceeds pi*sqrt(2) = {pi * np.sqrt(2):.4f}; "
            "the Fermi circle reaches BZ3 and this routine only handles BZ1+BZ2."
        )

    theta = np.linspace(0.0, 2.0 * pi, n_samples, endpoint=False)
    x_ext = k_F * np.cos(theta)
    y_ext = k_F * np.sin(theta)

    # Region classification. The diagonal BZ2 boundaries |y| <= 2*pi - x etc.
    # are not crossed for k_F < 2*pi, so we only need the axial walls.
    in_bz1 = (np.abs(x_ext) <= pi) & (np.abs(y_ext) <= pi)
    code = np.full(n_samples, -1, dtype=np.int8)
    code[in_bz1] = 0
    code[~in_bz1 & (x_ext > pi)] = 1   # right BZ2  -> fold by (-2*pi, 0)
    code[~in_bz1 & (y_ext > pi)] = 2   # top   BZ2  -> fold by (0, -2*pi)
    code[~in_bz1 & (x_ext < -pi)] = 3  # left  BZ2  -> fold by (+2*pi, 0)
    code[~in_bz1 & (y_ext < -pi)] = 4  # bottom BZ2 -> fold by (0, +2*pi)
    assert np.all(code >= 0), "unclassified samples — k_F may exceed BZ1+BZ2"

    x_fold = x_ext.copy()
    y_fold = y_ext.copy()
    fold_shifts: dict[int, tuple[float, float]] = {
        1: (-2.0 * pi, 0.0),
        2: (0.0, -2.0 * pi),
        3: (+2.0 * pi, 0.0),
        4: (0.0, +2.0 * pi),
    }
    for c, (dx, dy) in fold_shifts.items():
        mask = code == c
        x_fold[mask] += dx
        y_fold[mask] += dy

    # Split into runs of consecutive samples with the same BZ-id.
    transitions = np.where(np.diff(code) != 0)[0] + 1
    seg_indices = np.split(np.arange(n_samples), transitions)

    # Wrap-around stitching: if the first and last segments share a code,
    # they are two halves of one physical arc split by theta = 0.
    if len(seg_indices) > 1 and code[seg_indices[0][0]] == code[seg_indices[-1][0]]:
        seg_indices[0] = np.concatenate([seg_indices[-1], seg_indices[0]])
        seg_indices.pop()

    band1_arcs: list[np.ndarray] = []
    band2_arcs: list[np.ndarray] = []
    for idx in seg_indices:
        if idx.size == 0:
            continue
        arc = np.column_stack([x_fold[idx], y_fold[idx]])
        c = code[idx[0]]
        if c == 0:
            band1_arcs.append(arc)
        else:
            band2_arcs.append(arc)
    return band1_arcs, band2_arcs
