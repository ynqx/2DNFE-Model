"""Plane-wave Hamiltonian for the 2D NFE model.

The Hamiltonian in the plane-wave basis ``{|k + G>}`` is

    H[i, j] = |k + G_i|^2 * delta_{ij}  +  V(G_i - G_j)

where ``V(G)`` is the Fourier coefficient of the periodic potential.
The caller supplies ``V`` as a closure ``V_func(G) -> complex`` so
:func:`build_H` is agnostic to the potential shape (cosine, anisotropic,
higher-harmonic, ...). The signature accepts integer ``(m, n)`` tuples,
matching the storage convention in :mod:`lattice`.
"""

from typing import Callable

import numpy as np

from .lattice import Lattice, generate_G_vectors

VFunc = Callable[[tuple[int, int]], complex]


def build_V_block(V_func: VFunc, N_cut: int = 3) -> np.ndarray:
    """Construct the k-independent potential block ``V[G - G']``.

    This is the off-diagonal part of ``H`` and does not depend on ``k``.
    Factored out so :func:`build_H_batch` can build it once and reuse it
    across all k-points.

    Returns
    -------
    V_block : ndarray of shape ``(N_pw, N_pw)``, dtype ``complex128``
        Diagonal entries are zero; off-diagonal entries are ``V_func((dm, dn))``.
    """
    G = generate_G_vectors(N_cut)
    N_pw = G.shape[0]
    V_block = np.zeros((N_pw, N_pw), dtype=np.complex128)
    for i in range(N_pw):
        for j in range(N_pw):
            if i == j:
                continue
            dG = (int(G[i, 0] - G[j, 0]), int(G[i, 1] - G[j, 1]))
            V_block[i, j] = V_func(dG)
    return V_block


def build_H(
    k: np.ndarray,
    V_func: VFunc,
    N_cut: int = 3,
    lattice: Lattice = Lattice(),
) -> np.ndarray:
    """Build the NFE Hamiltonian at a single k-point.

    Parameters
    ----------
    k : array-like of shape ``(2,)``
    V_func : callable
        Integer ``(m, n)`` -> complex Fourier coefficient ``V_G``.
    N_cut : int, default 3
        Plane-wave cutoff; basis size is ``(2*N_cut + 1) ** 2``.
    lattice : Lattice, default ``Lattice(1.0, 1.0)``
        Real-space lattice constants. ``G_phys = (m * b_x, n * b_y)`` with
        ``b_x = 2*pi/a_x``, ``b_y = 2*pi/a_y``.

    Returns
    -------
    H : ndarray of shape ``(N_pw, N_pw)``, dtype ``complex128``
    """
    k = np.asarray(k, dtype=np.float64)
    G = generate_G_vectors(N_cut)
    b = np.array([lattice.b_x, lattice.b_y])
    kpG = k[None, :] + G * b[None, :]
    diag = np.sum(kpG ** 2, axis=1)

    H = build_V_block(V_func, N_cut)
    np.fill_diagonal(H, diag)
    return H


def build_H_batch(
    k_pts: np.ndarray,
    V_func: VFunc,
    N_cut: int = 3,
    lattice: Lattice = Lattice(),
) -> np.ndarray:
    """Batched ``build_H`` over many k-points (see :func:`build_H`).

    Returns
    -------
    H : ndarray of shape ``(N_k, N_pw, N_pw)``, dtype ``complex128``
    """
    k_pts = np.asarray(k_pts, dtype=np.float64)
    G = generate_G_vectors(N_cut)
    N_pw = G.shape[0]
    N_k = k_pts.shape[0]

    V_block = build_V_block(V_func, N_cut)
    b = np.array([lattice.b_x, lattice.b_y])

    # Diagonal: (N_k, 1, 2) + (1, N_pw, 2) -> (N_k, N_pw, 2) via broadcasting.
    kpG = k_pts[:, None, :] + G[None, :, :] * b[None, None, :]
    diag = np.sum(kpG ** 2, axis=2)  # (N_k, N_pw)

    H = np.broadcast_to(V_block, (N_k, N_pw, N_pw)).copy()
    np.einsum("kii->ki", H)[...] = diag
    return H


def cosine_potential(V_0: float) -> VFunc:
    """Return a ``V_func`` for ``V(r) = -V_0 [cos(2*pi*x) + cos(2*pi*y)]``.

    The only nonzero Fourier coefficients are ``V_G = -V_0 / 2`` for
    ``G in {(+/-1, 0), (0, +/-1)}`` (in integer ``(m, n)`` units).
    """
    table: dict[tuple[int, int], complex] = {
        (1, 0): -V_0 / 2 + 0j,
        (-1, 0): -V_0 / 2 + 0j,
        (0, 1): -V_0 / 2 + 0j,
        (0, -1): -V_0 / 2 + 0j,
    }

    def V_func(G: tuple[int, int]) -> complex:
        return table.get(G, 0.0 + 0.0j)

    return V_func


def anisotropic_cosine_potential(V_0: float, alpha: float = 0.3) -> VFunc:
    """Return a ``V_func`` for ``V(r) = -V_0 [cos(2*pi*x) + alpha cos(2*pi*y)]``.

    Anisotropic strength: the gap at X (from ``V_{(+/-1, 0)} = -V_0/2``) is
    ``V_0``, while the gap at Y (from ``V_{(0, +/-1)} = -alpha V_0/2``) is
    ``alpha * V_0``. Default ``alpha = 0.3`` gives a Y-gap one third of the
    X-gap, producing visibly different pocket geometry at X vs Y.
    """
    table: dict[tuple[int, int], complex] = {
        (1, 0): -V_0 / 2 + 0j,
        (-1, 0): -V_0 / 2 + 0j,
        (0, 1): -V_0 * alpha / 2 + 0j,
        (0, -1): -V_0 * alpha / 2 + 0j,
    }

    def V_func(G: tuple[int, int]) -> complex:
        return table.get(G, 0.0 + 0.0j)

    return V_func


def harmonic_cosine_potential(V_0: float, beta: float = 0.25) -> VFunc:
    """Return a ``V_func`` for cosine + second-harmonic potential.

    ``V(r) = -V_0 [cos(2*pi*x) + cos(2*pi*y)] - beta V_0 [cos(4*pi*x) + cos(4*pi*y)]``.

    Nonzero Fourier coefficients live at first-shell G's (``+/-1`` along an axis)
    and second-shell G's (``+/-2`` along an axis). The second harmonic adds
    gaps midway between Gamma and M (where ``G = (+/-2, 0)`` etc. become resonant).
    """
    table: dict[tuple[int, int], complex] = {
        (1, 0): -V_0 / 2 + 0j,
        (-1, 0): -V_0 / 2 + 0j,
        (0, 1): -V_0 / 2 + 0j,
        (0, -1): -V_0 / 2 + 0j,
        (2, 0): -V_0 * beta / 2 + 0j,
        (-2, 0): -V_0 * beta / 2 + 0j,
        (0, 2): -V_0 * beta / 2 + 0j,
        (0, -2): -V_0 * beta / 2 + 0j,
    }

    def V_func(G: tuple[int, int]) -> complex:
        return table.get(G, 0.0 + 0.0j)

    return V_func


def is_hermitian(H: np.ndarray, atol: float = 1e-12) -> bool:
    """Return ``True`` if ``H`` equals its conjugate transpose within ``atol``."""
    return np.allclose(H, H.conj().T, atol=atol, rtol=0.0)
