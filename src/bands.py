"""Diagonalize the NFE Hamiltonian along k-paths.

Provides :func:`compute_bands_along_path` for band-structure plots along
piecewise-linear paths in the BZ (e.g. ``Gamma -> X -> M -> Gamma``).
Eigenvalues are obtained with :func:`numpy.linalg.eigvalsh` (Hermitian
LAPACK routine), which returns real values sorted ascending.
"""

import numpy as np

from .hamiltonian import VFunc, build_H
from .lattice import generate_G_vectors


def sample_k_path(
    path_nodes: list[np.ndarray], n_per_segment: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a piecewise-linear k-path, deduplicating shared endpoints.

    Returns
    -------
    k_pts : ndarray of shape ``(n_total, 2)``
        Sampled k-points along the path.
    k_dist : ndarray of shape ``(n_total,)``
        Cumulative Euclidean distance along the path.
    node_distances : ndarray of shape ``(len(path_nodes),)``
        Cumulative distance at each input vertex; used for vertical guides.
    """
    nodes = [np.asarray(n, dtype=np.float64) for n in path_nodes]
    segments: list[np.ndarray] = []
    node_dists = [0.0]
    for i in range(len(nodes) - 1):
        seg = np.linspace(nodes[i], nodes[i + 1], n_per_segment, endpoint=True)
        if i > 0:
            seg = seg[1:]  # drop shared corner so it is not double-counted
        segments.append(seg)
        node_dists.append(node_dists[-1] + np.linalg.norm(nodes[i + 1] - nodes[i]))

    k_pts = np.concatenate(segments, axis=0)
    step_lens = np.linalg.norm(np.diff(k_pts, axis=0), axis=1)
    k_dist = np.concatenate([[0.0], np.cumsum(step_lens)])
    return k_pts, k_dist, np.array(node_dists)


def compute_bands_along_path(
    path_nodes: list[np.ndarray],
    n_per_segment: int,
    V_func: VFunc,
    N_cut: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize ``H(k)`` along a piecewise-linear k-path.

    Parameters
    ----------
    path_nodes : list of length-2 array-likes
        Vertices of the path, e.g. ``[Gamma, X, M, Gamma]``.
    n_per_segment : int
        Points per segment, including both endpoints. Shared corners are
        deduplicated, so a path of ``N`` nodes yields ``(N-1)*(n-1)+1`` points.
    V_func : callable
        Potential Fourier-coefficient function; passed straight to ``build_H``.
    N_cut : int, default 3
        Plane-wave cutoff.

    Returns
    -------
    k_dist : ndarray of shape ``(n_total,)``
        Cumulative path distance, suitable as the x-axis for band plots.
    energies : ndarray of shape ``(n_total, N_pw)``
        Eigenvalues sorted ascending along axis 1, one row per k-point.
    node_distances : ndarray of shape ``(len(path_nodes),)``
        x-positions of the path vertices; use for vertical dashed guides.
    """
    k_pts, k_dist, node_distances = sample_k_path(path_nodes, n_per_segment)
    N_pw = (2 * N_cut + 1) ** 2

    energies = np.empty((k_pts.shape[0], N_pw), dtype=np.float64)
    for i, k in enumerate(k_pts):
        H = build_H(k, V_func, N_cut)
        energies[i, :] = np.linalg.eigvalsh(H)

    return k_dist, energies, node_distances


def free_electron_energies_along_path(
    path_nodes: list[np.ndarray],
    n_per_segment: int,
    N_cut: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic free-electron bands along a k-path.

    For each k on the path returns the sorted-ascending set
    ``{|k + G|^2}`` over all G in the basis. With ``V_0 = 0`` this must
    coincide with :func:`compute_bands_along_path` to within floating
    precision; the difference is a useful regression check.

    Returns
    -------
    k_dist : ndarray of shape ``(n_total,)``
    energies : ndarray of shape ``(n_total, N_pw)``, sorted ascending
    node_distances : ndarray of shape ``(len(path_nodes),)``
    """
    k_pts, k_dist, node_distances = sample_k_path(path_nodes, n_per_segment)
    G = generate_G_vectors(N_cut)

    # Vectorize: k_pts (n_total, 1, 2) + G (1, N_pw, 2) -> (n_total, N_pw, 2).
    kpG = k_pts[:, None, :] + 2.0 * np.pi * G[None, :, :]
    energies = np.sum(kpG ** 2, axis=2)
    energies.sort(axis=1)
    return k_dist, energies, node_distances
