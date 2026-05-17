"""Reciprocal-lattice geometry for the 2D square NFE model.

Natural units: lattice constant ``a = 1``, so the primitive reciprocal
vectors are ``b1 = (2*pi, 0)`` and ``b2 = (0, 2*pi)``.

G-vectors are stored as integer ``(m, n)`` pairs. Multiplication by
``2*pi`` is performed by the caller (typically inside :func:`build_H`
when constructing the diagonal). Keeping integer form means equality
comparisons used as dictionary keys for ``V_func`` lookups are exact.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lattice:
    """2D rectangular lattice parameters.

    ``a_x``, ``a_y`` are the real-space lattice constants. Reciprocal
    primitive vectors and derived geometry are exposed as properties so
    callers never compute ``2*pi/a`` by hand.

    Default ``Lattice()`` is the unit square ``(1, 1)`` used everywhere
    in M1-M6, so existing code keeps working unchanged.
    """

    a_x: float = 1.0
    a_y: float = 1.0

    @property
    def b_x(self) -> float:
        return 2.0 * np.pi / self.a_x

    @property
    def b_y(self) -> float:
        return 2.0 * np.pi / self.a_y

    @property
    def cell_area(self) -> float:
        return self.a_x * self.a_y

    @property
    def bz_area(self) -> float:
        return self.b_x * self.b_y


def generate_G_vectors(N_cut: int) -> np.ndarray:
    """Return all integer ``(m, n)`` with ``|m|, |n| <= N_cut``.

    Parameters
    ----------
    N_cut : int
        Plane-wave cutoff.

    Returns
    -------
    G : ndarray of shape ``(N_pw, 2)``, dtype ``int64``
        One row per plane wave, ordered lexicographically in ``(m, n)`` so
        the index mapping is deterministic. With ``N_cut = 3`` the basis
        contains ``(2*3 + 1) ** 2 = 49`` plane waves.
    """
    rng = range(-N_cut, N_cut + 1)
    G = np.array([(m, n) for m in rng for n in rng], dtype=np.int64)
    return G


def first_bz_corners(lattice: Lattice = Lattice()) -> np.ndarray:
    """Return the four corners of the first Brillouin zone rectangle.

    BZ1 is ``[-b_x/2, b_x/2] x [-b_y/2, b_y/2]``. For the default square
    unit lattice this reduces to ``[-pi, pi] x [-pi, pi]``.
    """
    bx2 = lattice.b_x / 2.0
    by2 = lattice.b_y / 2.0
    return np.array(
        [
            [-bx2, -by2],
            [bx2, -by2],
            [bx2, by2],
            [-bx2, by2],
        ],
        dtype=np.float64,
    )


def high_symmetry_points(lattice: Lattice = Lattice()) -> dict[str, np.ndarray]:
    """Return the canonical high-symmetry k-points (Gamma, X, M).

    Convention: ``X = (b_x/2, 0)`` and ``M = (b_x/2, b_y/2)``. For the
    default unit square these are ``(pi, 0)`` and ``(pi, pi)`` —
    backwards-compatible with notebooks 01-06.
    """
    bx2 = lattice.b_x / 2.0
    by2 = lattice.b_y / 2.0
    return {
        "Gamma": np.array([0.0, 0.0]),
        "X": np.array([bx2, 0.0]),
        "M": np.array([bx2, by2]),
    }


def high_symmetry_points_rectangular(lattice: Lattice) -> dict[str, np.ndarray]:
    """Return the four high-symmetry k-points (Gamma, X, Y, M) for a rectangular BZ.

    Distinct from :func:`high_symmetry_points` only in the inclusion of
    ``Y = (0, b_y/2)``. For ``a_y > a_x`` this Y point is closer to Gamma
    than X is (``b_y < b_x``).
    """
    bx2 = lattice.b_x / 2.0
    by2 = lattice.b_y / 2.0
    return {
        "Gamma": np.array([0.0, 0.0]),
        "X": np.array([bx2, 0.0]),
        "Y": np.array([0.0, by2]),
        "M": np.array([bx2, by2]),
    }


def high_symmetry_path(lattice: Lattice = Lattice()) -> list[np.ndarray]:
    """Return the standard band-structure path ``Gamma -> X -> M -> Gamma``."""
    pts = high_symmetry_points(lattice)
    return [pts["Gamma"], pts["X"], pts["M"], pts["Gamma"]]


def reciprocal_lattice_points(N_max: int) -> np.ndarray:
    """Return all reciprocal lattice points within ``|m|, |n| <= N_max``.

    Output is in physical units (multiplied by ``2*pi``), suitable for
    direct plotting on the ``(k_x, k_y)`` plane.
    """
    rng = np.arange(-N_max, N_max + 1)
    mm, nn = np.meshgrid(rng, rng, indexing="ij")
    return 2.0 * np.pi * np.stack([mm.ravel(), nn.ravel()], axis=1)


def perpendicular_bisectors_nearest_neighbors() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the 8 perpendicular-bisector lines of the origin's nearest neighbors.

    Each line is represented as a pair of points ``(p1, p2)`` lying on it,
    suitable for ``matplotlib.axes.Axes.axline``. The 8 nearest reciprocal
    neighbors are 4 axial at ``(+/-2*pi, 0), (0, +/-2*pi)`` (giving the BZ1
    edges ``x = +/-pi``, ``y = +/-pi``) and 4 diagonal at ``(+/-2*pi, +/-2*pi)``
    (giving the BZ2 outer edges ``x +/- y = +/-2*pi``).
    """
    pi = np.pi
    two_pi = 2.0 * pi
    lines: list[tuple[np.ndarray, np.ndarray]] = []

    for c in (pi, -pi):
        lines.append((np.array([c, 0.0]), np.array([c, 1.0])))  # vertical x = c
        lines.append((np.array([0.0, c]), np.array([1.0, c])))  # horizontal y = c

    for c in (two_pi, -two_pi):
        lines.append((np.array([c, 0.0]), np.array([0.0, c])))   # x + y = c
        lines.append((np.array([c, 0.0]), np.array([0.0, -c])))  # x - y = c

    return lines


def second_bz_triangles() -> list[np.ndarray]:
    """Return the four right triangles that compose the second Brillouin zone.

    Each triangle is a ``(3, 2)`` array of vertices in order, ready to pass to
    ``matplotlib.patches.Polygon``. The four triangles sit outside BZ1 and
    inside the diagonal perpendicular-bisector lines.
    """
    pi = np.pi
    two_pi = 2.0 * pi
    return [
        np.array([[pi, pi], [pi, -pi], [two_pi, 0.0]]),     # right
        np.array([[-pi, pi], [pi, pi], [0.0, two_pi]]),     # top
        np.array([[-pi, pi], [-pi, -pi], [-two_pi, 0.0]]),  # left
        np.array([[-pi, -pi], [pi, -pi], [0.0, -two_pi]]),  # bottom
    ]
