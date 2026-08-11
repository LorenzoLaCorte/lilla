"""Stable six-correlator coordinates for (possibly unnormalized) real X-states.

The coordinate order is

    (II, IZ, ZI, ZZ, XX, YY),

where ``x_PQ = Tr[rho (P tensor Q)]``.  Consequently

    rho = 1/4 sum_PQ x_PQ (P tensor Q)

on the real-X subspace, and ``x_II = Tr(rho)``.  Keeping the trace as a
coordinate is useful for the probability-weighted states used by the repeater
convolutions.
"""

import numpy as np

from src.core.pauli_fourier.state import lambda_to_mu
from src.core.states import QuantumState


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)

COORDINATE_LABELS = ("II", "IZ", "ZI", "ZZ", "XX", "YY")
XFunc = np.ndarray
_PRODUCTS = (
    np.kron(I2, I2),
    np.kron(I2, Z),
    np.kron(Z, I2),
    np.kron(Z, Z),
    np.kron(X, X),
    np.kron(Y, Y),
)


def _as_coordinates(coordinates, *, atol=1.0e-12):
    raw = np.asarray(coordinates)
    if raw.shape != (6,):
        raise ValueError("real-X states require six coordinates")
    try:
        finite = np.all(np.isfinite(raw))
    except TypeError as error:
        raise ValueError("real-X coordinates must be real numbers") from error
    if not finite:
        raise ValueError("real-X coordinates must be finite")
    if np.iscomplexobj(raw):
        if np.max(np.abs(raw.imag)) > atol:
            raise ValueError("real-X coordinates must be real")
        raw = raw.real
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("real-X coordinates must be finite")
    return result


def coordinates_to_matrix(coordinates):
    """Reconstruct a 4x4 matrix from six Pauli correlators."""
    coordinates = _as_coordinates(coordinates)
    matrix = sum(
        (coefficient * product for coefficient, product in zip(coordinates, _PRODUCTS)),
        np.zeros((4, 4), dtype=complex),
    ) / 4.0
    return np.real_if_close(matrix, tol=1000)


def matrix_to_coordinates(matrix, *, atol=1.0e-12):
    """Encode a real X-matrix, rejecting components outside the subspace."""
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (4, 4):
        raise ValueError("a two-qubit state must be a 4x4 matrix")
    if not np.allclose(matrix, matrix.conj().T, atol=atol, rtol=0.0):
        raise ValueError("the matrix must be Hermitian")

    coordinates = np.array(
        [np.trace(matrix @ product) for product in _PRODUCTS],
        dtype=complex,
    )
    if np.max(np.abs(coordinates.imag)) > atol:
        raise ValueError("the matrix is not a real X-state")
    coordinates = coordinates.real
    if not np.allclose(
        coordinates_to_matrix(coordinates), matrix, atol=atol, rtol=0.0
    ):
        raise ValueError("the matrix has components outside the real-X subspace")
    return coordinates


def state_trace(coordinates):
    """Return Tr(rho), which is the II coordinate."""
    return float(_as_coordinates(coordinates)[0])


def phi_plus_fidelity(coordinates):
    """Return phi+ fidelity for a physical state, conditioning on its trace."""
    coordinates = _as_coordinates(coordinates)
    trace = coordinates[0]
    if trace <= 0.0:
        raise ValueError("fidelity is undefined for a non-positive trace")
    return float(
        (coordinates[0] + coordinates[3] + coordinates[4] - coordinates[5])
        / (4.0 * trace)
    )


def is_physical(coordinates, *, atol=1.0e-12):
    """Check positive semidefiniteness and a nonnegative trace."""
    coordinates = _as_coordinates(coordinates)
    if coordinates[0] < -atol:
        return False
    return bool(_minimum_real_x_eigenvalue(coordinates) >= -atol)


def _real_x_matrix_entries(values):
    """Return the six independent real-X matrix entries, vectorized."""
    n, z_r, z_l, c_zz, c_xx, c_yy = np.moveaxis(values, -1, 0)
    return (
        (n + z_r + z_l + c_zz) / 4.0,
        (n - z_r + z_l - c_zz) / 4.0,
        (n + z_r - z_l - c_zz) / 4.0,
        (n - z_r - z_l + c_zz) / 4.0,
        (c_xx - c_yy) / 4.0,
        (c_xx + c_yy) / 4.0,
    )


def _minimum_real_x_eigenvalue(values):
    """Smallest eigenvalue of each real-X matrix, without 4x4 matrices."""
    a, b, c, d, w, z = _real_x_matrix_entries(values)
    even_min = (a + d - np.hypot(a - d, 2.0 * w)) / 2.0
    odd_min = (b + c - np.hypot(b - c, 2.0 * z)) / 2.0
    return np.minimum(even_min, odd_min)


def bell_weights_to_coordinates(lambdas):
    """Embed Bell weights in the real-X coordinates.

    The Bell ordering is ``(phi+, psi+, psi-, phi-)``, consistently with the
    Bell and Pauli-Fourier backends.  If their Pauli-channel eigenvalues are
    ``(1, mu_x, mu_y, mu_z)``, the corresponding correlators are
    ``(1, 0, 0, mu_z, mu_x, -mu_y)``.
    """
    mu = lambda_to_mu(np.asarray(lambdas, dtype=float))
    if mu.shape != (4,):
        raise ValueError("Bell weights must contain four entries")
    return np.array([mu[0], 0.0, 0.0, mu[3], mu[1], -mu[2]], dtype=float)


class RealXState(QuantumState):
    """Normalized two-qubit state in the invariant real-X subspace."""

    width = 6

    def __init__(self, *, coordinates=None, matrix=None, lambdas=None):
        supplied = sum(value is not None for value in (coordinates, matrix, lambdas))
        if supplied != 1:
            raise ValueError("provide exactly one of coordinates, matrix, or lambdas")

        if matrix is not None:
            coordinates = matrix_to_coordinates(matrix)
        elif lambdas is not None:
            coordinates = bell_weights_to_coordinates(lambdas)
        coordinates = _as_coordinates(coordinates)

        if not np.isclose(coordinates[0], 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("a normalized real-X state must have II=1")
        if not is_physical(coordinates):
            raise ValueError("coordinates do not describe a positive semidefinite state")
        self.coordinates = coordinates.astype(float, copy=True)

    def __repr__(self):
        return f"RealXState(coordinates={self.coordinates})"

    def get_generation_sf(self, t_trunc, i=None) -> XFunc:
        del i
        return np.tile(self.coordinates, (t_trunc, 1))


def phi_plus_fidelity_func(coordinate_func):
    """Vectorized phi+ fidelity for normalized real-X state functions."""
    values = np.asarray(coordinate_func, dtype=float)
    if values.shape[-1] != 6:
        raise ValueError("real-X state functions must have last dimension six")
    trace = values[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (values[..., 0] + values[..., 3] + values[..., 4] - values[..., 5]) / (
            4.0 * trace
        )


def polish_x_func(coordinate_func, *, atol=1.0e-10):
    """Remove roundoff while preserving the real-X block geometry.

    Tiny negative eigenvalues of either real-X 2x2 block are projected away.
    Materially non-physical states raise instead of being silently clipped.
    """
    values = np.asarray(coordinate_func)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("real-X state functions must have shape (T, 6)")
    values = np.real_if_close(values, tol=1000)
    if np.iscomplexobj(values):
        raise ValueError("real-X state functions must be real")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("real-X state functions must be finite")

    trace_tolerance = max(10.0 * atol, 1.0e-9)
    invalid_trace = np.flatnonzero(
        ~np.isclose(values[:, 0], 1.0, atol=trace_tolerance, rtol=0.0)
    )
    if len(invalid_trace):
        index = int(invalid_trace[0])
        raise ValueError(
            f"real-X state at index {index} is not normalized "
            f"(II={values[index, 0]})"
        )

    # Normalize all rows in one operation.  This is equivalent to the old
    # matrix reconstruction followed by division by Tr(rho), but avoids a
    # Python loop and 4x4 allocations.
    if np.any(values[:, 0] <= atol):
        index = int(np.flatnonzero(values[:, 0] <= atol)[0])
        raise ValueError(f"real-X state at index {index} has zero trace")
    result = values / values[:, 0, np.newaxis]
    result[:, 0] = 1.0

    minimum = _minimum_real_x_eigenvalue(result)
    invalid = np.flatnonzero(minimum < -atol)
    if len(invalid):
        index = int(invalid[0])
        raise ValueError(
            f"real-X state at index {index} is not positive semidefinite "
            f"(minimum eigenvalue {minimum[index]:.3e})"
        )

    # Projection is needed only for rows with signed roundoff.  Each parity
    # block has the form 1/2 [[s+u,v],[v,s-u]], so clipping its two
    # eigenvalues has a closed form and needs no matrix construction/eigh.
    repair = np.flatnonzero(minimum < 0.0)
    if len(repair):
        n, z_right, z_left, zz, xx, yy = result[repair].T
        sector_data = (
            ((n + zz) / 2.0, (z_right + z_left) / 2.0, (xx - yy) / 2.0),
            ((n - zz) / 2.0, (z_left - z_right) / 2.0, (xx + yy) / 2.0),
        )
        projected_sectors = []
        for sector_trace, diagonal_bias, coherence in sector_data:
            radius = np.hypot(diagonal_bias, coherence)
            eigenvalue_plus = np.maximum((sector_trace + radius) / 2.0, 0.0)
            eigenvalue_minus = np.maximum((sector_trace - radius) / 2.0, 0.0)
            projected_trace = eigenvalue_plus + eigenvalue_minus
            scale = np.divide(
                eigenvalue_plus - eigenvalue_minus,
                radius,
                out=np.zeros_like(radius),
                where=radius > 0.0,
            )
            projected_sectors.append(
                (projected_trace, scale * diagonal_bias, scale * coherence)
            )

        (trace_even, bias_even, coherence_even), (
            trace_odd,
            bias_odd,
            coherence_odd,
        ) = projected_sectors
        repaired_trace = trace_even + trace_odd
        repaired = np.column_stack(
            (
                repaired_trace,
                bias_even - bias_odd,
                bias_even + bias_odd,
                trace_even - trace_odd,
                coherence_even + coherence_odd,
                coherence_odd - coherence_even,
            )
        )
        result[repair] = repaired / repaired_trace[:, np.newaxis]
        result[repair, 0] = 1.0
    return result[0] if one_dimensional else result


def normalize_weighted_x_func(weighted_coordinates, pmf, *, atol=1.0e-10):
    """Condition probability-weighted correlators on their completion time.

    The trace coordinate is independently produced by the quantum map, while
    ``pmf`` is produced by the classical success/failure accounting.  Checking
    their equality here is an important end-to-end invariant.
    """
    weighted = np.asarray(weighted_coordinates, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    if weighted.ndim != 2 or weighted.shape[1] != 6:
        raise ValueError("weighted real-X functions must have shape (T, 6)")
    if pmf.shape != (weighted.shape[0],):
        raise ValueError("pmf and weighted state function have incompatible shapes")
    if not np.all(np.isfinite(pmf)) or np.any(pmf < 0.0):
        raise ValueError("success PMFs must be finite and nonnegative")
    if not np.allclose(weighted[:, 0], pmf, atol=atol, rtol=atol):
        difference = float(np.max(np.abs(weighted[:, 0] - pmf)))
        raise ValueError(
            "weighted II coordinate does not match the success PMF "
            f"(maximum discrepancy {difference:.3e})"
        )

    normalized = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (len(pmf), 1)
    )
    # Every positive bin may feed a later protocol level.  Replacing a tiny
    # positive bin by a dummy state would therefore lose physical information
    # even if its contribution is negligible at the present level.
    populated = pmf > 0.0
    normalized[populated] = weighted[populated] / pmf[populated, np.newaxis]
    return polish_x_func(normalized, atol=max(atol * 10.0, 1.0e-10))
