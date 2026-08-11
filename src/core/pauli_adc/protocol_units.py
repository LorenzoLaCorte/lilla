"""Noise, swapping, and recurrence-distillation maps on real-X correlators."""

from dataclasses import dataclass

import numpy as np

from src.core.pauli_adc.state import _as_coordinates


_VALIDATION_ATOL = 1.0e-12


def _as_finite_real_array(values, *, name):
    raw = np.asarray(values)
    try:
        finite = np.all(np.isfinite(raw))
    except TypeError as error:
        raise ValueError(f"{name} must be real numbers") from error
    if not finite:
        raise ValueError(f"{name} must be finite")
    if np.iscomplexobj(raw):
        if np.max(np.abs(raw.imag)) > _VALIDATION_ATOL:
            raise ValueError(f"{name} must be real")
        raw = raw.real
    return np.asarray(raw, dtype=float)


@dataclass(frozen=True)
class LocalAffineChannel:
    """Low-level affine Pauli-axis map on Bloch coordinates.

    The action is ``(x,y,z) -> (scale_x*x, scale_y*y, scale_z*z + shift_z)``.
    Finiteness is enforced here; CPTP validity is guaranteed by the factory
    functions in this module, but is the caller's responsibility when this
    dataclass is instantiated directly.
    """

    scale_x: float
    scale_y: float
    scale_z: float
    shift_z: float

    def __post_init__(self):
        values = _as_finite_real_array(
            [self.scale_x, self.scale_y, self.scale_z, self.shift_z],
            name="affine-map parameters",
        )
        for field, value in zip(
            ("scale_x", "scale_y", "scale_z", "shift_z"), values
        ):
            object.__setattr__(self, field, float(value))

    @property
    def z_block(self):
        """Restriction of the PTM to the ordered basis (I,Z)."""
        return np.array(
            [[1.0, 0.0], [self.shift_z, self.scale_z]],
            dtype=float,
        )

    @property
    def fixed_point_bias(self):
        """Stationary Bloch-z value c/(1-d), when it is unique."""
        denominator = 1.0 - self.scale_z
        if denominator == 0.0:
            if self.shift_z == 0.0:
                raise ValueError("the channel has no unique z fixed point")
            raise ValueError("invalid affine channel: d=1 with nonzero translation")
        return self.shift_z / denominator


def pauli_eigenvalues(probabilities):
    """Convert (p_I,p_X,p_Y,p_Z) to Pauli-channel eigenvalues mu."""
    probabilities = _as_finite_real_array(
        probabilities, name="Pauli probabilities"
    )
    if probabilities.shape != (4,):
        raise ValueError("a Pauli channel requires four probabilities")
    if np.any(probabilities < 0.0) or not np.isclose(
        np.sum(probabilities), 1.0, atol=_VALIDATION_ATOL, rtol=0.0
    ):
        raise ValueError("Pauli probabilities must be nonnegative and sum to one")
    p_i, p_x, p_y, p_z = probabilities
    return np.array(
        [
            p_i + p_x + p_y + p_z,
            p_i + p_x - p_y - p_z,
            p_i - p_x + p_y - p_z,
            p_i - p_x - p_y + p_z,
        ],
        dtype=float,
    )


def _validate_pauli_eigenvalues(mu):
    if mu.shape != (4,) or not np.isclose(
        mu[0], 1.0, atol=_VALIDATION_ATOL, rtol=0.0
    ):
        raise ValueError("mu must contain four eigenvalues with mu_0=1")
    inverse_probabilities = np.array(
        [
            1.0 + mu[1] + mu[2] + mu[3],
            1.0 + mu[1] - mu[2] - mu[3],
            1.0 - mu[1] + mu[2] - mu[3],
            1.0 - mu[1] - mu[2] + mu[3],
        ]
    ) / 4.0
    if np.any(inverse_probabilities < -_VALIDATION_ATOL):
        raise ValueError("mu does not define a completely positive Pauli channel")


def sequential_pauli_adc_channel(mu, eta, *, order="pauli_after_adc"):
    """Build one ordered Pauli/ADC layer from its PTM parameters."""
    mu = _as_finite_real_array(mu, name="Pauli eigenvalues")
    _validate_pauli_eigenvalues(mu)
    eta = float(_as_finite_real_array([eta], name="eta")[0])
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0,1]")

    if order == "pauli_after_adc":
        shift_z = mu[3] * (1.0 - eta)
    elif order == "adc_after_pauli":
        shift_z = 1.0 - eta
    else:
        raise ValueError("unknown concatenation order")
    return LocalAffineChannel(
        scale_x=mu[1] * np.sqrt(eta),
        scale_y=mu[2] * np.sqrt(eta),
        scale_z=mu[3] * eta,
        shift_z=shift_z,
    )


def _validate_pauli_lindblad_rates(rates, *, atol=1.0e-14):
    gamma_x, gamma_y, gamma_z = rates
    jump_rates = np.array(
        [
            gamma_y + gamma_z - gamma_x,
            gamma_x + gamma_z - gamma_y,
            gamma_x + gamma_y - gamma_z,
        ]
    ) / 4.0
    if np.any(jump_rates < -atol):
        raise ValueError("Pauli decay rates do not define a CP Pauli semigroup")


def joint_pauli_adc_channel(pauli_decay_rates, adc_rate, time):
    """Exponentiate a simultaneous Pauli-Lindblad plus ADC generator.

    ``pauli_decay_rates`` are the decay rates of X, Y and Z Pauli modes.  A
    scalar denotes isotropic depolarization.  The triangle constraints are
    checked so that the rates correspond to nonnegative Pauli jump rates.
    """
    rates = _as_finite_real_array(
        pauli_decay_rates, name="Pauli decay rates"
    )
    if rates.ndim == 0:
        rates = np.repeat(rates, 3)
    if rates.shape != (3,):
        raise ValueError("provide one or three Pauli decay rates")
    adc_rate, time = _as_finite_real_array(
        [adc_rate, time], name="ADC rate and time"
    )
    if np.any(rates < 0.0) or adc_rate < 0.0 or time < 0.0:
        raise ValueError("rates and time must be nonnegative")
    _validate_pauli_lindblad_rates(rates)

    gamma_x, gamma_y, gamma_z = rates
    longitudinal_rate = gamma_z + adc_rate
    scale_z = np.exp(-longitudinal_rate * time)
    if longitudinal_rate == 0.0:
        shift_z = 0.0
    else:
        fixed_point_bias = adc_rate / longitudinal_rate
        shift_z = fixed_point_bias * -np.expm1(-longitudinal_rate * time)
    return LocalAffineChannel(
        scale_x=np.exp(-(gamma_x + adc_rate / 2.0) * time),
        scale_y=np.exp(-(gamma_y + adc_rate / 2.0) * time),
        scale_z=scale_z,
        shift_z=shift_z,
    )


def compose_channels(after, before):
    """Return ``after o before`` for two affine Pauli-axis channels."""
    return LocalAffineChannel(
        scale_x=after.scale_x * before.scale_x,
        scale_y=after.scale_y * before.scale_y,
        scale_z=after.scale_z * before.scale_z,
        shift_z=after.shift_z + after.scale_z * before.shift_z,
    )


def apply_local_channels(coordinates, left, right=None):
    """Apply possibly different local channels to a real-X state."""
    coordinates = _as_coordinates(coordinates)
    if right is None:
        right = left

    z_sector = np.array(
        [
            [coordinates[0], coordinates[1]],
            [coordinates[2], coordinates[3]],
        ],
        dtype=float,
    )
    z_out = left.z_block @ z_sector @ right.z_block.T
    return np.array(
        [
            z_out[0, 0],
            z_out[0, 1],
            z_out[1, 0],
            z_out[1, 1],
            left.scale_x * right.scale_x * coordinates[4],
            left.scale_y * right.scale_y * coordinates[5],
        ],
        dtype=float,
    )


def phi_plus_swap(left_coordinates, right_coordinates, *, conditioned=False):
    """Contract the inner qubits with phi+ and return (state, branch mass).

    Inputs are ordered as (outer, measured) for the left link and (measured,
    outer) for the right link.  The returned state is unnormalized unless
    ``conditioned=True``.  The branch mass is a conditional probability only
    for trace-one inputs; weighted inputs produce weighted probability mass.
    """
    left_coordinates = _as_coordinates(left_coordinates)
    right_coordinates = _as_coordinates(right_coordinates)

    left_z = np.array(
        [
            [left_coordinates[0], left_coordinates[1]],
            [left_coordinates[2], left_coordinates[3]],
        ]
    )
    right_z = np.array(
        [
            [right_coordinates[0], right_coordinates[1]],
            [right_coordinates[2], right_coordinates[3]],
        ]
    )
    z_out = (left_z @ right_z) / 4.0
    output = np.array(
        [
            z_out[0, 0],
            z_out[0, 1],
            z_out[1, 0],
            z_out[1, 1],
            left_coordinates[4] * right_coordinates[4] / 4.0,
            -left_coordinates[5] * right_coordinates[5] / 4.0,
        ],
        dtype=float,
    )
    probability = float(output[0])
    if conditioned:
        if probability <= 0.0:
            raise ValueError("cannot condition on a zero-probability outcome")
        output = output / probability
    return output, probability


def psi_corrected_swap(left_coordinates, right_coordinates, *, conditioned=False):
    """Return one corrected psi Bell-measurement branch.

    The Pauli correction is applied to the right outer qubit.  After this
    correction, psi+ and psi- have the same six-coordinate map.
    """
    left = _as_coordinates(left_coordinates)
    right = _as_coordinates(right_coordinates)
    output = np.array(
        [
            (left[0] * right[0] - left[1] * right[2]) / 4.0,
            (-left[0] * right[1] + left[1] * right[3]) / 4.0,
            (left[2] * right[0] - left[3] * right[2]) / 4.0,
            (-left[2] * right[1] + left[3] * right[3]) / 4.0,
            left[4] * right[4] / 4.0,
            -left[5] * right[5] / 4.0,
        ],
        dtype=float,
    )
    probability = float(output[0])
    if conditioned:
        if probability <= 0.0:
            raise ValueError("cannot condition on a zero-probability outcome")
        output = output / probability
    return output, probability


def accepted_bell_swap(
    left_coordinates,
    right_coordinates,
    *,
    outcomes="all_corrected",
    conditioned=False,
):
    """Sum an explicitly selected set of corrected Bell outcomes.

    Supported policies are ``phi_plus`` (one outcome), ``phi_pair``,
    ``psi_pair``, and ``all_corrected``.  The result is unnormalized by
    default, so its ``II`` coordinate is the intrinsic accepted-outcome mass.
    """
    left = _as_coordinates(left_coordinates)
    right = _as_coordinates(right_coordinates)
    if outcomes == "all_corrected":
        # Do not construct an unused phi+ branch on this common path.
        output = np.array(
            [
                left[0] * right[0],
                left[1] * right[3],
                left[2] * right[0],
                left[3] * right[3],
                left[4] * right[4],
                -left[5] * right[5],
            ],
            dtype=float,
        )
    elif outcomes == "psi_pair":
        psi, _ = psi_corrected_swap(left, right)
        output = 2.0 * psi
    elif outcomes in ("phi_plus", "phi_pair"):
        phi, _ = phi_plus_swap(left, right)
        output = phi if outcomes == "phi_plus" else 2.0 * phi
    else:
        raise ValueError(
            "outcomes must be one of phi_plus, phi_pair, psi_pair, "
            "or all_corrected"
        )

    probability = float(output[0])
    if conditioned:
        if probability <= 0.0:
            raise ValueError("cannot condition on a zero-probability outcome")
        output = output / probability
    return output, probability


def recurrence_distillation(
    kept_coordinates,
    auxiliary_coordinates,
    *,
    conditioned=False,
    werner_twirl=False,
):
    """Ideal bilateral-CNOT recurrence with equal target outcomes accepted.

    The first state is the kept/control pair and the second the measured
    auxiliary/target pair.  The 00 and 11 branches are summed without an
    outcome-dependent correction.  This role ordering matters when ADC has
    produced nonzero local Z biases.
    """
    kept = _as_coordinates(kept_coordinates)
    auxiliary = _as_coordinates(auxiliary_coordinates)
    output = np.array(
        [
            (kept[0] * auxiliary[0] + kept[3] * auxiliary[3]) / 2.0,
            (kept[1] * auxiliary[0] + kept[2] * auxiliary[3]) / 2.0,
            (kept[2] * auxiliary[0] + kept[1] * auxiliary[3]) / 2.0,
            (kept[3] * auxiliary[0] + kept[0] * auxiliary[3]) / 2.0,
            (kept[4] * auxiliary[4] + kept[5] * auxiliary[5]) / 2.0,
            (kept[5] * auxiliary[4] + kept[4] * auxiliary[5]) / 2.0,
        ],
        dtype=float,
    )
    if werner_twirl:
        omega = (output[3] + output[4] - output[5]) / 3.0
        output = np.array(
            [output[0], 0.0, 0.0, omega, omega, -omega], dtype=float
        )
    probability = float(output[0])
    if conditioned:
        if probability <= 0.0:
            raise ValueError("cannot condition on a zero-probability outcome")
        output = output / probability
    return output, probability
