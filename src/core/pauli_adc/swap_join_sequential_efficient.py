"""Exact finite-sum efficient join for sequential Pauli/ADC noise.

The direct join in :mod:`swap_join` is the correctness oracle.  This module
changes only the order of summation: each finite sequential waiting channel is
expanded into exponential kernels, those kernels are filtered over the
memory-time window, and the fixed six correlators are reconstructed before the
swap or distillation map is applied.

This finite channel family is not treated as a semigroup.  The endpoint
translation is retained as a duration-dependent finite sum.
"""

import numpy as np
from scipy.signal import lfilter

from src.core.pauli_adc.swap_join import (
    AttemptTerms,
    _validate_attempt_inputs,
)


def _next_power_of_two(value):
    return 1 << max(0, int(value - 1).bit_length())


def _fft_convolve_truncated(left, right, length):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if length <= 0:
        return np.empty(0, dtype=float)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("FFT convolution inputs must be one-dimensional")
    if len(left) == 0 or len(right) == 0:
        return np.zeros(length, dtype=float)
    transform_length = _next_power_of_two(len(left) + len(right) - 1)
    product = np.fft.rfft(left, transform_length) * np.fft.rfft(
        right, transform_length
    )
    convolution = np.fft.irfft(product, transform_length)
    result = np.zeros(length, dtype=float)
    result[: min(length, len(convolution))] = convolution[:length]
    return result


def _power_series_inverse(coefficients, length):
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) < length:
        raise ValueError("power-series coefficients have an invalid shape")
    if coefficients[0] <= 1.0e-15:
        raise ValueError("renewal has a unit-probability zero-duration failure loop")
    inverse = np.array([1.0 / coefficients[0]], dtype=float)
    order = 1
    while order < length:
        next_order = min(2 * order, length)
        product = _fft_convolve_truncated(
            coefficients[:next_order], inverse, next_order
        )
        correction = -product
        correction[0] += 2.0
        inverse = _fft_convolve_truncated(inverse, correction, next_order)
        order = next_order
    return inverse[:length]


def _prepare_renewal(failure_kernel, first_success, shift):
    failure = np.asarray(failure_kernel, dtype=float)
    first = np.asarray(first_success, dtype=float)
    if (
        failure.ndim != 1
        or first.ndim not in (1, 2)
        or len(first) != len(failure)
    ):
        raise ValueError("renewal kernels must have the same leading dimension")
    if (
        not np.all(np.isfinite(failure))
        or np.any(failure < -1.0e-12)
        or np.sum(failure) > 1.0 + 1.0e-10
        or not np.all(np.isfinite(first))
    ):
        raise ValueError("renewal kernels contain invalid probability data")
    if not isinstance(shift, (int, np.integer)) or shift < 0:
        raise ValueError("renewal shift must be a nonnegative integer")
    failure = np.maximum(failure, 0.0)
    size = len(failure)
    shifted = np.zeros(size, dtype=float)
    if shift < size:
        shifted[shift:] = failure[: size - shift]
    denominator = np.zeros(size, dtype=float)
    denominator[0] = 1.0
    denominator -= shifted
    inverse = _power_series_inverse(denominator, size)
    return first, inverse


def _convolve_with_resolvent(first, inverse):
    size = len(inverse)
    transform_length = _next_power_of_two(len(first) + len(inverse) - 1)
    inverse_spectrum = np.fft.rfft(inverse, transform_length)
    if first.ndim == 1:
        result = np.fft.irfft(
            np.fft.rfft(first, transform_length) * inverse_spectrum,
            transform_length,
        )[:size]
    else:
        result = np.fft.irfft(
            np.fft.rfft(first, transform_length, axis=0)
            * inverse_spectrum[:, np.newaxis],
            transform_length,
            axis=0,
        )[:size]
    leading_nonzero = np.flatnonzero(
        np.any(first != 0.0, axis=1) if first.ndim == 2 else first != 0.0
    )
    if len(leading_nonzero) == 0:
        result[...] = 0.0
    else:
        result[: leading_nonzero[0]] = 0.0
    return np.real_if_close(result, tol=1000).astype(float, copy=False)


def renewal_sum_sequential_efficient(failure_kernel, first_success, *, shift=0):
    first, inverse = _prepare_renewal(failure_kernel, first_success, shift)
    result = _convolve_with_resolvent(first, inverse)
    columns = (
        [result]
        if result.ndim == 1
        else [result[:, index] for index in range(result.shape[1])]
    )
    first_columns = (
        [first]
        if first.ndim == 1
        else [first[:, index] for index in range(first.shape[1])]
    )
    factor = 64.0 * np.finfo(float).eps * (
        1.0 + np.ceil(np.log2(max(2, len(inverse))))
    )
    for values, initial in zip(columns, first_columns):
        if np.all(initial >= 0.0):
            tolerance = factor * np.max(np.abs(values), initial=0.0)
            tiny_negative = (values < 0.0) & (values >= -tolerance)
            values[tiny_negative] = 0.0
    return result


def renewal_sum_sequential_efficient_weighted(
    failure_kernel, weighted_first, *, shift=0
):
    first, inverse = _prepare_renewal(failure_kernel, weighted_first, shift)
    if first.ndim != 2 or first.shape[1] != 6:
        raise ValueError("weighted real-X renewal inputs must have shape (T, 6)")
    if np.any(first[:, 0] < -1.0e-12):
        raise ValueError("weighted real-X traces must be nonnegative")
    weighted = _convolve_with_resolvent(first, inverse)
    pmf = np.array(weighted[:, 0], copy=True)
    scale = np.max(np.abs(pmf), initial=0.0)
    roundoff_floor = (
        64.0
        * np.finfo(float).eps
        * (1.0 + np.ceil(np.log2(max(2, len(pmf)))))
        * scale
    )
    if np.any(pmf < -roundoff_floor):
        minimum = float(np.min(pmf))
        raise ValueError(
            f"FFT renewal produced a material negative probability ({minimum:.3e})"
        )
    unreliable = pmf <= roundoff_floor
    pmf[unreliable] = 0.0
    weighted[unreliable] = 0.0
    weighted[:, 0] = pmf
    return pmf, weighted


def _rolling_sum(values, cutoff):
    """Return inclusive rolling sums without converting large cutoffs to floats."""
    result = np.cumsum(values, axis=0)
    age = int(cutoff) + 1
    if age < len(values):
        result[age:] -= result[:-age].copy()
    return result


def _endpoint_exponential_terms(pauli_rates, adc_rate):
    """Return finite exponential terms for one endpoint affine channel.

    For duration ``t`` the endpoint coefficients are

    ``a_x(t) = exp(-(Gamma_x + gamma/2)t)``,
    ``a_y(t) = exp(-(Gamma_y + gamma/2)t)``,
    ``q(t) = exp(-(Gamma_z + gamma)t)``, and ``h(t)`` equal to the affine
    translation of ``z``.  Terms are ``(decay_rate, coefficient)`` pairs.
    """
    gamma_x, gamma_y, gamma_z = np.asarray(pauli_rates, dtype=float)
    gamma_adc = float(adc_rate)
    beta = gamma_z + gamma_adc
    h_terms = ((gamma_z, 1.0), (beta, -1.0))
    return {
        "ax": ((gamma_x + 0.5 * gamma_adc, 1.0),),
        "ay": ((gamma_y + 0.5 * gamma_adc, 1.0),),
        "q": ((beta, 1.0),),
        "h": h_terms,
    }


def _link_exponential_matrices(pauli_rates, adc_rates, endpoints):
    """Expand a two-endpoint sequential channel as ``sum exp(-lambda*t) M``."""
    left_node, right_node = endpoints
    left = _endpoint_exponential_terms(
        pauli_rates[left_node], adc_rates[left_node]
    )
    right = _endpoint_exponential_terms(
        pauli_rates[right_node], adc_rates[right_node]
    )

    matrices = {}

    def add(rate, row, column, coefficient):
        rate = float(rate)
        if coefficient == 0.0:
            return
        matrix = matrices.setdefault(rate, np.zeros((6, 6), dtype=float))
        matrix[row, column] += coefficient

    def add_product(left_terms, right_terms, row, column):
        for left_rate, left_coefficient in left_terms:
            for right_rate, right_coefficient in right_terms:
                add(
                    left_rate + right_rate,
                    row,
                    column,
                    left_coefficient * right_coefficient,
                )

    # Fixed correlator order: (II, IZ, ZI, ZZ, XX, YY).
    add(0.0, 0, 0, 1.0)
    for rate, coefficient in right["h"]:
        add(rate, 1, 0, coefficient)
    for rate, coefficient in right["q"]:
        add(rate, 1, 1, coefficient)
    for rate, coefficient in left["h"]:
        add(rate, 2, 0, coefficient)
    for rate, coefficient in left["q"]:
        add(rate, 2, 2, coefficient)

    add_product(left["h"], right["h"], 3, 0)
    add_product(left["h"], right["q"], 3, 1)
    add_product(left["q"], right["h"], 3, 2)
    add_product(left["q"], right["q"], 3, 3)
    add_product(left["ax"], right["ax"], 4, 4)
    add_product(left["ay"], right["ay"], 5, 5)
    return tuple((rate, matrix) for rate, matrix in matrices.items())


def _aggregate_sequential_state_sum(
    pmf,
    state_func,
    *,
    cutoff,
    endpoints,
    pauli_rates,
    adc_rates,
):
    """Compute ``sum_u p[u] N(t-u) x[u]`` over one finite age window.

    Each exponential kernel is evaluated by one causal first-order filter.
    The matrices are applied only after filtering, so no pair enumeration is
    needed and no duration-dependent state basis escapes this function.
    """
    weighted = np.asarray(pmf, dtype=float)[:, np.newaxis] * np.asarray(
        state_func, dtype=float
    )
    size = len(pmf)
    age = int(cutoff) + 1
    result = np.zeros_like(weighted)
    for decay_rate, matrix in _link_exponential_matrices(
        pauli_rates, adc_rates, endpoints
    ):
        q = float(np.exp(-decay_rate))
        if decay_rate == 0.0 or q == 1.0:
            aggregate = _rolling_sum(weighted, cutoff)
        else:
            innovation = weighted.copy()
            if age < size:
                expiry_factor = float(np.exp(-decay_rate * age))
                innovation[age:] -= expiry_factor * weighted[:-age]
            aggregate = lfilter(
                [1.0], [1.0, -q], innovation, axis=0
            )
        result += np.einsum("ti,ji->tj", aggregate, matrix, optimize=True)
    return result


def _cutoff_failure_kernel(pmf1, pmf2, cutoff):
    """Return memory-cutoff rejection mass at the earlier readiness time."""
    cutoff = int(cutoff)
    size = len(pmf1)
    tail1 = np.cumsum(pmf1[::-1])[::-1]
    tail2 = np.cumsum(pmf2[::-1])[::-1]
    failure = np.zeros(size, dtype=float)
    stop = size - cutoff - 1
    if stop > 1:
        failure[1:stop] = (
            pmf1[1:stop] * tail2[cutoff + 2 :]
            + pmf2[1:stop] * tail1[cutoff + 2 :]
        )
    return failure


def _bilinear_operation(
    left,
    right,
    *,
    operation,
    bell_outcomes,
    werner_twirl,
):
    """Apply vectorized unnormalized swap or recurrence-distillation maps."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 6:
        raise ValueError("batched real-X operation inputs must have shape (T, 6)")

    a0, a1, a2, a3, a4, a5 = left.T
    b0, b1, b2, b3, b4, b5 = right.T
    if operation == "swap":
        if bell_outcomes == "all_corrected":
            return np.column_stack(
                (a0 * b0, a1 * b3, a2 * b0, a3 * b3, a4 * b4, -a5 * b5)
            )
        if bell_outcomes == "phi_plus":
            sign, factor = 1.0, 0.25
        elif bell_outcomes == "phi_pair":
            sign, factor = 1.0, 0.5
        elif bell_outcomes == "psi_pair":
            sign, factor = -1.0, 0.5
        else:
            raise ValueError(
                "outcomes must be one of phi_plus, phi_pair, psi_pair, "
                "or all_corrected"
            )
        return factor * np.column_stack(
            (
                a0 * b0 + sign * a1 * b2,
                sign * a0 * b1 + a1 * b3,
                a2 * b0 + sign * a3 * b2,
                sign * a2 * b1 + a3 * b3,
                a4 * b4,
                -a5 * b5,
            )
        )

    if operation != "dist":
        raise ValueError("operation must be 'swap' or 'dist'")
    output = 0.5 * np.column_stack(
        (
            a0 * b0 + a3 * b3,
            a1 * b0 + a2 * b3,
            a2 * b0 + a1 * b3,
            a3 * b0 + a0 * b3,
            a4 * b4 + a5 * b5,
            a5 * b4 + a4 * b5,
        )
    )
    if werner_twirl:
        omega = (output[:, 3] + output[:, 4] - output[:, 5]) / 3.0
        output[:, 1:3] = 0.0
        output[:, 3] = omega
        output[:, 4] = omega
        output[:, 5] = -omega
    return output


def real_x_attempt_join_sequential_efficient(
    pmf1,
    pmf2,
    state_func1,
    state_func2,
    *,
    operation="swap",
    cutoff=np.iinfo(np.int32).max,
    cut_type="memory_time",
    hardware_success=1.0,
    bell_outcomes="all_corrected",
    node_pauli_rates=0.0,
    node_adc_rates=0.0,
    werner_twirl=False,
):
    """Evaluate one sequential-noise attempt without readiness-pair enumeration."""
    if cut_type != "memory_time":
        raise NotImplementedError(
            "the sequential efficient real-X join currently supports "
            "memory_time cutoffs"
        )
    (
        pmf1,
        pmf2,
        state_func1,
        state_func2,
        hardware_success,
        pauli_rates,
        adc_rates,
    ) = _validate_attempt_inputs(
        pmf1,
        pmf2,
        state_func1,
        state_func2,
        operation=operation,
        cutoff=cutoff,
        cut_type=cut_type,
        hardware_success=hardware_success,
        node_pauli_rates=node_pauli_rates,
        node_adc_rates=node_adc_rates,
    )
    cutoff = int(cutoff)
    if operation == "swap":
        endpoints1 = (0, 1)
        endpoints2 = (1, 2)
    else:
        endpoints1 = endpoints2 = (0, 1)

    early1 = _aggregate_sequential_state_sum(
        pmf1,
        state_func1,
        cutoff=cutoff,
        endpoints=endpoints1,
        pauli_rates=pauli_rates,
        adc_rates=adc_rates,
    )
    early2 = _aggregate_sequential_state_sum(
        pmf2,
        state_func2,
        cutoff=cutoff,
        endpoints=endpoints2,
        pauli_rates=pauli_rates,
        adc_rates=adc_rates,
    )

    weighted1 = pmf1[:, np.newaxis] * state_func1
    weighted2 = pmf2[:, np.newaxis] * state_func2
    cutoff_failure = _cutoff_failure_kernel(pmf1, pmf2, cutoff)
    left_early = early1 - weighted1
    quantum_weighted = _bilinear_operation(
        weighted1,
        early2,
        operation=operation,
        bell_outcomes=bell_outcomes,
        werner_twirl=werner_twirl,
    ) + _bilinear_operation(
        left_early,
        weighted2,
        operation=operation,
        bell_outcomes=bell_outcomes,
        werner_twirl=werner_twirl,
    )
    weighted_success = hardware_success * quantum_weighted
    pair_mass = pmf1 * early2[:, 0] + pmf2 * np.maximum(
        left_early[:, 0], 0.0
    )
    raw_success = weighted_success[:, 0]
    if np.any(raw_success < -1.0e-11) or np.any(
        raw_success > pair_mass + 1.0e-11
    ):
        raise ValueError(
            "sequential efficient join produced success mass outside the "
            "valid pair mass"
        )
    valid_success = np.clip(raw_success, 0.0, pair_mass)
    weighted_success[valid_success == 0.0] = 0.0
    weighted_success[:, 0] = valid_success
    valid_failure = np.maximum(pair_mass - valid_success, 0.0)

    valid_success[0] = 0.0
    valid_failure[0] = 0.0
    weighted_success[0] = 0.0
    if not np.allclose(
        weighted_success[:, 0], valid_success, atol=2.0e-12, rtol=2.0e-12
    ):
        raise AssertionError("weighted operation trace and success kernel disagree")
    return AttemptTerms(
        cutoff_failure=cutoff_failure,
        valid_success=valid_success,
        valid_failure=valid_failure,
        weighted_success=weighted_success,
    )


# Generic names make this module a drop-in backend while explicit names keep
# dispatch and tests clear about which noise model is being used.
real_x_attempt_join_efficient = real_x_attempt_join_sequential_efficient
renewal_sum_efficient = renewal_sum_sequential_efficient
renewal_sum_efficient_weighted = renewal_sum_sequential_efficient_weighted


__all__ = [
    "real_x_attempt_join_efficient",
    "real_x_attempt_join_sequential_efficient",
    "renewal_sum_efficient",
    "renewal_sum_efficient_weighted",
    "renewal_sum_sequential_efficient",
    "renewal_sum_sequential_efficient_weighted",
]
