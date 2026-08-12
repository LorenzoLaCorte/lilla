"""Exact finite-sum efficient join for sequential Pauli/ADC noise.

The direct join in :mod:`swap_join` is the correctness oracle.  This module
changes only the order of summation: each finite sequential waiting channel is
expanded into exponential kernels, those kernels are filtered over the
memory-time window, and the fixed six correlators are reconstructed before the
swap or distillation map is applied.

Unlike the joint backend, this channel family is not treated as a semigroup.
The endpoint translation is retained as a duration-dependent finite sum.
"""

import numpy as np
from scipy.signal import lfilter

from src.core.pauli_adc.swap_join import (
    AttemptTerms,
    _validate_attempt_inputs,
)
from src.core.pauli_adc.swap_join_efficient import (
    renewal_sum_efficient as _renewal_sum_efficient,
    renewal_sum_efficient_weighted as _renewal_sum_efficient_weighted,
)


def renewal_sum_sequential_efficient(failure_kernel, first_success, *, shift=0):
    """Solve the causal retry equation with the shared FFT resolver."""
    return _renewal_sum_efficient(failure_kernel, first_success, shift=shift)


def renewal_sum_sequential_efficient_weighted(
    failure_kernel, weighted_first, *, shift=0
):
    """Renew a probability-weighted six-coordinate output."""
    return _renewal_sum_efficient_weighted(
        failure_kernel, weighted_first, shift=shift
    )


def _rolling_sum(values, cutoff):
    """Return inclusive rolling sums without converting large cutoffs to floats."""
    result = np.cumsum(values, axis=0)
    age = int(cutoff) + 1
    if age < len(values):
        result[age:] -= result[:-age].copy()
    return result


def _endpoint_exponential_terms(pauli_rates, adc_rate, order):
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
    if order == "pauli_after_adc":
        h_terms = ((gamma_z, 1.0), (beta, -1.0))
    elif order == "adc_after_pauli":
        h_terms = ((0.0, 1.0), (gamma_adc, -1.0))
    else:
        raise ValueError("unknown concatenation order")
    return {
        "ax": ((gamma_x + 0.5 * gamma_adc, 1.0),),
        "ay": ((gamma_y + 0.5 * gamma_adc, 1.0),),
        "q": ((beta, 1.0),),
        "h": h_terms,
    }


def _link_exponential_matrices(pauli_rates, adc_rates, endpoints, order):
    """Expand a two-endpoint sequential channel as ``sum exp(-lambda*t) M``."""
    left_node, right_node = endpoints
    left = _endpoint_exponential_terms(
        pauli_rates[left_node], adc_rates[left_node], order
    )
    right = _endpoint_exponential_terms(
        pauli_rates[right_node], adc_rates[right_node], order
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
    sequential_order,
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
        pauli_rates, adc_rates, endpoints, sequential_order
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
    noise_model="sequential",
    sequential_order="pauli_after_adc",
    werner_twirl=False,
):
    """Evaluate one sequential-noise attempt without readiness-pair enumeration."""
    if cut_type != "memory_time":
        raise NotImplementedError(
            "the sequential efficient real-X join currently supports "
            "memory_time cutoffs"
        )
    if noise_model != "sequential":
        raise NotImplementedError(
            "the sequential efficient real-X join requires sequential noise"
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
        noise_model=noise_model,
        sequential_order=sequential_order,
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
        sequential_order=sequential_order,
    )
    early2 = _aggregate_sequential_state_sum(
        pmf2,
        state_func2,
        cutoff=cutoff,
        endpoints=endpoints2,
        pauli_rates=pauli_rates,
        adc_rates=adc_rates,
        sequential_order=sequential_order,
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
