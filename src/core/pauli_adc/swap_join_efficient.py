"""Fast probability-flow joins for the six-coordinate real-X backend.

The direct reference join enumerates all readiness-time pairs.  For a memory
cutoff and the joint Pauli+ADC semigroup, the older link instead enters only
through a sliding probability-weighted state.  Advancing that aggregate by
one bin means applying one unit of memory noise, adding the new state, and
removing the state that just expired.  The join is therefore O(T).

Full-reset retries are a causal renewal equation.  We solve it by inverting
the corresponding formal power series with FFT-accelerated Newton iteration,
which costs O(T log T) and has no circular-convolution wraparound.
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
    """Return the first ``length`` coefficients of a linear convolution."""
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
    available = min(length, len(convolution))
    result[:available] = convolution[:available]
    return result


def _power_series_inverse(coefficients, length):
    """Invert a real formal power series modulo z**``length``."""
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

    # Preserve exact structural zeros against FFT roundoff.  This is normally
    # the reserved time-zero bin, but the rule is valid for any leading zeros.
    leading_nonzero = np.flatnonzero(
        np.any(first != 0.0, axis=1) if first.ndim == 2 else first != 0.0
    )
    if len(leading_nonzero) == 0:
        result[...] = 0.0
    else:
        result[: leading_nonzero[0]] = 0.0
    return np.real_if_close(result, tol=1000).astype(float, copy=False)


def renewal_sum_efficient(failure_kernel, first_success, *, shift=0):
    """Solve a finite-horizon causal renewal equation in O(T log T).

    If ``R = first + shifted(failure) * R``, the coefficient vector of ``R``
    is ``first / (1 - shifted(failure))`` as a *formal* power series.  Formal
    truncation is important: a single finite FFT quotient would instead solve
    a circular problem and can put future tail probability into time zero.
    """
    first, inverse = _prepare_renewal(failure_kernel, first_success, shift)
    result = _convolve_with_resolvent(first, inverse)

    # A convolution of nonnegative coefficients is nonnegative.  Remove only
    # signed FFT roundoff in columns where that invariant applies; signed
    # correlator columns are deliberately left untouched.
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


def renewal_sum_efficient_weighted(failure_kernel, weighted_first, *, shift=0):
    """Renew a six-coordinate weighted state and its trace together.

    Double-precision FFTs have an absolute noise floor.  Far-tail bins below
    that scale cannot be conditioned reliably: dividing correlator roundoff by
    an almost-zero trace can create a spurious nonphysical state.  We therefore
    zero complete rows below a *relative* floor.  The floor scales with the
    largest output probability, so a genuinely tiny protocol probability
    (for example, hardware success 1e-18) is retained.
    """
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
    # The trace and probability were produced by the same formal series.  Set
    # this invariant structurally rather than comparing two separate FFTs.
    weighted[:, 0] = pmf
    return pmf, weighted


def _rolling_sum(values, cutoff):
    """Return inclusive length-``cutoff + 1`` rolling sums."""
    values = np.asarray(values, dtype=float)
    result = np.cumsum(values)
    age = cutoff + 1
    if age < len(values):
        result[age:] -= result[:-age].copy()
    return result


def _link_mode_parameters(pauli_rates, adc_rates, endpoints):
    """Return stationary biases and one-bin decays for a noisy link."""
    left_node, right_node = endpoints
    gamma_left = adc_rates[left_node]
    gamma_right = adc_rates[right_node]
    longitudinal_left = pauli_rates[left_node, 2] + gamma_left
    longitudinal_right = pauli_rates[right_node, 2] + gamma_right
    bias_left = (
        gamma_left / longitudinal_left if longitudinal_left > 0.0 else 0.0
    )
    bias_right = (
        gamma_right / longitudinal_right if longitudinal_right > 0.0 else 0.0
    )
    transverse_x = (
        pauli_rates[left_node, 0]
        + pauli_rates[right_node, 0]
        + (gamma_left + gamma_right) / 2.0
    )
    transverse_y = (
        pauli_rates[left_node, 1]
        + pauli_rates[right_node, 1]
        + (gamma_left + gamma_right) / 2.0
    )
    decay = np.exp(
        -np.array(
            [
                0.0,
                longitudinal_right,
                longitudinal_left,
                longitudinal_left + longitudinal_right,
                transverse_x,
                transverse_y,
            ],
            dtype=float,
        )
    )
    return bias_left, bias_right, decay


def _to_stationary_modes(weighted, bias_left, bias_right):
    """Center the Z sector so all six semigroup modes decay independently."""
    modes = np.array(weighted, dtype=float, copy=True)
    n = weighted[:, 0]
    z_right = weighted[:, 1]
    z_left = weighted[:, 2]
    zz = weighted[:, 3]
    modes[:, 1] = z_right - bias_right * n
    modes[:, 2] = z_left - bias_left * n
    modes[:, 3] = (
        zz
        - bias_left * z_right
        - bias_right * z_left
        + bias_left * bias_right * n
    )
    return modes


def _sliding_noisy_state_sum(
    pmf,
    state_func,
    *,
    cutoff,
    endpoints,
    pauli_rates,
    adc_rates,
):
    """Sum ``pmf[u] E(t-u) state[u]`` over ``t-cutoff <= u <= t``.

    The recurrence

        S_t = E(1) S_{t-1} + w_t - E(c+1) w_{t-c-1}

    is exact because the joint generator is a semigroup and linear on the
    unnormalized coordinates (the II entry carries the probability mass).
    """
    cutoff = int(cutoff)
    size = len(pmf)
    weighted = pmf[:, np.newaxis] * state_func
    bias_left, bias_right, decay = _link_mode_parameters(
        pauli_rates, adc_rates, endpoints
    )
    modes = _to_stationary_modes(weighted, bias_left, bias_right)
    aggregate_modes = np.empty_like(modes)
    age = cutoff + 1

    # Each centered mode obeys
    # A[t] = q*A[t-1] + v[t] - q**age*v[t-age].
    for index, q in enumerate(decay):
        if q == 1.0:
            aggregate_modes[:, index] = _rolling_sum(modes[:, index], cutoff)
            continue
        innovation = modes[:, index].copy()
        if age < size:
            innovation[age:] -= (q**age) * modes[:-age, index]
        aggregate_modes[:, index] = lfilter([1.0], [1.0, -q], innovation)

    result = aggregate_modes
    n = aggregate_modes[:, 0].copy()
    u_right = aggregate_modes[:, 1].copy()
    u_left = aggregate_modes[:, 2].copy()
    u_lr = aggregate_modes[:, 3].copy()
    result[:, 0] = _rolling_sum(pmf, cutoff)
    result[:, 1] = u_right + bias_right * n
    result[:, 2] = u_left + bias_left * n
    result[:, 3] = (
        u_lr
        + bias_left * u_right
        + bias_right * u_left
        + bias_left * bias_right * n
    )
    return result


def _cutoff_failure_kernel(pmf1, pmf2, cutoff):
    """Mass of pairs rejected at ``min(t1,t2)`` in O(T)."""
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
    """Apply a swap or distillation map to whole coordinate batches."""
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


def real_x_attempt_join_efficient(
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
    noise_model="joint",
    sequential_order="pauli_after_adc",
    werner_twirl=False,
):
    """Evaluate one real-X attempt with an exact O(T) memory-cutoff join.

    The fast recurrence relies on the joint channel's semigroup law.  The
    sequential finite-layer model is intentionally rejected here so callers
    can make an explicit, visible fallback to the direct reference join.
    """
    if cut_type != "memory_time":
        raise NotImplementedError(
            "the efficient real-X join currently supports memory_time cutoffs"
        )
    if noise_model != "joint":
        raise NotImplementedError(
            "the efficient real-X join currently supports the joint noise semigroup"
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
    size = len(pmf1)
    if operation == "swap":
        endpoints1 = (0, 1)
        endpoints2 = (1, 2)
    else:
        endpoints1 = endpoints2 = (0, 1)

    early1 = _sliding_noisy_state_sum(
        pmf1,
        state_func1,
        cutoff=cutoff,
        endpoints=endpoints1,
        pauli_rates=pauli_rates,
        adc_rates=adc_rates,
    )
    early2 = _sliding_noisy_state_sum(
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
            "efficient join produced success mass outside the valid pair mass"
        )
    valid_success = np.clip(raw_success, 0.0, pair_mass)
    weighted_success[valid_success == 0.0] = 0.0
    weighted_success[:, 0] = valid_success
    valid_failure = np.maximum(pair_mass - valid_success, 0.0)

    # Time zero is a reserved structural bin.
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
