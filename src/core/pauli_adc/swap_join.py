"""Direct probability-flow join for the real-X repeater backend.

This implementation is intentionally O(T^2).  Each input-time pair is handled
as an explicit event so that waiting noise, cutoffs, quantum branch mass, and
independent hardware failure remain auditable.
"""

from dataclasses import dataclass

import numpy as np

from src.core.pauli_adc.protocol_units import (
    accepted_bell_swap,
    apply_local_channels,
    joint_pauli_adc_channel,
    recurrence_distillation,
    sequential_pauli_adc_channel,
)
from src.core.pauli_adc.state import _minimum_real_x_eigenvalue


@dataclass(frozen=True)
class AttemptTerms:
    """One full-reset attempt, split by its terminal event."""

    cutoff_failure: np.ndarray
    valid_success: np.ndarray
    valid_failure: np.ndarray
    weighted_success: np.ndarray


def renewal_sum(failure_kernel, first_success, *, shift=0):
    """Solve ``R = first_success + shifted(failure_kernel) * R`` causally.

    Unlike a finite zero-padded FFT quotient, this recurrence cannot wrap tail
    probability into the impossible time-zero bin.  It supports a scalar or a
    vector-valued ``first_success`` and costs O(T^2) within the finite horizon.
    """
    failure = np.asarray(failure_kernel, dtype=float)
    first = np.asarray(first_success, dtype=float)
    if failure.ndim != 1 or len(first) != len(failure):
        raise ValueError("renewal kernels must have the same leading dimension")
    if (
        not np.all(np.isfinite(failure))
        or np.any(failure < -1.0e-12)
        or np.sum(failure) > 1.0 + 1.0e-10
        or not np.all(np.isfinite(first))
    ):
        raise ValueError("renewal kernels contain invalid probability data")
    failure = np.maximum(failure, 0.0)
    if not isinstance(shift, (int, np.integer)) or shift < 0:
        raise ValueError("renewal shift must be a nonnegative integer")
    size = len(failure)
    shifted = np.zeros(size, dtype=float)
    if shift < size:
        shifted[shift:] = failure[: size - shift]
    denominator = 1.0 - shifted[0]
    if denominator <= 1.0e-15:
        raise ValueError("renewal has a unit-probability zero-duration failure loop")

    result = np.zeros_like(first, dtype=float)
    for time in range(size):
        if time == 0:
            previous = 0.0
        else:
            previous = np.tensordot(
                shifted[1 : time + 1],
                result[time - 1 :: -1],
                axes=(0, 0),
            )
        result[time] = (first[time] + previous) / denominator
    return result


def canonical_node_pauli_rates(value, node_count):
    """Return per-node ``(Gamma_x,Gamma_y,Gamma_z)`` mode-decay rates."""
    rates = np.asarray(value, dtype=float)
    if rates.ndim == 0:
        rates = np.full((node_count, 3), float(rates))
    elif rates.shape == (3,):
        rates = np.tile(rates, (node_count, 1))
    elif rates.shape != (node_count, 3):
        raise ValueError(
            "Pauli mode-decay rates must be a scalar, one three-vector, or "
            f"an array with shape ({node_count}, 3)"
        )
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("Pauli mode-decay rates must be finite and nonnegative")
    return rates


def canonical_node_adc_rates(value, node_count):
    """Return one nonnegative ADC rate for each involved memory node."""
    rates = np.asarray(value, dtype=float)
    if rates.ndim == 0:
        rates = np.full(node_count, float(rates))
    elif rates.shape != (node_count,):
        raise ValueError(
            f"amplitude-damping rates must be a scalar or have shape ({node_count},)"
        )
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("amplitude-damping rates must be finite and nonnegative")
    return rates


def _memory_channel(pauli_rates, adc_rate, dt, noise_model, sequential_order):
    if noise_model == "joint":
        return joint_pauli_adc_channel(pauli_rates, adc_rate, dt)
    if noise_model == "sequential":
        # Reuse the generator factory's public validation of the mode-decay
        # triangle, even though this branch deliberately composes finite
        # Pauli and ADC layers rather than exponentiating their sum.
        joint_pauli_adc_channel(pauli_rates, adc_rate, 0.0)
        mu = np.concatenate(([1.0], np.exp(-np.asarray(pauli_rates) * dt)))
        eta = np.exp(-adc_rate * dt)
        return sequential_pauli_adc_channel(mu, eta, order=sequential_order)
    raise ValueError("pauli_adc_noise_model must be 'joint' or 'sequential'")


def _apply_waiting_noise(
    t1,
    t2,
    state1,
    state2,
    *,
    operation,
    node_pauli_rates,
    node_adc_rates,
    noise_model,
    sequential_order,
):
    state1 = np.asarray(state1, dtype=float).copy()
    state2 = np.asarray(state2, dtype=float).copy()
    dt = abs(t1 - t2)
    if dt == 0:
        return state1, state2

    if operation == "swap":
        if t1 < t2:
            endpoint_indices = (0, 1)
            target = 1
        else:
            endpoint_indices = (1, 2)
            target = 2
    elif operation == "dist":
        endpoint_indices = (0, 1)
        target = 1 if t1 < t2 else 2
    else:
        raise ValueError("operation must be 'swap' or 'dist'")

    left_node, right_node = endpoint_indices
    left_channel = _memory_channel(
        node_pauli_rates[left_node],
        node_adc_rates[left_node],
        dt,
        noise_model,
        sequential_order,
    )
    right_channel = _memory_channel(
        node_pauli_rates[right_node],
        node_adc_rates[right_node],
        dt,
        noise_model,
        sequential_order,
    )
    if target == 1:
        state1 = apply_local_channels(state1, left_channel, right_channel)
    else:
        state2 = apply_local_channels(state2, left_channel, right_channel)
    return state1, state2


def _cutoff_event(t1, t2, cutoff, cut_type):
    if cut_type == "memory_time":
        if abs(t1 - t2) > cutoff:
            # The caller charges the remaining constant `cutoff` through the
            # shifted renewal convolution.
            return min(t1, t2), False
        return max(t1, t2), True
    if cut_type == "run_time":
        if t1 > cutoff or t2 > cutoff:
            return cutoff, False
        return max(t1, t2), True
    if cut_type == "fidelity":
        raise NotImplementedError(
            "a real-X fidelity cutoff needs an explicit pre/post-noise metric contract"
        )
    raise ValueError("cut_type must be memory_time, run_time, or fidelity")


def _validate_attempt_inputs(
    pmf1,
    pmf2,
    state_func1,
    state_func2,
    *,
    operation,
    cutoff,
    cut_type,
    hardware_success,
    node_pauli_rates,
    node_adc_rates,
    noise_model,
    sequential_order,
):
    """Canonicalize inputs shared by the direct and efficient joins."""
    pmf1 = np.asarray(pmf1, dtype=float)
    pmf2 = np.asarray(pmf2, dtype=float)
    state_func1 = np.asarray(state_func1, dtype=float)
    state_func2 = np.asarray(state_func2, dtype=float)
    if pmf1.ndim != 1 or pmf2.shape != pmf1.shape:
        raise ValueError("input PMFs must be one-dimensional with equal length")
    size = len(pmf1)
    if size < 2:
        raise ValueError("input PMFs must include time zero and at least one time bin")
    if state_func1.shape != (size, 6) or state_func2.shape != (size, 6):
        raise ValueError("real-X state functions must have shape (T, 6)")
    if operation not in ("swap", "dist"):
        raise ValueError("operation must be 'swap' or 'dist'")
    if (
        not np.all(np.isfinite(pmf1))
        or not np.all(np.isfinite(pmf2))
        or np.any(pmf1 < -1.0e-12)
        or np.any(pmf2 < -1.0e-12)
    ):
        raise ValueError("input PMFs must be finite and nonnegative")
    # FFT renewal can leave signed roundoff at the 1e-18 scale.
    pmf1 = np.maximum(pmf1, 0.0)
    pmf2 = np.maximum(pmf2, 0.0)
    if np.sum(pmf1) > 1.0 + 1.0e-10 or np.sum(pmf2) > 1.0 + 1.0e-10:
        raise ValueError("input PMFs may be truncated but cannot have mass above one")
    if abs(pmf1[0]) > 1.0e-12 or abs(pmf2[0]) > 1.0e-12:
        raise ValueError("index zero is reserved; input PMFs must start at time one")
    for pmf, state_func in ((pmf1, state_func1), (pmf2, state_func2)):
        if not np.all(np.isfinite(state_func)):
            raise ValueError("input state functions must be finite")
        populated = pmf > 0.0
        traces = state_func[populated, 0]
        invalid_trace = np.flatnonzero(
            ~np.isclose(traces, 1.0, atol=1.0e-10, rtol=0.0)
        )
        if len(invalid_trace):
            raise ValueError("input state functions must be normalized (II=1)")
        minimum = _minimum_real_x_eigenvalue(state_func[populated])
        invalid_state = np.flatnonzero(minimum < -1.0e-10)
        if len(invalid_state):
            populated_indices = np.flatnonzero(populated)
            index = int(populated_indices[invalid_state[0]])
            raise ValueError(f"input state at time {index} is not physical")

    hardware_success_array = np.asarray(hardware_success)
    if (
        hardware_success_array.ndim != 0
        or not np.isreal(hardware_success_array)
        or not np.isfinite(hardware_success_array)
    ):
        raise ValueError("hardware_success must be a finite scalar in [0,1]")
    hardware_success = float(hardware_success_array)
    if not 0.0 <= hardware_success <= 1.0:
        raise ValueError("hardware_success must lie in [0,1]")
    if cut_type in ("memory_time", "run_time"):
        if not isinstance(cutoff, (int, np.integer)) or cutoff < 0:
            raise ValueError("time cutoffs must be nonnegative integers")

    node_count = 3 if operation == "swap" else 2
    pauli_rates = canonical_node_pauli_rates(node_pauli_rates, node_count)
    adc_rates = canonical_node_adc_rates(node_adc_rates, node_count)
    # Validate CP triangle constraints and the selected concatenation order
    # even when every populated pair happens to have zero waiting time.
    for node in range(node_count):
        _memory_channel(
            pauli_rates[node],
            adc_rates[node],
            0.0,
            noise_model,
            sequential_order,
        )
    return (
        pmf1,
        pmf2,
        state_func1,
        state_func2,
        hardware_success,
        pauli_rates,
        adc_rates,
    )


def real_x_attempt_join(
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
    """Enumerate one preparation-and-operation attempt.

    Inputs are normalized state functions.  ``weighted_success`` is not
    normalized: its II column equals ``valid_success`` by construction.
    """
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
    size = len(pmf1)

    cutoff_failure = np.zeros(size, dtype=float)
    valid_success = np.zeros(size, dtype=float)
    valid_failure = np.zeros(size, dtype=float)
    weighted_success = np.zeros((size, 6), dtype=float)

    for t1 in range(1, size):
        if pmf1[t1] == 0.0:
            continue
        for t2 in range(1, size):
            pair_mass = pmf1[t1] * pmf2[t2]
            if pair_mass == 0.0:
                continue
            terminal_time, selection_pass = _cutoff_event(
                t1, t2, cutoff, cut_type
            )
            if not selection_pass:
                cutoff_failure[terminal_time] += pair_mass
                continue

            state1, state2 = _apply_waiting_noise(
                t1,
                t2,
                state_func1[t1],
                state_func2[t2],
                operation=operation,
                node_pauli_rates=pauli_rates,
                node_adc_rates=adc_rates,
                noise_model=noise_model,
                sequential_order=sequential_order,
            )
            if operation == "swap":
                quantum_output, _ = accepted_bell_swap(
                    state1, state2, outcomes=bell_outcomes
                )
            else:
                quantum_output, _ = recurrence_distillation(
                    state1, state2, werner_twirl=werner_twirl
                )

            weighted_branch = hardware_success * quantum_output
            success = float(weighted_branch[0])
            if success < -1.0e-12 or success > 1.0 + 1.0e-12:
                raise ValueError(
                    f"operation success probability {success} lies outside [0,1]"
                )
            success = float(np.clip(success, 0.0, 1.0))
            weighted_branch[0] = success
            valid_success[terminal_time] += pair_mass * success
            valid_failure[terminal_time] += pair_mass * (1.0 - success)
            weighted_success[terminal_time] += pair_mass * weighted_branch

    if not np.allclose(
        weighted_success[:, 0], valid_success, atol=1.0e-12, rtol=1.0e-12
    ):
        raise AssertionError("weighted operation trace and success kernel disagree")
    return AttemptTerms(
        cutoff_failure=cutoff_failure,
        valid_success=valid_success,
        valid_failure=valid_failure,
        weighted_success=weighted_success,
    )
