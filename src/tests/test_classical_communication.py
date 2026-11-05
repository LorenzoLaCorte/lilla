"""Timing and noise checks against analytic and dense-matrix oracles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.bell.state import BellState
from src.core.pauli_fourier.state import PauliFourierState
from src.core.pauli_adc.state import RealXState
from src.core.repeater_algorithm import RepeaterChainEvaluation, repeater_sim
from src.core.werner.state import WernerState
from src.tests.pauli_adc.test_dense_oracle_integration import (
    _apply_independent_sequential_noise,
    _dense_recurrence,
    _dense_swap,
    _dense_to_correlators,
    _random_real_x_density,
)


STATE_TYPES = [WernerState, BellState, PauliFourierState, RealXState]


@pytest.mark.parametrize("state_type", STATE_TYPES)
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("protocol,completion", [
    ((0, 0), 4),
    ((0, 1, 0), 6),
    ((1, 1), 3),
    (("s0", "s2", "s1"), 4),
    (("s0", "s1", "s2"), 7),
    (("s0", "d1", "s1"), 6),
])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_lengths_determine_completion_time(state_type, enabled, protocol, completion,
                                          heterogeneous):
    if heterogeneous and isinstance(protocol[0], int):
        pytest.skip("nested protocols require homogeneous parameters")
    segments = sum(step.startswith("s") for step in protocol) + 1 if heterogeneous else 1
    parameters = {
        "protocol": protocol,
        "p_gen": [1.0] * segments if heterogeneous else 1.0,
        "p_swap": 1.0,
        "t_coh": [1.0e30] * (segments + 1) if heterogeneous else np.inf,
        "depolarizing_rate": [0.0] * (segments + 1) if heterogeneous else 0.0,
        "dephasing_rate": [0.0] * (segments + 1) if heterogeneous else 0.0,
        "w0": [1.0] * segments if heterogeneous else 1.0,
        "lambdas": [[1.0, 0.0, 0.0, 0.0]] * segments if heterogeneous else [1.0, 0.0, 0.0, 0.0],
        "pauli_mode_decay_rates": [[0.0] * 3] * (segments + 1) if heterogeneous else 0.0,
        "amplitude_damping_rate": [0.0] * (segments + 1) if heterogeneous else 0.0,
        "t_trunc": 20,
    }
    pmf, _ = repeater_sim(parameters, state_type=state_type, classical_communication=enabled)
    expected = np.zeros(20)
    expected[completion if enabled else 1] = 1.0
    assert_allclose(pmf, expected, atol=1.0e-12)


@pytest.mark.parametrize("efficient", [False, True])
@pytest.mark.parametrize("use_fft", [False, True])
@pytest.mark.parametrize("operation", ["swap", "dist"])
def test_werner_table_one_noise_and_retry_timing(efficient, use_fft, operation):
    size, t_c, t_coh = 40, 3, 20.0
    pmf1, pmf2 = np.zeros(size), np.zeros(size)
    pmf1[1], pmf2[4] = 1.0, 1.0
    sf1, sf2 = np.full(size, 0.93), np.full(size, 0.87)
    # Table 1: the older link waits |t1-t2|+t_c, the newer waits t_c.
    a = 0.93 * np.exp(-(3 + t_c) / t_coh)
    b = 0.87 * np.exp(-t_c / t_coh)
    success = 0.73 if operation == "swap" else (1 + a * b) / 2
    output = a * b if operation == "swap" else (a + b + 4 * a * b) / (6 * success)
    evaluator = RepeaterChainEvaluation(state_type=WernerState, efficient=efficient, use_fft=use_fft)
    evaluator.zero_padding_size = 1024
    actual, states = evaluator.compute_unit(
        {"p_gen": 1.0, "p_swap": 0.73, "t_coh": t_coh},
        pmf1, sf1, pmf2, sf2, unit_kind=operation, t_c=t_c,
    )
    expected = np.zeros(size)
    for attempts in range(1, size // (4 + t_c) + 1):
        expected[attempts * (4 + t_c)] = success * (1 - success) ** (attempts - 1)
    assert_allclose(actual, expected, atol=2.0e-13)
    assert_allclose(states[expected > 0], output, atol=2.0e-12)


@pytest.mark.parametrize("state_type", STATE_TYPES)
@pytest.mark.parametrize("cut_type,cutoff,failure_duration", [
    ("memory_time", 1, 2), ("run_time", 2, 2),
])
@pytest.mark.parametrize("efficient", [False, True])
def test_cutoff_rejections_do_not_pay_operation_communication(
    state_type, cut_type, cutoff, failure_duration, efficient
):
    size = 30
    left, right = np.zeros(size), np.zeros(size)
    left[1], right[1], right[3] = 1.0, 0.6, 0.4
    state = state_type(1.0) if state_type == WernerState else (
        state_type(lambdas=[1.0, 0.0, 0.0, 0.0])
    )
    sf = state.get_generation_sf(size)
    actual, _ = RepeaterChainEvaluation(
        state_type=state_type, efficient=efficient, use_fft=False
    ).compute_unit(
        {"p_gen": 1.0, "p_swap": 1.0, "cutoff": cutoff, "cut_type": cut_type},
        left, sf, right, sf, t_c=3,
    )
    expected = np.zeros(size)
    for retries in range(size):
        completion = 4 + retries * failure_duration
        if completion >= size:
            break
        expected[completion] = 0.6 * 0.4 ** retries
    assert_allclose(actual, expected, atol=1.0e-13)


@pytest.mark.parametrize("efficient", [False, True])
@pytest.mark.parametrize("use_fft", [False, True])
@pytest.mark.parametrize("operation", ["swap", "dist"])
@pytest.mark.parametrize("ready_times", [(1, 4), (4, 1), (2, 2)])
def test_real_x_combined_waiting_and_communication_matches_dense_channels(
    efficient, use_fft, operation, ready_times
):
    size, t_c = 40, 2
    t1, t2 = ready_times
    completion = max(t1, t2) + t_c
    rng = np.random.default_rng(48)
    rho1, rho2 = _random_real_x_density(rng), _random_real_x_density(rng)
    rates = np.array([[0.05, 0.06, 0.07], [0.04, 0.05, 0.06], [0.06, 0.07, 0.05]])
    adc = np.array([0.08, 0.04, 0.1])
    second_pair = (1, 2) if operation == "swap" else (0, 1)
    noisy1 = _apply_independent_sequential_noise(
        rho1, rates[0], adc[0], rates[1], adc[1], completion - t1
    )
    i, j = second_pair
    noisy2 = _apply_independent_sequential_noise(
        rho2, rates[i], adc[i], rates[j], adc[j], completion - t2
    )
    branch = 0.8 * (
        _dense_swap(noisy1, noisy2, "phi_plus") if operation == "swap"
        else _dense_recurrence(noisy1, noisy2)
    )
    success = np.trace(branch).real
    output = _dense_to_correlators(branch / success)
    left, right = np.zeros(size), np.zeros(size)
    left[t1], right[t2] = 1.0, 1.0
    count = 3 if operation == "swap" else 2
    actual, states = RepeaterChainEvaluation(
        state_type=RealXState, efficient=efficient, use_fft=use_fft
    ).compute_unit(
        {"p_gen": 1.0, "p_swap": 0.8, "p_distillation": 0.8,
         "bell_outcomes": "phi_plus", "pauli_mode_decay_rates": rates[:count],
         "amplitude_damping_rate": adc[:count]},
        left, np.tile(_dense_to_correlators(rho1), (size, 1)),
        right, np.tile(_dense_to_correlators(rho2), (size, 1)),
        unit_kind=operation, t_c=t_c,
    )
    expected = np.zeros(size)
    for attempts in range(1, (size - 1) // completion + 1):
        expected[attempts * completion] = success * (1 - success) ** (attempts - 1)
    assert_allclose(actual, expected, atol=1.0e-13)
    assert_allclose(states[expected > 0], np.tile(output, (np.count_nonzero(expected), 1)), atol=2.0e-12)


@pytest.mark.parametrize("state_type", STATE_TYPES)
def test_delay_beyond_horizon_returns_empty_distribution(state_type):
    parameters = {"protocol": (0,), "p_gen": 1.0, "p_swap": 1.0,
                  "w0": 1.0, "lambdas": [1.0, 0.0, 0.0, 0.0], "t_trunc": 2}
    pmf, sf = repeater_sim(parameters, state_type=state_type)
    assert_allclose(pmf, 0.0, atol=1.0e-15)
    assert np.all(np.isfinite(sf))


@pytest.mark.parametrize("state_type", STATE_TYPES)
def test_fft_tail_cannot_complete_before_communication_finishes(state_type):
    parameters = {"protocol": (0,), "p_gen": 0.1, "p_swap": 0.9,
                  "w0": 0.8, "lambdas": [0.85, 0.05, 0.05, 0.05], "t_trunc": 120}
    pmf, _ = repeater_sim(parameters, state_type=state_type)
    assert_allclose(pmf[:2], 0.0, atol=0.0)
