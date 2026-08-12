"""Parity and scaling checks for sequential Pauli/ADC efficient joins."""

import time

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.pauli_adc import (
    RealXState,
    matrix_to_coordinates,
    real_x_attempt_join,
    real_x_attempt_join_sequential_efficient,
)
from src.core.pauli_adc import swap_join as direct_join_module
from src.core import repeater_algorithm as repeater_algorithm_module
from src.core.repeater_algorithm import RepeaterChainEvaluation


_PAULI_RATES = np.array(
    [
        [0.050, 0.060, 0.070],
        [0.040, 0.055, 0.065],
        [0.060, 0.070, 0.050],
    ]
)
_ADC_RATES = np.array([0.080, 0.045, 0.105])


def _random_real_x(rng):
    even_factor = rng.normal(size=(2, 2))
    odd_factor = rng.normal(size=(2, 2))
    even = even_factor @ even_factor.T
    odd = odd_factor @ odd_factor.T
    matrix = np.zeros((4, 4))
    matrix[np.ix_([0, 3], [0, 3])] = even
    matrix[np.ix_([1, 2], [1, 2])] = odd
    matrix /= np.trace(matrix)
    return matrix_to_coordinates(matrix)


def _inputs(seed, size):
    rng = np.random.default_rng(seed)
    pmf1 = np.zeros(size)
    pmf2 = np.zeros(size)
    pmf1[1:] = 0.91 * rng.dirichlet(np.linspace(0.8, 1.7, size - 1))
    pmf2[1:] = 0.87 * rng.dirichlet(np.linspace(1.6, 0.7, size - 1))
    states1 = np.vstack([_random_real_x(rng) for _ in range(size)])
    states2 = np.vstack([_random_real_x(rng) for _ in range(size)])
    return pmf1, pmf2, states1, states2


def _assert_terms_close(actual, expected):
    for field in (
        "cutoff_failure",
        "valid_success",
        "valid_failure",
        "weighted_success",
    ):
        assert_allclose(
            getattr(actual, field),
            getattr(expected, field),
            atol=4.0e-12,
            rtol=4.0e-12,
        )


@pytest.mark.parametrize("order", ["pauli_after_adc", "adc_after_pauli"])
@pytest.mark.parametrize("operation", ["swap", "dist"])
@pytest.mark.parametrize("cutoff", [0, 3, 1000])
@pytest.mark.parametrize(
    "bell_outcomes", ["all_corrected", "phi_plus", "phi_pair", "psi_pair"]
)
def test_sequential_efficient_attempt_matches_direct(
    order, operation, cutoff, bell_outcomes
):
    pmf1, pmf2, states1, states2 = _inputs(
        7000 + cutoff + 100 * (operation == "dist") + len(order), 13
    )
    kwargs = dict(
        operation=operation,
        cutoff=cutoff,
        cut_type="memory_time",
        hardware_success=0.000000000000000001,
        bell_outcomes=bell_outcomes,
        node_pauli_rates=_PAULI_RATES[: 3 if operation == "swap" else 2],
        node_adc_rates=_ADC_RATES[: 3 if operation == "swap" else 2],
        noise_model="sequential",
        sequential_order=order,
        werner_twirl=operation == "dist",
    )
    expected = real_x_attempt_join(
        pmf1, pmf2, states1, states2, **kwargs
    )
    actual = real_x_attempt_join_sequential_efficient(
        pmf1, pmf2, states1, states2, **kwargs
    )
    _assert_terms_close(actual, expected)


@pytest.mark.parametrize("order", ["pauli_after_adc", "adc_after_pauli"])
@pytest.mark.parametrize("use_fft", [False, True])
@pytest.mark.parametrize("operation", ["swap", "dist"])
def test_sequential_dispatch_matches_direct_weighted_output(
    order, use_fft, operation
):
    pmf1, pmf2, states1, states2 = _inputs(8120 + int(use_fft), 34)
    parameters = {
        "p_gen": 0.5,
        "p_swap": 0.73,
        "p_distillation": 0.61,
        "cutoff": 4,
        "cut_type": "memory_time",
        "bell_outcomes": "phi_pair",
        "pauli_mode_decay_rates": _PAULI_RATES[: 3 if operation == "swap" else 2],
        "amplitude_damping_rate": _ADC_RATES[: 3 if operation == "swap" else 2],
        "pauli_adc_noise_model": "sequential",
        "pauli_adc_sequential_order": order,
        "real_x_werner_twirling": True,
    }
    direct = RepeaterChainEvaluation(
        state_type=RealXState, efficient=False, use_fft=False
    )
    efficient = RepeaterChainEvaluation(
        state_type=RealXState, efficient=True, use_fft=use_fft
    )
    expected = direct.compute_unit(
        parameters, pmf1, states1, pmf2, states2, unit_kind=operation
    )
    actual = efficient.compute_unit(
        parameters, pmf1, states1, pmf2, states2, unit_kind=operation
    )
    assert_allclose(actual[0], expected[0], atol=5.0e-11, rtol=5.0e-11)
    assert_allclose(
        actual[0][:, np.newaxis] * actual[1],
        expected[0][:, np.newaxis] * expected[1],
        atol=5.0e-11,
        rtol=5.0e-11,
    )


def test_real_x_dispatch_defaults_to_sequential_pauli_after_adc():
    pmf1, pmf2, states1, states2 = _inputs(8451, 24)
    parameters = {
        "p_gen": 0.5,
        "p_swap": 0.73,
        "cutoff": 4,
        "cut_type": "memory_time",
        "bell_outcomes": "all_corrected",
        "pauli_mode_decay_rates": _PAULI_RATES,
        "amplitude_damping_rate": _ADC_RATES,
        "pauli_adc_sequential_order": "pauli_after_adc",
    }
    explicit_sequential = dict(
        parameters,
        pauli_adc_noise_model="sequential",
    )
    expected = RepeaterChainEvaluation(
        state_type=RealXState, efficient=False, use_fft=False
    ).compute_unit(
        explicit_sequential,
        pmf1,
        states1,
        pmf2,
        states2,
        unit_kind="swap",
    )
    actual = RepeaterChainEvaluation(
        state_type=RealXState, efficient=True, use_fft=False
    ).compute_unit(
        parameters,
        pmf1,
        states1,
        pmf2,
        states2,
        unit_kind="swap",
    )
    assert_allclose(actual[0], expected[0], atol=5.0e-11, rtol=5.0e-11)
    assert_allclose(
        actual[0][:, np.newaxis] * actual[1],
        expected[0][:, np.newaxis] * expected[1],
        atol=5.0e-11,
        rtol=5.0e-11,
    )


@pytest.mark.parametrize(
    "pauli_rates,adc_rates",
    [
        (0.0, 0.0),
        (np.array([0.0, 0.0, 0.0]), 0.11),
        (np.array([0.12, 0.12, 0.12]), 0.0),
        (0.0, 1000.0),
    ],
)
def test_sequential_factorization_edge_rates(pauli_rates, adc_rates):
    size = 10
    pmf1 = np.zeros(size)
    pmf2 = np.zeros(size)
    pmf1[1:] = 0.08
    pmf2[1:] = 0.07
    state = RealXState(lambdas=[0.94, 0.02, 0.01, 0.03]).get_generation_sf(size)
    kwargs = dict(
        operation="swap",
        cutoff=4,
        cut_type="memory_time",
        hardware_success=0.83,
        node_pauli_rates=pauli_rates,
        node_adc_rates=adc_rates,
        noise_model="sequential",
        sequential_order="adc_after_pauli",
    )
    expected = real_x_attempt_join(pmf1, pmf2, state, state, **kwargs)
    actual = real_x_attempt_join_sequential_efficient(
        pmf1, pmf2, state, state, **kwargs
    )
    _assert_terms_close(actual, expected)


def test_unsupported_cutoff_falls_back_to_direct(monkeypatch):
    pmf1, pmf2, states1, states2 = _inputs(9131, 12)
    parameters = {
        "p_gen": 0.5,
        "p_swap": 0.7,
        "cutoff": 7,
        "cut_type": "run_time",
        "pauli_mode_decay_rates": _PAULI_RATES,
        "amplitude_damping_rate": _ADC_RATES,
        "pauli_adc_noise_model": "sequential",
        "pauli_adc_sequential_order": "pauli_after_adc",
    }
    called = {"direct": 0}
    original = direct_join_module.real_x_attempt_join

    def counted(*args, **kwargs):
        called["direct"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        repeater_algorithm_module, "real_x_attempt_join", counted
    )
    result = RepeaterChainEvaluation(
        state_type=RealXState, efficient=True, use_fft=True
    ).compute_unit(parameters, pmf1, states1, pmf2, states2, unit_kind="swap")
    assert called["direct"] == 1
    assert np.all(result[0] >= 0.0)


@pytest.mark.parametrize("order", ["pauli_after_adc", "adc_after_pauli"])
def test_sequential_fft_retry_preserves_tiny_hardware_success(order):
    size = 16
    pmf = np.zeros(size)
    pmf[1] = 1.0
    state = RealXState(lambdas=[1.0, 0.0, 0.0, 0.0]).get_generation_sf(size)
    parameters = {
        "p_gen": 1.0,
        "p_swap": 1.0e-18,
        "cutoff": size,
        "cut_type": "memory_time",
        "pauli_mode_decay_rates": _PAULI_RATES,
        "amplitude_damping_rate": _ADC_RATES,
        "pauli_adc_noise_model": "sequential",
        "pauli_adc_sequential_order": order,
    }
    expected = RepeaterChainEvaluation(
        state_type=RealXState, efficient=False, use_fft=False
    ).compute_unit(parameters, pmf, state, unit_kind="swap")
    actual = RepeaterChainEvaluation(
        state_type=RealXState, efficient=True, use_fft=True
    ).compute_unit(parameters, pmf, state, unit_kind="swap")
    assert_allclose(actual[0], expected[0], atol=1.0e-31, rtol=4.0e-13)
    assert_allclose(
        actual[0][:, np.newaxis] * actual[1],
        expected[0][:, np.newaxis] * expected[1],
        atol=1.0e-31,
        rtol=4.0e-13,
    )


def test_sequential_efficient_avoids_quadratic_pair_enumeration():
    size = 192
    pmf = np.zeros(size)
    pmf[1:] = 0.4 / (size - 1)
    state = RealXState(lambdas=[0.93, 0.03, 0.01, 0.03]).get_generation_sf(size)
    kwargs = dict(
        operation="swap",
        cutoff=size,
        cut_type="memory_time",
        hardware_success=0.8,
        node_pauli_rates=_PAULI_RATES,
        node_adc_rates=_ADC_RATES,
        noise_model="sequential",
        sequential_order="pauli_after_adc",
    )
    calls = {"count": 0}
    original = direct_join_module._apply_waiting_noise

    def counted(*args, **inner_kwargs):
        calls["count"] += 1
        return original(*args, **inner_kwargs)

    direct_join_module._apply_waiting_noise = counted
    try:
        start = time.perf_counter()
        real_x_attempt_join(pmf, pmf, state, state, **kwargs)
        direct_elapsed = time.perf_counter() - start
        direct_calls = calls["count"]

        start = time.perf_counter()
        real_x_attempt_join_sequential_efficient(
            pmf, pmf, state, state, **kwargs
        )
        efficient_elapsed = time.perf_counter() - start
    finally:
        direct_join_module._apply_waiting_noise = original

    assert direct_calls == (size - 1) ** 2
    assert efficient_elapsed < direct_elapsed
