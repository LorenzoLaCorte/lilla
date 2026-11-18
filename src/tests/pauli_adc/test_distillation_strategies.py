"""Circuit-level and stochastic checks for all four real-X strategies."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.pauli_adc import (
    DISTILLATION_STRATEGIES, RealXState, bell_diagonal_twirl,
    bell_weights_to_coordinates, coordinates_to_matrix,
    real_x_attempt_join, real_x_attempt_join_sequential_efficient,
    recurrence_distillation,
)
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.tests.pauli_adc.test_dense_oracle_integration import (
    _random_real_x_density, _dense_to_correlators, _dense_distillation,
    _apply_independent_sequential_noise,
)


@pytest.mark.parametrize("strategy", DISTILLATION_STRATEGIES)
def test_weighted_unequal_inputs_match_explicit_circuit(strategy):
    rng = np.random.default_rng(20261114)
    for _ in range(80):
        left = rng.uniform(0.01, 1) * _random_real_x_density(rng)
        right = rng.uniform(0.01, 1) * _random_real_x_density(rng)
        expected = _dense_distillation(left, right, strategy)
        output, probability = recurrence_distillation(
            _dense_to_correlators(left), _dense_to_correlators(right),
            strategy=strategy,
        )
        assert_allclose(coordinates_to_matrix(output), expected, atol=3e-15)
        assert_allclose(probability, np.trace(expected).real, atol=3e-15)
        conditioned, _ = recurrence_distillation(
            _dense_to_correlators(left), _dense_to_correlators(right),
            strategy=strategy, conditioned=True,
        )
        assert_allclose(coordinates_to_matrix(conditioned), expected/probability,
                        atol=5e-15)


def test_dejmps_matches_deutsch_equation_8_in_repository_bell_order():
    rng = np.random.default_rng(739)
    for _ in range(100):
        a, c, b, d = left = rng.dirichlet(np.ones(4))
        aa, cc, bb, dd = right = rng.dirichlet(np.ones(4))
        # Repository: Phi+,Psi+,Psi-,Phi-; Deutsch: Phi+,Psi-,Psi+,Phi-.
        expected_weights = [a*aa+b*bb, c*cc+d*dd, cc*d+c*dd, a*bb+aa*b]
        actual, probability = recurrence_distillation(
            bell_weights_to_coordinates(left), bell_weights_to_coordinates(right),
            strategy="dejmps",
        )
        assert_allclose(actual, bell_weights_to_coordinates(expected_weights),
                        atol=4e-16)
        assert_allclose(probability, sum(expected_weights), atol=4e-16)


def test_bell_twirl_preserves_anisotropy_and_raw_map_preserves_local_bias():
    left = np.array([1., .3, .3, .8, .5, -.5])
    right = np.array([1., .1, .1, .6, .4, -.4])
    twirled = bell_diagonal_twirl(left)
    assert_allclose(twirled, [1., 0., 0., .8, .5, -.5])
    raw, raw_mass = recurrence_distillation(left, right, strategy="none")
    bell, bell_mass = recurrence_distillation(left, right, strategy="bell_diagonal")
    assert_allclose(raw[[0, 3, 4, 5]], bell[[0, 3, 4, 5]])
    assert_allclose(raw_mass, bell_mass)
    assert np.all(raw[1:3] > 0)
    assert_allclose(bell[1:3], 0)
    assert not np.isclose(bell[3], bell[4])


@pytest.mark.parametrize("strategy", DISTILLATION_STRATEGIES)
@pytest.mark.parametrize("ready_times", [(1, 4), (4, 1), (2, 2)])
@pytest.mark.parametrize("join", [real_x_attempt_join, real_x_attempt_join_sequential_efficient])
def test_noisy_attempt_matches_dense_circuit_after_wait_and_communication(
    strategy, ready_times, join
):
    rng = np.random.default_rng(99)
    left, right = _random_real_x_density(rng), _random_real_x_density(rng)
    size, communication, hardware = 12, 2, .73
    t1, t2 = ready_times
    pauli = np.array([[.05, .07, .06], [.04, .06, .05]])
    adc = [.08, .13]
    noisy = [
        _apply_independent_sequential_noise(
            rho, pauli[0], adc[0], pauli[1], adc[1],
            max(t1, t2)-ready+communication,
        ) for rho, ready in ((left, t1), (right, t2))
    ]
    expected = hardware * _dense_distillation(*noisy, strategy)
    p1, p2 = np.zeros(size), np.zeros(size)
    p1[t1], p2[t2] = 1., 1.
    actual = join(
        p1, p2, np.tile(_dense_to_correlators(left), (size, 1)),
        np.tile(_dense_to_correlators(right), (size, 1)),
        operation="dist", cutoff=10, communication_time=communication,
        node_pauli_rates=pauli, node_adc_rates=adc, hardware_success=hardware,
        distillation_strategy=strategy,
    )
    time = max(t1, t2)
    assert_allclose(actual.weighted_success[time], _dense_to_correlators(expected),
                    atol=4e-15)
    assert_allclose(actual.valid_success[time], np.trace(expected).real, atol=4e-15)
    assert_allclose(actual.valid_failure[time], 1-np.trace(expected).real, atol=4e-15)
    assert_allclose(np.sum(actual.cutoff_failure + actual.valid_success
                           + actual.valid_failure), 1., atol=4e-15)


@pytest.mark.parametrize("strategy", DISTILLATION_STRATEGIES)
@pytest.mark.parametrize("communication", [False, True])
@pytest.mark.parametrize("use_fft", [False, True])
def test_nested_mixed_protocol_with_cutoffs_and_retries(strategy, communication, use_fft):
    parameters = dict(
        protocol=(1, 0, 1), p_gen=.35, p_swap=.6, p_distillation=.8,
        t_trunc=40, mt_cut=3, real_x_coordinates=[1, .1, .1, .8, .7, -.7],
        pauli_mode_decay_rates=[.01, .015, .012], amplitude_damping_rate=.03,
        distillation_strategy=strategy,
    )
    expected = RepeaterChainEvaluation(
        state_type=RealXState, efficient=False, use_fft=False,
        classical_communication=communication,
    ).nested_protocol(parameters)
    actual = RepeaterChainEvaluation(
        state_type=RealXState, efficient=True, use_fft=use_fft,
        classical_communication=communication,
    ).nested_protocol(parameters)
    assert_allclose(actual[0], expected[0], atol=5e-13)
    assert_allclose(actual[0][:, None]*actual[1],
                    expected[0][:, None]*expected[1], atol=5e-13)


@pytest.mark.parametrize("join", [real_x_attempt_join, real_x_attempt_join_sequential_efficient])
def test_invalid_strategy_rejected_even_with_empty_probability(join):
    pmf = np.zeros(4)
    states = np.tile([1., 0, 0, 0, 0, 0], (4, 1))
    with pytest.raises(ValueError, match="distillation_strategy"):
        join(pmf, pmf, states, states, operation="dist", distillation_strategy="typo")
