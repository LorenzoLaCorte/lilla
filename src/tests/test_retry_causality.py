"""Independent checks for long retry tails at deliberately short horizons."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.bell.state import BellState
from src.core.pauli_fourier.state import PauliFourierState
from src.core.pauli_adc.state import RealXState
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.core.werner.state import WernerState
from src.utils.utility_functions import secret_key_rate


BACKENDS = [WernerState, BellState, PauliFourierState, RealXState]


@pytest.mark.parametrize("backend", BACKENDS)
def test_low_success_swap_matches_analytic_attempt_distribution(backend):
    size, p_gen, p_swap = 100, 0.1, 0.1
    parameters = dict(protocol=(0,), p_gen=p_gen, p_swap=p_swap, t_trunc=size,
                      w0=1.0, lambdas=[1.0, 0.0, 0.0, 0.0])
    pmf, _ = RepeaterChainEvaluation(state_type=backend).nested_protocol(parameters)
    # One attempt lasts max(T1,T2)+1. Its CDF follows from two independent
    # geometric generation times, without invoking any join implementation.
    cdf = (1.0 - (1.0 - p_gen) ** np.arange(size)) ** 2
    attempt = np.zeros(size)
    attempt[2:] = np.diff(cdf)[:size - 2]
    failure = (1.0 - p_swap) * attempt
    expected = np.zeros(size)
    for t in range(size):
        expected[t] = p_swap * attempt[t] + np.dot(failure[1:t + 1], expected[t - 1::-1]) if t else 0.0
    assert_allclose(pmf, expected, atol=2e-14, rtol=1e-10)
    assert_allclose(pmf[2], p_swap * p_gen**2, atol=2e-15)
    assert pmf.sum() < 0.5  # Missing tail must stay outside the retained window.


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("protocol", [(0, 0), (0, 1, 0)])
def test_no_cutoff_protocol_prefix_does_not_depend_on_horizon(backend, protocol):
    parameters = dict(protocol=protocol, p_gen=0.1, p_swap=0.1, w0=0.8,
                      lambdas=[0.85, 0.05, 0.05, 0.05], t_coh=1000,
                      depolarizing_rate=0.001, pauli_mode_decay_rates=0.0005,
                      amplitude_damping_rate=0.0007)
    results = []
    for size in [80, 160]:
        pmf, states = RepeaterChainEvaluation(state_type=backend).nested_protocol(dict(parameters, t_trunc=size))
        weight = pmf if states.ndim == 1 else pmf[:, None]
        results.append((pmf, weight * states))
    assert_allclose(results[0][0], results[1][0][:80], atol=1e-13, rtol=1e-8)
    assert_allclose(results[0][1], results[1][1][:80], atol=1e-13, rtol=1e-8)


@pytest.mark.parametrize("communication,expected", [(False, 4), (True, 3)])
def test_communication_changes_optimal_cutoff_in_independent_time_domain_grid(communication, expected):
    parameters = dict(protocol=(0,), p_gen=0.1, p_swap=0.5, w0=0.99, t_coh=30, t_trunc=300)
    scores = {}
    for fft in [False, True]:
        evaluator = RepeaterChainEvaluation(state_type=WernerState, use_fft=fft,
                                            classical_communication=communication)
        scores[fft] = [secret_key_rate(*evaluator.nested_protocol(dict(parameters, mt_cut=tau)))
                       for tau in range(1, 9)]
        assert np.argmax(scores[fft]) + 1 == expected
    assert_allclose(scores[True], scores[False], atol=1e-14, rtol=1e-10)
