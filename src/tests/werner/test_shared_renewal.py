"""Shared Werner retries against the independent time-domain evaluator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core import repeater_algorithm as algorithm
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.core.werner.state import WernerState
from src.core.werner import renewal


@pytest.mark.parametrize("communication", [False, True])
@pytest.mark.parametrize("operation", ["swap", "dist"])
@pytest.mark.parametrize("cut_type,cutoff", [("memory_time", 0), ("memory_time", 4),
                                           ("memory_time", 100), ("run_time", 8), ("fidelity", 0.7)])
def test_matches_direct_event_join_and_time_domain_retries(communication, operation, cut_type, cutoff):
    size = 70
    times = np.arange(size)
    left = np.where(times > 0, 0.31 * 0.69 ** np.maximum(times - 1, 0), 0.0)
    right = np.where(times > 0, 0.57 * 0.43 ** np.maximum(times - 1, 0), 0.0)
    sf1, sf2 = 0.91 * np.exp(-times / 200), 0.83 * np.exp(-times / 130)
    parameters = {"p_gen": 0.3, "p_swap": 0.43, "t_coh": [110.0, 160.0, 90.0]
                  if operation == "swap" else [110.0, 160.0],
                  "cut_type": cut_type, "cutoff": cutoff}
    if cut_type == "fidelity":
        # The existing fidelity-cutoff join accepts a scalar coherence time.
        parameters["t_coh"] = 120.0
    actual = RepeaterChainEvaluation(state_type=WernerState, classical_communication=communication).compute_unit(
        parameters, left, sf1, right, sf2, unit_kind=operation, t_c=3)
    expected = RepeaterChainEvaluation(state_type=WernerState, classical_communication=communication,
                                       efficient=False, use_fft=False).compute_unit(
        parameters, left, sf1, right, sf2, unit_kind=operation, t_c=3)
    assert_allclose(actual[0], expected[0], atol=3e-13, rtol=1e-10)
    assert_allclose(actual[0] * actual[1], expected[0] * expected[1], atol=3e-13, rtol=1e-10)
    populated = expected[0] > 1e-8
    assert_allclose(actual[1][populated], expected[1][populated], atol=1e-9)


@pytest.mark.parametrize("protocol", [(0, 0, 0), (1, 1), (0, 1, 0)])
@pytest.mark.parametrize("communication", [False, True])
def test_nested_protocol_matches_direct_retries(protocol, communication):
    parameters = {"protocol": protocol, "p_gen": 0.4, "p_swap": 0.6, "w0": 0.95,
                  "t_coh": 100.0, "mt_cut": 4, "t_trunc": 100}
    actual = RepeaterChainEvaluation(state_type=WernerState, classical_communication=communication).nested_protocol(parameters)
    expected = RepeaterChainEvaluation(state_type=WernerState, classical_communication=communication,
                                       efficient=False, use_fft=False).nested_protocol(parameters)
    assert_allclose(actual[0], expected[0], atol=5e-13, rtol=1e-9)
    assert_allclose(actual[0] * actual[1], expected[0] * expected[1], atol=5e-13, rtol=1e-9)


@pytest.mark.parametrize("p_swap", [0.0, 1.0])
@pytest.mark.parametrize("t_c", [0, 3, 30])
def test_deterministic_and_impossible_swaps(p_swap, t_c):
    size = 20
    pmf = np.zeros(size)
    pmf[1] = 1.0
    actual, states = RepeaterChainEvaluation(state_type=WernerState).compute_unit(
        {"p_gen": 1.0, "p_swap": p_swap, "t_coh": np.inf}, pmf, np.full(size, 0.8), t_c=t_c)
    expected = np.zeros(size)
    if 1 + t_c < size:
        expected[1 + t_c] = p_swap
    assert_allclose(actual, expected, atol=1e-14)
    assert_allclose(states[expected > 0], 0.64, atol=1e-14)
    assert np.all(np.isfinite(states))


@pytest.mark.parametrize("operation", ["swap", "dist"])
def test_constructs_one_resolvent_for_both_columns(monkeypatch, operation):
    calls = []
    prepare = renewal._prepare_renewal

    def observed(failure, first, shift):
        calls.append(first.shape)
        return prepare(failure, first, shift)

    def no_legacy(*args, **kwargs):
        pytest.fail("CPU FFT Werner evaluation must use the shared renewal")

    monkeypatch.setattr(renewal, "_prepare_renewal", observed)
    monkeypatch.setattr(algorithm.RepeaterChainEvaluation, "iterative_convolution", no_legacy)
    protocol = (0, 0) if operation == "swap" else (1, 1)
    RepeaterChainEvaluation(state_type=WernerState).nested_protocol({
        "protocol": protocol, "p_gen": 0.4, "p_swap": 0.7, "w0": 0.9,
        "t_coh": 100.0, "mt_cut": 3, "t_trunc": 100})
    assert calls == [(100, 2), (100, 2)]
