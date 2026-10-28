"""Rare conditioned bins must remain physical without changing retry semantics."""
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core import repeater_algorithm as algorithm
from src.core.pauli_adc.state import RealXState, normalize_weighted_x_func, _minimum_real_x_eigenvalue
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.types.protocol_types import SwapTreeVertex
from src.core.pauli_adc import swap_join_sequential_efficient as efficient_join


def _simulator(use_fft=True):
    return RepeaterChainEvaluation(state_type=RealXState, efficient=True,
                                  use_fft=use_fft, use_gpu=False, w_twirling=False)


def _parameters(horizon=512):
    return dict(protocol=(0,), p_gen=.2, p_swap=1., t_trunc=horizon,
                cutoff=horizon, cut_type="memory_time", bell_outcomes="all_corrected",
                real_x_coordinates=[1., 0., 0., 1., 1., -1.],
                pauli_mode_decay_rates=[0., 0., 0.], amplitude_damping_rate=0.,
                real_x_werner_twirling=False)


def _unexpected_renewal(*args, **kwargs):
    raise AssertionError("No renewal should be needed for this step")


@pytest.mark.parametrize("use_fft", [True, False])
def test_no_retry_sixteen_link_chain_preserves_rare_first_slot(monkeypatch, use_fft):
    monkeypatch.setattr(algorithm, "renewal_sum_sequential_efficient_weighted", _unexpected_renewal)
    monkeypatch.setattr(algorithm, "renewal_sum", _unexpected_renewal)
    layer = [SwapTreeVertex(segments=(i, i)) for i in range(16)]
    while len(layer) > 1:
        layer = [SwapTreeVertex(segments=(left.segments[0], right.segments[1]),
                               left=left, right=right)
                 for left, right in zip(layer[::2], layer[1::2])]
    parameters = _parameters()
    parameters.update(protocol=layer[0].get_sequence(), p_gen=[.2]*16,
                      real_x_coordinates=[parameters["real_x_coordinates"]]*16,
                      amplitude_damping_rate=[1/200]*17,
                      pauli_mode_decay_rates=[[0., 0., 0.]]*17)
    pmf, x = _simulator(use_fft).asymmetric_heterogeneous_protocol(parameters, 16)
    # All 16 links are ready immediately, so no qubit has waited or relaxed.
    assert_allclose(pmf[1], .2**16, rtol=1e-13, atol=0)
    assert_allclose(x[1], [1, 0, 0, 1, 1, -1], rtol=0, atol=1e-14)
    # With no rejection or cutoff, completion is the maximum generation time.
    cdf = (1-(1-.2)**np.arange(len(pmf)))**16
    assert_allclose(np.cumsum(pmf), cdf, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("horizon", [512, 1024])
@pytest.mark.parametrize("policy,probability,state", [
    ("phi_pair", .625, [1, .8, .8, 1, .6, -.6]),
    ("psi_pair", .375, [1, 0, 0, 1, 1, -1]),
])
def test_noiseless_partial_sources_match_exact_state_and_causal_timing(horizon, policy, probability, state):
    parameters = _parameters(horizon)
    parameters.update(bell_outcomes=policy,
                      real_x_coordinates=[1, .5, .5, 1, np.sqrt(.75), -np.sqrt(.75)])
    pmf, x = _simulator().nested_protocol(parameters)
    reference_pmf, reference_x = _simulator(False).nested_protocol(parameters)
    assert_allclose(pmf, reference_pmf, atol=1e-13, rtol=1e-10)
    assert_allclose(pmf[:, None]*x, reference_pmf[:, None]*reference_x, atol=1e-13, rtol=1e-10)
    assert_allclose(x[pmf > 0], np.broadcast_to(state, x[pmf > 0].shape), atol=1e-9, rtol=0)
    assert pmf.sum() > .999999999
    # Every attempt prepares two geometric(.2) links; its mean is 65/9 slots.
    assert np.dot(pmf, np.arange(horizon)) == pytest.approx((65/9)/probability, rel=1e-9)


@pytest.mark.parametrize("failure", ["exception", "nonphysical", "nonfinite"])
def test_fft_failure_retries_only_the_renewal_and_keeps_fft_enabled(monkeypatch, caplog, failure):
    parameters = _parameters(32)
    parameters.update(p_swap=.8, cutoff=3,
                      real_x_coordinates=[1, 0, 0, .8, .8, -.8], amplitude_damping_rate=.03)
    reference = _simulator(False).nested_protocol(parameters)
    original_fft = algorithm.renewal_sum_sequential_efficient_weighted
    original_causal = algorithm.renewal_sum
    original_join = algorithm.real_x_attempt_join_sequential_efficient
    calls = {"join": 0, "causal": 0}

    def broken_fft(*args, **kwargs):
        if failure == "exception":
            raise ValueError("FFT renewal produced a material negative probability")
        pmf, weighted = original_fft(*args, **kwargs)
        weighted[1, 4] = 10*pmf[1] if failure == "nonphysical" else np.nan
        return pmf, weighted

    def counted_join(*args, **kwargs):
        calls["join"] += 1
        return original_join(*args, **kwargs)

    def counted_causal(kernel, first_success, **kwargs):
        calls["causal"] += 1
        assert first_success.shape == (32, 6)  # One renewal, including the trace.
        return original_causal(kernel, first_success, **kwargs)

    monkeypatch.setattr(algorithm, "renewal_sum_sequential_efficient_weighted", broken_fft)
    monkeypatch.setattr(algorithm, "real_x_attempt_join_sequential_efficient", counted_join)
    monkeypatch.setattr(algorithm, "renewal_sum", counted_causal)
    simulator = _simulator()
    snapshot = deepcopy(parameters)
    pmf, x = simulator.nested_protocol(parameters)
    assert calls == {"join": 1, "causal": 1}
    assert parameters == snapshot
    assert simulator.use_fft is True
    assert "recomputing this swap step with causal renewal" in caplog.text
    assert_allclose(pmf, reference[0], atol=0, rtol=0)
    assert_allclose(x, reference[1], atol=0, rtol=0)

    # The next healthy operation still takes the fast path.
    monkeypatch.setattr(algorithm, "renewal_sum_sequential_efficient_weighted", original_fft)
    monkeypatch.setattr(algorithm, "renewal_sum", _unexpected_renewal)
    pmf, x = simulator.nested_protocol(parameters)
    assert_allclose(pmf[:, None]*x, reference[0][:, None]*reference[1], atol=1e-12, rtol=1e-12)


def test_invalid_causal_output_still_raises(monkeypatch):
    parameters = _parameters(16)
    parameters["p_swap"] = .8

    def broken_fft(*args, **kwargs):
        raise ValueError("FFT numerical failure")

    original_causal = algorithm.renewal_sum

    def nonphysical_causal(*args, **kwargs):
        weighted = original_causal(*args, **kwargs)
        weighted[:, 4] = 10*weighted[:, 0]
        return weighted

    monkeypatch.setattr(algorithm, "renewal_sum_sequential_efficient_weighted", broken_fft)
    monkeypatch.setattr(algorithm, "renewal_sum", nonphysical_causal)
    with pytest.raises(ValueError, match="not positive semidefinite"):
        _simulator().nested_protocol(parameters)


def test_small_genuine_failure_probability_is_not_skipped(monkeypatch):
    parameters = _parameters(8)
    parameters.update(p_gen=1., p_swap=1-1e-10)
    calls = []
    original = algorithm.renewal_sum_sequential_efficient_weighted

    def counted(kernel, *args, **kwargs):
        calls.append(kernel.copy())
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(algorithm, "renewal_sum_sequential_efficient_weighted", counted)
    pmf, _ = _simulator().nested_protocol(parameters)
    assert len(calls) == 1 and calls[0][1] > 0
    assert pmf[2] == pytest.approx(parameters["p_swap"]*(1-parameters["p_swap"]), rel=1e-5, abs=0)


def test_subnormal_conditioning_keeps_probability_without_manufacturing_a_state():
    state = np.array([1., .5, .5, 1., np.sqrt(.75), -np.sqrt(.75)])
    pmf = np.array([0., 1e-320, 1e-200])
    weighted = pmf[:, None]*state
    # At subnormal precision the same rank-one state can round outside the PSD cone.
    assert _minimum_real_x_eigenvalue(weighted[1]/pmf[1]) < -1e-9
    snapshot_p, snapshot_w = pmf.copy(), weighted.copy()
    normalized = normalize_weighted_x_func(weighted, pmf)
    assert_allclose(normalized[1], [1, 0, 0, 0, 0, 0], atol=0)
    assert_allclose(normalized[2], state, atol=1e-14)
    assert_allclose(pmf, snapshot_p, atol=0, rtol=0)
    assert_allclose(weighted, snapshot_w, atol=0, rtol=0)


def test_tiny_normal_probability_still_rejects_nonphysical_correlators():
    pmf = np.array([0., 1e-200])
    weighted = pmf[:, None]*np.array([1., 0., 0., 1., 1.001, -1.001])
    with pytest.raises(ValueError, match="not positive semidefinite"):
        normalize_weighted_x_func(weighted, pmf)


def test_rare_selected_parity_in_deep_protocol_stays_physical():
    # Overnight failure at ordinary (not subnormal) probabilities: subtraction
    # in a rare Psi branch amplified signed population roundoff after AD.
    parameters = _parameters(512)
    parameters.update(protocol=('d4', 'd5', 's4', 'd5', 'd6', 's5', 'd3', 'd6', 's3',
                                'd6', 'd7', 's6', 'd2', 'd7', 's2', 'd1', 'd7', 's1',
                                'd0', 's0', 'd7'),
                      p_gen=[.2]*8, real_x_coordinates=[[1, 0, 0, 1, 1, -1]]*8,
                      pauli_mode_decay_rates=[[0., 0., 0.]]*9,
                      amplitude_damping_rate=[1/25]*9, bell_outcomes='psi_pair')
    pmf, states = _simulator().asymmetric_heterogeneous_protocol(parameters, 8)
    assert 0 < pmf.sum() < 1e-30  # Physical outputs do not imply sufficient coverage.
    assert np.min(_minimum_real_x_eigenvalue(states[pmf >= np.finfo(float).tiny])) >= -1e-10


@pytest.mark.parametrize("material", [False, True])
def test_fft_weighted_psd_repair_is_bounded_and_preserves_trace(monkeypatch, material):
    size = 32
    first = np.zeros((size, 6))
    first[1] = .25*np.array([1, 0, 0, 1, 1, -1])
    first[2] = .1*np.array([1, 0, 0, .5, .5, -.5])
    output = first.copy()
    probability = .1 if material else 1e-10
    coherence = 2. if material else 1+1e-8
    output[5] = probability*np.array([1, 0, 0, 1, coherence, -coherence])
    monkeypatch.setattr(efficient_join, "_convolve_with_resolvent", lambda *args: output.copy())
    if material:
        with pytest.raises(ValueError, match="material nonphysical weighted"):
            efficient_join.renewal_sum_sequential_efficient_weighted(np.zeros(size), first)
    else:
        pmf, weighted = efficient_join.renewal_sum_sequential_efficient_weighted(np.zeros(size), first)
        assert_allclose(pmf, output[:, 0], atol=0, rtol=0)
        states = normalize_weighted_x_func(weighted, pmf)  # Original strict check still passes.
        assert _minimum_real_x_eigenvalue(states[5]) >= -1e-14
        assert np.max(abs(weighted-output)) < 1e-17
