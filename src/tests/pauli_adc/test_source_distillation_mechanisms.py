"""Golden source examples: raw recurrence can reduce entanglement without noise."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.pauli_adc import (
    DISTILLATION_STRATEGIES, RealXState, bell_diagonal_twirl,
    bell_weights_to_coordinates, coordinates_to_matrix, phi_plus_fidelity,
    phi_plus_werner_twirl, recurrence_distillation,
)
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.scripts.compare_distillation_strategies import source_coordinates
from src.scripts.distillation_source_mechanisms import run_diagnostics, state_metrics
from src.tests.pauli_adc.test_dense_oracle_integration import _dense_distillation


def test_slide_dephased_bell_example_and_bell_twirl_control():
    # lilla-slides-overleaf/content.tex: "Twirling before recurrence".
    source = bell_weights_to_coordinates([.95, 0., 0., .05])
    raw, raw_probability = recurrence_distillation(source, source, strategy="none", conditioned=True)
    bell, bell_probability = recurrence_distillation(source, source, strategy="bell_diagonal", conditioned=True)
    werner = phi_plus_werner_twirl(source)
    input_twirl_only, probability = recurrence_distillation(werner, werner, strategy="none", conditioned=True)
    bbpssw, bbpssw_probability = recurrence_distillation(source, source, strategy="bbpssw", conditioned=True)
    assert_allclose(phi_plus_fidelity(source), .95)
    assert_allclose(phi_plus_fidelity(raw), .905)
    assert_allclose(state_metrics(raw)["negativity"], .405)
    assert_allclose(bell_diagonal_twirl(source), source)
    assert_allclose(bell, raw)
    assert_allclose([raw_probability, bell_probability], [1., 1.])
    expected_probability = .95**2+2*.95*.05/3+5*.05**2/9
    expected_fidelity = (.95**2+.05**2/9)/expected_probability
    assert_allclose(phi_plus_fidelity(input_twirl_only), expected_fidelity)
    assert_allclose(phi_plus_fidelity(bbpssw), expected_fidelity)
    assert_allclose([probability, bbpssw_probability], expected_probability)
    assert .96495 < expected_fidelity < .96497


@pytest.mark.parametrize("q", [0., .1, .2, .25, .35, .5, .8, 1.])
def test_raw_pure_pe_squares_coherence_and_reduces_negativity(q):
    ket = np.array([np.sqrt(1-q), 0., 0., np.sqrt(q)])
    rho = np.outer(ket, ket)
    source = source_coordinates("biased_x", 1., q)
    output, probability = recurrence_distillation(source, source, strategy="none", conditioned=True)
    c = 2*np.sqrt(q*(1-q))
    expected = np.array([
        [1-q, 0., 0., 2*q*(1-q)], [0., 0., 0., 0.],
        [0., 0., 0., 0.], [2*q*(1-q), 0., 0., q],
    ])
    assert_allclose(coordinates_to_matrix(source), rho, atol=2e-16)
    assert_allclose(probability, 1.)
    assert_allclose(coordinates_to_matrix(output), expected, atol=2e-16)
    assert_allclose(expected, _dense_distillation(rho, rho, "none"), atol=2e-16)
    assert_allclose(phi_plus_fidelity(output), (1+c*c)/2)
    assert_allclose(state_metrics(source)["negativity"], c/2)
    assert_allclose(state_metrics(output)["negativity"], c*c/2)
    if 0 < q < 1 and q != .5:
        assert phi_plus_fidelity(output) < phi_plus_fidelity(source)
        assert state_metrics(output)["negativity"] < state_metrics(source)["negativity"]


@pytest.mark.parametrize("q", [.1, .25, .35, .5])
def test_pure_pe_successful_branches_explain_loss_when_averaged(q):
    # Explicit four-qubit CNOT on |psi_q> tensor |psi_q>, resolved by target.
    # Outcome 00 leaves amplitudes (1-q,q); 11 leaves equal amplitudes.
    kept_00 = np.array([1-q, 0., 0., q])
    kept_11 = np.array([np.sqrt(q*(1-q)), 0., 0., np.sqrt(q*(1-q))])
    branch_00, branch_11 = np.outer(kept_00, kept_00), np.outer(kept_11, kept_11)
    p00, p11 = np.trace(branch_00), np.trace(branch_11)
    assert_allclose([p00, p11], [(1-q)**2+q*q, 2*q*(1-q)])
    phi = np.array([1., 0., 0., 1.])/np.sqrt(2)
    assert_allclose(branch_11/p11, np.outer(phi, phi), atol=3e-16)
    source = source_coordinates("biased_x", 1., q)
    output, probability = recurrence_distillation(source, source, strategy="none")
    assert_allclose(coordinates_to_matrix(output), branch_00+branch_11, atol=3e-16)
    assert_allclose(probability, p00+p11)
    assert_allclose(branch_00[3, 3]/p00, q*q/((1-q)**2+q*q))


@pytest.mark.parametrize("q", [.1, .25, .35, .5])
def test_repeated_raw_rounds_keep_population_but_destroy_nonmaximal_coherence(q):
    source = source_coordinates("biased_x", 1., q)
    c = 2*np.sqrt(q*(1-q))
    state = source
    for round_number in range(1, 6):
        state, probability = recurrence_distillation(state, state, strategy="none", conditioned=True)
        assert_allclose(probability, 1.)
        assert_allclose(state, [1., 1-2*q, 1-2*q, 1., c**(2**round_number), -c**(2**round_number)],
                        atol=4e-16)
        assert_allclose(state_metrics(state)["negativity"], c**(2**round_number)/2, atol=4e-16)


@pytest.mark.parametrize("q", [.1, .25, .35])
def test_bbpssw_and_dejmps_rescue_pure_pe_conditional_quality(q):
    source = source_coordinates("biased_x", 1., q)
    c = 2*np.sqrt(q*(1-q))
    dejmps, probability = recurrence_distillation(source, source, strategy="dejmps", conditioned=True)
    assert_allclose(probability, (1+c*c)/2)
    assert_allclose(phi_plus_fidelity(dejmps), .5+c/(1+c*c))
    assert_allclose(state_metrics(dejmps)["negativity"], c/(1+c*c))
    for strategy in ("bbpssw", "dejmps"):
        output, _ = recurrence_distillation(source, source, strategy=strategy, conditioned=True)
        assert phi_plus_fidelity(output) > phi_plus_fidelity(source)
        assert state_metrics(output)["negativity"] > state_metrics(source)["negativity"]


@pytest.mark.parametrize("strategy", DISTILLATION_STRATEGIES)
@pytest.mark.parametrize("efficient", [False, True])
@pytest.mark.parametrize("use_fft", [False, True])
def test_pure_pe_network_round_matches_circuit_state_and_geometric_retries(strategy, efficient, use_fft):
    # Keep nonzero retry masses above the FFT solver's roundoff clipping floor.
    size = 12
    source = source_coordinates("biased_x", 1., .25)
    rho = coordinates_to_matrix(source)
    expected = _dense_distillation(rho, rho, strategy)
    probability = float(np.trace(expected).real)
    conditional = expected/probability
    pmf, states = RepeaterChainEvaluation(
        state_type=RealXState, efficient=efficient, use_fft=use_fft,
        classical_communication=False,
    ).nested_protocol(dict(protocol=(1,), p_gen=1., p_swap=1., p_distillation=1.,
        t_trunc=size, real_x_coordinates=source, distillation_strategy=strategy,
        pauli_mode_decay_rates=0., amplitude_damping_rate=0.))
    golden_pmf = np.zeros(size)
    golden_pmf[1:] = probability*(1-probability)**np.arange(size-1)
    assert_allclose(pmf, golden_pmf, atol=3e-15)
    actual_weighted = np.array([p*coordinates_to_matrix(state) for p, state in zip(pmf, states)])
    assert_allclose(actual_weighted, golden_pmf[:, None, None]*conditional, atol=3e-15)


def test_diagnostic_round_zero_is_source_and_perfect_bell_is_fixed():
    rows = run_diagnostics(rounds=3)
    for row in rows:
        assert row["minimum_eigenvalue"] >= -1e-14
        if row["source_id"] == "pure_pe_q0.5":
            assert_allclose([row["fidelity"], row["negativity"], row["purity"]], [1., .5, 1.])
    initial = [r for r in rows if r["source_id"] == "pure_pe_q0.25" and r["round"] == 0]
    assert len(initial) == 4
    assert_allclose([r["fidelity"] for r in initial], [.5+np.sqrt(3)/4]*4)
