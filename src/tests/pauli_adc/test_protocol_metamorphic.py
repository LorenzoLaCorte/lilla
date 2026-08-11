"""Metamorphic checks for the real-X protocol layer.

These tests deliberately avoid copying the implementation's dense formulas.
They instead check reductions to the independently implemented Pauli-Fourier
backend, completeness of Bell branches, probability-mass conservation, and
the operational meaning of the kept/auxiliary ordering in recurrence.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.pauli_adc import (
    RealXState,
    accepted_bell_swap,
    bell_weights_to_coordinates,
    real_x_attempt_join,
    recurrence_distillation,
)
from src.core.pauli_fourier.protocol_units import get_dist_mu_out
from src.core.pauli_fourier.state import PauliFourierState, lambda_to_mu
from src.core.repeater_algorithm import RepeaterChainEvaluation


def _embed_weighted_mu(weighted_mu):
    """Embed weighted PF coordinates in (II,IZ,ZI,ZZ,XX,YY)."""
    mu0, mu1, mu2, mu3 = weighted_mu
    return np.array([mu0, 0.0, 0.0, mu3, mu1, -mu2])


def _constant_state_func(coordinates, size):
    return np.tile(np.asarray(coordinates, dtype=float), (size, 1))


def _geometric_pmf(probability, size):
    result = np.zeros(size)
    times = np.arange(1, size)
    result[1:] = probability * (1.0 - probability) ** (times - 1)
    return result


def _embed_mu_func(mu_func):
    mu_func = np.asarray(mu_func)
    return np.column_stack(
        (
            mu_func[:, 0],
            np.zeros(len(mu_func)),
            np.zeros(len(mu_func)),
            mu_func[:, 3],
            mu_func[:, 1],
            -mu_func[:, 2],
        )
    )


def test_corrected_swap_reduces_to_pauli_fourier_multiplication():
    rng = np.random.default_rng(20260803)

    for _ in range(100):
        left_weights = rng.dirichlet(np.ones(4))
        right_weights = rng.dirichlet(np.ones(4))
        left_mu = lambda_to_mu(left_weights)
        right_mu = lambda_to_mu(right_weights)
        left = bell_weights_to_coordinates(left_weights)
        right = bell_weights_to_coordinates(right_weights)

        expected = _embed_weighted_mu(left_mu * right_mu)
        all_corrected, all_mass = accepted_bell_swap(
            left, right, outcomes="all_corrected"
        )
        phi_one, phi_one_mass = accepted_bell_swap(
            left, right, outcomes="phi_plus"
        )
        phi_pair, phi_pair_mass = accepted_bell_swap(
            left, right, outcomes="phi_pair"
        )
        psi_pair, psi_pair_mass = accepted_bell_swap(
            left, right, outcomes="psi_pair"
        )

        # Four corrected Bell branches are complete.  Bell-diagonal inputs
        # additionally make all four conditioned output states identical.
        assert_allclose(all_corrected, phi_pair + psi_pair, atol=2.0e-16)
        assert_allclose(phi_pair, 2.0 * phi_one, atol=2.0e-16)
        assert_allclose(all_corrected, expected, atol=2.0e-16)
        assert_allclose(
            [phi_one_mass, phi_pair_mass, psi_pair_mass, all_mass],
            [0.25, 0.5, 0.5, 1.0],
            atol=2.0e-16,
        )
        for policy in ("phi_plus", "phi_pair", "psi_pair", "all_corrected"):
            conditioned, _ = accepted_bell_swap(
                left, right, outcomes=policy, conditioned=True
            )
            assert_allclose(conditioned, expected, atol=3.0e-16)


def test_recurrence_reduces_to_pauli_fourier_rule_with_and_without_twirl():
    rng = np.random.default_rng(20260804)

    for _ in range(100):
        kept_weights = rng.dirichlet(np.ones(4))
        auxiliary_weights = rng.dirichlet(np.ones(4))
        kept_mu = lambda_to_mu(kept_weights)
        auxiliary_mu = lambda_to_mu(auxiliary_weights)
        kept = bell_weights_to_coordinates(kept_weights)
        auxiliary = bell_weights_to_coordinates(auxiliary_weights)

        for twirl in (False, True):
            expected_mu = get_dist_mu_out(
                1,
                1,
                kept_mu,
                auxiliary_mu,
                depolar_rate=0.0,
                dephase_rate=0.0,
                w_twirling=twirl,
            )
            output, probability = recurrence_distillation(
                kept, auxiliary, werner_twirl=twirl
            )
            assert_allclose(output, _embed_weighted_mu(expected_mu), atol=3.0e-16)
            assert_allclose(probability, expected_mu[0], atol=2.0e-16)


def test_recurrence_kept_and_auxiliary_roles_are_operationally_asymmetric():
    zero_zero = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    one_one = np.array([1.0, -1.0, -1.0, 1.0, 0.0, 0.0])

    keep_zero, probability_zero = recurrence_distillation(zero_zero, one_one)
    keep_one, probability_one = recurrence_distillation(one_one, zero_zero)

    # Both target measurements agree with certainty, but recurrence keeps the
    # first/control pair.  Reversing the inputs therefore keeps |11> instead
    # of |00>; ADC-generated local Z biases make this distinction observable.
    assert_allclose(keep_zero, zero_zero)
    assert_allclose(keep_one, one_one)
    assert_allclose([probability_zero, probability_one], [1.0, 1.0])
    assert_allclose(keep_zero[[0, 3, 4, 5]], keep_one[[0, 3, 4, 5]])
    assert not np.allclose(keep_zero[[1, 2]], keep_one[[1, 2]])


def test_attempt_join_conserves_every_input_pair_mass():
    size = 9
    pmf1 = np.array([0.0, 0.20, 0.15, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0])
    pmf2 = np.array([0.0, 0.05, 0.15, 0.10, 0.10, 0.05, 0.0, 0.0, 0.0])
    left = bell_weights_to_coordinates([0.72, 0.08, 0.07, 0.13])
    right = bell_weights_to_coordinates([0.64, 0.11, 0.09, 0.16])
    state_func1 = _constant_state_func(left, size)
    state_func2 = _constant_state_func(right, size)
    pair_coverage = np.sum(pmf1) * np.sum(pmf2)

    for operation in ("swap", "dist"):
        attempt = real_x_attempt_join(
            pmf1,
            pmf2,
            state_func1,
            state_func2,
            operation=operation,
            cutoff=1,
            cut_type="memory_time",
            hardware_success=0.63,
            bell_outcomes="all_corrected",
        )
        classified_mass = np.sum(
            attempt.cutoff_failure
            + attempt.valid_success
            + attempt.valid_failure
        )
        assert_allclose(classified_mass, pair_coverage, atol=2.0e-16)
        assert_allclose(
            attempt.weighted_success[:, 0],
            attempt.valid_success,
            atol=2.0e-16,
        )


def test_attempt_join_rejects_arrays_with_more_than_unit_probability_mass():
    pmf = np.array([0.0, 0.8, 0.5, 0.0])
    state = bell_weights_to_coordinates([1.0, 0.0, 0.0, 0.0])
    state_func = _constant_state_func(state, len(pmf))
    with pytest.raises(ValueError, match="mass above one"):
        real_x_attempt_join(pmf, pmf, state_func, state_func)


def test_full_reset_retry_has_the_expected_geometric_law_and_state():
    size = 10
    pmf = np.zeros(size)
    pmf[1] = 1.0
    left_weights = np.array([0.72, 0.08, 0.07, 0.13])
    right_weights = np.array([0.64, 0.11, 0.09, 0.16])
    left = bell_weights_to_coordinates(left_weights)
    right = bell_weights_to_coordinates(right_weights)
    sf1 = _constant_state_func(left, size)
    sf2 = _constant_state_func(right, size)

    evaluator = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False
    )
    hardware_success = 0.6
    parameters = {
        "p_gen": 1.0,
        "swap_hardware_efficiency": hardware_success,
        "t_coh": np.inf,
        "cut_type": "memory_time",
        "mt_cut": np.iinfo(np.int32).max,
        "pauli_mode_decay_rates": 0.0,
        "amplitude_damping_rate": 0.0,
        "bell_outcomes": "all_corrected",
    }

    output_pmf, output_sf = evaluator.compute_unit(
        parameters, pmf, sf1, pmf, sf2, unit_kind="swap"
    )
    expected_pmf = np.zeros(size)
    expected_pmf[1:] = hardware_success * (1.0 - hardware_success) ** np.arange(
        size - 1
    )
    expected_mu = lambda_to_mu(left_weights) * lambda_to_mu(right_weights)
    expected_state = _embed_weighted_mu(expected_mu)

    assert_allclose(output_pmf, expected_pmf, atol=2.0e-16)
    assert_allclose(
        output_sf[1:],
        np.tile(expected_state, (size - 1, 1)),
        atol=3.0e-16,
    )


def test_direct_distillation_does_not_require_an_irrelevant_swap_efficiency():
    size = 7
    pmf = np.zeros(size)
    pmf[1] = 1.0
    state = bell_weights_to_coordinates([0.8, 0.1, 0.05, 0.05])
    state_func = _constant_state_func(state, size)
    parameters = {
        "p_gen": 1.0,
        "p_distillation": 1.0,
        "pauli_mode_decay_rates": 0.0,
        "amplitude_damping_rate": 0.0,
    }
    output_pmf, _ = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False
    ).compute_unit(parameters, pmf, state_func, unit_kind="dist")
    assert np.sum(output_pmf) > 0.0


def test_tiny_but_positive_hardware_efficiency_is_not_dropped():
    size = 5
    pmf = np.zeros(size)
    pmf[1] = 1.0
    state = bell_weights_to_coordinates([1.0, 0.0, 0.0, 0.0])
    state_func = _constant_state_func(state, size)
    efficiency = 1.0e-18
    parameters = {
        "p_gen": 1.0,
        "swap_hardware_efficiency": efficiency,
        "pauli_mode_decay_rates": 0.0,
        "amplitude_damping_rate": 0.0,
        "bell_outcomes": "all_corrected",
    }
    output_pmf, output_states = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False
    ).compute_unit(parameters, pmf, state_func, unit_kind="swap")
    assert_allclose(output_pmf[1:], efficiency, rtol=0.0, atol=1.0e-32)
    assert_allclose(output_states[1:], _constant_state_func(state, size - 1))


def test_end_to_end_pauli_fourier_reduction_through_cutoffs_and_retries():
    """PF and real-X backends agree for swap, distill, and a mixed stack."""
    size = 28
    weights = (
        np.array([0.72, 0.08, 0.07, 0.13]),
        np.array([0.64, 0.11, 0.09, 0.16]),
        np.array([0.68, 0.10, 0.08, 0.14]),
        np.array([0.61, 0.13, 0.10, 0.16]),
    )
    pmfs = tuple(
        _geometric_pmf(probability, size)
        for probability in (0.22, 0.37, 0.31, 0.18)
    )
    real_state_funcs = tuple(
        _constant_state_func(bell_weights_to_coordinates(value), size)
        for value in weights
    )
    pf_state_funcs = tuple(
        _constant_state_func(lambda_to_mu(value), size) for value in weights
    )

    real_parameters = {
        "p_gen": 0.2,  # compute_unit validates it; supplied PMFs remain authoritative.
        "swap_hardware_efficiency": 0.67,
        "p_distillation": 1.0,
        "t_coh": np.inf,
        "cut_type": "memory_time",
        "mt_cut": 3,
        # Per-node X/Y/Z mode rates.  ADC=0 leaves Bell-diagonal states in
        # the PF subspace, while unequal generation times exercise decay.
        "pauli_mode_decay_rates": [0.07, 0.07, 0.04],
        "amplitude_damping_rate": 0.0,
        "pauli_adc_noise_model": "joint",
        "bell_outcomes": "all_corrected",
        "real_x_werner_twirling": False,
    }

    def pf_parameters(node_count):
        # The old PF backend takes node depolarizing/dephasing rates.  Their
        # endpoint sums give Gamma_x=Gamma_y=0.07 and Gamma_z=0.04 per node.
        return {
            "p_gen": 0.2,
            "p_swap": 0.67,
            "t_coh": np.inf,
            "cut_type": "memory_time",
            "mt_cut": 3,
            "depolarizing_rate": [0.04] * node_count,
            "dephasing_rate": [0.03] * node_count,
        }

    real_evaluator = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False, w_twirling=False
    )
    pf_evaluator = RepeaterChainEvaluation(
        state_type=PauliFourierState,
        use_fft=False,
        efficient=False,
        w_twirling=False,
    )

    def assert_reduction(real_result, pf_result):
        real_pmf, real_states = real_result
        pf_pmf, pf_states = pf_result
        assert_allclose(real_pmf, pf_pmf, atol=3.0e-16, rtol=0.0)
        populated = np.maximum(real_pmf, pf_pmf) > 0.0
        assert np.any(populated)
        assert_allclose(
            real_states[populated],
            _embed_mu_func(pf_states)[populated],
            atol=8.0e-16,
            rtol=0.0,
        )

    # One-level units with deliberately unequal input waiting distributions.
    for operation, node_count in (("swap", 3), ("dist", 2)):
        real_result = real_evaluator.compute_unit(
            real_parameters,
            pmfs[0],
            real_state_funcs[0],
            pmfs[1],
            real_state_funcs[1],
            unit_kind=operation,
        )
        pf_result = pf_evaluator.compute_unit(
            pf_parameters(node_count),
            pmfs[0],
            pf_state_funcs[0],
            pmfs[1],
            pf_state_funcs[1],
            unit_kind=operation,
        )
        assert_reduction(real_result, pf_result)

    # Mixed two-level construction: make two unequal swapped links, then use
    # one as the kept pair and the other as the auxiliary pair in recurrence.
    real_swaps = tuple(
        real_evaluator.compute_unit(
            real_parameters,
            pmfs[2 * index],
            real_state_funcs[2 * index],
            pmfs[2 * index + 1],
            real_state_funcs[2 * index + 1],
            unit_kind="swap",
        )
        for index in range(2)
    )
    pf_swaps = tuple(
        pf_evaluator.compute_unit(
            pf_parameters(3),
            pmfs[2 * index],
            pf_state_funcs[2 * index],
            pmfs[2 * index + 1],
            pf_state_funcs[2 * index + 1],
            unit_kind="swap",
        )
        for index in range(2)
    )
    for real_result, pf_result in zip(real_swaps, pf_swaps):
        assert_reduction(real_result, pf_result)

    real_mixed = real_evaluator.compute_unit(
        real_parameters, *real_swaps[0], *real_swaps[1], unit_kind="dist"
    )
    pf_mixed = pf_evaluator.compute_unit(
        pf_parameters(2), *pf_swaps[0], *pf_swaps[1], unit_kind="dist"
    )
    assert_reduction(real_mixed, pf_mixed)
