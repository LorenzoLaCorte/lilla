"""Parity checks for the efficient six-coordinate real-X backend.

The direct join is kept as the small-horizon reference here.  Dense-matrix
oracles for that reference live in ``test_dense_oracle_integration.py``; these
tests focus on proving that the sliding semigroup join and formal-series
renewal preserve every weighted probability-flow quantity.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.pauli_adc import (
    RealXState,
    matrix_to_coordinates,
    real_x_attempt_join,
    real_x_attempt_join_efficient,
    renewal_sum,
    renewal_sum_efficient,
)
from src.core.repeater_algorithm import RepeaterChainEvaluation


_SWAP_PAULI_RATES = np.array(
    [
        [0.050, 0.060, 0.070],
        [0.040, 0.055, 0.065],
        [0.060, 0.070, 0.050],
    ],
    dtype=float,
)
_SWAP_ADC_RATES = np.array([0.080, 0.045, 0.105], dtype=float)


def _random_real_x_coordinates(rng):
    """Draw a physical real-X state from two random positive 2x2 blocks."""
    even_factor = rng.normal(size=(2, 2))
    odd_factor = rng.normal(size=(2, 2))
    even = even_factor @ even_factor.T
    odd = odd_factor @ odd_factor.T
    matrix = np.zeros((4, 4), dtype=float)
    matrix[np.ix_([0, 3], [0, 3])] = even
    matrix[np.ix_([1, 2], [1, 2])] = odd
    matrix /= np.trace(matrix)
    return matrix_to_coordinates(matrix)


def _random_inputs(seed, size, *, coverage1=0.91, coverage2=0.87):
    rng = np.random.default_rng(seed)
    pmf1 = np.zeros(size)
    pmf2 = np.zeros(size)
    pmf1[1:] = coverage1 * rng.dirichlet(np.linspace(0.8, 1.7, size - 1))
    pmf2[1:] = coverage2 * rng.dirichlet(np.linspace(1.6, 0.7, size - 1))
    state_func1 = np.vstack(
        [_random_real_x_coordinates(rng) for _ in range(size)]
    )
    state_func2 = np.vstack(
        [_random_real_x_coordinates(rng) for _ in range(size)]
    )
    return pmf1, pmf2, state_func1, state_func2


def _assert_attempt_terms_close(actual, expected, *, atol=3.0e-12):
    assert_allclose(
        actual.cutoff_failure,
        expected.cutoff_failure,
        atol=atol,
        rtol=2.0e-12,
    )
    assert_allclose(
        actual.valid_success, expected.valid_success, atol=atol, rtol=2.0e-12
    )
    assert_allclose(
        actual.valid_failure, expected.valid_failure, atol=atol, rtol=2.0e-12
    )
    assert_allclose(
        actual.weighted_success,
        expected.weighted_success,
        atol=atol,
        rtol=2.0e-12,
    )
    assert_allclose(
        actual.weighted_success[:, 0],
        actual.valid_success,
        atol=atol,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "bell_outcomes",
    ["all_corrected", "phi_plus", "phi_pair", "psi_pair"],
)
@pytest.mark.parametrize("cutoff", [0, 2, 15])
def test_efficient_swap_attempt_matches_direct_for_every_bell_policy(
    bell_outcomes,
    cutoff,
):
    size = 12
    pmf1, pmf2, state_func1, state_func2 = _random_inputs(
        1200 + 17 * cutoff + len(bell_outcomes), size
    )
    kwargs = dict(
        operation="swap",
        cutoff=cutoff,
        cut_type="memory_time",
        hardware_success=0.673,
        bell_outcomes=bell_outcomes,
        node_pauli_rates=_SWAP_PAULI_RATES,
        node_adc_rates=_SWAP_ADC_RATES,
        noise_model="joint",
    )
    expected = real_x_attempt_join(
        pmf1, pmf2, state_func1, state_func2, **kwargs
    )
    actual = real_x_attempt_join_efficient(
        pmf1, pmf2, state_func1, state_func2, **kwargs
    )
    _assert_attempt_terms_close(actual, expected)


@pytest.mark.parametrize("werner_twirl", [False, True])
@pytest.mark.parametrize("cutoff", [0, 3, 14])
def test_efficient_distillation_attempt_matches_direct_with_heterogeneous_rates(
    werner_twirl,
    cutoff,
):
    size = 11
    pmf1, pmf2, state_func1, state_func2 = _random_inputs(
        2300 + 19 * cutoff + int(werner_twirl), size
    )
    kwargs = dict(
        operation="dist",
        cutoff=cutoff,
        cut_type="memory_time",
        hardware_success=0.719,
        node_pauli_rates=_SWAP_PAULI_RATES[:2],
        node_adc_rates=_SWAP_ADC_RATES[:2],
        noise_model="joint",
        werner_twirl=werner_twirl,
    )
    expected = real_x_attempt_join(
        pmf1, pmf2, state_func1, state_func2, **kwargs
    )
    actual = real_x_attempt_join_efficient(
        pmf1, pmf2, state_func1, state_func2, **kwargs
    )
    _assert_attempt_terms_close(actual, expected)


@pytest.mark.parametrize("shift", [0, 4, 41])
@pytest.mark.parametrize("vector_first", [False, True])
def test_formal_series_renewal_matches_causal_reference(shift, vector_first):
    rng = np.random.default_rng(4400 + shift + 100 * int(vector_first))
    size = 37
    failure = rng.random(size)
    failure *= 0.72 / np.sum(failure)
    # Exercise the nontrivial constant denominator when shift is zero.
    failure[0] += 0.035
    failure *= 0.78 / np.sum(failure)

    if vector_first:
        first = rng.normal(scale=0.08, size=(size, 6))
        first[:3] = 0.0
        # Model the structural identity between weighted II and probability.
        first[:, 0] = np.abs(first[:, 0])
    else:
        first = rng.random(size) * 0.07
        first[:2] = 0.0

    expected = renewal_sum(failure, first, shift=shift)
    actual = renewal_sum_efficient(failure, first, shift=shift)
    assert_allclose(actual, expected, atol=2.0e-11, rtol=2.0e-11)


@pytest.mark.parametrize(
    "unit_kind,bell_outcomes,werner_twirl",
    [
        ("swap", "all_corrected", False),
        ("swap", "phi_plus", False),
        ("dist", "all_corrected", False),
        ("dist", "all_corrected", True),
    ],
)
def test_high_level_efficient_backend_matches_direct_weighted_output(
    unit_kind,
    bell_outcomes,
    werner_twirl,
):
    size = 38
    pmf1, pmf2, state_func1, state_func2 = _random_inputs(
        6700 + len(bell_outcomes) + 10 * int(werner_twirl),
        size,
        coverage1=0.94,
        coverage2=0.92,
    )
    node_count = 3 if unit_kind == "swap" else 2
    parameters = {
        "p_gen": 0.5,
        "swap_hardware_efficiency": 0.68,
        "p_distillation": 0.74,
        "cut_type": "memory_time",
        "mt_cut": 3,
        "bell_outcomes": bell_outcomes,
        "pauli_mode_decay_rates": _SWAP_PAULI_RATES[:node_count],
        "amplitude_damping_rate": _SWAP_ADC_RATES[:node_count],
        "pauli_adc_noise_model": "joint",
        "real_x_werner_twirling": werner_twirl,
    }

    direct = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False
    )
    fast = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=True, efficient=True
    )
    expected_pmf, expected_states = direct.compute_unit(
        parameters,
        pmf1,
        state_func1,
        pmf2,
        state_func2,
        unit_kind=unit_kind,
    )
    actual_pmf, actual_states = fast.compute_unit(
        parameters,
        pmf1,
        state_func1,
        pmf2,
        state_func2,
        unit_kind=unit_kind,
    )

    assert_allclose(actual_pmf, expected_pmf, atol=2.0e-11, rtol=2.0e-11)
    expected_weighted = expected_pmf[:, np.newaxis] * expected_states
    actual_weighted = actual_pmf[:, np.newaxis] * actual_states
    assert_allclose(
        actual_weighted, expected_weighted, atol=3.0e-11, rtol=3.0e-11
    )
    assert_allclose(
        actual_weighted[:, 0], actual_pmf, atol=3.0e-12, rtol=0.0
    )


def test_fft_retry_removes_unresolvable_tail_without_nonphysical_states():
    """Regression for signed FFT residue in a long geometric retry tail."""
    size = 1000
    pmf = np.zeros(size)
    pmf[2] = 1.0
    state_func = RealXState(lambdas=[1.0, 0.0, 0.0, 0.0]).get_generation_sf(size)
    parameters = {
        "p_gen": 1.0,
        "swap_hardware_efficiency": 0.3,
        "mt_cut": size,
        "pauli_mode_decay_rates": 0.0,
        "amplitude_damping_rate": 0.0,
        "bell_outcomes": "all_corrected",
    }
    actual_pmf, actual_states = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=True, efficient=True
    ).compute_unit(parameters, pmf, state_func, unit_kind="swap")

    expected = np.zeros(size)
    expected[2 : size : 2] = 0.3 * 0.7 ** np.arange((size - 1) // 2)
    assert np.all(actual_pmf >= 0.0)
    resolved = expected > 5.0e-14
    assert_allclose(
        actual_pmf[resolved], expected[resolved], atol=4.0e-15, rtol=3.0e-13
    )
    assert_allclose(np.sum(actual_pmf), np.sum(expected), atol=2.0e-13)
    populated = actual_pmf > 0.0
    assert_allclose(
        actual_states[populated],
        np.tile(state_func[0], (np.count_nonzero(populated), 1)),
        atol=2.0e-10,
    )


def test_relative_fft_floor_preserves_globally_tiny_success_probabilities():
    size = 7
    pmf = np.zeros(size)
    pmf[1] = 1.0
    state_func = RealXState(lambdas=[1.0, 0.0, 0.0, 0.0]).get_generation_sf(size)
    efficiency = 1.0e-18
    parameters = {
        "p_gen": 1.0,
        "swap_hardware_efficiency": efficiency,
        "mt_cut": size,
        "pauli_mode_decay_rates": 0.0,
        "amplitude_damping_rate": 0.0,
        "bell_outcomes": "all_corrected",
    }
    actual_pmf, _ = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=True, efficient=True
    ).compute_unit(parameters, pmf, state_func, unit_kind="swap")
    assert_allclose(actual_pmf[1:], efficiency, atol=1.0e-32, rtol=2.0e-14)


def test_numpy_integer_max_cutoff_does_not_overflow_fast_expiration_age():
    size = 9
    pmf1, pmf2, state_func1, state_func2 = _random_inputs(8811, size)
    kwargs = {
        "cutoff": np.int64(np.iinfo(np.int64).max),
        "hardware_success": 0.77,
        "node_pauli_rates": _SWAP_PAULI_RATES,
        "node_adc_rates": _SWAP_ADC_RATES,
    }
    expected = real_x_attempt_join(
        pmf1, pmf2, state_func1, state_func2, **kwargs
    )
    actual = real_x_attempt_join_efficient(
        pmf1, pmf2, state_func1, state_func2, **kwargs
    )
    _assert_attempt_terms_close(actual, expected)


def test_fast_join_with_non_fft_causal_retry_matches_full_reference():
    size = 29
    pmf1, pmf2, state_func1, state_func2 = _random_inputs(9912, size)
    parameters = {
        "p_gen": 0.5,
        "swap_hardware_efficiency": 0.61,
        "mt_cut": 4,
        "pauli_mode_decay_rates": _SWAP_PAULI_RATES,
        "amplitude_damping_rate": _SWAP_ADC_RATES,
        "pauli_adc_noise_model": "joint",
        "bell_outcomes": "phi_pair",
    }
    expected = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False
    ).compute_unit(
        parameters,
        pmf1,
        state_func1,
        pmf2,
        state_func2,
        unit_kind="swap",
    )
    actual = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=True
    ).compute_unit(
        parameters,
        pmf1,
        state_func1,
        pmf2,
        state_func2,
        unit_kind="swap",
    )
    assert_allclose(actual[0], expected[0], atol=3.0e-12, rtol=3.0e-12)
    assert_allclose(
        actual[0][:, None] * actual[1],
        expected[0][:, None] * expected[1],
        atol=4.0e-12,
        rtol=4.0e-12,
    )
