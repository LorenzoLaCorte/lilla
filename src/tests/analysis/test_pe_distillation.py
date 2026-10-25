"""Independent circuit and exact timing controls for the exploratory variants."""
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.analysis.pe_distillation import HH, evaluate, recurrence_tensor, recurrence_variant, source
from src.core.pauli_adc import coordinates_to_matrix, matrix_to_coordinates, recurrence_distillation
from src.core.pauli_adc import swap_join_sequential_efficient as join
from src.core.pauli_adc.state import RealXState
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.tests.pauli_adc.test_dense_oracle_integration import (
    _apply_independent_sequential_noise, _dense_recurrence,
)
from src.visualization.config import DEFAULT, expand_config
from src.visualization.metrics import binary_entropy


@pytest.mark.parametrize("mode", ["Z", "X"])
def test_tensor_matches_independent_dense_circuit_on_unequal_mixed_inputs(mode):
    a, b = source(.25, .9), source(.45, .8)
    rho, sigma = coordinates_to_matrix(a), coordinates_to_matrix(b)
    if mode == "X":
        expected = HH @ _dense_recurrence(HH@rho@HH, HH@sigma@HH) @ HH
    else:
        expected = _dense_recurrence(rho, sigma)
        assert_allclose(recurrence_distillation(a, b)[0], matrix_to_coordinates(expected), atol=1e-14)
    actual = np.einsum("i,j,kij->k", a, b, recurrence_tensor(mode))
    assert_allclose(actual, matrix_to_coordinates(expected), atol=1e-14)
    assert actual[0] <= 1 and np.linalg.eigvalsh(expected).min() >= -1e-14


@pytest.mark.parametrize("mode,probability,coordinates", [
    ("Z", 1., [1, .5, .5, 1, .75, -.75]),
    ("X", .875, [1, 1/7, 1/7, 1, 4*np.sqrt(3)/7, -4*np.sqrt(3)/7]),
    ("11", .375, [1, 0, 0, 1, 1, -1]),
])
def test_pure_partial_source_has_expected_concentration(mode, probability, coordinates):
    x = source(.25)
    out = np.einsum("i,j,kij->k", x, x, recurrence_tensor(mode))
    assert out[0] == pytest.approx(probability, abs=1e-14)
    assert_allclose(out/out[0], coordinates, atol=1e-14)


def test_x_recurrence_with_retries_matches_exact_four_link_timing():
    config = deepcopy(DEFAULT)
    config["initial_state"] = {"kind": "real_x", "values": source(.25).tolist()}
    config["hardware"].update(p_gen=1., p_swap=1., p_distillation=1.)
    config["memory"] = {"kind": "t1_t2", "t1_slots": None, "t2_slots": None}
    config["protocol"]["distillation_rounds"] = {f"{i}:{i}": 1 for i in range(4)}
    config["numerics"]["t_trunc"] = 64
    _, parameters, _ = expand_config(config)
    original = join._bilinear_operation
    with recurrence_variant("X"):
        pmf, x = RepeaterChainEvaluation(state_type=RealXState, efficient=True,
            use_fft=True, use_gpu=False, w_twirling=False).asymmetric_heterogeneous_protocol(parameters, 4)
    assert join._bilinear_operation is original
    # Four independent leaf distillations, geometric(.875) completion times.
    cdf = (1-(1-.875)**np.arange(64))**4
    assert_allclose(np.cumsum(pmf), cdf, atol=1e-12)
    expected = [1, 1/7, 1/7, 1, (4*np.sqrt(3)/7)**4, -(4*np.sqrt(3)/7)**4]
    # Compare weighted states: rare FFT tail bins can have poor relative
    # conditional accuracy while their absolute contribution is roundoff-sized.
    assert_allclose(pmf[:, None]*x, pmf[:, None]*expected, atol=1e-13, rtol=1e-11)
    mean_wait = np.dot(pmf, np.arange(64))
    assert mean_wait == pytest.approx(np.sum(1-cdf), abs=1e-11)
    assert 1-float(binary_entropy((1-expected[4])/2)) > .85


def test_prototype_map_is_restored_after_failure():
    original = join._bilinear_operation
    with pytest.raises(RuntimeError, match="deliberate"):
        with recurrence_variant("11"):
            raise RuntimeError("deliberate")
    assert join._bilinear_operation is original


def test_x_recurrence_applies_waiting_noise_before_basis_rotation():
    size = 7
    pmf1 = np.r_[0., .3*.7**np.arange(size-1)]
    pmf2 = np.r_[0., .4*.6**np.arange(size-1)]
    a, b = source(.25, .9), source(.45, .8)
    rho, sigma = coordinates_to_matrix(a), coordinates_to_matrix(b)
    expected = np.zeros((size, 6))
    for t1 in range(1, size):
        for t2 in range(1, size):
            kept, auxiliary = rho, sigma
            if t1 != t2:
                older = rho if t1 < t2 else sigma
                aged = _apply_independent_sequential_noise(
                    older, [0., 0., 0.], .05, [0., 0., 0.], .12, abs(t1-t2))
                if t1 < t2:
                    kept = aged
                else:
                    auxiliary = aged
            output = HH @ _dense_recurrence(HH@kept@HH, HH@auxiliary@HH) @ HH
            expected[max(t1, t2)] += pmf1[t1]*pmf2[t2]*matrix_to_coordinates(output)
    with recurrence_variant("X"):
        attempt = join.real_x_attempt_join_sequential_efficient(
            pmf1, pmf2, np.tile(a, (size, 1)), np.tile(b, (size, 1)),
            operation="dist", cutoff=size, cut_type="memory_time", hardware_success=1.,
            node_pauli_rates=[[0., 0., 0.]]*2, node_adc_rates=[.05, .12])
    assert_allclose(attempt.weighted_success, expected, atol=1e-13)
    assert_allclose(attempt.valid_success, expected[:, 0], atol=1e-13)


def test_population_bias_changes_the_preferred_root_operation():
    physical, blind = {}, {}
    for mode in ("Z", "X", "11"):
        actual = evaluate(.25, 100., "root", mode, "all_corrected")
        predicted = evaluate(.25, 100., "root", mode, "all_corrected", erase_bias=True)
        assert actual["converged"] and predicted["converged"]
        physical[mode] = actual["rate_bits_per_slot"]
        blind[mode] = predicted["rate_bits_per_slot"]
    assert max(physical, key=physical.get) == "11"
    assert max(blind, key=blind.get) == "X"
    assert blind["11"] == 0 and physical["11"] > 0
    assert blind["Z"] == pytest.approx(physical["Z"], abs=1e-12)
    assert blind["X"] == pytest.approx(physical["X"], abs=1e-12)
    regret = 1-physical["X"]/physical["11"]
    assert .10 < regret < .11
