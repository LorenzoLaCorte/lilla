"""Numerical oracle tests for the six-correlator Pauli+ADC reference kernel."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.pauli_adc import (
    apply_local_channels,
    compose_channels,
    coordinates_to_matrix,
    is_physical,
    joint_pauli_adc_channel,
    matrix_to_coordinates,
    pauli_eigenvalues,
    phi_plus_fidelity,
    phi_plus_swap,
    sequential_pauli_adc_channel,
)


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PAULIS = (I2, X, Y, Z)


def _single_qubit_kraus(probabilities, eta, order):
    adc = (
        np.diag([1.0, np.sqrt(eta)]).astype(complex),
        np.array([[0.0, np.sqrt(1.0 - eta)], [0.0, 0.0]], dtype=complex),
    )
    pauli = tuple(
        np.sqrt(probability) * operator
        for probability, operator in zip(probabilities, PAULIS)
    )
    if order == "pauli_after_adc":
        return tuple(p @ a for p in pauli for a in adc)
    if order == "adc_after_pauli":
        return tuple(a @ p for a in adc for p in pauli)
    raise ValueError(order)


def _apply_direct_local_channels(rho, left_kraus, right_kraus):
    output = np.zeros((4, 4), dtype=complex)
    for left in left_kraus:
        for right in right_kraus:
            operator = np.kron(left, right)
            output += operator @ rho @ operator.conj().T
    return output


def _bell_branch(rho_a, rho_b, bell, correction):
    tensor = np.kron(rho_a, rho_b).reshape((2,) * 8)
    bell = np.asarray(bell, dtype=complex).reshape(2, 2)
    output = np.einsum("mn,amnbAMNB,MN->abAB", bell, tensor, bell.conj())
    output = output.reshape(4, 4)
    correction = np.kron(I2, correction)
    return correction @ output @ correction.conj().T


def _random_real_x_state(rng):
    diagonal = rng.dirichlet(np.ones(4))
    a, b, c, d = diagonal
    w = rng.uniform(-1.0, 1.0) * np.sqrt(a * d)
    z = rng.uniform(-1.0, 1.0) * np.sqrt(b * c)
    return np.array(
        [
            [a, 0.0, 0.0, w],
            [0.0, b, z, 0.0],
            [0.0, z, c, 0.0],
            [w, 0.0, 0.0, d],
        ],
        dtype=complex,
    )


def test_worked_phi_plus_example_matches_full_kraus_calculation():
    phi_plus = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    rho = np.outer(phi_plus, phi_plus)
    coordinates = matrix_to_coordinates(rho)
    assert_allclose(coordinates, [1.0, 0.0, 0.0, 1.0, 1.0, -1.0])

    probabilities = np.array([0.925, 0.025, 0.025, 0.025])
    mu = pauli_eigenvalues(probabilities)
    channel = sequential_pauli_adc_channel(mu, 0.8, order="pauli_after_adc")
    actual_coordinates = apply_local_channels(coordinates, channel)
    expected_coordinates = np.array([1.0, 0.18, 0.18, 0.5508, 0.648, -0.648])
    assert_allclose(actual_coordinates, expected_coordinates, atol=1.0e-15)

    direct = _apply_direct_local_channels(
        rho,
        _single_qubit_kraus(probabilities, 0.8, "pauli_after_adc"),
        _single_qubit_kraus(probabilities, 0.8, "pauli_after_adc"),
    )
    assert_allclose(coordinates_to_matrix(actual_coordinates), direct, atol=1.0e-15)
    assert_allclose(phi_plus_fidelity(actual_coordinates), 0.7117, atol=1.0e-15)
    assert is_physical(actual_coordinates)


def test_asymmetric_noise_matches_full_density_matrix_for_both_orders():
    rho = np.array(
        [
            [0.4, 0.0, 0.0, 0.12],
            [0.0, 0.1, 0.03, 0.0],
            [0.0, 0.03, 0.2, 0.0],
            [0.12, 0.0, 0.0, 0.3],
        ],
        dtype=complex,
    )
    coordinates = matrix_to_coordinates(rho)
    left_probabilities = np.array([0.86, 0.06, 0.05, 0.03])
    right_probabilities = np.array([0.91, 0.02, 0.04, 0.03])

    for order in ("pauli_after_adc", "adc_after_pauli"):
        left = sequential_pauli_adc_channel(
            pauli_eigenvalues(left_probabilities), 0.73, order=order
        )
        right = sequential_pauli_adc_channel(
            pauli_eigenvalues(right_probabilities), 0.84, order=order
        )
        compact = apply_local_channels(coordinates, left, right)
        direct = _apply_direct_local_channels(
            rho,
            _single_qubit_kraus(left_probabilities, 0.73, order),
            _single_qubit_kraus(right_probabilities, 0.84, order),
        )
        assert_allclose(coordinates_to_matrix(compact), direct, atol=2.0e-15)


def test_phi_plus_swap_is_a_two_by_two_matrix_product_and_tracks_probability():
    noisy = np.array([1.0, 0.18, 0.18, 0.5508, 0.648, -0.648])
    unnormalized, probability = phi_plus_swap(noisy, noisy)
    expected = np.array(
        [0.2581, 0.069786, 0.069786, 0.08394516, 0.104976, -0.104976]
    )
    assert_allclose(unnormalized, expected, atol=1.0e-15)
    assert_allclose(probability, 0.2581, atol=1.0e-15)

    conditioned, conditioned_probability = phi_plus_swap(
        noisy, noisy, conditioned=True
    )
    assert_allclose(conditioned_probability, probability)
    assert_allclose(conditioned[0], 1.0)
    assert_allclose(phi_plus_fidelity(conditioned), 0.5346737311119721)

    rho = coordinates_to_matrix(noisy)
    phi_plus = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    direct = _bell_branch(rho, rho, phi_plus, I2)
    assert_allclose(coordinates_to_matrix(unnormalized), direct, atol=2.0e-15)


def test_adc_can_make_corrected_phi_and_psi_swap_branches_inequivalent():
    noisy = np.array([1.0, 0.18, 0.18, 0.5508, 0.648, -0.648])
    rho = coordinates_to_matrix(noisy)
    bell_states = {
        "phi+": (np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0), I2),
        "phi-": (np.array([1.0, 0.0, 0.0, -1.0]) / np.sqrt(2.0), Z),
        "psi+": (np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2.0), X),
        "psi-": (np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2.0), Y),
    }
    branches = {}
    for name, (bell, correction) in bell_states.items():
        branch = _bell_branch(rho, rho, bell, correction)
        probability = float(np.trace(branch).real)
        branches[name] = (probability, branch / probability)
        matrix_to_coordinates(branch)

    assert_allclose(branches["phi+"][0], 0.2581)
    assert_allclose(branches["phi-"][0], 0.2581)
    assert_allclose(branches["psi+"][0], 0.2419)
    assert_allclose(branches["psi-"][0], 0.2419)
    assert_allclose(branches["phi+"][1], branches["phi-"][1])
    assert_allclose(branches["psi+"][1], branches["psi-"][1])
    assert not np.allclose(branches["phi+"][1], branches["psi+"][1])


def test_all_corrected_bell_branches_close_on_random_real_x_inputs():
    rng = np.random.default_rng(20260802)
    bell_states = {
        "phi+": (np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0), I2),
        "phi-": (np.array([1.0, 0.0, 0.0, -1.0]) / np.sqrt(2.0), Z),
        "psi+": (np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2.0), X),
        "psi-": (np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2.0), Y),
    }

    for _ in range(100):
        rho_a = _random_real_x_state(rng)
        rho_b = _random_real_x_state(rng)
        coordinates_a = matrix_to_coordinates(rho_a)
        coordinates_b = matrix_to_coordinates(rho_b)
        branches = {
            name: _bell_branch(rho_a, rho_b, bell, correction)
            for name, (bell, correction) in bell_states.items()
        }

        for branch in branches.values():
            matrix_to_coordinates(branch)
        assert_allclose(branches["phi+"], branches["phi-"], atol=3.0e-16)
        assert_allclose(branches["psi+"], branches["psi-"], atol=3.0e-16)

        measured_bias_product = coordinates_a[1] * coordinates_b[2]
        expected_phi = (1.0 + measured_bias_product) / 4.0
        expected_psi = (1.0 - measured_bias_product) / 4.0
        assert_allclose(np.trace(branches["phi+"]).real, expected_phi)
        assert_allclose(np.trace(branches["psi+"]).real, expected_psi)
        assert_allclose(
            sum(np.trace(branch).real for branch in branches.values()), 1.0
        )


def test_joint_generator_is_a_semigroup_but_ordered_layers_need_not_be():
    gamma_depolarizing = 0.04
    gamma_adc = 0.08
    total_time = 5.0

    joint_full = joint_pauli_adc_channel(
        gamma_depolarizing, gamma_adc, total_time
    )
    joint_half = joint_pauli_adc_channel(
        gamma_depolarizing, gamma_adc, total_time / 2.0
    )
    assert_allclose(
        list(vars(joint_full).values()),
        list(vars(compose_channels(joint_half, joint_half)).values()),
        atol=1.0e-15,
    )

    mu_full = np.array(
        [1.0, *([np.exp(-gamma_depolarizing * total_time)] * 3)]
    )
    mu_half = np.array(
        [1.0, *([np.exp(-gamma_depolarizing * total_time / 2.0)] * 3)]
    )
    ordered_full = sequential_pauli_adc_channel(
        mu_full,
        np.exp(-gamma_adc * total_time),
        order="pauli_after_adc",
    )
    ordered_half = sequential_pauli_adc_channel(
        mu_half,
        np.exp(-gamma_adc * total_time / 2.0),
        order="pauli_after_adc",
    )
    ordered_two_steps = compose_channels(ordered_half, ordered_half)
    assert_allclose(ordered_full.shift_z, 0.2699191169839555)
    assert_allclose(ordered_two_steps.shift_z, 0.2855276072958546)
    assert not np.isclose(ordered_full.shift_z, ordered_two_steps.shift_z)
    assert_allclose(joint_full.shift_z, 0.3007922426039824)


def test_random_compact_kernels_match_dense_oracles():
    rng = np.random.default_rng(20260801)
    phi_plus = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)

    for index in range(250):
        rho_a = _random_real_x_state(rng)
        rho_b = _random_real_x_state(rng)
        coordinates_a = matrix_to_coordinates(rho_a)
        coordinates_b = matrix_to_coordinates(rho_b)

        probabilities_left = rng.dirichlet(np.ones(4))
        probabilities_right = rng.dirichlet(np.ones(4))
        eta_left, eta_right = rng.random(2)
        order = "pauli_after_adc" if index % 2 == 0 else "adc_after_pauli"
        left_channel = sequential_pauli_adc_channel(
            pauli_eigenvalues(probabilities_left), eta_left, order=order
        )
        right_channel = sequential_pauli_adc_channel(
            pauli_eigenvalues(probabilities_right), eta_right, order=order
        )
        compact_noise = apply_local_channels(
            coordinates_a, left_channel, right_channel
        )
        dense_noise = _apply_direct_local_channels(
            rho_a,
            _single_qubit_kraus(probabilities_left, eta_left, order),
            _single_qubit_kraus(probabilities_right, eta_right, order),
        )
        assert_allclose(
            coordinates_to_matrix(compact_noise),
            dense_noise,
            atol=8.0e-16,
            rtol=0.0,
        )

        compact_swap, probability = phi_plus_swap(
            coordinates_a, coordinates_b
        )
        dense_swap = _bell_branch(rho_a, rho_b, phi_plus, I2)
        assert_allclose(
            coordinates_to_matrix(compact_swap),
            dense_swap,
            atol=3.0e-16,
            rtol=0.0,
        )
        assert_allclose(
            probability,
            np.trace(dense_swap).real,
            atol=2.0e-16,
            rtol=0.0,
        )


def test_fixed_correlators_cover_zero_and_complete_damping_without_a_branch():
    phi_plus_coordinates = np.array([1.0, 0.0, 0.0, 1.0, 1.0, -1.0])
    identity = sequential_pauli_adc_channel(
        [1.0, 1.0, 1.0, 1.0], 1.0
    )
    assert_allclose(
        apply_local_channels(phi_plus_coordinates, identity),
        phi_plus_coordinates,
    )

    complete_adc = sequential_pauli_adc_channel(
        [1.0, 1.0, 1.0, 1.0], 0.0
    )
    assert_allclose(
        apply_local_channels(phi_plus_coordinates, complete_adc),
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    )


def test_channel_factories_reject_nonphysical_or_ambiguous_inputs():
    with pytest.raises(ValueError, match="completely positive"):
        sequential_pauli_adc_channel([1.0, 1.0, 1.0, -1.0], 0.8)
    with pytest.raises(ValueError, match="sum to one"):
        pauli_eigenvalues([0.25, 0.25, 0.25, 0.250009])
    with pytest.raises(ValueError, match="must be real"):
        apply_local_channels(
            [1.0, 0.0, 0.0, 1.0, 1.0 + 1.0j, -1.0],
            sequential_pauli_adc_channel([1.0, 1.0, 1.0, 1.0], 1.0),
        )


def test_joint_generator_is_accurate_near_the_identity():
    channel = joint_pauli_adc_channel(0.04, 0.08, 1.0e-8)
    assert_allclose(channel.fixed_point_bias, 2.0 / 3.0, rtol=2.0e-8)

    tiny_time = 1.0e-12
    tiny = joint_pauli_adc_channel(0.04, 0.08, tiny_time)
    expected_shift = (0.08 / 0.12) * -np.expm1(-0.12 * tiny_time)
    assert_allclose(tiny.shift_z, expected_shift, rtol=0.0, atol=1.0e-30)
