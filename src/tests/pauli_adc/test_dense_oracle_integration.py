"""Independent dense-matrix oracles for the RealXState repeater backend.

The expected values below are deliberately built without calling the compact
real-X channel, swap, or distillation rules.  Those compact routines are the
subjects under test: the oracle instead evolves ordinary density matrices,
projects Bell states, and applies an explicit four-qubit bilateral-CNOT
circuit.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from src.core.pauli_adc import (
    RealXState,
    normalize_weighted_x_func,
    polish_x_func,
    real_x_attempt_join,
    recurrence_distillation,
)
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.core.pauli_adc.state import _minimum_real_x_eigenvalue


I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)

_CORRELATOR_OPERATORS = (
    np.kron(I2, I2),
    np.kron(I2, Z),
    np.kron(Z, I2),
    np.kron(Z, Z),
    np.kron(X, X),
    np.kron(Y, Y),
)

_BELL_DATA = {
    "phi+": (
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex) / np.sqrt(2.0),
        I2,
    ),
    "phi-": (
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex) / np.sqrt(2.0),
        Z,
    ),
    "psi+": (
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex) / np.sqrt(2.0),
        X,
    ),
    "psi-": (
        np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex) / np.sqrt(2.0),
        Y,
    ),
}


def _dense_to_correlators(rho):
    """Encode a dense (possibly unnormalized) real-X matrix independently."""
    values = np.array(
        [np.trace(rho @ operator) for operator in _CORRELATOR_OPERATORS]
    )
    assert_allclose(values.imag, 0.0, atol=2.0e-13)
    return values.real


def _correlators_to_dense(coordinates):
    """Local reconstruction used only by assertions in this test module."""
    return sum(
        coefficient * operator
        for coefficient, operator in zip(coordinates, _CORRELATOR_OPERATORS)
    ) / 4.0


def _random_real_x_density(rng):
    diagonal = rng.dirichlet(np.ones(4))
    a, b, c, d = diagonal
    even_coherence = rng.uniform(-1.0, 1.0) * np.sqrt(a * d)
    odd_coherence = rng.uniform(-1.0, 1.0) * np.sqrt(b * c)
    return np.array(
        [
            [a, 0.0, 0.0, even_coherence],
            [0.0, b, odd_coherence, 0.0],
            [0.0, odd_coherence, c, 0.0],
            [even_coherence, 0.0, 0.0, d],
        ],
        dtype=complex,
    )


def _joint_single_qubit_basis_images(pauli_decay_rates, adc_rate, time):
    """Exponentiate the full Lindblad generator on the matrix-unit basis.

    This is intentionally separate from the affine-PTM implementation.  The
    Pauli jump rates are reconstructed from the requested X/Y/Z mode-decay
    rates, and amplitude damping is represented by its lowering operator.
    """
    gamma_x, gamma_y, gamma_z = np.asarray(pauli_decay_rates, dtype=float)
    jump_rates = np.array(
        [
            gamma_y + gamma_z - gamma_x,
            gamma_x + gamma_z - gamma_y,
            gamma_x + gamma_y - gamma_z,
        ]
    ) / 4.0
    assert np.min(jump_rates) >= -1.0e-14

    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    excited_projector = lowering.conj().T @ lowering

    def generator(matrix):
        result = np.zeros((2, 2), dtype=complex)
        for rate, pauli in zip(jump_rates, (X, Y, Z)):
            result += rate * (pauli @ matrix @ pauli - matrix)
        result += adc_rate * (
            lowering @ matrix @ lowering.conj().T
            - 0.5 * (excited_projector @ matrix + matrix @ excited_projector)
        )
        return result

    matrix_units = []
    generator_matrix = np.zeros((4, 4), dtype=complex)
    for row in range(2):
        for column in range(2):
            matrix_unit = np.zeros((2, 2), dtype=complex)
            matrix_unit[row, column] = 1.0
            matrix_units.append(matrix_unit)
            generator_matrix[:, 2 * row + column] = generator(matrix_unit).reshape(-1)

    propagator = expm(time * generator_matrix)
    return tuple(
        (propagator @ matrix_unit.reshape(-1)).reshape(2, 2)
        for matrix_unit in matrix_units
    )


def _apply_independent_joint_noise(
    rho,
    left_pauli_rates,
    left_adc_rate,
    right_pauli_rates,
    right_adc_rate,
    time,
):
    left_images = _joint_single_qubit_basis_images(
        left_pauli_rates, left_adc_rate, time
    )
    right_images = _joint_single_qubit_basis_images(
        right_pauli_rates, right_adc_rate, time
    )
    output = np.zeros((4, 4), dtype=complex)
    for left_row in range(2):
        for left_column in range(2):
            for right_row in range(2):
                for right_column in range(2):
                    coefficient = rho[
                        2 * left_row + right_row,
                        2 * left_column + right_column,
                    ]
                    output += coefficient * np.kron(
                        left_images[2 * left_row + left_column],
                        right_images[2 * right_row + right_column],
                    )
    return output


def _single_qubit_sequential_kraus(pauli_decay_rates, adc_rate, time, order):
    """Construct an ordered Pauli/ADC layer directly from Kraus operators."""
    mu_x, mu_y, mu_z = np.exp(-np.asarray(pauli_decay_rates, dtype=float) * time)
    probabilities = np.array(
        [
            1.0 + mu_x + mu_y + mu_z,
            1.0 + mu_x - mu_y - mu_z,
            1.0 - mu_x + mu_y - mu_z,
            1.0 - mu_x - mu_y + mu_z,
        ]
    ) / 4.0
    assert np.min(probabilities) >= -2.0e-14
    probabilities = np.maximum(probabilities, 0.0)
    pauli_kraus = tuple(
        np.sqrt(probability) * pauli
        for probability, pauli in zip(probabilities, (I2, X, Y, Z))
    )

    eta = np.exp(-adc_rate * time)
    adc_kraus = (
        np.diag([1.0, np.sqrt(eta)]).astype(complex),
        np.array([[0.0, np.sqrt(1.0 - eta)], [0.0, 0.0]], dtype=complex),
    )
    if order == "pauli_after_adc":
        return tuple(pauli @ adc for pauli in pauli_kraus for adc in adc_kraus)
    if order == "adc_after_pauli":
        return tuple(adc @ pauli for adc in adc_kraus for pauli in pauli_kraus)
    raise ValueError(order)


def _apply_independent_sequential_noise(
    rho,
    left_pauli_rates,
    left_adc_rate,
    right_pauli_rates,
    right_adc_rate,
    time,
    order,
):
    left_kraus = _single_qubit_sequential_kraus(
        left_pauli_rates, left_adc_rate, time, order
    )
    right_kraus = _single_qubit_sequential_kraus(
        right_pauli_rates, right_adc_rate, time, order
    )
    output = np.zeros((4, 4), dtype=complex)
    for left in left_kraus:
        for right in right_kraus:
            operator = np.kron(left, right)
            output += operator @ rho @ operator.conj().T
    return output


def _renewal_solution(failure_kernel, first_success):
    """Solve R = first_success + failure_kernel * R bin by bin.

    The failure kernel is scalar while ``first_success`` may carry arbitrary
    trailing matrix/state axes.  This recurrence is an implementation
    independent of ``RepeaterChainEvaluation.iterative_convolution``.
    """
    failure_kernel = np.asarray(failure_kernel, dtype=float)
    first_success = np.asarray(first_success)
    result = np.zeros_like(first_success)
    for time in range(len(failure_kernel)):
        result[time] = first_success[time]
        for failure_time in range(1, time + 1):
            result[time] += (
                failure_kernel[failure_time] * result[time - failure_time]
            )
    return result


def _assert_attempt_mass_conservation(attempt, input_mass=1.0):
    assert_allclose(
        np.sum(attempt.cutoff_failure)
        + np.sum(attempt.valid_success)
        + np.sum(attempt.valid_failure),
        input_mass,
        atol=3.0e-14,
    )
    assert_allclose(
        attempt.weighted_success[:, 0],
        attempt.valid_success,
        atol=3.0e-14,
    )


def _corrected_bell_branch(rho_left, rho_right, bell, correction):
    """Project the two inner qubits and correct the right outer qubit."""
    left = rho_left.reshape(2, 2, 2, 2)
    right = rho_right.reshape(2, 2, 2, 2)
    branch = np.einsum(
        "mn,amcp,ndqh,pq->adch",
        bell.conj(),
        left,
        right,
        bell,
    ).reshape(4, 4)
    correction_operator = np.kron(I2, correction)
    return correction_operator @ branch @ correction_operator.conj().T


def _dense_swap(rho_left, rho_right, outcomes):
    branches = {
        name: _corrected_bell_branch(rho_left, rho_right, bell, correction)
        for name, (bell, correction) in _BELL_DATA.items()
    }
    if outcomes == "phi_plus":
        return branches["phi+"]
    if outcomes == "all_corrected":
        return sum(branches.values(), np.zeros((4, 4), dtype=complex))
    raise ValueError(outcomes)


def _basis_index(bits):
    index = 0
    for bit in bits:
        index = 2 * index + bit
    return index


def _dense_recurrence(kept, auxiliary):
    """Explicit bilateral CNOT followed by 00/11 target postselection."""
    # ``input_for_output`` is the computational-basis permutation induced by
    # the two CNOT gates.  Indexing the density matrix by that permutation is
    # exactly U rho U^dagger, without relying on any compact state rule.
    input_for_output = np.empty(16, dtype=int)
    for kept_left in range(2):
        for kept_right in range(2):
            for auxiliary_left in range(2):
                for auxiliary_right in range(2):
                    before = (
                        kept_left,
                        kept_right,
                        auxiliary_left,
                        auxiliary_right,
                    )
                    after = (
                        kept_left,
                        kept_right,
                        auxiliary_left ^ kept_left,
                        auxiliary_right ^ kept_right,
                    )
                    input_for_output[_basis_index(after)] = _basis_index(before)

    four_qubit_state = np.kron(kept, auxiliary)
    evolved = four_qubit_state[np.ix_(input_for_output, input_for_output)]
    accepted = np.zeros((4, 4), dtype=complex)
    for outcome in (0, 1):
        for kept_left in range(2):
            for kept_right in range(2):
                for kept_left_bra in range(2):
                    for kept_right_bra in range(2):
                        output_row = 2 * kept_left + kept_right
                        output_column = 2 * kept_left_bra + kept_right_bra
                        row = _basis_index(
                            (kept_left, kept_right, outcome, outcome)
                        )
                        column = _basis_index(
                            (kept_left_bra, kept_right_bra, outcome, outcome)
                        )
                        accepted[output_row, output_column] += evolved[row, column]
    return accepted


def _phi_plus_density():
    phi_plus = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
    return np.outer(phi_plus, phi_plus.conj())


def test_state_construction_and_weighted_normalization_match_dense_states():
    rho = np.array(
        [
            [0.46, 0.0, 0.0, 0.22],
            [0.0, 0.09, 0.035, 0.0],
            [0.0, 0.035, 0.16, 0.0],
            [0.22, 0.0, 0.0, 0.29],
        ],
        dtype=complex,
    )
    expected = _dense_to_correlators(rho)
    assert_allclose(RealXState(matrix=rho).coordinates, expected, atol=1.0e-15)
    assert_allclose(
        RealXState(coordinates=expected).coordinates, expected, atol=1.0e-15
    )

    bell_weights = np.array([0.61, 0.17, 0.08, 0.14])
    bell_order = ("phi+", "psi+", "psi-", "phi-")
    bell_mixture = sum(
        weight * np.outer(bell.reshape(-1), bell.reshape(-1).conj())
        for weight, name in zip(bell_weights, bell_order)
        for bell in [_BELL_DATA[name][0]]
    )
    assert_allclose(
        RealXState(lambdas=bell_weights).coordinates,
        _dense_to_correlators(bell_mixture),
        atol=1.0e-15,
    )

    rho_second = np.array(
        [
            [0.31, 0.0, 0.0, -0.11],
            [0.0, 0.21, 0.06, 0.0],
            [0.0, 0.06, 0.18, 0.0],
            [-0.11, 0.0, 0.0, 0.30],
        ],
        dtype=complex,
    )
    pmf = np.zeros(8)
    pmf[2] = 0.37
    pmf[5] = 0.41
    weighted = np.zeros((8, 6))
    weighted[2] = pmf[2] * expected
    weighted[5] = pmf[5] * _dense_to_correlators(rho_second)
    normalized = normalize_weighted_x_func(weighted, pmf)
    assert_allclose(normalized[2], expected, atol=1.0e-15)
    assert_allclose(
        normalized[5], _dense_to_correlators(rho_second), atol=1.0e-15
    )
    assert_allclose(
        normalized[[0, 1, 3, 4, 6, 7]],
        np.tile([1, 0, 0, 0, 0, 0], (6, 1)),
    )

    broken_weight = weighted.copy()
    broken_weight[5, 0] += 1.0e-4
    with pytest.raises(ValueError, match="weighted II coordinate"):
        normalize_weighted_x_func(broken_weight, pmf)


def test_batched_polish_matches_dense_parity_block_projection():
    """The fast analytic repair is the same PSD-cone projection as eigh."""
    rng = np.random.default_rng(20260808)
    coordinates = []
    expected = []
    for index in range(200):
        angle_even, angle_odd = rng.uniform(-np.pi, np.pi, size=2)
        rotations = [
            np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            for angle in (angle_even, angle_odd)
        ]
        negative = -rng.uniform(1.0e-13, 5.0e-11)
        positive = rng.dirichlet(np.ones(3)) * (1.0 - negative)
        even_eigenvalues = (
            np.array([negative, positive[0]])
            if index % 2 == 0
            else positive[:2]
        )
        odd_eigenvalues = (
            positive[1:]
            if index % 2 == 0
            else np.array([negative, positive[2]])
        )
        # For odd-index rows the unused positive eigenvalue keeps trace one.
        if index % 2:
            even_eigenvalues *= (1.0 - negative) / np.sum(even_eigenvalues)
            odd_eigenvalues[1] = 1.0 - negative - np.sum(even_eigenvalues)

        blocks = [
            rotation @ np.diag(eigenvalues) @ rotation.T
            for rotation, eigenvalues in zip(
                rotations, (even_eigenvalues, odd_eigenvalues)
            )
        ]
        matrix = np.zeros((4, 4), dtype=float)
        matrix[np.ix_([0, 3], [0, 3])] = blocks[0]
        matrix[np.ix_([1, 2], [1, 2])] = blocks[1]
        matrix /= np.trace(matrix)
        coordinates.append(_dense_to_correlators(matrix))

        projected = np.zeros((4, 4), dtype=float)
        for indices in ([0, 3], [1, 2]):
            block = matrix[np.ix_(indices, indices)]
            eigenvalues, eigenvectors = np.linalg.eigh(block)
            block = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            projected[np.ix_(indices, indices)] = block
        projected /= np.trace(projected)
        expected.append(_dense_to_correlators(projected))

    actual = polish_x_func(np.asarray(coordinates), atol=1.0e-10)
    assert_allclose(actual, expected, atol=8.0e-16, rtol=0.0)


def test_batched_analytic_minimum_eigenvalue_matches_dense_spectrum():
    rng = np.random.default_rng(20260809)
    coordinates = rng.normal(size=(500, 6))
    dense = np.stack([_correlators_to_dense(row) for row in coordinates])
    expected = np.linalg.eigvalsh(dense)[:, 0]
    assert_allclose(
        _minimum_real_x_eigenvalue(coordinates),
        expected,
        atol=7.0e-16,
        rtol=2.0e-15,
    )


def test_batched_polish_reports_first_materially_nonphysical_bin():
    values = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], (7, 1))
    values[4] = [1.0, 0.0, 0.0, 1.0 + 5.0e-7, 0.0, 0.0]
    with pytest.raises(ValueError, match="index 4.*minimum eigenvalue"):
        polish_x_func(values)


@pytest.mark.parametrize(
    "outcomes,t_left,t_right,hardware_success,older_link,node_pair",
    [
        ("all_corrected", 1, 4, 0.73, "left", (0, 1)),
        ("phi_plus", 5, 2, 0.61, "right", (1, 2)),
    ],
)
def test_one_attempt_swap_with_endpoint_local_joint_noise_matches_dense_oracle(
    outcomes,
    t_left,
    t_right,
    hardware_success,
    older_link,
    node_pair,
):
    rho_left = np.array(
        [
            [0.41, 0.0, 0.0, 0.15],
            [0.0, 0.12, 0.04, 0.0],
            [0.0, 0.04, 0.18, 0.0],
            [0.15, 0.0, 0.0, 0.29],
        ],
        dtype=complex,
    )
    rho_right = np.array(
        [
            [0.36, 0.0, 0.0, -0.13],
            [0.0, 0.20, 0.055, 0.0],
            [0.0, 0.055, 0.14, 0.0],
            [-0.13, 0.0, 0.0, 0.30],
        ],
        dtype=complex,
    )
    node_pauli_rates = np.array(
        [
            [0.050, 0.060, 0.070],
            [0.040, 0.055, 0.065],
            [0.060, 0.070, 0.050],
        ]
    )
    node_adc_rates = np.array([0.080, 0.045, 0.105])
    size = 9
    pmf_left = np.zeros(size)
    pmf_right = np.zeros(size)
    pmf_left[t_left] = 1.0
    pmf_right[t_right] = 1.0
    sf_left = np.tile(_dense_to_correlators(rho_left), (size, 1))
    sf_right = np.tile(_dense_to_correlators(rho_right), (size, 1))

    actual = real_x_attempt_join(
        pmf_left,
        pmf_right,
        sf_left,
        sf_right,
        operation="swap",
        cutoff=size,
        hardware_success=hardware_success,
        bell_outcomes=outcomes,
        node_pauli_rates=node_pauli_rates,
        node_adc_rates=node_adc_rates,
        noise_model="joint",
    )

    first_node, second_node = node_pair
    time = abs(t_left - t_right)
    if older_link == "left":
        rho_left = _apply_independent_joint_noise(
            rho_left,
            node_pauli_rates[first_node],
            node_adc_rates[first_node],
            node_pauli_rates[second_node],
            node_adc_rates[second_node],
            time,
        )
    else:
        rho_right = _apply_independent_joint_noise(
            rho_right,
            node_pauli_rates[first_node],
            node_adc_rates[first_node],
            node_pauli_rates[second_node],
            node_adc_rates[second_node],
            time,
        )
    accepted = _dense_swap(rho_left, rho_right, outcomes)
    terminal_time = max(t_left, t_right)
    success = hardware_success * np.trace(accepted).real

    expected_success = np.zeros(size)
    expected_failure = np.zeros(size)
    expected_weighted = np.zeros((size, 6))
    expected_success[terminal_time] = success
    expected_failure[terminal_time] = 1.0 - success
    expected_weighted[terminal_time] = (
        hardware_success * _dense_to_correlators(accepted)
    )
    assert_allclose(actual.cutoff_failure, 0.0, atol=1.0e-15)
    assert_allclose(actual.valid_success, expected_success, atol=2.0e-13)
    assert_allclose(actual.valid_failure, expected_failure, atol=2.0e-13)
    assert_allclose(actual.weighted_success, expected_weighted, atol=2.0e-13)


@pytest.mark.parametrize("t_kept,t_auxiliary", [(2, 6), (7, 3)])
def test_unequal_time_distillation_with_endpoint_joint_noise_matches_dense_oracle(
    t_kept,
    t_auxiliary,
):
    """Waiting noise follows the older copy without changing kept/aux roles."""
    rho_kept = np.array(
        [
            [0.43, 0.0, 0.0, 0.19],
            [0.0, 0.11, 0.028, 0.0],
            [0.0, 0.028, 0.17, 0.0],
            [0.19, 0.0, 0.0, 0.29],
        ],
        dtype=complex,
    )
    rho_auxiliary = np.array(
        [
            [0.34, 0.0, 0.0, -0.12],
            [0.0, 0.19, 0.052, 0.0],
            [0.0, 0.052, 0.16, 0.0],
            [-0.12, 0.0, 0.0, 0.31],
        ],
        dtype=complex,
    )
    endpoint_pauli_rates = np.array(
        [[0.045, 0.060, 0.055], [0.070, 0.052, 0.064]]
    )
    endpoint_adc_rates = np.array([0.085, 0.035])
    hardware_success = 0.68
    size = 11
    pmf_kept = np.zeros(size)
    pmf_auxiliary = np.zeros(size)
    pmf_kept[t_kept] = 1.0
    pmf_auxiliary[t_auxiliary] = 1.0
    sf_kept = np.tile(_dense_to_correlators(rho_kept), (size, 1))
    sf_auxiliary = np.tile(_dense_to_correlators(rho_auxiliary), (size, 1))

    actual = real_x_attempt_join(
        pmf_kept,
        pmf_auxiliary,
        sf_kept,
        sf_auxiliary,
        operation="dist",
        cutoff=size,
        hardware_success=hardware_success,
        node_pauli_rates=endpoint_pauli_rates,
        node_adc_rates=endpoint_adc_rates,
        noise_model="joint",
        werner_twirl=False,
    )

    waiting_time = abs(t_kept - t_auxiliary)
    if t_kept < t_auxiliary:
        rho_kept = _apply_independent_joint_noise(
            rho_kept,
            endpoint_pauli_rates[0],
            endpoint_adc_rates[0],
            endpoint_pauli_rates[1],
            endpoint_adc_rates[1],
            waiting_time,
        )
    else:
        rho_auxiliary = _apply_independent_joint_noise(
            rho_auxiliary,
            endpoint_pauli_rates[0],
            endpoint_adc_rates[0],
            endpoint_pauli_rates[1],
            endpoint_adc_rates[1],
            waiting_time,
        )
    accepted = _dense_recurrence(rho_kept, rho_auxiliary)
    terminal_time = max(t_kept, t_auxiliary)
    success = hardware_success * np.trace(accepted).real

    expected_success = np.zeros(size)
    expected_failure = np.zeros(size)
    expected_weighted = np.zeros((size, 6))
    expected_success[terminal_time] = success
    expected_failure[terminal_time] = 1.0 - success
    expected_weighted[terminal_time] = (
        hardware_success * _dense_to_correlators(accepted)
    )
    assert_allclose(actual.cutoff_failure, 0.0, atol=1.0e-15)
    assert_allclose(actual.valid_success, expected_success, atol=2.0e-13)
    assert_allclose(actual.valid_failure, expected_failure, atol=2.0e-13)
    assert_allclose(actual.weighted_success, expected_weighted, atol=2.0e-13)
    _assert_attempt_mass_conservation(actual)


@pytest.mark.parametrize("order", ["pauli_after_adc", "adc_after_pauli"])
def test_sequential_order_join_matches_independent_kraus_oracle(order):
    rho_left = np.array(
        [
            [0.39, 0.0, 0.0, 0.14],
            [0.0, 0.13, 0.045, 0.0],
            [0.0, 0.045, 0.19, 0.0],
            [0.14, 0.0, 0.0, 0.29],
        ],
        dtype=complex,
    )
    rho_right = np.array(
        [
            [0.35, 0.0, 0.0, -0.10],
            [0.0, 0.22, 0.050, 0.0],
            [0.0, 0.050, 0.15, 0.0],
            [-0.10, 0.0, 0.0, 0.28],
        ],
        dtype=complex,
    )
    node_pauli_rates = np.array(
        [[0.040, 0.052, 0.060], [0.065, 0.050, 0.058], [0.055, 0.068, 0.061]]
    )
    node_adc_rates = np.array([0.080, 0.047, 0.096])
    size = 9
    t_left, t_right = 1, 5
    pmf_left = np.zeros(size)
    pmf_right = np.zeros(size)
    pmf_left[t_left] = 1.0
    pmf_right[t_right] = 1.0
    sf_left = np.tile(_dense_to_correlators(rho_left), (size, 1))
    sf_right = np.tile(_dense_to_correlators(rho_right), (size, 1))
    hardware_success = 0.77

    actual = real_x_attempt_join(
        pmf_left,
        pmf_right,
        sf_left,
        sf_right,
        operation="swap",
        cutoff=size,
        hardware_success=hardware_success,
        bell_outcomes="phi_plus",
        node_pauli_rates=node_pauli_rates,
        node_adc_rates=node_adc_rates,
        noise_model="sequential",
        sequential_order=order,
    )
    noisy_left = _apply_independent_sequential_noise(
        rho_left,
        node_pauli_rates[0],
        node_adc_rates[0],
        node_pauli_rates[1],
        node_adc_rates[1],
        t_right - t_left,
        order,
    )
    accepted = _dense_swap(noisy_left, rho_right, "phi_plus")
    success = hardware_success * np.trace(accepted).real
    expected_weighted = hardware_success * _dense_to_correlators(accepted)
    assert_allclose(actual.valid_success[t_right], success, atol=2.0e-13)
    assert_allclose(actual.valid_failure[t_right], 1.0 - success, atol=2.0e-13)
    assert_allclose(actual.weighted_success[t_right], expected_weighted, atol=2.0e-13)
    _assert_attempt_mass_conservation(actual)


def test_memory_cutoff_boundary_and_shifted_renewal_timing():
    coordinates = _dense_to_correlators(_phi_plus_density())
    size = 30
    cutoff = 3
    state_func = np.tile(coordinates, (size, 1))

    def deterministic_attempt(t_left, t_right):
        pmf_left = np.zeros(size)
        pmf_right = np.zeros(size)
        pmf_left[t_left] = 1.0
        pmf_right[t_right] = 1.0
        return real_x_attempt_join(
            pmf_left,
            pmf_right,
            state_func,
            state_func,
            cutoff=cutoff,
            cut_type="memory_time",
            bell_outcomes="all_corrected",
        )

    boundary_pass = deterministic_attempt(2, 2 + cutoff)
    assert_allclose(boundary_pass.valid_success[2 + cutoff], 1.0)
    assert_allclose(boundary_pass.cutoff_failure, 0.0)
    _assert_attempt_mass_conservation(boundary_pass)

    just_outside = deterministic_attempt(2, 2 + cutoff + 1)
    # The unshifted attempt kernel records when the first link arrived.
    assert_allclose(just_outside.cutoff_failure[2], 1.0)
    assert_allclose(just_outside.valid_success, 0.0)
    assert_allclose(just_outside.valid_failure, 0.0)
    _assert_attempt_mass_conservation(just_outside)

    # Mix a valid pair with the just-outside pair.  The evaluator must charge
    # the timeout at min(t1,t2)+cutoff = 2+3 = 5 before restarting.
    valid_probability = 0.60
    pmf_left = np.zeros(size)
    pmf_right = np.zeros(size)
    pmf_left[2] = 1.0
    pmf_right[2] = valid_probability
    pmf_right[2 + cutoff + 1] = 1.0 - valid_probability
    evaluator = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False
    )
    parameters = {
        "p_gen": 1.0,
        "p_swap": 1.0,
        "cut_type": "memory_time",
        "mt_cut": cutoff,
        "bell_outcomes": "all_corrected",
        "pauli_mode_decay_rates": 0.0,
        "amplitude_damping_rate": 0.0,
    }
    pmf, states = evaluator.compute_unit(
        parameters, pmf_left, state_func, pmf_right, state_func, unit_kind="swap"
    )
    expected = np.zeros(size)
    failure_duration = 2 + cutoff
    for failure_count in range(size):
        completion_time = 2 + failure_count * failure_duration
        if completion_time >= size:
            break
        expected[completion_time] = (
            valid_probability * (1.0 - valid_probability) ** failure_count
        )
    assert_allclose(pmf, expected, atol=3.0e-15)
    populated = expected > 0.0
    assert_allclose(
        states[populated],
        np.tile(coordinates, (np.count_nonzero(populated), 1)),
        atol=3.0e-15,
    )


def test_run_time_cutoff_boundary_and_renewal_timing():
    coordinates = _dense_to_correlators(_phi_plus_density())
    size = 25
    cutoff = 4
    state_func = np.tile(coordinates, (size, 1))

    def deterministic_attempt(t_right):
        pmf_left = np.zeros(size)
        pmf_right = np.zeros(size)
        pmf_left[1] = 1.0
        pmf_right[t_right] = 1.0
        return real_x_attempt_join(
            pmf_left,
            pmf_right,
            state_func,
            state_func,
            cutoff=cutoff,
            cut_type="run_time",
            bell_outcomes="all_corrected",
        )

    boundary_pass = deterministic_attempt(cutoff)
    assert_allclose(boundary_pass.valid_success[cutoff], 1.0)
    assert_allclose(boundary_pass.cutoff_failure, 0.0)
    _assert_attempt_mass_conservation(boundary_pass)

    just_outside = deterministic_attempt(cutoff + 1)
    assert_allclose(just_outside.cutoff_failure[cutoff], 1.0)
    assert_allclose(just_outside.valid_success, 0.0)
    _assert_attempt_mass_conservation(just_outside)

    valid_probability = 0.70
    pmf_left = np.zeros(size)
    pmf_right = np.zeros(size)
    pmf_left[1] = 1.0
    pmf_right[cutoff] = valid_probability
    pmf_right[cutoff + 1] = 1.0 - valid_probability
    evaluator = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False
    )
    parameters = {
        "p_gen": 1.0,
        "p_swap": 1.0,
        "cut_type": "run_time",
        "rt_cut": cutoff,
        "bell_outcomes": "all_corrected",
        "pauli_mode_decay_rates": 0.0,
        "amplitude_damping_rate": 0.0,
    }
    pmf, _ = evaluator.compute_unit(
        parameters, pmf_left, state_func, pmf_right, state_func, unit_kind="swap"
    )
    expected = np.zeros(size)
    for attempt_count in range(1, size):
        completion_time = attempt_count * cutoff
        if completion_time >= size:
            break
        expected[completion_time] = (
            valid_probability * (1.0 - valid_probability) ** (attempt_count - 1)
        )
    assert_allclose(pmf, expected, atol=3.0e-15)


def test_stochastic_two_leaf_protocol_matches_dense_full_reset_renewal_oracle():
    """End-to-end comparison including generation, waiting noise, and retries."""
    rho_left = np.array(
        [
            [0.44, 0.0, 0.0, 0.18],
            [0.0, 0.10, 0.032, 0.0],
            [0.0, 0.032, 0.17, 0.0],
            [0.18, 0.0, 0.0, 0.29],
        ],
        dtype=complex,
    )
    rho_right = np.array(
        [
            [0.37, 0.0, 0.0, -0.115],
            [0.0, 0.20, 0.049, 0.0],
            [0.0, 0.049, 0.15, 0.0],
            [-0.115, 0.0, 0.0, 0.28],
        ],
        dtype=complex,
    )
    node_pauli_rates = np.array(
        [[0.040, 0.050, 0.060], [0.055, 0.065, 0.050], [0.070, 0.060, 0.055]]
    )
    node_adc_rates = np.array([0.075, 0.042, 0.091])
    cutoff = 2
    hardware_success = 0.81
    size = 14
    p_left, p_right = 0.48, 0.61
    times = np.arange(1, size)
    pmf_left = np.concatenate(
        ([0.0], p_left * (1.0 - p_left) ** (times - 1))
    )
    pmf_right = np.concatenate(
        ([0.0], p_right * (1.0 - p_right) ** (times - 1))
    )
    sf_left = np.tile(_dense_to_correlators(rho_left), (size, 1))
    sf_right = np.tile(_dense_to_correlators(rho_right), (size, 1))

    # First construct the complete one-attempt kernels using only dense
    # matrices.  Caching by waiting time keeps the oracle easy to inspect.
    noisy_left = {0: rho_left}
    noisy_right = {0: rho_right}
    for waiting_time in range(1, cutoff + 1):
        noisy_left[waiting_time] = _apply_independent_joint_noise(
            rho_left,
            node_pauli_rates[0],
            node_adc_rates[0],
            node_pauli_rates[1],
            node_adc_rates[1],
            waiting_time,
        )
        noisy_right[waiting_time] = _apply_independent_joint_noise(
            rho_right,
            node_pauli_rates[1],
            node_adc_rates[1],
            node_pauli_rates[2],
            node_adc_rates[2],
            waiting_time,
        )

    cutoff_failure = np.zeros(size)
    valid_success = np.zeros(size)
    valid_failure = np.zeros(size)
    weighted_success = np.zeros((size, 4, 4), dtype=complex)
    for t_left in range(1, size):
        for t_right in range(1, size):
            pair_mass = pmf_left[t_left] * pmf_right[t_right]
            waiting_time = abs(t_left - t_right)
            if waiting_time > cutoff:
                cutoff_failure[min(t_left, t_right)] += pair_mass
                continue

            if t_left < t_right:
                left_state = noisy_left[waiting_time]
                right_state = rho_right
            elif t_right < t_left:
                left_state = rho_left
                right_state = noisy_right[waiting_time]
            else:
                left_state = rho_left
                right_state = rho_right
            accepted = _dense_swap(left_state, right_state, "phi_plus")
            weighted_branch = hardware_success * accepted
            success = np.trace(weighted_branch).real
            terminal_time = max(t_left, t_right)
            valid_success[terminal_time] += pair_mass * success
            valid_failure[terminal_time] += pair_mass * (1.0 - success)
            weighted_success[terminal_time] += pair_mass * weighted_branch

    input_pair_mass = np.sum(pmf_left) * np.sum(pmf_right)
    assert_allclose(
        np.sum(cutoff_failure) + np.sum(valid_success) + np.sum(valid_failure),
        input_pair_mass,
        atol=3.0e-14,
    )
    assert_allclose(
        np.trace(weighted_success, axis1=1, axis2=2).real,
        valid_success,
        atol=3.0e-14,
    )

    # A memory-time cutoff recorded at min(t1,t2) consumes ``cutoff`` more
    # time units.  Solve both renewal layers directly as R = G + F * R.
    shifted_cutoff_failure = np.zeros(size)
    shifted_cutoff_failure[cutoff:] = cutoff_failure[: size - cutoff]
    success_after_cutoffs = _renewal_solution(
        shifted_cutoff_failure, valid_success
    )
    failure_after_cutoffs = _renewal_solution(
        shifted_cutoff_failure, valid_failure
    )
    weighted_after_cutoffs = _renewal_solution(
        shifted_cutoff_failure, weighted_success
    )
    expected_pmf = _renewal_solution(failure_after_cutoffs, success_after_cutoffs)
    expected_weighted = _renewal_solution(
        failure_after_cutoffs, weighted_after_cutoffs
    )
    assert_allclose(
        np.trace(expected_weighted, axis1=1, axis2=2).real,
        expected_pmf,
        atol=2.0e-13,
    )

    evaluator = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False
    )
    parameters = {
        "p_gen": [p_left, p_right],
        "p_swap": hardware_success,
        "cut_type": "memory_time",
        "mt_cut": cutoff,
        "bell_outcomes": "phi_plus",
        "pauli_mode_decay_rates": node_pauli_rates,
        "amplitude_damping_rate": node_adc_rates,
        "pauli_adc_noise_model": "joint",
    }
    actual_pmf, actual_states = evaluator.compute_unit(
        parameters,
        pmf_left,
        sf_left,
        pmf_right,
        sf_right,
        unit_kind="swap",
    )
    actual_weighted = np.stack(
        [
            actual_pmf[time] * _correlators_to_dense(actual_states[time])
            for time in range(size)
        ]
    )
    assert_allclose(actual_pmf, expected_pmf, atol=5.0e-13)
    assert_allclose(actual_weighted, expected_weighted, atol=5.0e-13)


def test_random_recurrence_distillation_matches_explicit_four_qubit_circuit():
    rng = np.random.default_rng(20260802)
    for _ in range(100):
        rho_kept = _random_real_x_density(rng)
        rho_auxiliary = _random_real_x_density(rng)
        expected_matrix = _dense_recurrence(rho_kept, rho_auxiliary)
        expected = _dense_to_correlators(expected_matrix)

        actual, probability = recurrence_distillation(
            _dense_to_correlators(rho_kept),
            _dense_to_correlators(rho_auxiliary),
        )
        assert_allclose(actual, expected, atol=5.0e-15)
        assert_allclose(probability, np.trace(expected_matrix).real, atol=5.0e-15)

        conditioned, conditioned_probability = recurrence_distillation(
            _dense_to_correlators(rho_kept),
            _dense_to_correlators(rho_auxiliary),
            conditioned=True,
        )
        assert_allclose(conditioned_probability, probability, atol=5.0e-15)
        assert_allclose(
            _correlators_to_dense(conditioned),
            expected_matrix / probability,
            atol=5.0e-15,
        )


def test_retry_pmf_for_phi_plus_postselection_has_geometric_golden_values():
    rho = _phi_plus_density()
    coordinates = _dense_to_correlators(rho)
    size = 18
    input_pmf = np.zeros(size)
    input_pmf[1] = 1.0
    state_func = np.tile(coordinates, (size, 1))
    evaluator = RepeaterChainEvaluation(
        state_type=RealXState, use_fft=False, efficient=False
    )
    parameters = {
        "p_gen": 1.0,
        "p_swap": 1.0,
        "mt_cut": size + 1,
        "bell_outcomes": "phi_plus",
        "pauli_mode_decay_rates": 0.0,
        "amplitude_damping_rate": 0.0,
    }
    pmf, states = evaluator.compute_unit(
        parameters,
        input_pmf,
        state_func,
        unit_kind="swap",
    )

    expected_pmf = np.zeros(size)
    for attempt_count in range(1, size):
        expected_pmf[attempt_count] = 0.25 * 0.75 ** (attempt_count - 1)
    assert_allclose(pmf, expected_pmf, atol=2.0e-15)
    assert_allclose(
        states[pmf > 0.0],
        np.tile(coordinates, (np.count_nonzero(pmf > 0.0), 1)),
        atol=2.0e-15,
    )


def test_small_nested_swap_then_distillation_matches_dense_circuit_and_retries():
    rho = np.array(
        [
            [0.46, 0.0, 0.0, 0.22],
            [0.0, 0.09, 0.035, 0.0],
            [0.0, 0.035, 0.16, 0.0],
            [0.22, 0.0, 0.0, 0.29],
        ],
        dtype=complex,
    )
    coordinates = _dense_to_correlators(rho)
    size = 18
    evaluator = RepeaterChainEvaluation(
        state_type=RealXState,
        use_fft=False,
        efficient=False,
        w_twirling=False,
    )
    parameters = {
        "protocol": (0, 1),
        "p_gen": 1.0,
        "p_swap": 1.0,
        "p_distillation": 1.0,
        "t_trunc": size,
        "mt_cut": size + 1,
        "real_x_coordinates": coordinates,
        "bell_outcomes": "all_corrected",
        "pauli_mode_decay_rates": 0.0,
        "amplitude_damping_rate": 0.0,
        "real_x_werner_twirling": False,
    }
    levels = evaluator.nested_protocol(parameters, all_level=True)
    assert len(levels) == 3

    swap_matrix = _dense_swap(rho, rho, "all_corrected")
    assert_allclose(np.trace(swap_matrix).real, 1.0, atol=2.0e-15)
    swap_coordinates = _dense_to_correlators(swap_matrix)
    swap_pmf, swap_states = levels[1]
    expected_swap_pmf = np.zeros(size)
    expected_swap_pmf[1] = 1.0
    assert_allclose(swap_pmf, expected_swap_pmf, atol=2.0e-15)
    assert_allclose(swap_states[1], swap_coordinates, atol=2.0e-15)

    distilled_matrix = _dense_recurrence(swap_matrix, swap_matrix)
    distillation_success = np.trace(distilled_matrix).real
    conditioned_coordinates = _dense_to_correlators(
        distilled_matrix / distillation_success
    )
    expected_distillation_pmf = np.zeros(size)
    for attempt_count in range(1, size):
        expected_distillation_pmf[attempt_count] = (
            distillation_success * (1.0 - distillation_success) ** (attempt_count - 1)
        )

    distillation_pmf, distillation_states = levels[2]
    assert_allclose(distillation_pmf, expected_distillation_pmf, atol=3.0e-15)
    assert_allclose(
        distillation_states[distillation_pmf > 0.0],
        np.tile(
            conditioned_coordinates,
            (np.count_nonzero(distillation_pmf > 0.0), 1),
        ),
        atol=3.0e-15,
    )
