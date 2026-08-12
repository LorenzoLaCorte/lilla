"""Symbolic checks for Pauli noise concatenated with amplitude damping.

These tests ground the analytical real-X-state derivation used in the research
notes.  The convention is

    E = P o A_eta,

so amplitude damping acts first and the Pauli channel acts second.  Pauli
channel eigenvalues are denoted by mu, while xi denotes eigenvalues of E.
"""

import sympy as sp


I2 = sp.eye(2)
X = sp.Matrix([[0, 1], [1, 0]])
Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
Z = sp.diag(1, -1)
PAULIS = (I2, X, Y, Z)


def assert_zero(expr):
    assert sp.simplify(sp.expand(expr)) == 0


def assert_matrix_zero(matrix):
    for expr in matrix:
        assert_zero(expr)


def pauli_channel(operator, probabilities):
    return sum(
        (
            probability * pauli * operator * pauli
            for probability, pauli in zip(probabilities, PAULIS)
        ),
        sp.zeros(2),
    )


def amplitude_damping(operator, eta):
    k0 = sp.diag(1, sp.sqrt(eta))
    k1 = sp.Matrix([[0, sp.sqrt(1 - eta)], [0, 0]])
    return k0 * operator * k0.T + k1 * operator * k1.T


def concatenated_channel(operator, probabilities, eta):
    return pauli_channel(amplitude_damping(operator, eta), probabilities)


def reversed_concatenated_channel(operator, probabilities, eta):
    return amplitude_damping(pauli_channel(operator, probabilities), eta)


def pauli_transfer_matrix(channel):
    return sp.Matrix(
        4,
        4,
        lambda row, column: sp.trace(PAULIS[row] * channel(PAULIS[column])) / 2,
    )


def channel_from_transfer_matrix(transfer_matrix, operator):
    input_coefficients = sp.Matrix(
        [sp.trace(pauli * operator) / 2 for pauli in PAULIS]
    )
    output_coefficients = transfer_matrix * input_coefficients
    return sum(
        (
            coefficient * pauli
            for coefficient, pauli in zip(output_coefficients, PAULIS)
        ),
        sp.zeros(2),
    )


def apply_same_local_channel(two_qubit_operator, channel):
    """Apply channel tensor channel without introducing symbolic Kraus roots."""
    matrix_units = (
        sp.Matrix([[1, 0], [0, 0]]),
        sp.Matrix([[0, 1], [0, 0]]),
        sp.Matrix([[0, 0], [1, 0]]),
        sp.Matrix([[0, 0], [0, 1]]),
    )
    result = sp.zeros(4)
    for left_row in range(2):
        for left_column in range(2):
            left_unit = matrix_units[2 * left_row + left_column]
            left_output = channel(left_unit)
            for right_row in range(2):
                for right_column in range(2):
                    coefficient = two_qubit_operator[
                        2 * left_row + right_row,
                        2 * left_column + right_column,
                    ]
                    if coefficient == 0:
                        continue
                    right_unit = matrix_units[2 * right_row + right_column]
                    result += coefficient * sp.kronecker_product(
                        left_output, channel(right_unit)
                    )
    return result


def phi_plus_swap(rho_a, rho_b):
    """Project the inner qubits of rho_a tensor rho_b onto phi+."""
    result = sp.zeros(4)
    output_index = lambda left, right: 2 * left + right
    link_index = lambda first, second: 2 * first + second

    for left in range(2):
        for right in range(2):
            for left_prime in range(2):
                for right_prime in range(2):
                    result[
                        output_index(left, right),
                        output_index(left_prime, right_prime),
                    ] = sum(
                        (
                            rho_a[
                                link_index(left, measured),
                                link_index(left_prime, measured_prime),
                            ]
                            * rho_b[
                                link_index(measured, right),
                                link_index(measured_prime, right_prime),
                            ]
                            / 2
                            for measured in range(2)
                            for measured_prime in range(2)
                        ),
                        sp.S.Zero,
                    )
    return result


def test_pauli_channel_is_diagonal_in_the_pauli_right_eigenoperators():
    p_i, p_x, p_y, p_z = sp.symbols("p_i p_x p_y p_z")
    probabilities = (p_i, p_x, p_y, p_z)
    mu = (
        p_i + p_x + p_y + p_z,
        p_i + p_x - p_y - p_z,
        p_i - p_x + p_y - p_z,
        p_i - p_x - p_y + p_z,
    )

    for pauli, eigenvalue in zip(PAULIS, mu):
        assert_matrix_zero(pauli_channel(pauli, probabilities) - eigenvalue * pauli)

    expected_transfer_matrix = sp.diag(*mu)
    actual_transfer_matrix = pauli_transfer_matrix(
        lambda operator: pauli_channel(operator, probabilities)
    )
    assert_matrix_zero(actual_transfer_matrix - expected_transfer_matrix)


def test_concatenated_transfer_matrix_and_channel_order():
    p_i, p_x, p_y, p_z, eta = sp.symbols("p_i p_x p_y p_z eta")
    probabilities = (p_i, p_x, p_y, p_z)
    mu1 = p_i + p_x - p_y - p_z
    mu2 = p_i - p_x + p_y - p_z
    mu3 = p_i - p_x - p_y + p_z

    actual_transfer_matrix = pauli_transfer_matrix(
        lambda operator: concatenated_channel(operator, probabilities, eta)
    )
    expected_transfer_matrix = sp.Matrix(
        [
            [p_i + p_x + p_y + p_z, 0, 0, 0],
            [0, mu1 * sp.sqrt(eta), 0, 0],
            [0, 0, mu2 * sp.sqrt(eta), 0],
            [mu3 * (1 - eta), 0, 0, mu3 * eta],
        ]
    )
    assert_matrix_zero(actual_transfer_matrix - expected_transfer_matrix)

    pauli_after_adc = concatenated_channel(I2 / 2, probabilities, eta)
    adc_after_pauli = amplitude_damping(
        pauli_channel(I2 / 2, probabilities), eta
    )
    mu0 = p_i + p_x + p_y + p_z
    expected_order_difference = (mu3 - mu0) * (1 - eta) * Z / 2
    assert_matrix_zero(
        pauli_after_adc - adc_after_pauli - expected_order_difference
    )


def test_reversed_concatenation_transfer_matrix_and_right_eigenoperators():
    p_i, p_x, p_y, p_z, eta = sp.symbols("p_i p_x p_y p_z eta")
    probabilities = (p_i, p_x, p_y, p_z)
    mu0 = p_i + p_x + p_y + p_z
    mu1 = p_i + p_x - p_y - p_z
    mu2 = p_i - p_x + p_y - p_z
    mu3 = p_i - p_x - p_y + p_z

    actual_transfer_matrix = pauli_transfer_matrix(
        lambda operator: reversed_concatenated_channel(
            operator, probabilities, eta
        )
    )
    expected_transfer_matrix = sp.Matrix(
        [
            [mu0, 0, 0, 0],
            [0, mu1 * sp.sqrt(eta), 0, 0],
            [0, 0, mu2 * sp.sqrt(eta), 0],
            [mu0 * (1 - eta), 0, 0, mu3 * eta],
        ]
    )
    assert_matrix_zero(actual_transfer_matrix - expected_transfer_matrix)

    normalized_transfer_matrix = sp.Matrix(
        [
            [1, 0, 0, 0],
            [0, mu1 * sp.sqrt(eta), 0, 0],
            [0, 0, mu2 * sp.sqrt(eta), 0],
            [1 - eta, 0, 0, mu3 * eta],
        ]
    )
    channel = lambda operator: channel_from_transfer_matrix(
        normalized_transfer_matrix, operator
    )
    right_eigenoperators = (
        (1 - mu3 * eta) * I2 + (1 - eta) * Z,
        X,
        Y,
        Z,
    )
    eigenvalues = (
        1,
        mu1 * sp.sqrt(eta),
        mu2 * sp.sqrt(eta),
        mu3 * eta,
    )
    for operator, eigenvalue in zip(right_eigenoperators, eigenvalues):
        assert_matrix_zero(channel(operator) - eigenvalue * operator)


def test_concatenated_channel_right_eigenoperators():
    mu1, mu2, mu3, eta = sp.symbols("mu1 mu2 mu3 eta")
    transfer_matrix = sp.Matrix(
        [
            [1, 0, 0, 0],
            [0, mu1 * sp.sqrt(eta), 0, 0],
            [0, 0, mu2 * sp.sqrt(eta), 0],
            [mu3 * (1 - eta), 0, 0, mu3 * eta],
        ]
    )
    channel = lambda operator: channel_from_transfer_matrix(
        transfer_matrix, operator
    )

    right_eigenoperators = (
        (1 - mu3 * eta) * I2 + mu3 * (1 - eta) * Z,
        X,
        Y,
        Z,
    )
    eigenvalues = (
        1,
        mu1 * sp.sqrt(eta),
        mu2 * sp.sqrt(eta),
        mu3 * eta,
    )
    for operator, eigenvalue in zip(right_eigenoperators, eigenvalues):
        assert_matrix_zero(channel(operator) - eigenvalue * operator)


def test_real_x_state_expansion_in_the_right_eigenoperator_basis():
    a, b, c, w, z = sp.symbols("a b c w z")
    mu3, eta = sp.symbols("mu3 eta")
    d = 1 - a - b - c
    rho = sp.Matrix(
        [
            [a, 0, 0, w],
            [0, b, z, 0],
            [0, z, c, 0],
            [w, 0, 0, d],
        ]
    )

    cap_a = 1 - mu3 * eta
    cap_b = mu3 * (1 - eta)
    r0 = cap_a * I2 + cap_b * Z
    u = a + b - c - d
    v = a - b + c - d
    t = a - b - c + d
    alpha = 1 / (4 * cap_a**2)
    beta = (v * cap_a - cap_b) / (4 * cap_a**2)
    gamma = (u * cap_a - cap_b) / (4 * cap_a**2)
    delta = (
        t - cap_b * (u + v) / cap_a + cap_b**2 / cap_a**2
    ) / 4
    r = (z + w) / 2
    s = (z - w) / 2

    reconstructed = (
        alpha * sp.kronecker_product(r0, r0)
        + beta * sp.kronecker_product(r0, Z)
        + gamma * sp.kronecker_product(Z, r0)
        + delta * sp.kronecker_product(Z, Z)
        + r * sp.kronecker_product(X, X)
        + s * sp.kronecker_product(Y, Y)
    )
    assert_matrix_zero(reconstructed - rho)


def test_real_x_coefficients_follow_the_diagonal_decay_rule():
    alpha, beta, gamma, delta, r, s = sp.symbols(
        "alpha beta gamma delta r s"
    )
    mu1, mu2, mu3, eta = sp.symbols("mu1 mu2 mu3 eta")
    cap_a = 1 - mu3 * eta
    cap_b = mu3 * (1 - eta)
    r0 = cap_a * I2 + cap_b * Z

    transfer_matrix = sp.Matrix(
        [
            [1, 0, 0, 0],
            [0, mu1 * sp.sqrt(eta), 0, 0],
            [0, 0, mu2 * sp.sqrt(eta), 0],
            [mu3 * (1 - eta), 0, 0, mu3 * eta],
        ]
    )
    channel = lambda operator: channel_from_transfer_matrix(
        transfer_matrix, operator
    )
    rho = (
        alpha * sp.kronecker_product(r0, r0)
        + beta * sp.kronecker_product(r0, Z)
        + gamma * sp.kronecker_product(Z, r0)
        + delta * sp.kronecker_product(Z, Z)
        + r * sp.kronecker_product(X, X)
        + s * sp.kronecker_product(Y, Y)
    )
    directly_evolved = apply_same_local_channel(rho, channel)
    expected = (
        alpha * sp.kronecker_product(r0, r0)
        + mu3 * eta * beta * sp.kronecker_product(r0, Z)
        + mu3 * eta * gamma * sp.kronecker_product(Z, r0)
        + (mu3 * eta) ** 2 * delta * sp.kronecker_product(Z, Z)
        + mu1**2 * eta * r * sp.kronecker_product(X, X)
        + mu2**2 * eta * s * sp.kronecker_product(Y, Y)
    )
    assert_matrix_zero(directly_evolved - expected)

    gamma_depol, gamma_adc, time = sp.symbols(
        "gamma_depol gamma_adc time"
    )
    depolar_decay = sp.exp(-gamma_depol * time)
    adc_survival = sp.exp(-gamma_adc * time)
    assert_zero(
        depolar_decay * adc_survival
        - sp.exp(-(gamma_depol + gamma_adc) * time)
    )
    assert_zero(
        (depolar_decay * adc_survival) ** 2
        - sp.exp(-2 * (gamma_depol + gamma_adc) * time)
    )
    assert_zero(
        depolar_decay**2 * adc_survival
        - sp.exp(-(2 * gamma_depol + gamma_adc) * time)
    )
    assert_zero(
        1
        - depolar_decay * adc_survival
        - (1 - sp.exp(-(gamma_depol + gamma_adc) * time))
    )
    assert_zero(
        depolar_decay * (1 - adc_survival)
        - (
            sp.exp(-gamma_depol * time)
            - sp.exp(-(gamma_depol + gamma_adc) * time)
        )
    )


def test_phi_plus_swap_closes_on_the_six_right_eigenoperator_products():
    cap_a_a, cap_b_a, cap_a_b, cap_b_b = sp.symbols(
        "cap_a_a cap_b_a cap_a_b cap_b_b"
    )
    alpha_a, beta_a, gamma_a, delta_a, r_a, s_a = sp.symbols(
        "alpha_a beta_a gamma_a delta_a r_a s_a"
    )
    alpha_b, beta_b, gamma_b, delta_b, r_b, s_b = sp.symbols(
        "alpha_b beta_b gamma_b delta_b r_b s_b"
    )
    r0_a = cap_a_a * I2 + cap_b_a * Z
    r0_b = cap_a_b * I2 + cap_b_b * Z

    rho_a = (
        alpha_a * sp.kronecker_product(r0_a, r0_a)
        + beta_a * sp.kronecker_product(r0_a, Z)
        + gamma_a * sp.kronecker_product(Z, r0_a)
        + delta_a * sp.kronecker_product(Z, Z)
        + r_a * sp.kronecker_product(X, X)
        + s_a * sp.kronecker_product(Y, Y)
    )
    rho_b = (
        alpha_b * sp.kronecker_product(r0_b, r0_b)
        + beta_b * sp.kronecker_product(r0_b, Z)
        + gamma_b * sp.kronecker_product(Z, r0_b)
        + delta_b * sp.kronecker_product(Z, Z)
        + r_b * sp.kronecker_product(X, X)
        + s_b * sp.kronecker_product(Y, Y)
    )

    g00 = 2 * (cap_a_a * cap_a_b + cap_b_a * cap_b_b)
    g03 = 2 * cap_b_a
    g30 = 2 * cap_b_b
    alpha_out = (
        g00 * alpha_a * alpha_b
        + g03 * alpha_a * gamma_b
        + g30 * beta_a * alpha_b
        + 2 * beta_a * gamma_b
    ) / 2
    beta_out = (
        g00 * alpha_a * beta_b
        + g03 * alpha_a * delta_b
        + g30 * beta_a * beta_b
        + 2 * beta_a * delta_b
    ) / 2
    gamma_out = (
        g00 * gamma_a * alpha_b
        + g03 * gamma_a * gamma_b
        + g30 * delta_a * alpha_b
        + 2 * delta_a * gamma_b
    ) / 2
    delta_out = (
        g00 * gamma_a * beta_b
        + g03 * gamma_a * delta_b
        + g30 * delta_a * beta_b
        + 2 * delta_a * delta_b
    ) / 2
    expected = (
        alpha_out * sp.kronecker_product(r0_a, r0_b)
        + beta_out * sp.kronecker_product(r0_a, Z)
        + gamma_out * sp.kronecker_product(Z, r0_b)
        + delta_out * sp.kronecker_product(Z, Z)
        + r_a * r_b * sp.kronecker_product(X, X)
        - s_a * s_b * sp.kronecker_product(Y, Y)
    )
    assert_matrix_zero(phi_plus_swap(rho_a, rho_b) - expected)

    factorized_alpha = (
        cap_a_a * alpha_a * cap_a_b * alpha_b
        + (cap_b_a * alpha_a + beta_a)
        * (cap_b_b * alpha_b + gamma_b)
    )
    assert_zero(alpha_out - factorized_alpha)
