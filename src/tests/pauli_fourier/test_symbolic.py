"""Symbolic checks for the Pauli-Fourier Bell-diagonal representation.

These tests verify that the Bell-diagonal backend can be reimplemented by
propagating Pauli-Fourier coordinates mu instead of Bell weights lambda.
They use the implemented Bell order (phi+, psi+, psi-, phi-) = (I, X, Y, Z)
and the reference/Werner convention that the smaller completion time is the
stored older link.
"""

import sympy as sp


def mu_to_lambda(mu):
    """Bell weights in order (phi+, psi+, psi-, phi-) from mu=(1,mu1,mu2,mu3)."""
    mu0, mu1, mu2, mu3 = mu
    return sp.Matrix(
        [
            (mu0 + mu1 + mu2 + mu3) / 4,
            (mu0 + mu1 - mu2 - mu3) / 4,
            (mu0 - mu1 + mu2 - mu3) / 4,
            (mu0 - mu1 - mu2 + mu3) / 4,
        ]
    )


def lambda_to_mu(lam):
    """Pauli-Fourier coordinates from Bell weights in code order."""
    l0, l1, l2, l3 = lam
    return sp.Matrix(
        [
            l0 + l1 + l2 + l3,
            l0 + l1 - l2 - l3,
            l0 - l1 + l2 - l3,
            l0 - l1 - l2 + l3,
        ]
    )


def swap_lambda(a, b):
    """Current Bell-diagonal swap rule."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return sp.Matrix(
        [
            a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 + a3 * b2,
            a0 * b2 + a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0,
        ]
    )


def dist_lambda_numerators(a, b):
    """Current untwirled Bell-diagonal p_success * lambda_out rule."""
    return sp.Matrix(
        [
            a[0] * b[0] + a[3] * b[3],
            a[1] * b[1] + a[2] * b[2],
            a[1] * b[2] + a[2] * b[1],
            a[0] * b[3] + a[3] * b[0],
        ]
    )


def assert_zero(expr):
    assert sp.simplify(expr) == 0


def assert_vec_zero(vec):
    for expr in vec:
        assert_zero(expr)


def test_mu_lambda_maps_are_inverse_with_code_bell_order():
    mu1, mu2, mu3 = sp.symbols("mu1 mu2 mu3")
    p_i, p_x, p_y, p_z = sp.symbols("p_i p_x p_y p_z")

    mu = sp.Matrix([1, mu1, mu2, mu3])
    assert_vec_zero(lambda_to_mu(mu_to_lambda(mu)) - mu)

    pauli_probabilities_in_code_bell_order = sp.Matrix([p_i, p_x, p_y, p_z])
    expected_mu = sp.Matrix(
        [
            p_i + p_x + p_y + p_z,
            p_i + p_x - p_y - p_z,
            p_i - p_x + p_y - p_z,
            p_i - p_x - p_y + p_z,
        ]
    )
    assert_vec_zero(lambda_to_mu(pauli_probabilities_in_code_bell_order) - expected_mu)


def test_depolarizing_and_dephasing_are_diagonal_in_mu():
    mu1, mu2, mu3, eta_d, eta_z = sp.symbols("mu1 mu2 mu3 eta_d eta_z")
    lam = mu_to_lambda(sp.Matrix([1, mu1, mu2, mu3]))

    depol_in_lambdas = sp.Matrix([eta_d * li + (1 - eta_d) / 4 for li in lam])
    depol_in_mu = mu_to_lambda(sp.Matrix([1, eta_d * mu1, eta_d * mu2, eta_d * mu3]))
    assert_vec_zero(depol_in_lambdas - depol_in_mu)

    p_flip = (1 - eta_z) / 2
    dephase_in_lambdas = sp.Matrix(
        [
            (1 - p_flip) * lam[0] + p_flip * lam[3],
            (1 - p_flip) * lam[1] + p_flip * lam[2],
            (1 - p_flip) * lam[2] + p_flip * lam[1],
            (1 - p_flip) * lam[3] + p_flip * lam[0],
        ]
    )
    dephase_in_mu = mu_to_lambda(sp.Matrix([1, eta_z * mu1, eta_z * mu2, mu3]))
    assert_vec_zero(dephase_in_lambdas - dephase_in_mu)


def test_swap_is_componentwise_multiplication_in_mu_even_with_waiting_noise():
    mu1, mu2, mu3 = sp.symbols("mu1 mu2 mu3")
    nu1, nu2, nu3 = sp.symbols("nu1 nu2 nu3")
    r1, r2, r3 = sp.symbols("r1 r2 r3")

    mu = sp.Matrix([1, mu1, mu2, mu3])
    nu = sp.Matrix([1, nu1, nu2, nu3])
    lam_a = mu_to_lambda(mu)
    lam_b = mu_to_lambda(nu)

    direct = swap_lambda(lam_a, lam_b)
    mu_only = mu_to_lambda(sp.Matrix([1, mu1 * nu1, mu2 * nu2, mu3 * nu3]))
    assert_vec_zero(direct - mu_only)

    noisy_nu = sp.Matrix([1, r1 * nu1, r2 * nu2, r3 * nu3])
    direct_noisy = swap_lambda(lam_a, mu_to_lambda(noisy_nu))
    mu_only_noisy = mu_to_lambda(sp.Matrix([1, r1 * mu1 * nu1, r2 * mu2 * nu2, r3 * mu3 * nu3]))
    assert_vec_zero(direct_noisy - mu_only_noisy)


def test_distillation_weighted_mu_numerators_need_only_mu():
    mu1, mu2, mu3 = sp.symbols("mu1 mu2 mu3")
    nu1, nu2, nu3 = sp.symbols("nu1 nu2 nu3")
    r1, r2, r3 = sp.symbols("r1 r2 r3")

    mu = sp.Matrix([1, mu1, mu2, mu3])
    nu = sp.Matrix([1, nu1, nu2, nu3])

    weighted_mu = lambda_to_mu(dist_lambda_numerators(mu_to_lambda(mu), mu_to_lambda(nu)))
    expected_weighted_mu = sp.Matrix(
        [
            (1 + mu3 * nu3) / 2,
            (mu1 * nu1 + mu2 * nu2) / 2,
            (mu1 * nu2 + mu2 * nu1) / 2,
            (mu3 + nu3) / 2,
        ]
    )
    assert_vec_zero(weighted_mu - expected_weighted_mu)

    noisy_nu = sp.Matrix([1, r1 * nu1, r2 * nu2, r3 * nu3])
    noisy_weighted_mu = lambda_to_mu(dist_lambda_numerators(mu_to_lambda(mu), mu_to_lambda(noisy_nu)))
    expected_noisy_weighted_mu = sp.Matrix(
        [
            (1 + mu3 * r3 * nu3) / 2,
            (mu1 * r1 * nu1 + mu2 * r2 * nu2) / 2,
            (mu1 * r2 * nu2 + mu2 * r1 * nu1) / 2,
            (mu3 + r3 * nu3) / 2,
        ]
    )
    assert_vec_zero(noisy_weighted_mu - expected_noisy_weighted_mu)

    p_success = expected_noisy_weighted_mu[0]
    conditioned_mu = noisy_weighted_mu / p_success
    expected_conditioned_mu = sp.Matrix(
        [
            1,
            (mu1 * r1 * nu1 + mu2 * r2 * nu2) / (1 + mu3 * r3 * nu3),
            (mu1 * r2 * nu2 + mu2 * r1 * nu1) / (1 + mu3 * r3 * nu3),
            (mu3 + r3 * nu3) / (1 + mu3 * r3 * nu3),
        ]
    )
    assert_vec_zero(conditioned_mu - expected_conditioned_mu)


def test_existing_w_twirling_option_can_be_expressed_in_weighted_mu_form():
    mu1, mu2, mu3 = sp.symbols("mu1 mu2 mu3")
    nu1, nu2, nu3 = sp.symbols("nu1 nu2 nu3")

    mu = sp.Matrix([1, mu1, mu2, mu3])
    nu = sp.Matrix([1, nu1, nu2, nu3])
    numerators = dist_lambda_numerators(mu_to_lambda(mu), mu_to_lambda(nu))
    p_success = sum(numerators)

    twirled_lambda_numerator = sp.Matrix(
        [
            numerators[0],
            (p_success - numerators[0]) / 3,
            (p_success - numerators[0]) / 3,
            (p_success - numerators[0]) / 3,
        ]
    )
    weighted_twirled_mu = lambda_to_mu(twirled_lambda_numerator)

    phi_plus_numerator_from_mu = (
        (1 + mu3) * (1 + nu3) + (mu1 + mu2) * (nu1 + nu2)
    ) / 8
    expected_nontrivial_mu = (4 * phi_plus_numerator_from_mu - p_success) / 3
    expected_weighted_twirled_mu = sp.Matrix(
        [
            p_success,
            expected_nontrivial_mu,
            expected_nontrivial_mu,
            expected_nontrivial_mu,
        ]
    )
    assert_vec_zero(weighted_twirled_mu - expected_weighted_twirled_mu)
