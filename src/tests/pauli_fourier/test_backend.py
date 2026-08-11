import numpy as np
from numpy.testing import assert_allclose

from src.core.bell.protocol_units import (
    dephasing_noise,
    depolarizing_noise,
    get_dist_lambda_out,
    get_dist_prob_suc as get_bell_dist_prob_suc,
    get_swap_lambda_out,
)
from src.core.bell.state import BellState
from src.core.pauli_fourier.protocol_units import (
    get_dist_mu_out,
    get_dist_prob_suc,
    get_swap_mu_out,
    pauli_fourier_noise,
)
from src.core.pauli_fourier.state import (
    PauliFourierState,
    lambda_to_mu,
    mu_func_to_lambda_func,
    mu_to_lambda,
    polish_mu_func,
)
from src.core.repeater_algorithm import RepeaterChainEvaluation, repeater_sim
from src.core.werner.state import WernerState
from src.utils.utility_functions import bell_to_fid, secret_key_rate, werner_to_fid


def _assert_state_funcs_close_by_probability(pmf, actual, expected, atol=1.0e-8):
    assert_allclose(pmf, pmf, atol=0.0)
    for index, probability in enumerate(pmf):
        if probability < 1.0e-11:
            continue
        assert_allclose(actual[index], expected[index], atol=atol, rtol=1.0e-7)


def test_lambda_mu_numeric_maps_are_inverse():
    lambdas = np.array(
        [
            [0.85, 0.05, 0.07, 0.03],
            [0.25, 0.25, 0.25, 0.25],
            [0.10, 0.20, 0.60, 0.10],
        ],
        dtype=float,
    )
    assert_allclose(mu_to_lambda(lambda_to_mu(lambdas)), lambdas, atol=1.0e-15)


def test_pauli_fourier_state_validates_physical_inputs():
    state = PauliFourierState([0.85, 0.05, 0.07, 0.03])
    assert_allclose(mu_to_lambda(state.mu), [0.85, 0.05, 0.07, 0.03])

    try:
        PauliFourierState([0.9, 0.2, -0.05, -0.05])
    except ValueError:
        pass
    else:
        raise AssertionError("non-physical Bell weights should be rejected")


def test_polish_mu_func_preserves_physical_negative_mu_components():
    lambdas = np.array([[0.10, 0.20, 0.60, 0.10]], dtype=float)
    mu = lambda_to_mu(lambdas)
    assert mu[0, 1] < 0.0
    mu_with_drift = mu.copy()
    mu_with_drift[0, 0] = 0.999999999
    polished = polish_mu_func(mu_with_drift)
    assert polished[0, 1] < 0.0
    assert_allclose(mu_to_lambda(polished), lambdas, atol=1.0e-9)


def test_pauli_fourier_noise_matches_bell_depolarizing_and_dephasing():
    lambdas = np.array([0.85, 0.05, 0.075, 0.025], dtype=float)
    mu = lambda_to_mu(lambdas)
    dt = 7
    depolar_rate = 0.031
    dephase_rate = 0.047
    expected = dephasing_noise(
        depolarizing_noise(lambdas, dt, depolar_rate),
        dt,
        dephase_rate,
    )
    actual = mu_to_lambda(pauli_fourier_noise(mu, dt, depolar_rate, dephase_rate))
    assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_direct_protocol_units_match_bell_backend_with_scalar_and_heterogeneous_rates():
    examples = [
        (
            np.array([0.85, 0.05, 0.075, 0.025], dtype=float),
            np.array([0.70, 0.10, 0.12, 0.08], dtype=float),
            7,
            2,
            0.031,
            0.047,
        ),
        (
            np.array([0.61, 0.17, 0.13, 0.09], dtype=float),
            np.array([0.78, 0.11, 0.07, 0.04], dtype=float),
            1,
            9,
            [0.01, 0.02, 0.03],
            [0.04, 0.01, 0.02],
        ),
    ]
    for lam_a, lam_b, t1, t2, depolar_rate, dephase_rate in examples:
        mu_a = lambda_to_mu(lam_a)
        mu_b = lambda_to_mu(lam_b)

        bell_swap = get_swap_lambda_out(
            t1, t2, lam_a.copy(), lam_b.copy(), depolar_rate, dephase_rate, w_twirling=False
        )
        pf_swap = mu_to_lambda(
            get_swap_mu_out(t1, t2, mu_a.copy(), mu_b.copy(), depolar_rate, dephase_rate, w_twirling=False)
        )
        assert_allclose(pf_swap, bell_swap, atol=1.0e-12, rtol=1.0e-12)

        bell_ps = get_bell_dist_prob_suc(
            t1, t2, lam_a.copy(), lam_b.copy(), depolar_rate, dephase_rate, w_twirling=False
        )
        pf_ps = get_dist_prob_suc(
            t1, t2, mu_a.copy(), mu_b.copy(), depolar_rate, dephase_rate, w_twirling=False
        )
        assert_allclose(pf_ps, bell_ps, atol=1.0e-12, rtol=1.0e-12)

        for w_twirling in (False, True):
            bell_dist = get_dist_lambda_out(
                t1, t2, lam_a.copy(), lam_b.copy(), depolar_rate, dephase_rate, w_twirling=w_twirling
            )
            pf_dist = mu_to_lambda(
                get_dist_mu_out(t1, t2, mu_a.copy(), mu_b.copy(), depolar_rate, dephase_rate, w_twirling=w_twirling)
            )
            assert_allclose(pf_dist, bell_dist, atol=1.0e-12, rtol=1.0e-12)


def test_repeater_sim_infers_pauli_fourier_for_lambda_inputs_and_matches_bell():
    parameters = {
        "protocol": (0, 1),
        "lambdas": np.array([0.86, 0.05, 0.06, 0.03], dtype=float),
        "p_gen": 0.08,
        "p_swap": 0.75,
        "t_trunc": 160,
        "mt_cut": 20,
        "depolarizing_rate": 0.01,
        "dephasing_rate": 0.015,
    }

    pmf_pf, mu_func = repeater_sim(parameters)
    pmf_bell, lambda_func = repeater_sim(parameters, state_type=BellState)
    assert isinstance(RepeaterChainEvaluation()._resolve_state_type(parameters), type)
    assert RepeaterChainEvaluation()._resolve_state_type(parameters) == PauliFourierState
    assert_allclose(pmf_pf, pmf_bell, atol=1.0e-8, rtol=1.0e-7)
    _assert_state_funcs_close_by_probability(
        pmf_pf,
        mu_func_to_lambda_func(mu_func),
        lambda_func,
        atol=5.0e-7,
    )

    assert_allclose(
        secret_key_rate(pmf_pf, mu_func, state_type=PauliFourierState),
        secret_key_rate(pmf_bell, lambda_func, state_type=BellState),
        atol=1.0e-8,
        rtol=1.0e-7,
    )


def test_heterogeneous_pauli_fourier_matches_legacy_bell_backend():
    protocol = ("s0", "d1", "s1")
    lambdas = [
        [0.82, 0.08, 0.06, 0.04],
        [0.79, 0.09, 0.07, 0.05],
        [0.88, 0.04, 0.05, 0.03],
    ]
    parameters = {
        "protocol": protocol,
        "lambdas": lambdas,
        "p_gen": [0.08, 0.07, 0.06],
        "p_swap": 0.8,
        "t_trunc": 120,
        "depolarizing_rate": [0.002, 0.003, 0.004, 0.005],
        "dephasing_rate": [0.001, 0.002, 0.003, 0.004],
    }
    pmf_pf, mu_func = repeater_sim(parameters)
    pmf_bell, lambda_func = repeater_sim(parameters, state_type=BellState)
    assert_allclose(pmf_pf, pmf_bell, atol=1.0e-8, rtol=1.0e-7)
    _assert_state_funcs_close_by_probability(
        pmf_pf,
        mu_func_to_lambda_func(mu_func),
        lambda_func,
        atol=5.0e-7,
    )


def test_pauli_fourier_matches_werner_for_werner_initial_lambdas():
    w0 = 0.8
    fid = werner_to_fid(w0)
    parameters_pf = {
        "protocol": (0,),
        "lambdas": np.array([fid, (1.0 - fid) / 3.0, (1.0 - fid) / 3.0, (1.0 - fid) / 3.0]),
        "p_gen": 0.1,
        "p_swap": 0.9,
        "t_trunc": 120,
        "depolarizing_rate": 1.0 / 1000.0,
    }
    parameters_werner = {
        "protocol": (0,),
        "w0": w0,
        "p_gen": 0.1,
        "p_swap": 0.9,
        "t_trunc": 120,
        "t_coh": 1000,
    }
    pmf_pf, mu_func = repeater_sim(parameters_pf)
    pmf_w, w_func = repeater_sim(parameters_werner, state_type=WernerState)
    assert_allclose(pmf_pf, pmf_w, atol=1.0e-8, rtol=1.0e-7)
    pf_fids = bell_to_fid(mu_func_to_lambda_func(mu_func))
    w_fids = werner_to_fid(w_func)
    for probability, f_pf, f_w in zip(pmf_pf, pf_fids, w_fids):
        if probability < 1.0e-11:
            continue
        assert_allclose(f_pf, f_w, atol=1.0e-6, rtol=1.0e-6)
