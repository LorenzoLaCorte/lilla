import numpy as np
from numpy.testing import assert_allclose

from src.core.bell.protocol_units_efficient import bell_join_efficient
from src.core.pauli_fourier.protocol_units import pauli_fourier_join
from src.core.pauli_fourier.protocol_units_efficient import pauli_fourier_join_efficient
from src.core.pauli_fourier.state import lambda_to_mu, mu_to_lambda


def _build_test_inputs():
    pmf1 = np.array([0.0, 0.31, 0.24, 0.18, 0.12, 0.08, 0.05, 0.02], dtype=np.float64)
    pmf2 = np.array([0.0, 0.27, 0.25, 0.19, 0.13, 0.09, 0.05, 0.02], dtype=np.float64)
    time_axis = np.arange(len(pmf1), dtype=np.float64)
    lambda_func1 = np.column_stack(
        (
            0.70 - 0.02 * time_axis,
            0.10 + 0.01 * time_axis,
            0.12 + 0.005 * time_axis,
            0.08 + 0.005 * time_axis,
        )
    )
    lambda_func2 = np.column_stack(
        (
            0.64 - 0.015 * time_axis,
            0.16 + 0.006 * time_axis,
            0.11 + 0.004 * time_axis,
            0.09 + 0.005 * time_axis,
        )
    )
    return pmf1, pmf2, lambda_func1, lambda_func2, lambda_to_mu(lambda_func1), lambda_to_mu(lambda_func2)


def _assert_pf_direct_matches_efficient(
    evaluate_func,
    depolar_rate=0.0,
    dephase_rate=0.0,
    ycut=True,
    w_twirling=False,
):
    pmf1, pmf2, _, _, mu_func1, mu_func2 = _build_test_inputs()
    expected = pauli_fourier_join(
        pmf1,
        pmf2,
        mu_func1,
        mu_func2,
        cutoff=2,
        ycut=ycut,
        cut_type="memory_time",
        evaluate_func=evaluate_func,
        depolar_rate=depolar_rate,
        dephase_rate=dephase_rate,
        w_twirling=w_twirling,
    )
    actual = pauli_fourier_join_efficient(
        pmf1,
        pmf2,
        mu_func1,
        mu_func2,
        cutoff=2,
        ycut=ycut,
        cut_type="memory_time",
        evaluate_func=evaluate_func,
        depolar_rate=depolar_rate,
        dephase_rate=dephase_rate,
        w_twirling=w_twirling,
    )
    if actual.ndim == 1 and expected.ndim == 2:
        assert_allclose(
            expected,
            np.repeat(expected[:, [0]], expected.shape[1], axis=1),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        expected = expected[:, 0]
    assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_pauli_fourier_efficient_probability_matches_direct():
    _assert_pf_direct_matches_efficient("one_rule", depolar_rate=0.04, ycut=True)
    pmf1, pmf2, _, _, mu_func1, mu_func2 = _build_test_inputs()
    actual = pauli_fourier_join_efficient(
        pmf1,
        pmf2,
        mu_func1,
        mu_func2,
        cutoff=2,
        ycut=True,
        cut_type="memory_time",
        evaluate_func="one_rule",
        depolar_rate=0.04,
    )
    assert actual.ndim == 1


def test_pauli_fourier_efficient_swap_matches_direct():
    _assert_pf_direct_matches_efficient("swap_sf_rule", depolar_rate=0.04, ycut=True)
    _assert_pf_direct_matches_efficient("swap_sf_rule", dephase_rate=0.07, ycut=True)
    _assert_pf_direct_matches_efficient(
        "swap_sf_rule",
        depolar_rate=[0.01, 0.02, 0.03],
        dephase_rate=[0.04, 0.01, 0.02],
        ycut=True,
    )


def test_pauli_fourier_efficient_distillation_matches_direct():
    _assert_pf_direct_matches_efficient("dist_ps_rule", depolar_rate=0.04, ycut=True)
    _assert_pf_direct_matches_efficient("dist_pf_rule", dephase_rate=0.07, ycut=True)
    _assert_pf_direct_matches_efficient(
        "dist_sf_rule",
        depolar_rate=0.04,
        dephase_rate=0.07,
        ycut=True,
        w_twirling=False,
    )
    _assert_pf_direct_matches_efficient(
        "dist_sf_rule",
        depolar_rate=0.04,
        dephase_rate=0.07,
        ycut=True,
        w_twirling=True,
    )


def test_pauli_fourier_efficient_weighted_states_match_bell_after_conversion():
    pmf1, pmf2, lambda_func1, lambda_func2, mu_func1, mu_func2 = _build_test_inputs()
    cases = [
        ("swap_sf_rule", "f1f2", False),
        ("dist_sf_rule", "f1+f2+4f1f2", False),
        ("dist_sf_rule", "f1+f2+4f1f2", True),
    ]
    for pf_eval, bell_eval, w_twirling in cases:
        pf_result = pauli_fourier_join_efficient(
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            cutoff=2,
            ycut=True,
            cut_type="memory_time",
            evaluate_func=pf_eval,
            depolar_rate=0.023,
            dephase_rate=0.041,
            w_twirling=w_twirling,
        )
        bell_result = bell_join_efficient(
            pmf1,
            pmf2,
            lambda_func1,
            lambda_func2,
            cutoff=2,
            ycut=True,
            cut_type="memory_time",
            evaluate_func=bell_eval,
            depolar_rate=0.023,
            dephase_rate=0.041,
            w_twirling=w_twirling,
        )
        assert_allclose(mu_to_lambda(pf_result), bell_result, atol=1.0e-12, rtol=1.0e-12)
