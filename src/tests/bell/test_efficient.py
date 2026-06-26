import time

import numpy as np
from numpy.testing import assert_allclose

from src.core.bell.protocol_units import bell_join
from src.core.bell.protocol_units_efficient import bell_join_efficient
from src.core.bell.state import BellState
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.core.werner.state import WernerState
from src.utils.utility_functions import bell_to_fid, secret_key_rate


def _build_bell_test_inputs():
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
    return pmf1, pmf2, lambda_func1, lambda_func2


def _assert_direct_matches_efficient(
    *,
    evaluate_func,
    depolar_rate=0.0,
    dephase_rate=0.0,
    ycut=True,
    w_twirling=False,
):
    pmf1, pmf2, lambda_func1, lambda_func2 = _build_bell_test_inputs()
    cutoff = 2
    cut_type = "memory_time"

    expected = bell_join(
        pmf1,
        pmf2,
        lambda_func1,
        lambda_func2,
        cutoff=cutoff,
        ycut=ycut,
        cut_type=cut_type,
        evaluate_func=evaluate_func,
        depolar_rate=depolar_rate,
        dephase_rate=dephase_rate,
        w_twirling=w_twirling,
    )
    actual = bell_join_efficient(
        pmf1,
        pmf2,
        lambda_func1,
        lambda_func2,
        cutoff=cutoff,
        ycut=ycut,
        cut_type=cut_type,
        evaluate_func=evaluate_func,
        depolar_rate=depolar_rate,
        dephase_rate=dephase_rate,
        w_twirling=w_twirling,
    )

    assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_bell_swap_depolarizing_probability_matches_direct():
    _assert_direct_matches_efficient(evaluate_func="1", depolar_rate=0.04, ycut=True)


def test_bell_swap_depolarizing_state_matches_direct():
    _assert_direct_matches_efficient(evaluate_func="f1f2", depolar_rate=0.04, ycut=True)


def test_bell_swap_dephasing_state_matches_direct():
    _assert_direct_matches_efficient(evaluate_func="f1f2", dephase_rate=0.07, ycut=True)


def test_bell_distillation_depolarizing_matches_direct():
    _assert_direct_matches_efficient(evaluate_func="0.5+0.5f1f2", depolar_rate=0.04, ycut=True)
    _assert_direct_matches_efficient(
        evaluate_func="f1+f2+4f1f2",
        depolar_rate=0.04,
        ycut=True,
        w_twirling=False,
    )


def test_bell_distillation_dephasing_matches_direct():
    _assert_direct_matches_efficient(evaluate_func="0.5+0.5f1f2", dephase_rate=0.07, ycut=True)
    _assert_direct_matches_efficient(
        evaluate_func="f1+f2+4f1f2",
        dephase_rate=0.07,
        ycut=True,
        w_twirling=False,
    )


def _assert_repeater_protocol_matches_direct(label, protocol, depolar_rate=0.0, dephase_rate=0.0):
    parameters = {
        "protocol": protocol,
        "lambdas": np.array([0.95, 0.02, 0.02, 0.01], dtype=np.float64),
        "p_gen": 0.01,
        "p_swap": 0.85,
        "t_trunc": 5000,
    }
    
    if depolar_rate != 0.0:
        parameters["depolarizing_rate"] = depolar_rate
    if dephase_rate != 0.0:
        parameters["dephasing_rate"] = dephase_rate

    direct = RepeaterChainEvaluation(state_type=BellState, use_fft=True, efficient=False)
    efficient = RepeaterChainEvaluation(state_type=BellState, use_fft=True, efficient=True)

    direct_start = time.perf_counter()
    expected_pmf, expected_state = direct.nested_protocol(parameters)
    direct_elapsed = time.perf_counter() - direct_start

    efficient_start = time.perf_counter()
    actual_pmf, actual_state = efficient.nested_protocol(parameters)
    efficient_elapsed = time.perf_counter() - efficient_start

    print(
        f"{label}: direct={direct_elapsed:.6f}s "
        f"efficient={efficient_elapsed:.6f}s"
    )

    # assert_allclose(actual_pmf, expected_pmf, rtol=1.0e-5, atol=1.0e-7)
    # assert_allclose(actual_state, expected_state, rtol=1.0e-5, atol=1.0e-7)

    fids_actual = [bell_to_fid(l) for l in actual_state]
    fids_expected = [bell_to_fid(l) for l in expected_state]

    for i, (p_actual, p_expected, f_actual, f_expected) in enumerate(zip(actual_pmf[1:], expected_pmf[1:], fids_actual[1:], fids_expected[1:]), start=1):
        assert np.isclose(p_actual, p_expected, atol=1e-5), f"PMF mismatch between Bell and Werner states, index {i}"
        if p_actual < 1e-10 and p_expected < 1e-10:
            continue  # skip fidelity check for negligible probabilities
        if i > 300 and i < 400: 
            print(p_actual, p_expected)
            print(f_actual, f_expected)
        assert np.isclose(f_actual, f_expected, atol=1e-5), f"Fidelity mismatch between Bell ({f_actual}) and Werner ({f_expected}) states, index {i}"

    # skr_bell = secret_key_rate(actual_pmf, actual_state, state_type=BellState)
    # skr_werner = secret_key_rate(expected_pmf, expected_state, state_type=WernerState)
    # print(f"\tSecret key rates (Bell vs Werner): {skr_bell:.6f} (B) vs {skr_werner:.6f} (W)")
    # assert np.isclose(skr_bell, skr_werner, atol=1e-5), "Secret key rate mismatch between Bell and Werner states"


def test_repeater_swap_both_efficient_matches_direct():
    _assert_repeater_protocol_matches_direct("swap-both", (0,), depolar_rate=0.04, dephase_rate=0.07)


def test_repeater_swap_depolarizing_efficient_matches_direct():
    _assert_repeater_protocol_matches_direct("swap-depolarizing", (0,), depolar_rate=0.04)


def test_repeater_swap_dephasing_efficient_matches_direct():
    _assert_repeater_protocol_matches_direct("swap-dephasing", (0,), dephase_rate=0.07)


def test_repeater_distillation_depolarizing_efficient_matches_direct():
    _assert_repeater_protocol_matches_direct("distillation-depolarizing", (1,), depolar_rate=0.04)


def test_repeater_distillation_dephasing_efficient_matches_direct():
    _assert_repeater_protocol_matches_direct("distillation-dephasing", (1,), dephase_rate=0.07)
