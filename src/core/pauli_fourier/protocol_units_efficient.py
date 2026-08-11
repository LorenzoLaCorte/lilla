import logging

import numba as nb
import numpy as np

from src.core.pauli_fourier.protocol_units import get_links_rate
from src.core.werner.protocol_units_efficient import (
    join_with_fail_cutoff,
    join_with_suc_cutoff,
)

__all__ = ["pauli_fourier_join_efficient"]


def _extract_probability_column(pmf):
    pmf = np.asarray(pmf, dtype=np.float64)
    if pmf.ndim == 1:
        return pmf
    return pmf[:, 0]


def _is_rate_sequence(rate):
    return isinstance(rate, (list, tuple, np.ndarray))


def _get_link_rates(rate):
    if _is_rate_sequence(rate):
        link_rates = get_links_rate(rate)
        return link_rates[0], link_rates[1]
    return rate, rate


def _component_rates(depolar_rate, dephase_rate, rate_kind):
    if rate_kind == "zero":
        return 0.0, 0.0
    gamma1, gamma2 = _get_link_rates(depolar_rate)
    if rate_kind == "gamma":
        return gamma1, gamma2
    if rate_kind == "gamma_delta":
        delta1, delta2 = _get_link_rates(dephase_rate)
        return gamma1 + delta1, gamma2 + delta2
    raise ValueError(rate_kind)


def _warn_for_large_exponents(size, depolar_rate, dephase_rate):
    gamma1, gamma2 = _get_link_rates(depolar_rate)
    delta1, delta2 = _get_link_rates(dephase_rate)
    max_exponent = (size - 1) * max(gamma1, gamma2, gamma1 + delta1, gamma2 + delta2)
    if max_exponent > 300:
        logging.warning("Overflow in the Pauli-Fourier efficient exponential factors may occur.")


def _run_scalar_join(cutoff, ycut, func1_early, func1_later, func2_early, func2_later):
    result = np.zeros_like(func1_early, dtype=np.float64)
    minus_result = np.zeros_like(func1_early, dtype=np.float64)

    if ycut:
        return join_with_suc_cutoff(
            cutoff,
            result,
            minus_result,
            np.cumsum(func1_early),
            func1_later,
            np.cumsum(func2_early),
            func2_later,
        )

    return join_with_fail_cutoff(
        cutoff,
        result,
        minus_result,
        func1_early,
        func1_later,
        np.cumsum(func1_later),
        func2_early,
        func2_later,
        np.cumsum(func2_later),
    )


@nb.jit(nopython=True, error_model="python")
def _join_with_suc_cutoff_decayed(
    cutoff,
    result,
    later1,
    early1,
    later2,
    early2,
    alpha1,
    alpha2,
):
    decay1 = np.exp(-alpha1)
    decay2 = np.exp(-alpha2)
    remove1 = np.exp(-alpha1 * (cutoff + 1))
    remove2 = np.exp(-alpha2 * (cutoff + 1))
    sum1 = 0.0
    sum2 = 0.0

    for t in range(1, len(result)):
        old = t - cutoff - 1
        sum1_excluding_t = decay1 * sum1
        sum2_including_t = decay2 * sum2
        if old >= 1:
            sum1_excluding_t -= early1[old] * remove1
            sum2_including_t -= early2[old] * remove2
        sum2_including_t += early2[t]

        result[t] = later1[t] * sum2_including_t + later2[t] * sum1_excluding_t
        sum1 = sum1_excluding_t + early1[t]
        sum2 = sum2_including_t
    return result


def _run_decayed_suc_join(cutoff, later1, early1, later2, early2, alpha1, alpha2):
    return _join_with_suc_cutoff_decayed(
        cutoff,
        np.zeros_like(later1, dtype=np.float64),
        later1,
        early1,
        later2,
        early2,
        alpha1,
        alpha2,
    )


def _one_term(cutoff, ycut, pmf1, pmf2):
    return _run_scalar_join(cutoff, ycut, pmf1, pmf1, pmf2, pmf2)


def _state_column(mu_func, component):
    if component is None:
        return np.ones(len(mu_func), dtype=np.float64)
    return np.asarray(mu_func[:, component], dtype=np.float64)


def _product_term(
    cutoff,
    ycut,
    pmf1,
    pmf2,
    mu_func1,
    mu_func2,
    depolar_rate,
    dephase_rate,
    left_component,
    right_component,
    rate_kind,
):
    alpha1, alpha2 = _component_rates(depolar_rate, dephase_rate, rate_kind)
    value1 = _state_column(mu_func1, left_component)
    value2 = _state_column(mu_func2, right_component)
    later1 = pmf1 * value1
    later2 = pmf2 * value2
    if ycut:
        return _run_decayed_suc_join(cutoff, later1, later1, later2, later2, alpha1, alpha2)

    time_axis = np.arange(len(pmf1), dtype=np.float64)
    link1_exp = np.exp(alpha1 * time_axis)
    link2_exp = np.exp(alpha2 * time_axis)
    return _run_scalar_join(
        cutoff,
        ycut,
        later1 * link1_exp,
        later1 / link2_exp,
        later2 * link2_exp,
        later2 / link1_exp,
    )


def _later_component_term(cutoff, ycut, pmf1, pmf2, mu_func1, mu_func2, component):
    if ycut:
        return _run_decayed_suc_join(
            cutoff,
            pmf1 * mu_func1[:, component],
            pmf1,
            pmf2 * mu_func2[:, component],
            pmf2,
            0.0,
            0.0,
        )
    return _run_scalar_join(
        cutoff,
        ycut,
        pmf1,
        pmf1 * mu_func1[:, component],
        pmf2,
        pmf2 * mu_func2[:, component],
    )


def _early_decayed_component_term(
    cutoff,
    ycut,
    pmf1,
    pmf2,
    mu_func1,
    mu_func2,
    depolar_rate,
    dephase_rate,
    component,
    rate_kind,
):
    alpha1, alpha2 = _component_rates(depolar_rate, dephase_rate, rate_kind)
    if ycut:
        return _run_decayed_suc_join(
            cutoff,
            pmf1,
            pmf1 * mu_func1[:, component],
            pmf2,
            pmf2 * mu_func2[:, component],
            alpha1,
            alpha2,
        )

    time_axis = np.arange(len(pmf1), dtype=np.float64)
    link1_exp = np.exp(alpha1 * time_axis)
    link2_exp = np.exp(alpha2 * time_axis)

    return _run_scalar_join(
        cutoff,
        ycut,
        pmf1 * mu_func1[:, component] * link1_exp,
        pmf1 / link2_exp,
        pmf2 * mu_func2[:, component] * link2_exp,
        pmf2 / link1_exp,
    )


def _rate_kind_for_mu_component(component):
    if component in (1, 2):
        return "gamma_delta"
    if component == 3:
        return "gamma"
    if component == 0:
        return "zero"
    raise ValueError(component)


def _dist_success(
    cutoff,
    ycut,
    pmf1,
    pmf2,
    mu_func1,
    mu_func2,
    depolar_rate,
    dephase_rate,
    sign,
):
    one = _one_term(cutoff, ycut, pmf1, pmf2)
    zz = _product_term(
        cutoff,
        ycut,
        pmf1,
        pmf2,
        mu_func1,
        mu_func2,
        depolar_rate,
        dephase_rate,
        3,
        3,
        "gamma",
    )
    return 0.5 * one + sign * 0.5 * zz


def _swap_state(
    cutoff,
    ycut,
    pmf1,
    pmf2,
    mu_func1,
    mu_func2,
    depolar_rate,
    dephase_rate,
):
    result = np.zeros((len(pmf1), 4), dtype=np.float64)
    result[:, 0] = _one_term(cutoff, ycut, pmf1, pmf2)
    for component in (1, 2, 3):
        result[:, component] = _product_term(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
            component,
            component,
            _rate_kind_for_mu_component(component),
        )
    return result


def _dist_weighted_state(
    cutoff,
    ycut,
    pmf1,
    pmf2,
    mu_func1,
    mu_func2,
    depolar_rate,
    dephase_rate,
    w_twirling,
):
    weighted = np.zeros((len(pmf1), 4), dtype=np.float64)
    weighted[:, 0] = _dist_success(
        cutoff,
        ycut,
        pmf1,
        pmf2,
        mu_func1,
        mu_func2,
        depolar_rate,
        dephase_rate,
        1.0,
    )
    weighted[:, 1] = 0.5 * (
        _product_term(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
            1,
            1,
            "gamma_delta",
        )
        + _product_term(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
            2,
            2,
            "gamma_delta",
        )
    )
    weighted[:, 2] = 0.5 * (
        _product_term(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
            1,
            2,
            "gamma_delta",
        )
        + _product_term(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
            2,
            1,
            "gamma_delta",
        )
    )
    weighted[:, 3] = 0.5 * (
        _later_component_term(cutoff, ycut, pmf1, pmf2, mu_func1, mu_func2, 3)
        + _early_decayed_component_term(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
            3,
            "gamma",
        )
    )

    if not w_twirling:
        return weighted

    result = np.empty_like(weighted)
    result[:, 0] = weighted[:, 0]
    result[:, 1:] = np.sum(weighted[:, 1:], axis=1, keepdims=True) / 3.0
    return result


def pauli_fourier_join_efficient(
    pmf1,
    pmf2,
    mu_func1,
    mu_func2,
    cutoff=np.iinfo(np.int32).max,
    ycut=True,
    cut_type=None,
    evaluate_func=None,
    depolar_rate=0.0,
    dephase_rate=0.0,
    w_twirling=True,
):
    if cut_type != "memory_time":
        raise NotImplementedError("Unknown cut-off type.")

    pmf1 = _extract_probability_column(pmf1)
    pmf2 = _extract_probability_column(pmf2)
    mu_func1 = np.asarray(mu_func1, dtype=np.float64)
    mu_func2 = np.asarray(mu_func2, dtype=np.float64)
    _warn_for_large_exponents(len(pmf1), depolar_rate, dephase_rate)

    if evaluate_func == "one_rule":
        return _one_term(cutoff, ycut, pmf1, pmf2)

    if evaluate_func == "swap_sf_rule":
        return _swap_state(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
        )

    if evaluate_func == "dist_ps_rule":
        return _dist_success(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
            1.0,
        )

    if evaluate_func == "dist_pf_rule":
        return _dist_success(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
            -1.0,
        )

    if evaluate_func == "dist_sf_rule":
        return _dist_weighted_state(
            cutoff,
            ycut,
            pmf1,
            pmf2,
            mu_func1,
            mu_func2,
            depolar_rate,
            dephase_rate,
            w_twirling,
        )

    raise ValueError(evaluate_func)
