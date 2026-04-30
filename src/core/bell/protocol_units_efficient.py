import logging

import numpy as np

from src.core.bell.protocol_units import get_links_rate
from src.core.werner.protocol_units_efficient import (
    join_with_fail_cutoff,
    join_with_suc_cutoff,
)

__all__ = ["bell_join_efficient"]


# Involution indexes, useful for dephasing
# the noise swaps weights 0 <-> 1 and 2 <-> 3
HAT = (1, 0, 3, 2)

# Index s_k(i)
# based on the k, decide the affected weights
SWAP_PERMUTATIONS = (
    (0, 1, 2, 3), # k=0
    (1, 0, 3, 2), # k=1
    (2, 3, 0, 1), # k=2
    (3, 2, 1, 0), # k=3
)
SWAP_HAT_PERMUTATIONS = tuple(tuple(HAT[index] for index in permutation) for permutation in SWAP_PERMUTATIONS)

# Indices for distillation numerators
# indeces of the form ((indeces of a), (indeces of b)) = ((i, j), (m, n)) 
# correspond to terms (a_i b'_m + a_j b'_n) in the numerator of H_k
DIST_PAIRS = (
    ((0, 1), (0, 1)), # k = 0
    ((0, 1), (1, 0)), # k = 1
    ((2, 3), (2, 3)), # k = 2
    ((2, 3), (3, 2)), # k = 3
)
SECTOR_INDICES = ((0, 1), (2, 3))

def _extract_probability_column(pmf):
    if pmf.ndim == 1:
        return pmf
    return pmf[:, 0]


def _tile_probability(pmf):
    if pmf.ndim == 2:
        return pmf
    return np.repeat(pmf[:, np.newaxis], 4, axis=1)

# TOVERIFY: checks if the noise rates are heterogeneous
def _is_rate_sequence(rate):
    return isinstance(rate, (list, tuple, np.ndarray))

# TOVERIFY: in case of heterogeneous noise, get separate link rates
def _get_link_rates(rate):
    if _is_rate_sequence(rate):
        values = list(np.asarray(rate).tolist())
        if len(values) == 0:
            return 0.0, 0.0
        if len(values) == 1:
            return values[0], values[0]
        link_rates = get_links_rate(values)
        return link_rates[0], link_rates[1]
    return rate, rate

# TOVERIFY: not sure what the actual check here is
def _warn_for_large_exponents(size, depolar_rate, dephase_rate):
    gamma1, gamma2 = _get_link_rates(depolar_rate)
    delta1, delta2 = _get_link_rates(dephase_rate)
    max_exponent = (size - 1) * max(gamma1, gamma2, gamma1 + delta1, gamma2 + delta2)
    if max_exponent > 300:
        logging.warning("Overflow in the Bell efficient exponential factors may occur.")


def _ones_components(size, dtype):
    return np.ones((size, 4), dtype=dtype)


def _basis_components(size, basis_index, dtype):
    result = np.zeros((size, 4), dtype=dtype)
    result[:, basis_index] = 1.0
    return result


def _cumsum_time_axis(array):
    return np.cumsum(array, axis=0)


def _run_join_term(cutoff, ycut, term_builder, result_shape, *builder_args):
    result = np.zeros(result_shape, dtype=np.float64)
    minus_result = np.zeros(result_shape, dtype=np.float64)
    func1_early, func1_later, func2_early, func2_later = term_builder(*builder_args)

    if ycut:
        cum_func1_early = _cumsum_time_axis(func1_early)
        cum_func2_early = _cumsum_time_axis(func2_early)
        return join_with_suc_cutoff(
            cutoff,
            result,
            minus_result,
            cum_func1_early,
            func1_later,
            cum_func2_early,
            func2_later,
        )

    cum_func1_later = _cumsum_time_axis(func1_later)
    cum_func2_later = _cumsum_time_axis(func2_later)
    return join_with_fail_cutoff(
        cutoff,
        result,
        minus_result,
        func1_early,
        func1_later,
        cum_func1_later,
        func2_early,
        func2_later,
        cum_func2_later,
    )


def _run_term_specs(cutoff, ycut, result_shape, term_specs):
    final_result = np.zeros(result_shape, dtype=np.float64)
    for coefficient, term_builder, builder_args in term_specs:
        final_result += coefficient * _run_join_term(
            cutoff,
            ycut,
            term_builder,
            result_shape,
            *builder_args,
        )
    return final_result


def get_one_bell(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate):
    del lambda_func1, lambda_func2, depolar_rate, dephase_rate
    pmf1 = _tile_probability(pmf1)
    pmf2 = _tile_probability(pmf2)
    return pmf1, pmf1, pmf2, pmf2


def get_constant_bell(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, constant):
    del lambda_func1, lambda_func2, depolar_rate, dephase_rate
    pmf1 = _extract_probability_column(pmf1)
    pmf2 = _extract_probability_column(pmf2)
    size = len(pmf1)
    dtype = np.result_type(pmf1.dtype, pmf2.dtype, np.float64)
    ones = _ones_components(size, dtype)
    constant_values = np.full((size, 4), constant, dtype=dtype)
    return (
        pmf1[:, np.newaxis] * constant_values,
        pmf1[:, np.newaxis] * ones,
        pmf2[:, np.newaxis] * constant_values,
        pmf2[:, np.newaxis] * ones,
    )


def get_swap_pair_average_term(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, lambda_index):
    '''
    TOVERIFY, WRITEMORE based on the ipynb
    We return for lambdas for each time t,
    all the 4 are appropriately computed based on the swap rule
    on separate terms, using the right swap permutations
    This is only one of the three terms of the swap composition
    '''
    pmf1 = _extract_probability_column(pmf1)
    pmf2 = _extract_probability_column(pmf2)
    size = len(pmf1)
    gamma1, gamma2 = _get_link_rates(depolar_rate)

    time_axis = np.arange(size, dtype=np.float64)
    link1_decay = np.exp(gamma1 * time_axis)
    link2_decay = np.exp(gamma2 * time_axis)

    lambdas_a = lambda_func1[:, lambda_index]
    lambdas_a_hat = lambda_func1[:, HAT[lambda_index]]
    lambdas_b = lambda_func2[:, SWAP_PERMUTATIONS[lambda_index]]
    lambdas_b_hat = lambda_func2[:, SWAP_HAT_PERMUTATIONS[lambda_index]]

    return (
        pmf1[:, np.newaxis] * lambdas_a[:, np.newaxis] * link2_decay[:, np.newaxis],
        pmf1[:, np.newaxis] * ((lambdas_a + lambdas_a_hat) / 2.0 - 0.25)[:, np.newaxis] / link1_decay[:, np.newaxis],
        pmf2[:, np.newaxis] * lambdas_b * link1_decay[:, np.newaxis],
        pmf2[:, np.newaxis] * ((lambdas_b + lambdas_b_hat) / 2.0 - 0.25) / link2_decay[:, np.newaxis],
    )


def get_swap_pair_difference_term(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, lambda_index):
    '''
    TOVERIFY, WRITEMORE based on the ipynb
    We return for lambdas for each time t,
    This is only one of the three terms of the swap composition
    '''
    pmf1 = _extract_probability_column(pmf1)
    pmf2 = _extract_probability_column(pmf2)
    size = len(pmf1)
    gamma1, gamma2 = _get_link_rates(depolar_rate)
    delta1, delta2 = _get_link_rates(dephase_rate)

    time_axis = np.arange(size, dtype=np.float64)
    link1_decay = np.exp((gamma1 + delta1) * time_axis)
    link2_decay = np.exp((gamma2 + delta2) * time_axis)

    lambdas_a = lambda_func1[:, lambda_index]
    lambdas_a_hat = lambda_func1[:, HAT[lambda_index]]
    lambdas_b = lambda_func2[:, SWAP_PERMUTATIONS[lambda_index]]
    lambdas_b_hat = lambda_func2[:, SWAP_HAT_PERMUTATIONS[lambda_index]]

    return (
        pmf1[:, np.newaxis] * lambdas_a[:, np.newaxis] * link2_decay[:, np.newaxis],
        pmf1[:, np.newaxis] * ((lambdas_a - lambdas_a_hat) / 2.0)[:, np.newaxis] / link1_decay[:, np.newaxis],
        pmf2[:, np.newaxis] * lambdas_b * link1_decay[:, np.newaxis],
        pmf2[:, np.newaxis] * ((lambdas_b - lambdas_b_hat) / 2.0) / link2_decay[:, np.newaxis],
    )


def get_dist_success_dynamic_term(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, sector_index):
    del dephase_rate
    pmf1 = _extract_probability_column(pmf1)
    pmf2 = _extract_probability_column(pmf2)
    size = len(pmf1)
    gamma1, gamma2 = _get_link_rates(depolar_rate)

    time_axis = np.arange(size, dtype=np.float64)
    link1_decay = np.exp(gamma1 * time_axis)
    link2_decay = np.exp(gamma2 * time_axis)

    i, j = SECTOR_INDICES[sector_index]
    a_sum = lambda_func1[:, i] + lambda_func1[:, j]
    b_sum = lambda_func2[:, i] + lambda_func2[:, j]

    return (
        pmf1[:, np.newaxis] * a_sum[:, np.newaxis] * link2_decay[:, np.newaxis],
        pmf1[:, np.newaxis] * (a_sum - 0.5)[:, np.newaxis] / link1_decay[:, np.newaxis],
        pmf2[:, np.newaxis] * b_sum[:, np.newaxis] * link1_decay[:, np.newaxis],
        pmf2[:, np.newaxis] * (b_sum - 0.5)[:, np.newaxis] / link2_decay[:, np.newaxis],
    )


def get_dist_numerator_constant_term(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, bell_index):
    del depolar_rate, dephase_rate
    pmf1 = _extract_probability_column(pmf1)
    pmf2 = _extract_probability_column(pmf2)
    size = len(pmf1)
    dtype = np.result_type(pmf1.dtype, pmf2.dtype, np.float64)
    ones = _ones_components(size, dtype)
    basis = _basis_components(size, bell_index, dtype)

    (i, j), (m, n) = DIST_PAIRS[bell_index]
    a_sum = lambda_func1[:, i] + lambda_func1[:, j]
    b_sum = lambda_func2[:, m] + lambda_func2[:, n]

    return (
        pmf1[:, np.newaxis] * a_sum[:, np.newaxis] * ones,
        pmf1[:, np.newaxis] * 0.25 * basis,
        pmf2[:, np.newaxis] * b_sum[:, np.newaxis] * ones,
        pmf2[:, np.newaxis] * 0.25 * basis,
    )


def get_dist_numerator_gamma_term(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, bell_index):
    del dephase_rate
    pmf1 = _extract_probability_column(pmf1)
    pmf2 = _extract_probability_column(pmf2)
    size = len(pmf1)
    dtype = np.result_type(pmf1.dtype, pmf2.dtype, np.float64)
    basis = _basis_components(size, bell_index, dtype)
    ones = _ones_components(size, dtype)
    gamma1, gamma2 = _get_link_rates(depolar_rate)

    time_axis = np.arange(size, dtype=np.float64)
    link1_decay = np.exp(gamma1 * time_axis)
    link2_decay = np.exp(gamma2 * time_axis)

    (i, j), (m, n) = DIST_PAIRS[bell_index]
    a_sum = lambda_func1[:, i] + lambda_func1[:, j]
    b_sum = lambda_func2[:, m] + lambda_func2[:, n]

    return (
        pmf1[:, np.newaxis] * a_sum[:, np.newaxis] * link2_decay[:, np.newaxis] * ones,
        pmf1[:, np.newaxis] * ((2.0 * a_sum - 1.0) / 4.0)[:, np.newaxis] * basis / link1_decay[:, np.newaxis],
        pmf2[:, np.newaxis] * b_sum[:, np.newaxis] * link1_decay[:, np.newaxis] * ones,
        pmf2[:, np.newaxis] * ((2.0 * b_sum - 1.0) / 4.0)[:, np.newaxis] * basis / link2_decay[:, np.newaxis],
    )


def get_dist_numerator_gamma_delta_term(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, bell_index):
    pmf1 = _extract_probability_column(pmf1)
    pmf2 = _extract_probability_column(pmf2)
    size = len(pmf1)
    dtype = np.result_type(pmf1.dtype, pmf2.dtype, np.float64)
    basis = _basis_components(size, bell_index, dtype)
    ones = _ones_components(size, dtype)
    gamma1, gamma2 = _get_link_rates(depolar_rate)
    delta1, delta2 = _get_link_rates(dephase_rate)

    time_axis = np.arange(size, dtype=np.float64)
    link1_decay = np.exp((gamma1 + delta1) * time_axis)
    link2_decay = np.exp((gamma2 + delta2) * time_axis)

    (i, j), (m, n) = DIST_PAIRS[bell_index]
    a_diff = lambda_func1[:, i] - lambda_func1[:, j]
    b_diff = lambda_func2[:, m] - lambda_func2[:, n]

    return (
        pmf1[:, np.newaxis] * (a_diff / 2.0)[:, np.newaxis] * link2_decay[:, np.newaxis] * ones,
        pmf1[:, np.newaxis] * a_diff[:, np.newaxis] * basis / link1_decay[:, np.newaxis],
        pmf2[:, np.newaxis] * (b_diff / 2.0)[:, np.newaxis] * link1_decay[:, np.newaxis] * ones,
        pmf2[:, np.newaxis] * b_diff[:, np.newaxis] * basis / link2_decay[:, np.newaxis],
    )


def _swap_term_specs(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate):
    '''
    TOVERIFY, WRITEMORE
    Here, we assemble the swap terms to get the correct result
    One term is constant 1/4 
    The other two terms are also in a sum over i
    '''
    term_specs = [
        (
            1.0,
            get_constant_bell,
            (pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, 0.25),
        ),
    ]
    for lambda_index in range(4):
        term_specs.append(
            (
                1.0,
                get_swap_pair_average_term,
                (pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, lambda_index),
            )
        )
        term_specs.append(
            (
                1.0,
                get_swap_pair_difference_term,
                (pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, lambda_index),
            )
        )
    return term_specs


def _dist_success_term_specs(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, sign):
    '''
    TOVERIFY, WRITEMORE
    Here, we assemble ...
    One term is constant 1/2 
    The other two terms depend on the two index sectors (psi- and phi-)
    '''
    term_specs = [
        (
            1.0,
            get_constant_bell,
            (pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, 0.5),
        ),
    ]
    for sector_index in range(2):
        term_specs.append(
            (
                sign,
                get_dist_success_dynamic_term,
                (pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, sector_index),
            )
        )
    return term_specs


def _dist_numerator_term_specs(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate):
    '''
    TOVERIFY, WRITEMORE
    Here, we assemble ...
    '''
    term_specs = []
    for bell_index in range(4):
        term_specs.append(
            (
                1.0,
                get_dist_numerator_constant_term,
                (pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, bell_index),
            )
        )
        term_specs.append(
            (
                1.0,
                get_dist_numerator_gamma_term,
                (pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, bell_index),
            )
        )
        term_specs.append(
            (
                1.0,
                get_dist_numerator_gamma_delta_term,
                (pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, bell_index),
            )
        )
    return term_specs


def bell_join_efficient(
    pmf1,
    pmf2,
    lambda_func1,
    lambda_func2,
    cutoff=np.iinfo(np.int32).max,
    ycut=True,
    cut_type=None,
    evaluate_func=None,
    depolar_rate=0.0,
    dephase_rate=0.0,
    twirling=True,
):
    """
    Efficient Bell-diagonal join for memory-time cutoffs.

    The implementation follows the separable derivations for Bell-diagonal
    swapping and distillation under depolarizing and dephasing noise.
    """
    if cut_type != "memory_time":
        raise NotImplementedError("Unknown cut-off type.")

    size = len(_extract_probability_column(pmf1))
    _warn_for_large_exponents(size, depolar_rate, dephase_rate)

    if evaluate_func == "1":
        result_shape = _tile_probability(pmf1).shape
        return _run_join_term(
            cutoff,
            ycut,
            get_one_bell,
            result_shape,
            pmf1,
            pmf2,
            lambda_func1,
            lambda_func2,
            depolar_rate,
            dephase_rate,
        )

    if evaluate_func == "f1f2":
        return _run_term_specs(
            cutoff,
            ycut,
            lambda_func1.shape,
            _swap_term_specs(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate),
        )

    if evaluate_func == "0.5+0.5f1f2":
        return _run_term_specs(
            cutoff,
            ycut,
            _tile_probability(pmf1).shape,
            _dist_success_term_specs(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, 1.0),
        )

    if evaluate_func == "0.5-0.5f1f2":
        return _run_term_specs(
            cutoff,
            ycut,
            _tile_probability(pmf1).shape,
            _dist_success_term_specs(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate, -1.0),
        )

    if evaluate_func == "f1+f2+4f1f2":
        numerators = _run_term_specs(
            cutoff,
            ycut,
            lambda_func1.shape,
            _dist_numerator_term_specs(pmf1, pmf2, lambda_func1, lambda_func2, depolar_rate, dephase_rate),
        )
        if not twirling:
            return numerators
        success_probability = np.sum(numerators, axis=1, keepdims=True)
        result = np.empty_like(numerators)
        result[:, 0] = numerators[:, 0]
        result[:, 1:] = (success_probability - numerators[:, [0]]) / 3.0
        return result

    raise ValueError(evaluate_func)
