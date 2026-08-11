import numpy as np
from numpy.testing import assert_allclose

from src.core.bell.state import BellState
from src.core.pauli_fourier.state import PauliFourierState, mu_func_to_lambda_func
from src.core.repeater_algorithm import repeater_sim
from src.utils.utility_functions import secret_key_rate


SYM_PROTOCOLS = [(1,), (0,), (0, 1), (0, 0), (1, 1), (0, 1, 0)]

ASYM_PROTOCOL_MAP = {
    (0,): ("s0",),
    (1,): ("d0",),
    (1, 0): ("d0", "d1", "s0"),
    (0, 1): ("s0", "d1"),
    (0, 0): ("s0", "s2", "s1"),
    (1, 0, 0): ("d0", "d1", "d2", "d3", "s0", "s2", "s1"),
    (0, 1, 0): ("s0", "s2", "d1", "d3", "s1"),
}

SYM_CASES = [
    {
        "p_gen": 0.08,
        "p_swap": 0.70,
        "t_trunc": 160,
        "lambdas": np.array([0.86, 0.05, 0.06, 0.03], dtype=float),
        "depolarizing_rate": 0.010,
        "dephasing_rate": 0.015,
        "w_twirling": True,
    },
    {
        "p_gen": 0.12,
        "p_swap": 0.85,
        "t_trunc": 140,
        "lambdas": np.array([0.78, 0.10, 0.08, 0.04], dtype=float),
        "depolarizing_rate": 0.004,
        "dephasing_rate": 0.000,
        "w_twirling": False,
    },
    {
        "p_gen": 0.10,
        "p_swap": 0.90,
        "t_trunc": 120,
        "lambdas": np.array([0.91, 0.03, 0.04, 0.02], dtype=float),
        "w_twirling": True,
    },
]

ASYM_CASES = [
    {
        "p_gen": 0.08,
        "p_swap": 0.75,
        "t_trunc": 150,
        "lambdas": np.array([0.84, 0.06, 0.07, 0.03], dtype=float),
        "depolarizing_rate": 0.006,
        "dephasing_rate": 0.011,
        "w_twirling": True,
    },
    {
        "p_gen": 0.11,
        "p_swap": 0.88,
        "t_trunc": 130,
        "lambdas": np.array([0.80, 0.09, 0.07, 0.04], dtype=float),
        "depolarizing_rate": 0.003,
        "dephasing_rate": 0.000,
        "w_twirling": False,
    },
]


def segments_for_protocol(protocol):
    return sum(1 for step in protocol if step.startswith("s")) + 1


def assert_pf_matches_bell(parameters, *, w_twirling=True, atol=7.0e-7):
    pmf_pf, mu_func = repeater_sim(parameters, w_twirling=w_twirling)
    pmf_bell, lambda_func = repeater_sim(
        parameters,
        state_type=BellState,
        w_twirling=w_twirling,
    )
    assert_allclose(pmf_pf, pmf_bell, atol=1.0e-8, rtol=1.0e-7)

    lambda_from_mu = mu_func_to_lambda_func(mu_func)
    for index, probability in enumerate(pmf_pf):
        if probability < 1.0e-10:
            continue
        assert_allclose(lambda_from_mu[index], lambda_func[index], atol=atol, rtol=1.0e-6)

    assert_allclose(
        secret_key_rate(pmf_pf, mu_func, state_type=PauliFourierState),
        secret_key_rate(pmf_bell, lambda_func, state_type=BellState),
        atol=1.0e-8,
        rtol=1.0e-7,
    )
