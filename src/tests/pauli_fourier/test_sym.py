import pytest

from src.core.pauli_fourier.state import PauliFourierState
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.tests.pauli_fourier.helpers import (
    SYM_CASES,
    SYM_PROTOCOLS,
    assert_pf_matches_bell,
)


@pytest.mark.parametrize("protocol", SYM_PROTOCOLS)
@pytest.mark.parametrize("case", SYM_CASES)
def test_pf_matches_bell_for_nested_protocol_grid(protocol, case):
    parameters = {
        "protocol": protocol,
        "lambdas": case["lambdas"],
        "p_gen": case["p_gen"],
        "p_swap": case["p_swap"],
        "t_trunc": case["t_trunc"],
    }
    if "depolarizing_rate" in case:
        parameters["depolarizing_rate"] = case["depolarizing_rate"]
    if "dephasing_rate" in case:
        parameters["dephasing_rate"] = case["dephasing_rate"]

    assert_pf_matches_bell(parameters, w_twirling=case["w_twirling"])


def test_repeater_chain_evaluation_infers_pf_state_type_for_lambda_protocols():
    parameters = {
        "protocol": (0,),
        "lambdas": [0.85, 0.05, 0.07, 0.03],
        "p_gen": 0.1,
        "p_swap": 0.8,
        "t_trunc": 80,
    }
    simulator = RepeaterChainEvaluation()
    simulator.nested_protocol(parameters)
    assert simulator.state_type == PauliFourierState
