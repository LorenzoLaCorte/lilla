import numpy as np
import pytest

from src.tests.pauli_fourier.helpers import (
    ASYM_CASES,
    ASYM_PROTOCOL_MAP,
    assert_pf_matches_bell,
    segments_for_protocol,
)


@pytest.mark.parametrize("sym_protocol, asym_protocol", ASYM_PROTOCOL_MAP.items())
@pytest.mark.parametrize("case", ASYM_CASES)
def test_pf_matches_bell_for_asymmetric_homogeneous_grid(sym_protocol, asym_protocol, case):
    del sym_protocol
    parameters = {
        "protocol": asym_protocol,
        "lambdas": case["lambdas"],
        "p_gen": case["p_gen"],
        "p_swap": case["p_swap"],
        "t_trunc": case["t_trunc"],
        "depolarizing_rate": case["depolarizing_rate"],
        "dephasing_rate": case["dephasing_rate"],
    }
    assert_pf_matches_bell(parameters, w_twirling=case["w_twirling"])


@pytest.mark.parametrize("sym_protocol, asym_protocol", ASYM_PROTOCOL_MAP.items())
def test_pf_matches_bell_for_asymmetric_heterogeneous_protocol_grid(sym_protocol, asym_protocol):
    del sym_protocol
    segments = segments_for_protocol(asym_protocol)
    base_lambdas = np.array(
        [
            [0.82, 0.08, 0.06, 0.04],
            [0.79, 0.09, 0.07, 0.05],
            [0.88, 0.04, 0.05, 0.03],
            [0.76, 0.11, 0.08, 0.05],
        ],
        dtype=float,
    )
    parameters = {
        "protocol": asym_protocol,
        "lambdas": base_lambdas[:segments].tolist(),
        "p_gen": [0.09 - 0.01 * index for index in range(segments)],
        "p_swap": 0.82,
        "t_trunc": 130 if segments <= 2 else 110,
        "depolarizing_rate": [0.0015 + 0.0005 * index for index in range(segments + 1)],
        "dephasing_rate": [0.0008 + 0.0004 * index for index in range(segments + 1)],
    }
    assert_pf_matches_bell(parameters, w_twirling=True, atol=1.0e-6)
