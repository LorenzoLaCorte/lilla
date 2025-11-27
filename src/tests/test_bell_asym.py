import copy
import numpy as np
import pytest

from src.core.repeater_algorithm import repeater_sim
from src.utils.utility_functions import bell_to_fid, secret_key_rate, werner_to_fid

from src.core.states import QuantumState
from src.core.bell.state import BellState
from src.core.werner.state import WernerState
from src.tests.test_bell_sym import (
    P_SWAP_LIST,
    P_GEN_LIST,
    PROTOCOL_NAMES,
    T_COH_LIST,
    T_TRUNC_LIST,
    W0_LIST,
    CUTOFF,
    CUT_TYPE,
)

PROTOCOL_LIST = [(0,), (1,), (0, 1), (0, 0), (1, 0, 0), (0, 1, 0)]
PROTOCOL_MAP = {
    (0,): ("s0",),
    (1,): ("d0",),
    (0, 1): ("s0", "d1"),
    (0, 0): ("s0", "s2", "s1"),
    (1, 0, 0): ("d0", "d1", "d2", "d3", "s0", "s2", "s1"),
    (0, 1, 0): ("s0", "s2", "d1", "d3", "s1"),
}
PROTOCOL_NAMES = {
    (0,): "swap",
    (1,): "dist",
    (0, 1): "swap-dist",
    (0, 0): "swap-swap",
    (1, 0, 0): "dist-dist-swap",
    (0, 1, 0): "swap-dist-swap",
}

PARAM_GRID = [
    (noise, protocol, p_swap, p_gen, t_coh, t_trunc, w_0)
    for t_trunc in T_TRUNC_LIST
    for p_swap in P_SWAP_LIST
    for p_gen in P_GEN_LIST if (p_gen > 0.01 or (p_swap > 0.5 and t_trunc > 1000))
    # limit longer protocols to smaller t_trunc for test speed
    for protocol in PROTOCOL_LIST if (len(protocol) < 2 or t_trunc <= 100)
    for noise in ([True]) # if 1 not in protocol else [False])  # only noise if no distillation
    for t_coh in T_COH_LIST
    for w_0 in W0_LIST
]

PARAM_IDS = [
    f"{PROTOCOL_NAMES.get(protocol, str(protocol))}"
    f"-noise={'on' if noise else 'off'}"
    f"-pswap={p_swap}-pgen={p_gen}-tcoh={t_coh}-ttrunc={t_trunc}-w0={w_0}"
    for (noise, protocol, p_swap, p_gen, t_coh, t_trunc, w_0) in PARAM_GRID
]

@pytest.mark.parametrize(
    ("noise", "sym_protocol", "p_swap", "p_gen", "t_coh", "t_trunc", "w_0"),
    PARAM_GRID,
    ids=PARAM_IDS,
)
def test_bell_vs_werner(noise, sym_protocol, p_swap, p_gen, t_coh, t_trunc, w_0):
    """
    Test the Bell and Werner states in asymmetric protocols.
    Using depolarizing rate (Bell diagonal) vs. time of coherence time (Werner).
    """
    asym_protocol = PROTOCOL_MAP[sym_protocol]
    print(f"\nCOMPARISON | Protocol: {PROTOCOL_NAMES[sym_protocol]}, noise={noise}, p_swap={p_swap}, p_gen={p_gen}, t_coh={t_coh}, t_trunc={t_trunc}, w_0={w_0}")

    # Bell diagonal protocol
    f_0 = werner_to_fid(w_0)
    parameters_bd = {
        "protocol": asym_protocol,
        "lambdas": np.array([f_0, (1 - f_0) / 3, (1 - f_0) / 3, (1 - f_0) / 3]),
        "p_gen": p_gen,
        "p_swap": p_swap,
        "t_trunc": t_trunc,
    }
    if noise:
        parameters_bd["depolarizing_rate"] = 1.0 / t_coh
    print(parameters_bd)
    pmf_bell, lambda_func = repeater_sim(parameters=parameters_bd, state_type=BellState)

    # Werner protocol
    parameters_w = {
        "protocol": asym_protocol,
        "w0": w_0,
        "p_gen": p_gen,
        "p_swap": p_swap,
        "t_trunc": t_trunc,
    }
    if noise:
        parameters_w["t_coh"] = t_coh
    pmf_werner, w_func = repeater_sim(parameters=parameters_w, state_type=WernerState)

    fids_bell = [bell_to_fid(l) for l in lambda_func]
    fids_werner = [werner_to_fid(w) for w in w_func]
    for i, (p_bell, p_werner, f_bell, f_werner) in enumerate(zip(pmf_bell[1:], pmf_werner[1:], fids_bell[1:], fids_werner[1:]), start=1):
        assert np.isclose(p_bell, p_werner, atol=1e-5), f"PMF mismatch between Bell and Werner states, index {i}"
        if p_bell < 1e-10 and p_werner < 1e-10:
            continue  # skip fidelity check for negligible probabilities
        assert np.isclose(f_bell, f_werner, atol=1e-2), f"Fidelity mismatch between Bell ({f_bell}) and Werner ({f_werner}) states, index {i}"

    skr_bell = secret_key_rate(pmf_bell, lambda_func, state_type=BellState)
    skr_werner = secret_key_rate(pmf_werner, w_func, state_type=WernerState)
    print(f"\tSecret key rates (Bell vs Werner): {skr_bell:.6f} (B) vs {skr_werner:.6f} (W)")
    assert np.isclose(skr_bell, skr_werner, atol=1e-5), "Secret key rate mismatch between Bell and Werner states"