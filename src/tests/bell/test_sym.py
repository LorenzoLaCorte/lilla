import copy
import numpy as np
import pytest

from src.core.repeater_algorithm import RepeaterChainEvaluation, repeater_sim
from src.utils.utility_functions import bell_to_fid, secret_key_rate, werner_to_fid

from src.core.bell.state import BellState, polish_l_func
from src.core.werner.state import WernerState

P_SWAP_LIST = [0.1, 0.5, 0.9]
P_GEN_LIST = [0.01, 0.1]
PROTOCOL_LIST = [(1,), (0,), (0, 1), (0, 0), (1, 1), (0, 1, 0)]
PROTOCOL_NAMES = {
    (0,): "swap",
    (1,): "dist",
    (0, 1): "swap-dist",
    (0, 0): "swap-swap",
    (1, 1): "dist-dist",
    (0, 1, 0): "swap-dist-swap",
}
T_COH_LIST = [1000]
T_TRUNC_LIST = [100, 500, 1000]
W0_LIST = [0.8, 0.9]

CUTOFF = 50
CUT_TYPE = "memory_time"

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

SWAP_PARAM_GRID = [
    (noise, p_swap, p_gen, t_coh, t_trunc, w_0)
    for p_swap in P_SWAP_LIST
    for p_gen in P_GEN_LIST
    for noise in [False, True]
    for t_coh in T_COH_LIST
    for t_trunc in T_TRUNC_LIST
    for w_0 in W0_LIST
]

SWAP_PARAM_IDS = [
    f"swap"
    f"-noise={'on' if noise else 'off'}"
    f"-pswap={p_swap}-pgen={p_gen}-tcoh={t_coh}-ttrunc={t_trunc}-w0={w_0}"
    for (noise, p_swap, p_gen, t_coh, t_trunc, w_0) in SWAP_PARAM_GRID
]

DIST_PARAM_GRID = [
    (noise, p_swap, p_gen, t_coh, t_trunc, w_0)
    for p_swap in P_SWAP_LIST
    for p_gen in P_GEN_LIST
    for noise in [True]
    for t_coh in T_COH_LIST
    for t_trunc in T_TRUNC_LIST
    for w_0 in W0_LIST
]

DIST_PARAM_IDS = [
    f"{PROTOCOL_NAMES[(1,)]}"
    f"-noise={'on' if noise else 'off'}"
    f"-pswap={p_swap}-pgen={p_gen}-tcoh={t_coh}-ttrunc={t_trunc}-w0={w_0}"
    for (noise, p_swap, p_gen, t_coh, t_trunc, w_0) in DIST_PARAM_GRID
]

@pytest.mark.parametrize(
    ("noise", "p_swap", "p_gen", "t_coh", "t_trunc", "w_0"),
    SWAP_PARAM_GRID,
    ids=SWAP_PARAM_IDS,
)
def test_swap(noise, p_swap, p_gen, t_coh, t_trunc, w_0):
    """
    Test the swap function of the RepeaterChainEvaluation class.
    Compare direct swap vs protocol (0,) across the same parameter grid used elsewhere.
    """
    protocol = (0,)
    print(f"\nSWAP | Protocol: {PROTOCOL_NAMES[protocol]}, noise={noise}, p_swap={p_swap}, p_gen={p_gen}, t_coh={t_coh}, t_trunc={t_trunc}, w_0={w_0}")

    f_0 = werner_to_fid(w_0)
    lambdas = np.array([f_0, (1 - f_0) / 3, (1 - f_0) / 3, (1 - f_0) / 3])

    cutoff, cut_type = CUTOFF, CUT_TYPE

    t_list = np.arange(1, t_trunc)
    pmf1 = p_gen * (1 - p_gen) ** (t_list - 1)
    pmf1 = np.concatenate((np.array([0.]), pmf1))
    pmf2 = pmf1.copy()
    lambda_func1 = np.tile(lambdas, (t_trunc, 1))
    lambda_func2 = lambda_func1.copy()

    # swap between two identical links
    repeater = RepeaterChainEvaluation(state_type=BellState)
    direct_params = {
        # noise control only via depolarizing_rate for Bell diagonal
        **({"depolarizing_rate": 1.0 / t_coh} if noise else {}),
    }
    pmf_swap, state_out = repeater.swapping(
        parameters=direct_params, pmf1=pmf1, sf1=lambda_func1, pmf2=pmf2, sf2=lambda_func2,
        p_swap=p_swap, cutoff=cutoff, t_coh=t_coh, cut_type=cut_type,
    )

    assert pmf_swap.shape == pmf1.shape, "PMF output shape mismatch"
    assert state_out.shape == lambda_func1.shape, "State output shape mismatch"

    # Now, repeat the process calling the repeater protocol (force protocol to (0,))
    parameters = {
    "protocol": protocol,
        "lambdas": lambdas,
        "p_gen": p_gen,
        "p_swap": p_swap,
        "t_trunc": t_trunc,
        "cutoff": cutoff,
        "cut_type": cut_type,
        **({"depolarizing_rate": 1.0 / t_coh} if noise else {}),
    }

    pmf_swap_protocol, state_out_protocol = repeater_sim(parameters=parameters, state_type=BellState)

    for i, (p_swap, p_swap_protocol, s_out, s_out_protocol) in enumerate(zip(pmf_swap[1:], pmf_swap_protocol[1:], state_out[1:], state_out_protocol[1:]), start=1):
        assert np.isclose(p_swap, p_swap_protocol, atol=1e-5), f"PMF mismatch between direct and protocol methods, index {i}"
        if p_swap < 1e-10 and p_swap_protocol < 1e-10:
            continue  # skip fidelity check for negligible probabilities
        assert np.allclose(s_out, s_out_protocol, atol=1e-2), f"Fidelity mismatch between direct ({s_out}) and protocol ({s_out_protocol}) methods, index {i}"


@pytest.mark.parametrize(
    ("noise", "p_swap", "p_gen", "t_coh", "t_trunc", "w_0"),
    DIST_PARAM_GRID,
    ids=DIST_PARAM_IDS,
)
def test_distillation(noise, p_swap, p_gen, t_coh, t_trunc, w_0):
    """
    Test the distillation function of the RepeaterChainEvaluation class.
    This test checks the output shapes and compares direct distillation vs protocol (1,).
    """
    protocol = (1,)
    print(f"\nDISTILL | Protocol: {PROTOCOL_NAMES[protocol]}, noise={noise}, p_swap={p_swap}, p_gen={p_gen}, t_coh={t_coh}, t_trunc={t_trunc}, w_0={w_0}")

    f_0 = werner_to_fid(w_0)
    lambdas = np.array([f_0, (1 - f_0) / 3, (1 - f_0) / 3, (1 - f_0) / 3])

    cutoff, cut_type = CUTOFF, CUT_TYPE

    t_list = np.arange(1, t_trunc)
    pmf1 = p_gen * (1 - p_gen) ** (t_list - 1)
    pmf1 = np.concatenate((np.array([0.]), pmf1))
    pmf2 = pmf1.copy()

    lambda_func1 = np.tile(lambdas, (t_trunc, 1))
    lambda_func2 = lambda_func1.copy()

    # distillation between two identical links
    repeater = RepeaterChainEvaluation(state_type=BellState)
    direct_params = {
        **({"depolarizing_rate": 1.0 / t_coh} if noise else {}),
    }
    pmf_dist, state_out = repeater.distillation(
        direct_params,
        pmf1, lambda_func1, pmf2, lambda_func2,
        cutoff, t_coh, cut_type,
    )
    state_out = polish_l_func(state_out)

    assert pmf_dist.shape == pmf1.shape, "PMF output shape mismatch"
    assert state_out.shape == lambda_func1.shape, "State output shape mismatch"

    # Now, repeat the process calling the repeater protocol (force protocol to (1,))
    parameters = {
    "protocol": protocol,
        "lambdas": lambdas,
        "p_gen": p_gen,
        "p_swap": p_swap,
        "t_trunc": t_trunc,
        "cutoff": cutoff,
        "cut_type": cut_type,
        **({"depolarizing_rate": 1.0 / t_coh} if noise else {}),
    }

    pmf_dist_protocol, state_out_protocol = repeater_sim(parameters=parameters, state_type=BellState)

    assert np.allclose(pmf_dist, pmf_dist_protocol), "PMF mismatch between direct and protocol methods"
    # assert np.allclose(state_out, state_out_protocol), "Lamdas mismatch between direct and protocol methods"
    for s1, s2 in zip(state_out, state_out_protocol):
        assert np.allclose(s1, s2), f"Lamdas mismatch between direct and protocol methods: {s1} vs {s2}, index {np.where(state_out==s1)[0][0]}"


@pytest.mark.parametrize(
    ("noise", "protocol", "p_swap", "p_gen", "t_coh", "t_trunc", "w_0"),
    PARAM_GRID,
    ids=PARAM_IDS,
)
def test_bell_vs_werner(noise, protocol, p_swap, p_gen, t_coh, t_trunc, w_0):
    """
    Test the output shapes of the Bell and Werner states.
    Using a 1/1000 depolarizing rate (Bell diagonal) vs. a 1000 joint time of coherence time (Werner).
    """
    print(f"\nCOMPARISON | Protocol: {protocol}, noise={noise}, p_swap={p_swap}, p_gen={p_gen}, t_coh={t_coh}, t_trunc={t_trunc}, w_0={w_0}")

    # Bell diagonal protocol
    f_0 = werner_to_fid(w_0)
    parameters_bd = {
        "protocol": protocol,
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
        "protocol": protocol,
        "w0": w_0, # lambda = 0.85 -> w_0 = 0.8 , lambda = 0.97 -> w_0 = 0.96
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