"""
This script is used to test the performance of different distillation strategies
 in an extensive way, by using a Gaussian Process to optimize the number of distillations.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from scipy.optimize import OptimizeResult
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from src.core.werner.state import WernerState
from src.core.bell.state import BellState  # Bell-diagonal state class
from src.core.pauli_fourier.state import PauliFourierState

from src.plotting.gp_plots import plot_optimization_process
from src.utils.gp_utils import (
    get_asym_protocol_space,
    get_protocol_from_center_spacing_symmetricity,
    set_heuristic_t_trunc,
    get_ordered_results,
)

from src.types.repeater_types import (
    optimizerType,
    OptimizerType,
    ThresholdExceededError,
    SimParameters,
)

from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.utils.utility_functions import pmf_to_cdf, secret_key_rate

logging.basicConfig(level=logging.INFO)
StateName = str  # {"werner", "bell", "pf"}


@dataclass(frozen=True)
class TruncationPolicy:
    fixed_t_trunc: Optional[int] = None
    adaptive_attempts: int = 3
    growth: float = 2.0
    max_t_trunc: Optional[int] = None


def write_results(filename: str, ordered_results: Sequence[Tuple[np.float64, Tuple[str, ...]]]) -> None:
    path = Path(filename)
    serializable: Dict[str, float] = {str(protocol): float(skr) for skr, protocol in ordered_results}

    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=False)
        f.write("\n\n")
        f.write(f"Unique values: {len(set(float(skr) for skr, _ in ordered_results))}\n")


def asym_protocol_runner(
    simulator: RepeaterChainEvaluation,
    parameters: SimParameters,
    nodes: int,
    cdf_threshold: float,
    truncation_policy: TruncationPolicy,
    idx: Optional[int] = None,
    space_len: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
        The function tests the performance of a single asymmetric protocol,
        by taking the number of distillations and the nesting level after which dist is applied as a parameter,
        and returning the secret key rate of the strategy.
    """
    protocol = parameters["protocol"]
    dists = sum(1 for item in protocol if 'd' in item)

    if truncation_policy.fixed_t_trunc is not None:
        parameters["t_trunc"] = truncation_policy.fixed_t_trunc
    else:
        set_heuristic_t_trunc(
            parameters,
            nodes,
            dists,
            cdf_threshold=cdf_threshold,
        )
        if truncation_policy.max_t_trunc is not None:
            parameters["t_trunc"] = min(int(parameters["t_trunc"]), truncation_policy.max_t_trunc)

    attempts = 1 if truncation_policy.fixed_t_trunc is not None else max(1, truncation_policy.adaptive_attempts)
    pmf: np.ndarray
    state_func: np.ndarray
    coverage = 0.0

    for attempt in range(attempts):
        if (idx is None) or (space_len is None):
            logging.info(f"\nRunning: {parameters}")
        else:
            logging.info(f"\n({idx + 1}/{space_len}) Running: {parameters}")

        if isinstance(parameters["p_gen"], Iterable):
            pmf, state_func = simulator.asymmetric_heterogeneous_protocol(parameters, nodes - 1)
        else:
            pmf, state_func = simulator.asymmetric_homogeneous_protocol(parameters, nodes - 1)

        coverage = float(pmf_to_cdf(pmf)[-1])
        if coverage >= cdf_threshold:
            break

        if attempt == attempts - 1:
            break

        old_t_trunc = int(parameters["t_trunc"])
        new_t_trunc = max(old_t_trunc + 1, int(np.ceil(old_t_trunc * truncation_policy.growth)))
        if truncation_policy.max_t_trunc is not None:
            new_t_trunc = min(new_t_trunc, truncation_policy.max_t_trunc)
        if new_t_trunc <= old_t_trunc:
            break

        logging.warning(
            "CDF coverage %.6f below threshold %.6f for %s; increasing t_trunc from %s to %s",
            coverage,
            cdf_threshold,
            parameters["protocol"],
            old_t_trunc,
            new_t_trunc,
        )
        parameters["t_trunc"] = new_t_trunc

    if coverage < cdf_threshold:
        logging.error(f"CDF coverage {coverage} below threshold {cdf_threshold}")
        raise ThresholdExceededError(extra_info={"cdf_coverage": coverage})

    skr = float(
        secret_key_rate(
            pmf,
            state_func,
            extrapolation=True,
            state_type=simulator.state_type,
        )
    )

    logging.info(f"Protocol {parameters['protocol']},\t r = {skr}\n")
    return skr, pmf, state_func


def brute_force_optimization(
    simulator: RepeaterChainEvaluation,
    parameters: SimParameters,
    nodes: int,
    max_dists: int,
    filename: str,
    cdf_threshold: float,
    truncation_policy: TruncationPolicy,
) -> None:
    """
    Brute-force all asymmetric protocols in the space and report the best one.
    """
    results: List[Tuple[np.float64, Tuple[str, ...]]] = []

    space = get_asym_protocol_space(nodes, max_dists)
    logging.info(f"The space of asymmetric protocols has size {len(space)}")

    for idx, protocol in enumerate(space):
        try:
            parameters["protocol"] = protocol
            skr, _, _ = asym_protocol_runner(
                simulator,
                parameters,
                nodes,
                cdf_threshold,
                truncation_policy,
                idx,
                len(space),
            )
            results.append((np.float64(skr), protocol))

        except ThresholdExceededError:
            logging.warning(
                f"Simulation under coverage for {protocol}\n"
                "Consider increasing truncation time (or loosening --cdf_threshold)."
            )

    if not results:
        raise RuntimeError("No protocol met the CDF coverage threshold; nothing to report.")

    ordered_results = sorted(results, key=lambda x: x[0], reverse=True)

    if filename:
        write_results(filename, ordered_results)

    best_results = (float(ordered_results[0][0]), ordered_results[0][1])
    logging.info(f"\nBest results: {best_results}")


# Cache results of the objective function to avoid re-evaluating the same point and speed up the optimization
cache_results: Dict[Tuple[str, ...], float] = {}
strategy_to_protocol: Dict[Tuple[int, float, float, float], Tuple[str, ...]] = {}


def objective_key_rate(
    space: Dict[str, Any],
    nodes: int,
    max_dists: int,
    shot_count: List[int],
    gp_shots: int,
    parameters: SimParameters,
    simulator: RepeaterChainEvaluation,
    cdf_threshold: float,
    truncation_policy: TruncationPolicy,
) -> float:
    """
    Objective function, consider the whole space of actions,
        returning a negative secret key rate
        in order for the optimizer to maximize the function.
    """
    shot_count[0] += 1

    gamma = float(space["gamma"])
    eta = float(space["eta"])
    tau = float(space["tau"])
    kappa = int(space["rounds of distillation"])

    logging.info(f"\n\nGenerating a protocol with gamma={gamma}, eta={eta}, tau={tau}, kappa={kappa}")
    parameters["protocol"] = get_protocol_from_center_spacing_symmetricity(
        nodes, max_dists, gamma, kappa, eta, tau
    )
    logging.info(f"Protocol generated: {parameters['protocol']}")
    strategy_to_protocol[(kappa, gamma, eta, tau)] = parameters['protocol']

    if parameters['protocol'] in cache_results:
        logging.info("Already evaluated protocol, returning cached result")
        return -cache_results[parameters['protocol']]
    
    try:
        secret_key_rate, pmf, _ = asym_protocol_runner(
            simulator,
            parameters,
            nodes,
            cdf_threshold,
            truncation_policy,
            idx=shot_count[0],
            space_len=gp_shots,
        )
    except ThresholdExceededError:
        logging.warning(
            "Simulation under coverage for %s after t_trunc attempts; penalizing GP sample.",
            parameters["protocol"],
        )
        cache_results[parameters["protocol"]] = 0.0
        space.update({'protocol': parameters["protocol"]})
        return 0.0

    cache_results[parameters["protocol"]] = secret_key_rate
    space.update({'protocol': parameters["protocol"]})
    # The gaussian process minimizes the function, so return the negative of the key rate 
    return -secret_key_rate


def is_gp_done(result: OptimizeResult) -> None:
    """
    Callback: attach the last sampled protocol to result.x_iters[-1]
    so downstream code (e.g., get_ordered_results) can retrieve it.
    """
    key = tuple(result.x_iters[-1])
    # key is (kappa, gamma, eta, tau)
    result.x_iters[-1].append(strategy_to_protocol[cast(Tuple[int, float, float, float], key)])


def gaussian_optimization(
    simulator: RepeaterChainEvaluation,
    parameters: SimParameters,
    nodes: int,
    max_dists: int,
    gp_shots: Optional[int],
    gp_initial_points: Optional[int],
    filename: str,
    cdf_threshold: float,
    truncation_policy: TruncationPolicy,
    random_state: Optional[int] = None,
) -> None:
    """
    Bayesian optimization over protocol-encoding parameters (gamma, eta, tau, kappa).
    """
    logging.info(f"\n\nNumber of nodes: {nodes}, max dists: {max_dists}, seed: {random_state}")
    logging.info(f"Gaussian process with {gp_shots} evaluations and {gp_initial_points} initial points\n\n")

    shot_count = [-1]
    v = 2 * (nodes - 1) - 1

    space = [
        Integer(0, v * max_dists, name="rounds of distillation")
        if max_dists != 0
        else Categorical([0], name="rounds of distillation"),
        Real(0, 1, name="gamma"),
        Real(-1, 1, name="eta"),
        Real(0, 1, name="tau"),
    ]

    @use_named_args(space)
    def wrapped_objective(**sp: Any) -> float:
        return objective_key_rate(
            space=sp,
            nodes=nodes,
            max_dists=max_dists,
            shot_count=shot_count,
            gp_shots=int(gp_shots),
            parameters=parameters,
            simulator=simulator,
            cdf_threshold=cdf_threshold,
            truncation_policy=truncation_policy,
        )

    ordered_results: List[Tuple[np.float64, Tuple[int, ...]]] = []

    try:
        result: OptimizeResult = gp_minimize(
            wrapped_objective,
            space,
            n_calls=gp_shots,
            n_initial_points=gp_initial_points,
            callback=[is_gp_done],
            acq_func='LCB',
            kappa=1.96,
            noise=1e-10,    # There is no noise in results
            random_state=random_state
        )

        ordered_results = get_ordered_results(
            result=result,
            space_type="asymmetric",
            number_of_swaps=None,
        )

        if filename:
            plot_optimization_process(
                min_dists=0,
                max_dists=max_dists,
                parameters=parameters,
                results=ordered_results,
                gp_result=result,
            )

    finally:
        cache_results.clear()

    if filename and ordered_results:
        write_results(filename, ordered_results)

    if ordered_results:
        best_results = (float(ordered_results[0][0]), ordered_results[0][1])
        logging.info(f"\nBest results: {best_results}")
    else:
        logging.warning("No valid GP results to report (all evaluations fell under CDF threshold).")


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes in the chain")
    parser.add_argument("--max_dists", type=int, default=2, help="Maximum round of distillations per segment per level")

    parser.add_argument("--optimizer", type=optimizerType, default="gp", help="Optimizer: {gp, bf}")
    parser.add_argument("--gp_shots", type=int, default=100, 
                        help=(  "Number of GP evaluations"
                                "If not specified, it is computed dynamically based on the protocol"))
    parser.add_argument("--gp_initial_points", type=int, default=10, help="Number of initial random points for GP")

    parser.add_argument("--filename", type=str, default="output.txt", help="Filename for output log")
    parser.add_argument(
        "--cdf_threshold",
        type=float,
        default=0.99,
        help="Discard simulations whose pmf CDF coverage is below this threshold",
    )
    parser.add_argument(
        "--t_trunc",
        "--t-trunc",
        dest="t_trunc",
        type=int,
        default=None,
        help="Use a fixed truncation time instead of the heuristic estimate.",
    )
    parser.add_argument(
        "--adaptive_t_trunc_attempts",
        "--adaptive-t-trunc-attempts",
        dest="adaptive_t_trunc_attempts",
        type=int,
        default=3,
        help="Number of heuristic t_trunc retries when CDF coverage is below threshold.",
    )
    parser.add_argument(
        "--t_trunc_growth",
        "--t-trunc-growth",
        dest="t_trunc_growth",
        type=float,
        default=2.0,
        help="Multiplicative growth factor for adaptive t_trunc retries.",
    )
    parser.add_argument(
        "--max_t_trunc",
        "--max-t-trunc",
        dest="max_t_trunc",
        type=int,
        default=None,
        help="Optional upper bound for the heuristic and adaptive t_trunc retries.",
    )
    parser.add_argument("--p_swap", type=float, default=0.85, help="Swapping success probability")
    parser.add_argument("--p_gen", type=float, nargs="+", default=[0.0009082], help="Generation success probability")

    parser.add_argument(
        "--state",
        type=str,
        choices=["werner", "bell", "pf"],
        default="werner",
        help="Quantum state model: werner, bell (legacy Bell weights), or pf (Pauli-Fourier Bell-diagonal)",
    )

    # Werner parameters
    parser.add_argument("--w0", type=float, nargs="+", default=[0.9523], help="Werner parameter(s)")

    # Bell-diagonal parameters
    # One segment:  --lambdas 0.85 0.08 0.05 0.02
    # Multiple segments: repeat the flag: --lambdas ... --lambdas ... (action=append)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs=4,
        action="append",
        default=None,
        help="Bell-diagonal weights (lambda_phi+, lambda_psi+, lambda_psi-, lambda_phi-). "
             "Repeat flag to provide one 4-tuple per segment.",
    )
    parser.add_argument(
        "--depolarizing_rate",
        type=float,
        nargs="+",
        default=[0.001],
        help="Depolarizing noise parameter (Bell).",
    )
    parser.add_argument(
        "--dephasing_rate",
        type=float,
        nargs="+",
        default=[0.0],
        help="Dephasing noise parameter (Bell).",
    )

    parser.add_argument(
        "--t_coh",
        type=int,
        nargs="+",
        default=[560000],
        help="Coherence time (Werner).",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed for GP optimization")
    parser.add_argument(
        "--w-twirling",
        dest="w_twirling",
        action="store_true",
        default=True,
        help="Use Werner twirling after swap/distillation (default True).",
    )
    parser.add_argument(
        "--no-w-twirling",
        dest="w_twirling",
        action="store_false",
        help="Disable Werner twirling after swap/distillation.",
    )

    args = parser.parse_args()

    if args.t_trunc is not None and args.t_trunc < 1:
        raise ValueError("--t_trunc must be positive")
    if args.adaptive_t_trunc_attempts < 1:
        raise ValueError("--adaptive_t_trunc_attempts must be at least 1")
    if args.t_trunc_growth <= 1.0:
        raise ValueError("--t_trunc_growth must be larger than 1")
    if args.max_t_trunc is not None and args.max_t_trunc < 1:
        raise ValueError("--max_t_trunc must be positive")
    if not 0.0 < args.cdf_threshold < 1.0:
        raise ValueError("--cdf_threshold must be between 0 and 1")

    nodes: int = args.nodes
    max_dists: int = args.max_dists

    optimizer: OptimizerType = args.optimizer
    gp_shots: int = args.gp_shots
    gp_initial_points: int = args.gp_initial_points

    filename: str = args.filename
    cdf_threshold: float = args.cdf_threshold

    p_swap: float = float(args.p_swap)
    p_gen: Union[float, List[float]] = args.p_gen if len(args.p_gen) > 1 else float(args.p_gen[0])
    random_state: Optional[int] = args.seed
    truncation_policy = TruncationPolicy(
        fixed_t_trunc=args.t_trunc,
        adaptive_attempts=args.adaptive_t_trunc_attempts,
        growth=args.t_trunc_growth,
        max_t_trunc=args.max_t_trunc,
    )

    def infer_state_type(state: StateName):
        state_l = state.lower().strip()
        if state_l == "werner":
            return WernerState
        if state_l in {"bell", "bell_diagonal", "belldiagonal"}:
            return BellState
        if state_l in {"pf", "pauli_fourier", "paulifourier"}:
            return PauliFourierState
        raise ValueError(f"Unknown state '{state}'. Use 'werner', 'bell', or 'pf'.")

    state_type = infer_state_type(args.state)

    parameters: SimParameters = {
        "p_gen": p_gen,
        "p_swap": p_swap,
    }

    if state_type == WernerState:
        w0: Union[float, List[float]] = args.w0 if len(args.w0) > 1 else float(args.w0[0])
        parameters["w0"] = w0
        t_coh: Union[int, List[int]] = args.t_coh if len(args.t_coh) > 1 else int(args.t_coh[0])
        parameters["t_coh"] = t_coh
    else:
        # Lambdas:
        # - If user provided --lambdas once: use that (and replicate if needed downstream).
        # - If provided multiple times: treat as per-segment lambdas.
        if args.lambdas is None:
            raise ValueError("For Bell-diagonal states you must provide --lambdas (4-value list).")
        parameters["lambdas"] = args.lambdas if len(args.lambdas) > 1 else args.lambdas[0]

        dep = args.depolarizing_rate
        depolarizing_rate: Union[float, List[float]] = dep if len(dep) > 1 else float(dep[0])
        parameters["depolarizing_rate"] = depolarizing_rate

        deph = args.dephasing_rate
        dephasing_rate: Union[float, List[float]] = deph if len(deph) > 1 else float(deph[0])
        parameters["dephasing_rate"] = dephasing_rate

    simulator = RepeaterChainEvaluation(state_type=state_type, w_twirling=bool(args.w_twirling))

    # Run
    if optimizer == "gp":
        gaussian_optimization(
            simulator=simulator,
            parameters=parameters,
            nodes=nodes,
            max_dists=max_dists,
            gp_shots=gp_shots,
            gp_initial_points=gp_initial_points,
            filename=filename,
            cdf_threshold=cdf_threshold,
            truncation_policy=truncation_policy,
            random_state=random_state,
        )
    elif optimizer == "bf":
        brute_force_optimization(
            simulator=simulator,
            parameters=parameters,
            nodes=nodes,
            max_dists=max_dists,
            filename=filename,
            cdf_threshold=cdf_threshold,
            truncation_policy=truncation_policy,
        )
