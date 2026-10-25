"""Small PE/distillation study: python -m src.analysis.pe_distillation.

Z is the existing recurrence, X surrounds it with bilateral Hadamards, and
11 retains only the Z-recurrence target outcome 11. The latter two are local
analysis prototypes: their exact maps replace only the instantaneous distillation
map inside a scoped context. Waiting noise and retries use the existing backend.
No production operation, app option, or cache is modified.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core.pauli_adc import coordinates_to_matrix, matrix_to_coordinates, RealXState
from src.core.pauli_adc import swap_join_sequential_efficient as join
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.visualization.config import DEFAULT, expand_config
from src.visualization.metrics import summarize, time_quantities, validate_outputs

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])
BASIS = tuple(np.kron(a, b) for a, b in ((I, I), (I, Z), (Z, I), (Z, Z), (X, X), (Y, Y)))
H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
HH = np.kron(H, H)
CNOTS = np.zeros((16, 16))
for i in range(16):
    a, b, c, d = [(i >> k) & 1 for k in (3, 2, 1, 0)]
    CNOTS[(a << 3) + (b << 2) + ((c ^ a) << 1) + (d ^ b), i] = 1


def source(q, eta=1.):
    return np.array([1, 1-2*q*eta, 1-2*q*eta, 1-4*q*eta*(1-eta),
                     2*np.sqrt(q*(1-q))*eta, -2*np.sqrt(q*(1-q))*eta])


def recurrence_kraus(mode):
    """Qubit order: kept A,B; auxiliary C,D. Targets C,D are measured."""
    if mode not in ("Z", "X", "11"):
        raise ValueError("recurrence mode must be Z, X or 11")
    operators = []
    for outcome in ((3,) if mode == "11" else (0, 3)):
        operator = CNOTS[np.arange(4)*4+outcome, :]
        if mode == "X":
            operator = HH @ operator @ np.kron(HH, HH)
        operators.append(operator)
    return operators


@lru_cache(maxsize=3)
def recurrence_tensor(mode):
    """Derive the bilinear six-coordinate map from the full dense circuit."""
    tensor = np.zeros((6, 6, 6))
    operators = recurrence_kraus(mode)
    for i, a in enumerate(BASIS):
        for j, b in enumerate(BASIS):
            out = sum(k @ np.kron(a, b) @ k.conj().T for k in operators)/16
            coordinates = matrix_to_coordinates(out)  # Also verifies real-X closure.
            tensor[:, i, j] = coordinates
    # Clifford circuits have exact dyadic coefficients. Remove only circuit
    # construction roundoff, without clipping simulated states or probabilities.
    exact = np.rint(4*tensor)/4
    assert np.max(abs(tensor-exact)) < 1e-13
    exact.setflags(write=False)
    return exact


@contextmanager
def recurrence_variant(mode, erase_bias=False):
    """Optionally project away biases immediately before distillation only.

    This is a matched prediction control, not a full state-projection model:
    it preserves trace and XX/YY/ZZ at each distillation input.
    """
    if mode == "none" or (mode == "Z" and not erase_bias):
        yield
        return
    tensor = recurrence_tensor(mode) if mode != "Z" else None
    original = join._bilinear_operation

    def operation(left, right, *, operation, **kwargs):
        if operation == "dist":
            if erase_bias:
                left, right = left.copy(), right.copy()
                left[:, 1:3] = right[:, 1:3] = 0.
            if kwargs.get("werner_twirl"):
                raise ValueError("This study excludes state twirling")
            if tensor is not None:
                return np.einsum("ni,nj,kij->nk", left, right, tensor, optimize=True)
        return original(left, right, operation=operation, **kwargs)

    with patch.object(join, "_bilinear_operation", operation):
        yield


def evaluate(q, t1, placement, mode, policy, erase_bias=False):
    config = deepcopy(DEFAULT)
    config["initial_state"] = {"kind": "real_x", "values": source(q).tolist()}
    config["hardware"].update(p_gen=.2, p_swap=1., p_distillation=1.)
    config["protocol"]["bell_outcomes"] = policy
    spans = {"none": [], "leaves": ["0:0", "1:1", "2:2", "3:3"],
             "middle": ["0:1", "2:3"], "root": ["0:3"]}[placement]
    config["protocol"]["distillation_rounds"] = {span: 1 for span in spans}
    config["memory"] = {"kind": "t1_t2", "t1_slots": t1, "t2_slots": 2*t1}
    previous, summary, converged, error = None, {}, False, None
    fields = ("rate_bits_per_slot", "mean_wait_repository_slots", "eX", "eZ")
    try:
        # Direct backend calls avoid cache collisions between prototype maps.
        with recurrence_variant(mode, erase_bias):
            simulator = RepeaterChainEvaluation(state_type=RealXState, efficient=True,
                use_fft=True, use_gpu=False, w_twirling=False)
            for horizon in (1024, 2048, 4096, 8192):
                config["numerics"]["t_trunc"] = horizon
                _, parameters, _ = expand_config(config)
                pmf, x = simulator.asymmetric_heterogeneous_protocol(parameters, 4)
                pmf, x, _ = validate_outputs(pmf, x, horizon)
                summary, _ = summarize(pmf, x, .999999)
                if (previous is not None and min(previous["coverage"], summary["coverage"]) >= .999999
                        and np.allclose([previous[k] for k in fields], [summary[k] for k in fields],
                                        rtol=1e-6, atol=1e-9)):
                    converged = True
                    break
                previous = summary
        if not converged:
            error = "Horizon check failed"
    except ValueError as exc:
        error = str(exc)
    if not converged:
        summary.update({k: np.nan for k in (*fields, "pooled_secret_fraction")})
        summary["status"] = "failed"
    return dict(q=q, T1=t1, placement=placement, mode=mode, policy=policy, bias_blind=erase_bias,
                converged=converged, error=error, horizon=horizon, **summary)


def main():
    start = perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(__file__).resolve().parents[2]/"results/analysis"/f"pe-distillation-{stamp}"
    output.mkdir(parents=True)
    logging.basicConfig(filename=output/"evaluation.log", level=logging.WARNING)
    controls = []
    for q in (.25, .45, .5):
        for eta in (1., .9, .8):
            initial = source(q, eta)
            controls.append(dict(q=q, eta=eta, mode="input", probability=1.,
                SKF=float(time_quantities(initial)["secret_fraction"])))
            for mode in ("Z", "X", "11"):
                rho = coordinates_to_matrix(initial)
                out = sum(k @ np.kron(rho, rho) @ k.conj().T for k in recurrence_kraus(mode))
                probability = float(np.trace(out).real)
                x = matrix_to_coordinates(out/probability)
                controls.append(dict(q=q, eta=eta, mode=mode, probability=probability,
                    SKF=float(time_quantities(x)["secret_fraction"])))
    pd.DataFrame(controls).to_csv(output/"state-controls.csv", index=False)
    rows = []
    for q in (.25, .45, .5):
        for t1 in (100., 400.):
            for placement in ("none", "leaves", "middle", "root"):
                for mode in (("none",) if placement == "none" else ("Z", "X", "11")):
                    for policy in (("all_corrected", "psi_pair") if placement == "none" else ("all_corrected",)):
                        rows.append(evaluate(q, t1, placement, mode, policy))
            print(f"q={q:g}, T1={t1:g} finished", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"timing.csv", index=False)
    eligible = frame[frame.converged]
    best = eligible.loc[eligible.groupby(["q", "T1"]).rate_bits_per_slot.idxmax()]
    best.to_csv(output/"best-tested.csv", index=False)
    bias_control = [evaluate(.25, 100., placement, mode, "all_corrected", erase_bias=True)
                    for placement, mode in (("none", "none"), ("root", "Z"), ("root", "X"), ("root", "11"))]
    pd.DataFrame(bias_control).to_csv(output/"bias-control.csv", index=False)
    settings = dict(links=4, tree=[[0, 1], [2, 3]], p_gen=.2, p_swap=1.,
        p_distillation=1., T2_over_T1=2., q=[.25, .45, .5], T1=[100., 400.],
        cutoffs=None, max_rounds_per_selected_node=1, instantaneous_ideal_operations=True,
        parallel_copy_preparation=True, resource_note="Distillation requires extra copies and memories; SKR is per slot, not per memory",
        modes={"Z": "Existing bilateral-CNOT recurrence; accept 00 and 11",
               "X": "Hadamards on all four inputs, Z recurrence, Hadamards on retained pair",
               "11": "Z recurrence accepting target outcome 11 only"},
        horizons=[1024, 2048, 4096, 8192], coverage_target=.999999,
        convergence_rtol=1e-6, convergence_atol=1e-9,
        controls="Fixed-age damped pairs, without preparation costs; timing study uses fresh sources",
        points=len(frame), converged=int(frame.converged.sum()),
        bias_control_points=len(bias_control), bias_control_converged=sum(r["converged"] for r in bias_control),
        bias_control="Erase IZ/ZI only at distillation inputs; preserve all correlators; not a full projection schedule",
        elapsed_seconds=perf_counter()-start)
    (output/"settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    print(best[["q", "T1", "placement", "mode", "rate_bits_per_slot"]].to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
