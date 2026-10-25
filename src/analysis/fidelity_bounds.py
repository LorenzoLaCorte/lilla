"""Illustrate the swapping fidelity bounds using the app's evaluation service.

Run from the repository: source venv/bin/activate && python -m src.analysis.fidelity_bounds
All sources are fresh, with ideal deterministic hardware, no memory noise,
cutoffs or distillation. Bell ordering: Phi+, Psi+, Psi-, Phi-.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService
from src.visualization.metrics import bell_populations
from src.visualization.protocols import balanced_tree
from src.visualization.storage import save_run


def configuration(name, links, initial, policy="all_corrected", tree=None):
    config = deepcopy(DEFAULT)
    config.update(name=name, links=links, initial_state=initial)
    config["hardware"].update(p_gen=1., p_swap=1., p_distillation=1.)
    config["memory"].update(t1_slots=None, t2_slots=None)
    config["protocol"].update(tree=balanced_tree(links) if tree is None else tree,
                              bell_outcomes=policy)
    config["numerics"].update(t_trunc=512, coverage_target=1-1e-10)
    return config


def pauli_convolution(weights):
    """Independent reference: Pauli labels add modulo phase (bitwise XOR)."""
    result = np.array([1., 0., 0., 0.])
    for row in weights:
        result = np.array([sum(result[i]*row[i ^ j] for i in range(4))
                           for j in range(4)])
    return result


def tight_lower(links, fidelity):
    """Equal fidelities, independent states, 1/2 <= F <= 1, N >= 2."""
    m, r = divmod(links, 3)
    v = 2*fidelity-1
    return (1+(3-r)*v**(links-m)+r*v**(links-m-1))/4


def main():
    directory = Path("results/analysis") / ("fidelity-bounds-" +
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    directory.mkdir(parents=True)
    service = EvaluationService(cache_size=0)
    rows, scaling = [], []

    def evaluate(config, save=False):
        run = service.evaluate(config)
        assert run.summary["status"] == "covered", run.summary
        row = {"case": config["name"], "links": config["links"],
               "policy": config["protocol"]["bell_outcomes"],
               **{key: run.summary[key] for key in ("average_fidelity", "eX", "eZ",
                  "pooled_secret_fraction", "rate_bits_per_slot",
                  "mean_wait_repository_slots", "coverage")}}
        if save:
            row["bundle"] = str(save_run(run, directory).relative_to(directory))
        rows.append(row)
        return run

    fidelity = .9
    for links in range(2, 9):
        candidates = []
        # With identical source fidelities, order is immaterial for ideal swaps.
        # Cover every count of the three possible single-type Bell errors.
        for nx in range(links+1):
            for ny in range(links-nx+1):
                counts = (nx, ny, links-nx-ny)
                errors = np.repeat(np.arange(1, 4), counts)
                weights = np.zeros((links, 4))
                weights[:, 0] = fidelity
                weights[np.arange(links), errors] = 1-fidelity
                run = evaluate(configuration(f"rank-two errors {counts}", links,
                               {"kind": "bell_weights", "values": weights.tolist()}),
                               save=links in (2, 4, 8) and (nx == links or counts == (1, 1, 0)))
                np.testing.assert_allclose(bell_populations(run.summary["pooled_coordinates"]),
                                           pauli_convolution(weights), atol=1e-12, rtol=0)
                candidates.append((run.summary["average_fidelity"], counts))
        werner = evaluate(configuration("Werner", links,
                         {"kind": "werner", "fidelity": fidelity}), save=links in (2, 4, 8))
        lower = fidelity**links
        upper = (1+(2*fidelity-1)**links)/2
        homogeneous_lower = tight_lower(links, fidelity)
        exact_werner = (1+3*((4*fidelity-1)/3)**links)/4
        np.testing.assert_allclose(werner.summary["average_fidelity"], exact_werner, atol=1e-12)
        np.testing.assert_allclose(max(candidates)[0], upper, atol=1e-12)
        np.testing.assert_allclose(min(candidates)[0], homogeneous_lower, atol=1e-12)
        assert min(candidates)[0] >= lower-1e-12
        scaling.append({"links": links, "source_fidelity": fidelity, "lower_bound": lower,
                        "smallest_tested": min(candidates)[0], "min_error_counts": min(candidates)[1],
                        "werner": exact_werner, "upper_bound": upper,
                        "tested_error_counts": len(candidates)})

    # Same Bell populations, with/without the coherences of a pure PE source.
    q = .25
    transverse = 2*np.sqrt(q*(1-q))
    f_pe = (1+transverse)/2
    initial_states = {
        "PE q=0.25": {"kind": "real_x", "values": [1, 1-2*q, 1-2*q, 1,
                                                       transverse, -transverse]},
        "Bell-twirled PE": {"kind": "bell_weights", "values": [f_pe, 0, 0, 1-f_pe]},
    }
    for policy in ("all_corrected", "phi_pair", "psi_pair"):
        for name, initial in initial_states.items():
            run = evaluate(configuration(name, 2, initial, policy), save=True)
            expected = ((1+transverse**2)/2 if policy == "all_corrected" or
                        name == "Bell-twirled PE" else
                        1. if policy == "psi_pair" else .8)
            np.testing.assert_allclose(run.summary["average_fidelity"], expected, atol=1e-12)

    # Change tree shape while keeping the physical inputs fixed.
    weights = [[fidelity, 1-fidelity, 0, 0]]*4
    for name, tree in (("left comb", [[[0, 1], 2], 3]),
                       ("balanced", [[0, 1], [2, 3]]),
                       ("right comb", [0, [1, [2, 3]]])):
        run = evaluate(configuration(name, 4, {"kind": "bell_weights", "values": weights}, tree=tree))
        np.testing.assert_allclose(run.summary["average_fidelity"], .7048, atol=1e-12)

    # Domain check: without a high-fidelity restriction the stated upper bound
    # need not hold. X, Y and Z together cancel even when errors dominate.
    low_fidelity = [[.1, .9, 0, 0], [.1, 0, .9, 0], [.1, 0, 0, .9]]
    run = evaluate(configuration("F=0.1 domain counterexample", 3,
                   {"kind": "bell_weights", "values": low_fidelity}), save=True)
    np.testing.assert_allclose(run.summary["average_fidelity"], .730, atol=1e-12)
    assert run.summary["average_fidelity"] > (1+(2*.1-1)**3)/2

    # Further examples for the explanatory notes, including both endpoints.
    examples = []
    for f in (.5, .6, .75, .98, 1.):
        for links in (2, 3, 4, 8):
            weights = np.zeros((links, 4))
            weights[:, 0] = f
            weights[np.arange(links), 1+np.arange(links) % 3] = 1-f
            values = {}
            for name, initial in (
                ("balanced error types", {"kind": "bell_weights", "values": weights.tolist()}),
                ("Werner", {"kind": "werner", "fidelity": f}),
                ("same error type", {"kind": "bell_weights", "values": [f, 1-f, 0, 0]}),
            ):
                config = configuration(f"Additional F={f}: {name}", links, initial)
                values[name] = evaluate(config).summary["average_fidelity"]
            np.testing.assert_allclose(values["balanced error types"], tight_lower(links, f), atol=1e-12)
            np.testing.assert_allclose(values["Werner"], (1+3*((4*f-1)/3)**links)/4, atol=1e-12)
            np.testing.assert_allclose(values["same error type"], (1+(2*f-1)**links)/2, atol=1e-12)
            examples.append({"links": links, "source_fidelity": f, "paper_lower": f**links,
                             **values})
    for pattern in ("XXX", "XXY", "XYZ", "XXXX", "XXYY", "XXYZ"):
        weights = [[.9]+[.1 if symbol == error else 0. for error in "XYZ"] for symbol in pattern]
        run = evaluate(configuration(pattern, len(pattern), {"kind": "bell_weights", "values": weights}))
        np.testing.assert_allclose(run.summary["average_fidelity"], pauli_convolution(weights)[0], atol=1e-12)
    pd.DataFrame(examples).to_csv(directory/"additional-examples.csv", index=False)

    # Sanity-check nonvertex error mixtures; the proof, rather than this random
    # sample, establishes global optimality over the continuous domain.
    rng = np.random.default_rng(20261025)
    for _ in range(1000):
        links = int(rng.integers(2, 25))
        f = float(rng.uniform(.5, 1))
        weights = np.column_stack([np.full(links, f), (1-f)*rng.dirichlet(np.ones(3), links)])
        exact = pauli_convolution(weights)[0]
        assert tight_lower(links, f)-1e-12 <= exact <= (1+(2*f-1)**links)/2+1e-12

    frame = pd.DataFrame(scaling)
    frame.to_csv(directory/"scaling.csv", index=False)
    pd.DataFrame(rows).to_csv(directory/"runs.csv", index=False)
    (directory/"setup.json").write_text(json.dumps({"source_fidelity": fidelity,
        "preparation": "Fresh inputs; p_gen=p_swap=1; no memory noise, distillation or cutoffs",
        "bell_order": ["Phi+", "Psi+", "Psi-", "Phi-"],
        "notes": "Count search covers rank-two sources only; all main runs accept all corrected outcomes. "
                 "The final F=0.1 run intentionally tests outside the high-fidelity domain.",
        "backend_calls": service.backend_calls}, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for column, label, style in (("upper_bound", "Same error type / upper bound", "o-"),
                                  ("werner", "Werner", "s-"),
                                  ("smallest_tested", "Smallest tested", "^-"),
                                  ("lower_bound", "Lower bound", "--")):
        axes[0].plot(frame.links, frame[column], style, label=label, markersize=4)
    axes[0].set(xlabel="Elementary links", ylabel=r"Final $\Phi^+$ fidelity",
                title=r"Same source fidelity $F=0.9$")
    axes[0].legend(fontsize=7)
    for name, marker in (("PE q=0.25", "o"), ("Bell-twirled PE", "s")):
        subset = [r for r in rows if r["case"] == name]
        axes[1].plot(["All corrected", r"$\Phi$ pair", r"$\Psi$ pair"],
                     [r["average_fidelity"] for r in subset], marker+"-", label=name)
    axes[1].axhline(.875, color="gray", linestyle=":", label="Non-postselected upper bound")
    axes[1].set(ylabel=r"Final $\Phi^+$ fidelity", title="Two links: scope of the bound")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(directory/"fidelity-bounds.png", dpi=180)
    plt.close(fig)
    print(frame.to_string(index=False))
    print(f"\nSaved {service.backend_calls} evaluations to {directory}")


if __name__ == "__main__":
    main()
