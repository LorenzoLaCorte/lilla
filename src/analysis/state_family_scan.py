"""Finite-candidate policy transfer across fidelity-matched state families.

Run after activating venv: python -m src.analysis.state_family_scan
Each physical family is compared with its Bell-state twirl and a Werner state
of the same Phi+ fidelity, all under the same physical T1/T2 memory channel.
"""
from copy import deepcopy
from itertools import product
import json
import logging
from pathlib import Path

from src.visualization.comparisons import near_tie
from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService
from src.visualization.protocols import ordered_trees

OUTPUT = Path("src/analysis/state_family_results.json")
logging.getLogger().setLevel(logging.ERROR)


def states(fidelity, family):
    f = fidelity
    werner = {"kind": "werner", "fidelity": f}
    if family == "r_state":
        physical = {"kind": "real_x", "values": [1, f-1, 1-f, 2*f-1, f, -f]}
        bell = {"kind": "bell_weights", "values": [f, (1-f)/2, (1-f)/2, 0]}
    elif family == "phase_error":
        physical = bell = {"kind": "bell_weights", "values": [f, 0, 0, 1-f]}
    elif family == "bit_error":
        physical = bell = {"kind": "bell_weights", "values": [f, 1-f, 0, 0]}
    else:
        raise ValueError(family)
    return {"physical": physical, "bell_projection": bell, "werner": werner}


def main():
    service = EvaluationService(cache_size=0)
    results = []
    for family, fidelity, t1 in product(
            ("r_state", "phase_error", "bit_error"), (.95, .98), (20, 50)):
        state_models = states(fidelity, family)
        case = {"family": family, "fidelity": fidelity, "t1_slots": t1,
                "t2_slots": 2*t1, "p_gen": .2, "p_swap": .9,
                "links": 3, "horizon": 1024,
                "state_models": state_models}
        base = deepcopy(DEFAULT)
        base["links"] = 3
        base["hardware"].update(p_gen=.2, p_swap=.9)
        base["memory"].update(t1_slots=t1, t2_slots=2*t1)
        base["numerics"].update(t_trunc=1024, coverage_target=.999)
        rows = []
        for tree, cutoff, outcome in product(
                ordered_trees(3), (None, 1, 2, 3, 4, 8, 16),
                ("all_corrected", "phi_pair", "psi_pair")):
            candidate = {"tree": tree, "cutoff": cutoff, "outcomes": outcome}
            predictions = {}
            for model, initial in state_models.items():
                c = deepcopy(base)
                c["name"] = family + " as " + model
                c["initial_state"] = deepcopy(initial)
                c["protocol"].update(tree=tree, memory_cutoff_slots=cutoff,
                                     bell_outcomes=outcome)
                try:
                    s = service.evaluate(c).summary
                    predictions[model] = {k: s[k] for k in (
                        "status", "coverage", "rate_bits_per_slot", "average_fidelity",
                        "eX", "eZ", "mean_wait_repository_slots")}
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    predictions[model] = {"status": "failed", "error": str(exc)}
            rows.append({"candidate": candidate, "predictions": predictions})
        eligible = [r for r in rows if all(p["status"] == "covered"
                    for p in r["predictions"].values())]
        if eligible:
            score = lambda r, m: r["predictions"][m]["rate_bits_per_slot"]
            top = {m: max(score(r, m) for r in eligible) for m in state_models}
            winners = {m: [r for r in eligible if near_tie(score(r, m), top[m])]
                       for m in state_models}
            selection = {
                "best_rates": top,
                "winners": {m: [r["candidate"] for r in v] for m, v in winners.items()},
                "physical_rate_of_baseline_winners": {
                    m: [score(r, "physical") for r in v]
                    for m, v in winners.items()},
                "regret_range": {
                    m: [top["physical"]-max(score(r, "physical") for r in v),
                        top["physical"]-min(score(r, "physical") for r in v)]
                    for m, v in winners.items()},
            }
        else:
            selection = None
        result = {"case": case, "n_candidates": len(rows),
                  "n_covered": len(eligible), "selection": selection, "rows": rows}
        results.append(result)
        OUTPUT.write_text(json.dumps({"results": results}, indent=2) + "\n")
        print(json.dumps({"family": family, "fidelity": fidelity, "t1": t1,
                          "covered": len(eligible),
                          "selection": selection}), flush=True)


if __name__ == "__main__":
    main()
