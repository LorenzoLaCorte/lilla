"""Fidelity-matched state-family comparison with recurrence placement.

Run after activating venv: python -m src.analysis.distillation_family_scan
"""
from copy import deepcopy
from itertools import product
import json
import logging
from pathlib import Path

from src.analysis.state_family_scan import states
from src.visualization.comparisons import near_tie
from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService

OUTPUT = Path("src/analysis/distillation_family_results.json")
logging.getLogger().setLevel(logging.ERROR)
DIST = {"none": {}, "left": {"0:0": 1}, "right": {"1:1": 1},
        "both": {"0:0": 1, "1:1": 1}, "root": {"0:1": 1}}


def main():
    service = EvaluationService(cache_size=0)
    results = []
    for family, fidelity, t1 in product(
            ("phase_error", "bit_error", "r_state"), (.95, .97), (30, 80)):
        models = states(fidelity, family)
        if family != "r_state":
            del models["bell_projection"]  # identical to physical input
        case = {"family": family, "fidelity": fidelity, "t1_slots": t1,
                "t2_slots": 2*t1, "p_gen": .2, "p_swap": .9,
                "links": 2, "horizon": 2048, "state_models": models}
        base = deepcopy(DEFAULT)
        base["links"] = 2
        base["hardware"].update(p_gen=.2, p_swap=.9)
        base["memory"].update(t1_slots=t1, t2_slots=2*t1)
        base["numerics"].update(t_trunc=2048, coverage_target=.999)
        base["protocol"]["tree"] = [0, 1]
        rows = []
        for cutoff, dist in product((None, 1, 2, 4, 8, 16, 32), DIST):
            candidate = {"cutoff": cutoff, "distillation": dist}
            predictions = {}
            for model, initial in models.items():
                c = deepcopy(base)
                c["initial_state"] = deepcopy(initial)
                c["protocol"].update(memory_cutoff_slots=cutoff,
                                     distillation_rounds=DIST[dist])
                try:
                    s = service.evaluate(c).summary
                    predictions[model] = {k: s[k] for k in (
                        "status", "coverage", "rate_bits_per_slot", "average_fidelity",
                        "eX", "eZ", "mean_wait_repository_slots")}
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    predictions[model] = {"status": "failed", "error": str(exc)}
            rows.append({"candidate": candidate, "predictions": predictions})
        covered = [r for r in rows if all(p["status"] == "covered"
                   for p in r["predictions"].values())]
        if covered:
            score = lambda r, m: r["predictions"][m]["rate_bits_per_slot"]
            top = {m: max(score(r, m) for r in covered) for m in models}
            winners = {m: [r for r in covered if near_tie(score(r, m), top[m])]
                       for m in models}
            selection = {
                "best_rates": top,
                "winners": {m: [r["candidate"] for r in v] for m, v in winners.items()},
                "physical_rate_of_baseline_winners": {
                    m: [score(r, "physical") for r in v] for m, v in winners.items()},
                "regret_range": {
                    m: [top["physical"]-max(score(r, "physical") for r in v),
                        top["physical"]-min(score(r, "physical") for r in v)]
                    for m, v in winners.items()},
            }
        else:
            selection = None
        result = {"case": case, "n_candidates": len(rows),
                  "n_covered": len(covered), "selection": selection, "rows": rows}
        results.append(result)
        OUTPUT.write_text(json.dumps({"results": results}, indent=2) + "\n")
        print(json.dumps({"family": family, "fidelity": fidelity, "t1": t1,
                          "covered": len(covered), "selection": selection}), flush=True)


if __name__ == "__main__":
    main()
