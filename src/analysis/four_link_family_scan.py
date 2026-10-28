"""Four-link check of fidelity-only protocol selection under T1/T2 memory.

Run after activating venv: python -m src.analysis.four_link_family_scan
"""
import argparse
from copy import deepcopy
from itertools import product
import json
import logging
from pathlib import Path

from src.analysis.state_family_scan import states
from src.visualization.comparisons import near_tie
from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService
from src.visualization.protocols import ordered_trees

OUTPUT = Path("src/analysis/four_link_family_results.json")
logging.getLogger().setLevel(logging.ERROR)
DIST = {"none": {}, "root": {"0:3": 1},
        "all_leaves": {f"{i}:{i}": 1 for i in range(4)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("phase_error", "bit_error"))
    parser.add_argument("--fidelity", type=float)
    parser.add_argument("--horizon", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    service = EvaluationService(cache_size=0)
    results = []
    for family, fidelity in product(("phase_error", "bit_error"), (.95, .98)):
        if args.family and family != args.family:
            continue
        if args.fidelity is not None and fidelity != args.fidelity:
            continue
        initial = states(fidelity, family)
        base = deepcopy(DEFAULT)
        base["links"] = 4
        base["hardware"].update(p_gen=.2, p_swap=.9)
        base["memory"].update(t1_slots=80, t2_slots=160)
        base["numerics"].update(t_trunc=args.horizon, coverage_target=.999)
        rows = []
        for tree, cutoff, dist in product(
                ordered_trees(4), (None, 1, 2, 4, 8, 16), DIST):
            candidate = {"tree": tree, "cutoff": cutoff, "distillation": dist}
            metrics = {}
            for model in ("physical", "werner"):
                c = deepcopy(base)
                c["initial_state"] = deepcopy(initial[model])
                c["protocol"].update(tree=tree, memory_cutoff_slots=cutoff,
                                     distillation_rounds=DIST[dist])
                try:
                    s = service.evaluate(c).summary
                    metrics[model] = {k: s[k] for k in (
                        "status", "coverage", "rate_bits_per_slot", "average_fidelity",
                        "eX", "eZ", "mean_wait_repository_slots")}
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    metrics[model] = {"status": "failed", "error": str(exc)}
            rows.append({"candidate": candidate, "metrics": metrics})
        covered = [r for r in rows if all(s["status"] == "covered"
                   for s in r["metrics"].values())]
        if covered:
            score = lambda r, m: r["metrics"][m]["rate_bits_per_slot"]
            best = {m: max(score(r, m) for r in covered)
                    for m in ("physical", "werner")}
            winners = {m: [r for r in covered if near_tie(score(r, m), best[m])]
                       for m in best}
            selection = {"best_rates": best,
                         "winners": {m: [r["candidate"] for r in v]
                                     for m, v in winners.items()},
                         "physical_rate_of_werner_winners": [score(r, "physical")
                                                             for r in winners["werner"]],
                         "regret_range": [best["physical"]-max(
                             score(r, "physical") for r in winners["werner"]),
                             best["physical"]-min(
                             score(r, "physical") for r in winners["werner"])]}
        else:
            selection = None
        result = {"case": {"family": family, "fidelity": fidelity,
                           "links": 4, "p_gen": .2, "p_swap": .9,
                           "t1_slots": 80, "t2_slots": 160,
                           "horizon": args.horizon},
                  "n_candidates": len(rows), "n_covered": len(covered),
                  "selection": selection, "rows": rows}
        results.append(result)
        args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
