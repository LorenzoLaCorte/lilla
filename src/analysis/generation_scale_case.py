"""Check the central state-family policy effect at a lower link-success rate.

Run after activating venv: python -m src.analysis.generation_scale_case
"""
from copy import deepcopy
from itertools import product
import json
import logging
from pathlib import Path

from src.analysis.distillation_family_scan import DIST
from src.visualization.comparisons import near_tie
from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService

OUTPUT = Path("src/analysis/generation_scale_results.json")
logging.getLogger().setLevel(logging.ERROR)


def main():
    service = EvaluationService(cache_size=0)
    base = deepcopy(DEFAULT)
    base["links"] = 2
    base["hardware"].update(p_gen=.02, p_swap=.9)
    base["memory"].update(t1_slots=300, t2_slots=600)
    base["numerics"].update(t_trunc=32768, coverage_target=.999)
    base["protocol"]["tree"] = [0, 1]
    initial = {"physical": {"kind": "bell_weights", "values": [.95, 0, 0, .05]},
               "werner": {"kind": "werner", "fidelity": .95}}
    rows = []
    for cutoff, dist in product((None, 10, 20, 40, 80, 160, 320), DIST):
        candidate = {"cutoff": cutoff, "distillation": dist}
        metrics = {}
        for model, state in initial.items():
            c = deepcopy(base)
            c["initial_state"] = state
            c["protocol"].update(memory_cutoff_slots=cutoff,
                                 distillation_rounds=DIST[dist])
            try:
                s = service.evaluate(c).summary
                metrics[model] = {k: s[k] for k in (
                    "status", "coverage", "rate_bits_per_slot",
                    "eX", "eZ", "mean_wait_repository_slots")}
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                metrics[model] = {"status": "failed", "error": str(exc)}
        rows.append({"candidate": candidate, "metrics": metrics})
    covered = [r for r in rows if all(s["status"] == "covered"
               for s in r["metrics"].values())]
    score = lambda r, m: r["metrics"][m]["rate_bits_per_slot"]
    tops = {m: max(score(r, m) for r in covered) for m in initial}
    winners = {m: [r for r in covered if near_tie(score(r, m), tops[m])]
               for m in initial}
    result = {"case": {"links": 2, "p_gen": .02, "p_swap": .9,
                       "fidelity": .95, "t1_slots": 300,
                       "t2_slots": 600, "horizon": 32768},
              "n_candidates": len(rows), "n_covered": len(covered),
              "best_rates": tops,
              "winners": {m: [r["candidate"] for r in v] for m, v in winners.items()},
              "physical_rate_of_werner_winners": [score(r, "physical")
                                                  for r in winners["werner"]],
              "regret_range": [tops["physical"]-max(score(r, "physical")
                            for r in winners["werner"]),
                               tops["physical"]-min(score(r, "physical")
                            for r in winners["werner"])],
              "rows": rows}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     indent=2))


if __name__ == "__main__":
    main()
