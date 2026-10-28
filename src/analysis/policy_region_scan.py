"""Grid audit of fidelity-matched Werner policy loss for two-link chains.

Run after activating venv: python -m src.analysis.policy_region_scan
Each of 100 scenarios enumerates the same 35 cutoffs/recurrence policies. The
output is checkpointed after every case and includes all candidate rates.
"""
from copy import deepcopy
from itertools import product
import json
import logging
from pathlib import Path

from src.analysis.distillation_family_scan import DIST
from src.analysis.state_family_scan import states
from src.visualization.comparisons import near_tie
from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService

OUTPUT = Path("src/analysis/policy_region_results.json")
logging.getLogger().setLevel(logging.ERROR)
CUTOFFS = (None, 1, 2, 4, 8, 16, 32)


def main():
    service = EvaluationService(cache_size=0)
    results = []
    for family, fidelity, t1, ratio in product(
            ("phase_error", "bit_error"), (.91, .93, .95, .97, .99),
            (15, 25, 40, 60, 100), (1, 2)):
        base = deepcopy(DEFAULT)
        base["links"] = 2
        base["hardware"].update(p_gen=.2, p_swap=.9)
        base["memory"].update(t1_slots=t1, t2_slots=t1*ratio)
        base["numerics"].update(t_trunc=2048, coverage_target=.999)
        base["protocol"]["tree"] = [0, 1]
        initial = states(fidelity, family)
        rows = []
        for cutoff, dist in product(CUTOFFS, DIST):
            candidate = {"cutoff": cutoff, "distillation": dist}
            outputs = {}
            for model in ("physical", "werner"):
                c = deepcopy(base)
                c["initial_state"] = initial[model]
                c["protocol"].update(memory_cutoff_slots=cutoff,
                                     distillation_rounds=DIST[dist])
                try:
                    summary = service.evaluate(c).summary
                    outputs[model] = {"status": summary["status"],
                                      "coverage": summary["coverage"],
                                      "rate": summary["rate_bits_per_slot"]}
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    outputs[model] = {"status": "failed", "error": str(exc)}
            rows.append({"candidate": candidate, "outputs": outputs})
        covered = [r for r in rows if all(s["status"] == "covered"
                   for s in r["outputs"].values())]
        if covered:
            physical = lambda r: r["outputs"]["physical"]["rate"]
            werner = lambda r: r["outputs"]["werner"]["rate"]
            top_full, top_werner = max(map(physical, covered)), max(map(werner, covered))
            full_winners = [r for r in covered if near_tie(physical(r), top_full)]
            werner_winners = [r for r in covered if near_tie(werner(r), top_werner)]
            losses = [top_full-physical(r) for r in werner_winners]
            selection = {"best_physical_rate": top_full,
                         "best_werner_prediction": top_werner,
                         "physical_winners": [r["candidate"] for r in full_winners],
                         "werner_winners": [r["candidate"] for r in werner_winners],
                         "regret_range": [min(losses), max(losses)],
                         "relative_regret_range": [min(losses)/top_full,
                                                   max(losses)/top_full]
                         if top_full > 0 else None}
        else:
            selection = None
        result = {"case": {"family": family, "fidelity": fidelity,
                           "t1_slots": t1, "t2_slots": t1*ratio,
                           "p_gen": .2, "p_swap": .9, "links": 2,
                           "horizon": 2048},
                  "n_candidates": len(rows), "n_covered": len(covered),
                  "selection": selection, "rows": rows}
        results.append(result)
        OUTPUT.write_text(json.dumps({"results": results}, separators=(",", ":")) + "\n")
        print(json.dumps({"case": result["case"], "covered": len(covered),
                          "physical_rate": selection["best_physical_rate"] if selection else None,
                          "relative_regret": selection["relative_regret_range"] if selection else None}),
              flush=True)


if __name__ == "__main__":
    main()
