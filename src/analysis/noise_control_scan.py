"""Separate state-family policy loss from memory-noise effects.

Run after activating venv: python -m src.analysis.noise_control_scan
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

OUTPUT = Path("src/analysis/noise_control_results.json")
logging.getLogger().setLevel(logging.ERROR)


def main():
    service = EvaluationService(cache_size=0)
    results = []
    memory_cases = {
        "no_memory_noise": (None, None),
        "pure_dephasing": (None, 60),
        "pure_relaxation": (30, 60),
    }
    initial_cases = {
        "physical_phase_error": {"kind": "bell_weights", "values": [.95, 0, 0, .05]},
        "werner_proxy": {"kind": "werner", "fidelity": .95},
    }
    for label, (t1, t2) in memory_cases.items():
        base = deepcopy(DEFAULT)
        base["links"] = 2
        base["hardware"].update(p_gen=.2, p_swap=.9)
        base["memory"].update(t1_slots=t1, t2_slots=t2)
        base["numerics"].update(t_trunc=2048, coverage_target=.999)
        base["protocol"]["tree"] = [0, 1]
        rows = []
        for cutoff, dist in product((None, 1, 2, 4, 8, 16, 32), DIST):
            candidate = {"cutoff": cutoff, "distillation": dist}
            metrics = {}
            for model, initial in initial_cases.items():
                c = deepcopy(base)
                c["initial_state"] = initial
                c["protocol"].update(memory_cutoff_slots=cutoff,
                                     distillation_rounds=DIST[dist])
                try:
                    s = service.evaluate(c).summary
                    metrics[model] = {k: s[k] for k in (
                        "status", "coverage", "rate_bits_per_slot", "eX", "eZ",
                        "mean_wait_repository_slots")}
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    metrics[model] = {"status": "failed", "error": str(exc)}
            rows.append({"candidate": candidate, "metrics": metrics})
        covered = [r for r in rows if all(s["status"] == "covered"
                   for s in r["metrics"].values())]
        score = lambda r, m: r["metrics"][m]["rate_bits_per_slot"]
        full = max(score(r, "physical_phase_error") for r in covered)
        proxy = max(score(r, "werner_proxy") for r in covered)
        full_winners = [r for r in covered if near_tie(score(r, "physical_phase_error"), full)]
        proxy_winners = [r for r in covered if near_tie(score(r, "werner_proxy"), proxy)]
        result = {"memory": label, "t1_slots": t1, "t2_slots": t2,
                  "n_candidates": len(rows), "n_covered": len(covered),
                  "best_physical_rate": full, "best_proxy_prediction": proxy,
                  "physical_winners": [r["candidate"] for r in full_winners],
                  "proxy_winners": [r["candidate"] for r in proxy_winners],
                  "physical_rate_of_proxy_winners": [score(r, "physical_phase_error")
                                                     for r in proxy_winners],
                  "regret_range": [full-max(score(r, "physical_phase_error")
                                            for r in proxy_winners),
                                   full-min(score(r, "physical_phase_error")
                                            for r in proxy_winners)],
                  "rows": rows}
        results.append(result)
        OUTPUT.write_text(json.dumps({"results": results}, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
