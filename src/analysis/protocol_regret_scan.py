"""Reproducible finite-candidate comparison of physical and channel-twirled policies.

Run from the repository root after activating venv::

    python -m src.analysis.protocol_regret_scan --output src/analysis/protocol_regret_results.json

The scan is exploratory. All entries retain coverage and candidate rates so a
reported model-selection loss can be audited without relying on optimizer luck.
"""
import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService
from src.visualization.protocols import ordered_trees


def run_scan(output):
    service = EvaluationService(cache_size=0)
    scenarios = []
    # T2=2*T1 isolates pure amplitude damping; T2=T1 includes equal-order
    # transverse dephasing. All times are relative to a generation slot.
    for p_gen, fidelity, t1, t2_ratio in product(
            (0.08, 0.2), (0.95, 0.99), (8.0, 20.0, 50.0), (1.0, 2.0)):
        base = deepcopy(DEFAULT)
        base["links"] = 3
        base["name"] = "matched Pauli-channel twirl comparison"
        base["hardware"].update(p_gen=p_gen, p_swap=0.9)
        base["initial_state"]["fidelity"] = fidelity
        base["memory"].update(t1_slots=t1, t2_slots=t1*t2_ratio)
        base["numerics"].update(t_trunc=1024, coverage_target=0.999)
        rows = []
        for tree, cutoff, outcomes in product(
                ordered_trees(3), (None, 1, 2, 4, 8, 16, 32),
                ("all_corrected", "phi_pair", "psi_pair")):
            candidate = {"tree": tree, "cutoff": cutoff, "outcomes": outcomes}
            predictions = {}
            for model in ("full_real_x", "local_channel_twirl"):
                c = deepcopy(base)
                c["prediction_model"] = model
                c["protocol"].update(tree=tree, memory_cutoff_slots=cutoff,
                                     bell_outcomes=outcomes)
                try:
                    s = service.evaluate(c).summary
                    predictions[model] = {k: s[k] for k in (
                        "status", "coverage", "rate_bits_per_slot", "average_fidelity",
                        "eX", "eZ", "mean_wait_repository_slots")}
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    predictions[model] = {"status": "failed", "error": str(exc)}
            rows.append({"candidate": candidate, "predictions": predictions})
        covered = [r for r in rows if all(
            r["predictions"][m]["status"] == "covered"
            for m in ("full_real_x", "local_channel_twirl"))]
        physical = lambda r: r["predictions"]["full_real_x"]["rate_bits_per_slot"]
        approx = lambda r: r["predictions"]["local_channel_twirl"]["rate_bits_per_slot"]
        best_full = max(covered, key=physical) if covered else None
        best_twirl = max(covered, key=approx) if covered else None
        # Ties within the comparison module's numerical tolerance are retained.
        def ties(metric, best):
            if best is None:
                return []
            top = metric(best)
            return [r for r in covered if abs(metric(r)-top) <=
                    1e-12 + 1e-3*max(abs(metric(r)), abs(top))]
        full_ties = ties(physical, best_full)
        twirl_ties = ties(approx, best_twirl)
        summary = {
            "n_candidates": len(rows), "n_covered": len(covered),
            "best_physical": best_full["candidate"] if best_full else None,
            "best_twirled": best_twirl["candidate"] if best_twirl else None,
            "best_physical_rate": physical(best_full) if best_full else None,
            "selected_physical_rate": physical(best_twirl) if best_twirl else None,
            "absolute_regret": (physical(best_full)-physical(best_twirl)
                                if best_full and best_twirl else None),
            "full_ties": [r["candidate"] for r in full_ties],
            "twirl_ties": [r["candidate"] for r in twirl_ties],
            "worst_tie_regret": (max(physical(a)-physical(b)
                                     for a in full_ties for b in twirl_ties)
                                 if full_ties and twirl_ties else None),
        }
        scenario = {"parameters": {"p_gen": p_gen, "fidelity": fidelity,
                                   "t1_slots": t1, "t2_slots": t1*t2_ratio,
                                   "p_swap": 0.9, "links": 3, "horizon": 1024},
                    "summary": summary, "rows": rows}
        scenarios.append(scenario)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"scenarios": scenarios}, indent=2) + "\n")
        print(json.dumps({"parameters": scenario["parameters"], **summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("src/analysis/protocol_regret_results.json"))
    args = parser.parse_args()
    run_scan(args.output)
