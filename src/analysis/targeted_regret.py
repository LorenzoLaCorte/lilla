"""Targeted, fully enumerated physical-vs-Pauli-twirled protocol comparisons.

Run after activating venv: python -m src.analysis.targeted_regret
The JSON file is checkpointed after every scenario. A scenario's candidate set
is part of the output and selection is restricted to candidates covered under
both models at the stated horizon.
"""
import argparse
from copy import deepcopy
from itertools import product
import json
import logging
from pathlib import Path

from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService
from src.visualization.protocols import ordered_trees
from src.visualization.comparisons import near_tie

OUTPUT = Path("src/analysis/targeted_regret_results.json")
logging.getLogger().setLevel(logging.ERROR)


def cases():
    for t1 in (8, 10, 12, 14, 16, 20, 30):
        yield {"kind": "cutoff_refinement", "links": 3, "p_gen": .2,
               "fidelity": .99, "t1": t1, "t2": 2*t1, "p_swap": .9,
               "cutoffs": (None, 1, 2, 3, 4, 6, 8, 12, 16, 32),
               "distillation": ("none",), "outcomes": ("all_corrected",),
               "horizon": 1024}
    for p in ((.05, .2, .2), (.2, .05, .2), (.2, .2, .05)):
        for t1 in (15, 30):
            yield {"kind": "heterogeneous_generation", "links": 3, "p_gen": p,
                   "fidelity": .99, "t1": t1, "t2": 2*t1, "p_swap": .9,
                   "cutoffs": (None, 1, 2, 4, 8, 16, 32),
                   "distillation": ("none",),
                   "outcomes": ("all_corrected", "phi_pair", "psi_pair"),
                   "horizon": 2048}
    for fidelity, t1 in product((.95, .97, .99), (15, 30, 80)):
        yield {"kind": "distillation_placement", "links": 2, "p_gen": .2,
               "fidelity": fidelity, "t1": t1, "t2": 2*t1, "p_swap": .9,
               "cutoffs": (None, 1, 2, 4, 8, 16),
               "distillation": ("none", "left", "right", "both", "root"),
               "outcomes": ("all_corrected",), "horizon": 2048}
    # Non-Bell-diagonal R-state family: F|Phi+><Phi+|+(1-F)|01><01|.
    for fidelity, t1 in product((.90, .95, .98), (10, 25, 60)):
        yield {"kind": "r_state_family", "links": 3, "p_gen": .2,
               "fidelity": fidelity, "t1": t1, "t2": 2*t1, "p_swap": .9,
               "cutoffs": (None, 1, 2, 4, 8, 16, 32),
               "distillation": ("none",),
               "outcomes": ("all_corrected", "phi_pair", "psi_pair"),
               "horizon": 1024}


def candidate_configs(case):
    base = deepcopy(DEFAULT)
    base["name"] = case["kind"]
    base["links"] = case["links"]
    base["hardware"].update(p_gen=case["p_gen"], p_swap=case["p_swap"])
    if case["kind"] == "r_state_family":
        f = case["fidelity"]
        base["initial_state"] = {"kind": "real_x",
                                 "values": [1, f-1, 1-f, 2*f-1, f, -f]}
    else:
        base["initial_state"]["fidelity"] = case["fidelity"]
    base["memory"].update(t1_slots=case["t1"], t2_slots=case["t2"])
    base["numerics"].update(t_trunc=case["horizon"], coverage_target=.999)
    for tree, cutoff, dist, outcome in product(
            ordered_trees(case["links"]), case["cutoffs"],
            case["distillation"], case["outcomes"]):
        c = deepcopy(base)
        rounds = {"left": {"0:0": 1}, "right": {"1:1": 1},
                  "both": {"0:0": 1, "1:1": 1},
                  "root": {"0:1": 1}, "none": {}}[dist]
        c["protocol"].update(tree=tree, memory_cutoff_slots=cutoff,
                             distillation_rounds=rounds, bell_outcomes=outcome)
        yield c, {"tree": tree, "cutoff": cutoff, "distillation": dist,
                  "outcomes": outcome}


def evaluate_case(service, case):
    rows = []
    for c, candidate in candidate_configs(case):
        predictions = {}
        for model in ("full_real_x", "local_channel_twirl"):
            c["prediction_model"] = model
            try:
                s = service.evaluate(c).summary
                predictions[model] = {k: s[k] for k in (
                    "status", "coverage", "rate_bits_per_slot", "average_fidelity",
                    "eX", "eZ", "mean_wait_repository_slots")}
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                predictions[model] = {"status": "failed", "error": str(exc)}
        rows.append({"candidate": candidate, "predictions": predictions})
    covered = [r for r in rows if all(r["predictions"][m]["status"] == "covered"
               for m in ("full_real_x", "local_channel_twirl"))]
    scores = lambda model: (lambda r: r["predictions"][model]["rate_bits_per_slot"])
    full, twirl = scores("full_real_x"), scores("local_channel_twirl")
    if covered:
        full_top, twirl_top = max(map(full, covered)), max(map(twirl, covered))
        full_winners = [r for r in covered if near_tie(full(r), full_top)]
        twirl_winners = [r for r in covered if near_tie(twirl(r), twirl_top)]
        selection = {
            "best_full_rate": full_top, "best_twirled_prediction": twirl_top,
            "full_winners": [r["candidate"] for r in full_winners],
            "twirl_winners": [r["candidate"] for r in twirl_winners],
            "physical_rate_of_twirled_winners": [full(r) for r in twirl_winners],
            "regret_range": [full_top-max(map(full, twirl_winners)),
                             full_top-min(map(full, twirl_winners))],
        }
    else:
        selection = None
    return {"case": case, "n_candidates": len(rows), "n_covered": len(covered),
            "selection": selection, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    service = EvaluationService(cache_size=0)
    results = []
    for case in cases():
        if args.kind is not None and case["kind"] != args.kind:
            continue
        result = evaluate_case(service, case)
        results.append(result)
        args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")
        s = result["selection"]
        print(json.dumps({"case": {k: v for k, v in case.items() if k not in
                                    ("cutoffs", "distillation", "outcomes")},
                          "covered": result["n_covered"],
                          "full_rate": s["best_full_rate"] if s else None,
                          "regret": s["regret_range"] if s else None,
                          "full_winners": s["full_winners"] if s else None,
                          "twirl_winners": s["twirl_winners"] if s else None}),
              flush=True)


if __name__ == "__main__":
    main()
