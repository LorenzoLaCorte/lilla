"""Matched-channel T1 key-threshold comparison for a three-link chain.

Run after activating venv: python -m src.analysis.key_threshold
Roots refer to the stated finite policy set and asymptotic pooled BB84 metric.
"""
from copy import deepcopy
import json
from math import log2
from pathlib import Path

from scipy.optimize import brentq

from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService

OUTPUT = Path("src/analysis/key_threshold_results.json")
CUTOFFS = (None, 1, 2, 3, 4, 6, 8, 12, 16)


def entropy(p):
    return -p*log2(p)-(1-p)*log2(1-p) if 0 < p < 1 else 0.0


def main():
    service = EvaluationService(cache_size=0)
    def evaluate(t1, cutoff, model):
        c = deepcopy(DEFAULT)
        c["links"] = 3
        c["hardware"].update(p_gen=.2, p_swap=.9)
        c["initial_state"]["fidelity"] = .99
        c["memory"].update(t1_slots=t1, t2_slots=2*t1)
        c["protocol"].update(tree=[[0, 1], 2], memory_cutoff_slots=cutoff)
        c["prediction_model"] = model
        c["numerics"].update(t_trunc=1024, coverage_target=.999)
        s = service.evaluate(c).summary
        return {"signed_fraction": 1-entropy(s["eX"])-entropy(s["eZ"]),
                "rate_bits_per_slot": s["rate_bits_per_slot"],
                "coverage": s["coverage"], "mean_wait": s["mean_wait_repository_slots"]}

    roots = {}
    for model in ("full_real_x", "local_channel_twirl"):
        roots[model] = brentq(lambda t: evaluate(t, 1, model)["signed_fraction"],
                             7.5, 9, xtol=1e-7)
    probes = []
    for t1 in (7.5, 8.0, roots["full_real_x"], 8.25,
               roots["local_channel_twirl"], 8.5, 9.0):
        rows = [{"cutoff": cutoff, "model": model, **evaluate(t1, cutoff, model)}
                for cutoff in CUTOFFS
                for model in ("full_real_x", "local_channel_twirl")]
        probes.append({"t1_slots": t1, "rows": rows})
    result = {"scenario": {"links": 3, "p_gen": .2, "p_swap": .9,
                           "initial_fidelity": .99, "t2_equals_2t1": True,
                           "tree": [[0, 1], 2], "horizon": 1024,
                           "cutoffs": CUTOFFS, "outcomes": "all_corrected",
                           "distillation": "none"},
              "cutoff_one_thresholds": roots,
              "threshold_gap_slots": roots["local_channel_twirl"]-
                                     roots["full_real_x"],
              "probes": probes}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "probes"},
                     indent=2))
    for point in probes:
        print(point["t1_slots"], [max(
            (r for r in point["rows"] if r["model"] == model),
            key=lambda r: r["rate_bits_per_slot"])["rate_bits_per_slot"]
            for model in ("full_real_x", "local_channel_twirl")])


if __name__ == "__main__":
    main()
