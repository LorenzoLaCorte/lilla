"""Compare efficient real-X join with direct time-pair join.

Run after activating venv: python -m src.analysis.backend_benchmark
Timing is illustrative; correctness is the purpose of this check.
"""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.core.pauli_adc.state import RealXState
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.visualization.config import DEFAULT, expand_config
from src.visualization.metrics import summarize, validate_outputs

OUTPUT = Path("src/analysis/backend_benchmark_results.json")


def run(horizon, efficient):
    c = deepcopy(DEFAULT)
    c["links"] = 2
    c["hardware"].update(p_gen=.2, p_swap=.9)
    c["initial_state"]["fidelity"] = .98
    c["memory"].update(t1_slots=30, t2_slots=60)
    c["protocol"].update(tree=[0, 1], memory_cutoff_slots=4,
                         distillation_rounds={"0:1": 1})
    c["numerics"].update(t_trunc=horizon, coverage_target=.999)
    _, p, _ = expand_config(c)
    evaluator = RepeaterChainEvaluation(state_type=RealXState,
                                        efficient=efficient, use_fft=True,
                                        use_gpu=False, w_twirling=False)
    start = perf_counter()
    pmf, x = evaluator.asymmetric_heterogeneous_protocol(p, c["links"])
    elapsed = perf_counter()-start
    pmf, x, _ = validate_outputs(pmf, x, horizon)
    return pmf, x, elapsed, summarize(pmf, x, .999)[0]


def main():
    rows = []
    for horizon in (128, 256, 512, 1024):
        direct = run(horizon, False)
        fast = run(horizon, True)
        p0, x0, t0, s0 = direct
        p1, x1, t1, s1 = fast
        mask = (p0 > 1e-9) & (p1 > 1e-9)
        row = {"horizon": horizon, "direct_seconds": t0,
               "efficient_seconds": t1, "speedup": t0/t1,
               "max_pmf_error": float(np.max(abs(p0-p1))),
               "max_populated_coordinate_error": float(np.max(abs(x0[mask]-x1[mask])))
               if np.any(mask) else None,
               "direct_rate": s0["rate_bits_per_slot"],
               "efficient_rate": s1["rate_bits_per_slot"],
               "direct_coverage": s0["coverage"],
               "efficient_coverage": s1["coverage"]}
        rows.append(row)
        OUTPUT.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
