"""Small converged baseline checks before the next optimization batch.

Run from the activated repository environment:
python -m src.analysis.metropolitan_reference
No Bayesian searches are launched. Results include full convergence histories.
"""
from dataclasses import asdict
import json
import logging
from pathlib import Path

import numpy as np

from src.optimization.pauli_ad import Evaluation, Hardware, Numerics, candidate, fixed_protocol, write_json
from src.scripts.pauli_ad_hardware import hardware, metropolitan_link

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/"results/pauliAD/metropolitan-reference-20261029"


def main():
    logging.disable(logging.WARNING)  # Intermediate horizon warnings are expected; histories remain saved.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    numerics = Numerics(2048, 524288)

    def evaluate(nodes, scale, p, slot, visibility, rounds, model, label):
        key = (nodes, scale, p, slot, visibility, rounds, model)
        if key in cache:
            return cache[key]
        t1, t2 = 2.4*scale/slot, 1.71*scale/slot
        mode_rates, ad = ((1/t2-.5/t1, 1/t2-.5/t1, 0.), 1/t1) if model == "ad" else ((1/t2, 1/t2, 1/t1), 0.)
        h = Hardware(nodes, (p,)*(nodes-1), .85, 1., (mode_rates,)*nodes, (ad,)*nodes, visibility)
        choice = candidate([*fixed_protocol(nodes-1), *[f"d{nodes-2}"]*rounds], .5, ["all_corrected"]*(nodes-2))
        result = Evaluation(h, numerics).run(choice)
        row = dict(label=label, nodes=nodes, memory_scale=scale, p_gen=p, slot_seconds=slot,
                   visibility=visibility, rounds=rounds, model=model, hardware=asdict(h),
                   candidate=choice, result=result)
        rows.append(row)
        write_json(OUTPUT/"evaluations.json", rows)
        if result["status"] != "converged":
            raise RuntimeError(f"pilot candidate did not converge: {row}")
        cache[key] = row
        return row

    # Estimate sufficient thresholds for these particular fixed protocols,
    # rather than calling them optima over all possible protocols.
    brackets = []
    for nodes, rounds, low, high in ((5, 0, .002, .02), (5, 2, .002, .02), (9, 2, .005, .04)):
        def margin(p):
            return evaluate(nodes, 1., p, .00025, .96, rounds, "ad", "old-slot-threshold")["result"]["bb84_margin"]
        assert margin(low) < 0 < margin(high)
        for _ in range(8):
            middle = np.sqrt(low*high)
            if margin(middle) > 0:
                high = middle
            else:
                low = middle
        brackets.append(dict(nodes=nodes, rounds=rounds, p_nonpositive=float(low), p_positive=float(high),
                             generation_rate_positive_hz=float(high/.00025)))
    baselines = []
    for regime in ("metro97", "metro99"):
        for nodes in (5, 9):
            for scale in (.5, 1., 2.):
                h = hardware(regime, "network", nodes, scale)
                for model in ("ad", "unital"):
                    for rounds in (0, 2):
                        r = evaluate(nodes, scale, h["p_gen"][0], h["slot_seconds"], h["visibility"], rounds, model, regime)
                        baselines.append(dict(regime=regime, nodes=nodes, memory_scale=scale, model=model,
                                              rounds=rounds, p_gen=h["p_gen"][0], slot_seconds=h["slot_seconds"],
                                              skf=r["result"]["skf"], skr_bits_per_second=r["result"]["skr"]/h["slot_seconds"],
                                              margin=r["result"]["bb84_margin"],
                                              wait_seconds=r["result"]["mean_wait"]*h["slot_seconds"]))
    def h2(x):
        return -x*np.log2(x)-(1-x)*np.log2(1-x)
    v = .96**8
    noiseless_margin = float(1-2*h2((1-v)/2))
    summary = dict(link=metropolitan_link(), old_slot_threshold_brackets=brackets,
                   eight_link_F097_swaps_only_noiseless_margin=noiseless_margin,
                   metropolitan_baselines=baselines, numerics=asdict(numerics),
                   scope="Fixed balanced Bell-source, all-corrected anchors; not optimization or a device calibration")
    write_json(OUTPUT/"summary.json", summary)
    import csv
    with (OUTPUT/"baselines.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baselines[0]))
        writer.writeheader()
        writer.writerows(baselines)
    print(json.dumps(dict(link=summary["link"], thresholds=brackets,
                         noiseless_eight_link_margin=noiseless_margin,
                         positive_baselines=sum(r["skf"] > 0 for r in baselines),
                         baseline_count=len(baselines)), indent=2))


if __name__ == "__main__":
    main()
