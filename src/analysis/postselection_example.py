"""Positive-key filtering example: both inputs are already relaxed Bell pairs.

Run: python -m src.analysis.postselection_example.
These are prescribed generated states, with no further waiting noise. Rates
include swap rejection/retries, but do not model the prior state-preparation cost.
"""
from copy import deepcopy

import numpy as np

from src.analysis.postselection import POLICIES, evaluate, save_plot
from src.core.pauli_adc.protocol_units import apply_local_channels, sequential_pauli_adc_channel
from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService


def prepared_config(eta, policy="all_corrected"):
    # Apply equal local AD to Phi+: x=(1, 1-eta, 1-eta, eta²+(1-eta)², eta, -eta).
    channel = sequential_pauli_adc_channel([1, 1, 1, 1], eta)
    x = apply_local_channels([1, 0, 0, 1, 1, -1], channel)
    config = deepcopy(DEFAULT)
    config.update(name="Two already-relaxed Bell pairs", links=2)
    config["initial_state"] = {"kind": "real_x", "values": x.tolist()}
    config["memory"] = {"kind": "t1_t2", "t1_slots": None, "t2_slots": None}
    config["hardware"].update(p_gen=1.0, p_swap=1.0)
    config["protocol"].update(tree=[0, 1], bell_outcomes=policy)
    config["numerics"]["t_trunc"] = 64
    return config


def main():
    service, rows = EvaluationService(), []
    for eta in np.linspace(0.905, 0.925, 81):
        for policy in POLICIES:
            rows.append({"eta": float(eta), "policy": policy,
                         **evaluate(service, prepared_config(float(eta), policy))})
    example = prepared_config(0.9145, "psi_pair")
    output = save_plot(rows, example, "eta",
                       "Two already-relaxed Bell pairs · deterministic generation/swap hardware\n"
                       "No further waiting noise; prior preparation time excluded", "prepared-pair-example")
    print("At eta=0.9145:")
    for policy in POLICIES:
        summary = evaluate(service, prepared_config(0.9145, policy))
        print(f"  {policy}: score={summary['bb84_score']:.8f}, "
              f"SKR={summary['rate_bits_per_slot']:.8f}, "
              f"mean={summary['mean_wait_repository_slots']:.6f}")
    print(f"Import {output/'config.json'} into the app to reproduce the positive Psi case.")


if __name__ == "__main__":
    main()
