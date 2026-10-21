"""Controlled comparisons of a bounded, manually chosen candidate set."""
from hashlib import sha256

from src.core.pauli_adc.state import bell_weights_to_coordinates
from src.visualization.config import MODELS, canonical, converted, expand_config


def scenario_key(config):
    c, p, _ = expand_config(converted(config, model="full_real_x"))
    initial = p.get("real_x_coordinates")
    if initial is None:
        initial = [bell_weights_to_coordinates(row).tolist() for row in p["lambdas"]]
    return canonical({"links": c["links"], "initial": initial, "p_gen": p["p_gen"],
                      "p_swap": p["p_swap"], "p_distillation": p["p_distillation"],
                      "pauli": p["pauli_mode_decay_rates"], "adc": p["amplitude_damping_rate"],
                      "slot_duration_seconds": c["slot_duration_seconds"]})


def candidate_key(config):
    c, _, ops = expand_config(config)
    # Explicit zero rounds and shared-vs-override spellings describe the same candidate.
    return sha256(canonical({"tree": c["protocol"]["tree"],
                             "operations": [(o["id"], o["cutoff_slots"]) for o in ops],
                             "outcomes": c["protocol"]["bell_outcomes"]}).encode()).hexdigest()


def comparison_plan(configs):
    if not configs or len(configs) > 12:
        raise ValueError("select between 1 and 12 saved runs")
    if len({scenario_key(c) for c in configs}) != 1:
        raise ValueError("controlled comparisons require the same physical scenario and units")
    if len({c["numerics"]["t_trunc"] for c in configs}) != 1:
        raise ValueError("controlled comparisons require identical horizons")
    unique = {candidate_key(c): c for c in configs}
    return [converted(c, model=model) for c in unique.values() for model in MODELS]


def near_tie(a, b):
    return abs(a-b) <= 1e-12 + 1e-3*max(abs(a), abs(b))


def selection_loss(runs, expected_candidates=None):
    if not runs:
        raise ValueError("candidate comparison matrix is incomplete")
    if len({scenario_key(r.config) for r in runs}) != 1:
        raise ValueError("controlled comparisons require the same physical scenario and units")
    if len({r.metadata["horizon"] for r in runs}) != 1:
        raise ValueError("controlled comparisons require identical horizons")
    comparison_plan(list({candidate_key(r.config): r.config for r in runs}.values()))
    if len({r.metadata["source_fingerprint"] for r in runs}) != 1:
        raise ValueError("selection loss requires a common code fingerprint")
    matrix = {}
    for run in runs:
        key = candidate_key(run.config)
        model = run.config["prediction_model"]
        if model in matrix.setdefault(key, {}):
            raise ValueError("duplicate candidate/model row")
        matrix[key][model] = run
    if expected_candidates is not None and set(matrix) != set(expected_candidates):
        raise ValueError("candidate comparison matrix is incomplete")
    if any(set(row) != set(MODELS) for row in matrix.values()):
        raise ValueError("candidate comparison matrix is incomplete")
    full = {k: row["full_real_x"].summary["rate_bits_per_slot"] for k, row in matrix.items()}
    twirl = {k: row["local_channel_twirl"].summary["rate_bits_per_slot"] for k, row in matrix.items()}
    best_full, best_twirl = max(full.values()), max(twirl.values())
    winners_full = [k for k, v in full.items() if near_tie(v, best_full)]
    winners_twirl = [k for k, v in twirl.items() if near_tie(v, best_twirl)]
    selected = [full[k] for k in winners_twirl]
    # Include all pairs of near-tied full and twirled winners; positive loss only.
    losses = [max(0, full[a]-full[b]) for a in winners_full for b in winners_twirl]
    return {"candidates": len(matrix), "full_winners": winners_full,
            "twirl_winners": winners_twirl, "near_ties": len(winners_full)>1 or len(winners_twirl)>1,
            "full_winner_rate_range": [min(full[k] for k in winners_full), best_full],
            "prediction_errors_bits_per_slot": {k: twirl[k]-full[k] for k in matrix},
            "selected_full_rate_range": [min(selected), max(selected)],
            "absolute_selection_loss_range": [min(losses), max(losses)],
            "relative_selection_loss_range": ([min(losses)/best_full, max(losses)/best_full]
                                              if best_full > 0 else None),
            "provisional": any(r.metadata["diagnostics"]["provisional"] for r in runs),
            "tie_tolerance": "1e-12 + 1e-3*max(abs(a),abs(b)) bits/slot; ranking ambiguity"}


def comparison_rows(runs):
    rows = []
    for r in runs:
        p = r.config["protocol"]
        rows.append({"run": r.metadata["run_id"][:8], "name": r.config["name"],
                     "candidate": candidate_key(r.config)[:8], "tree": canonical(p["tree"]),
                     "distillation": canonical(p["distillation_rounds"]),
                     "cutoffs": canonical({o["id"]: o["cutoff_slots"] for o in r.metadata["operations"]}),
                     "outcomes": p["bell_outcomes"], "model": r.config["prediction_model"],
                     "rate_bits_per_slot": r.summary["rate_bits_per_slot"],
                     "mean_wait_slots": r.summary["mean_wait_repository_slots"],
                     "fidelity": r.summary["average_fidelity"], "coverage": r.summary["coverage"],
                     "horizon": r.metadata["horizon"], "status": r.summary["status"]})
    return rows


def parameter_differences(runs):
    def flatten(value, prefix=""):
        if isinstance(value, dict):
            return {k: v for key, item in value.items()
                    for k, v in flatten(item, f"{prefix}.{key}" if prefix else key).items()}
        return {prefix: canonical(value)}
    values = [flatten(r.config) for r in runs]
    return [{"parameter": key, **{r.metadata["run_id"][:8]: v.get(key, "missing")
                                  for r, v in zip(runs, values)}}
            for key in sorted(set().union(*(v.keys() for v in values)))
            if len({v.get(key) for v in values}) > 1]
