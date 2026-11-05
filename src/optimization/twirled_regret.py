"""Finite, matched-policy transfer experiments, independent of the visualization app.

Every model scores exactly the same candidate list. The full-model column is
the physical execution of *every* approximate-model policy, not its prediction.
"""
from __future__ import annotations

from argparse import ArgumentParser
from functools import lru_cache
import hashlib
from itertools import product
import json
import logging
from pathlib import Path

import numpy as np

from src.core.pauli_adc.state import polish_x_func
from src.optimization.pauli_ad import (
    Evaluation, Hardware, Journal, Numerics, candidate, digest,
    fixed_protocol, write_json,
)

MODELS = ("full", "bell_source", "werner_source", "channel_twirl", "bell_channel_twirl", "bell_each_stage")
SOURCES = ("werner", "phase_error", "bit_error", "r_state", "pe")
SPACES = ("swaps", "distillation", "postselection", "joint")


def project(coordinates, kind):
    """Linear maps on normalized or success-weighted real-X arrays; preserve II."""
    x = np.array(coordinates, dtype=float, copy=True)
    x[..., 1:3] = 0
    if kind == "werner":
        w = (x[..., 3] + x[..., 4] - x[..., 5])/3
        x[..., 3], x[..., 4], x[..., 5] = w, w, -w
    elif kind != "bell":
        raise ValueError(f"unknown state projection: {kind}")
    return x


def initial_state(family, fidelity, q=.25):
    """F fixes Phi+ fidelity, except PE where it fixes the Bell-reference visibility."""
    if family not in SOURCES or not .5 <= fidelity <= 1 or not 0 < q <= .5:
        raise ValueError("invalid source family, fidelity or q")
    f = float(fidelity)
    if family == "werner":
        v = (4*f-1)/3
        x = [1, 0, 0, v, v, -v]
    elif family == "phase_error":
        x = [1, 0, 0, 1, 2*f-1, 1-2*f]
    elif family == "bit_error":
        x = [1, 0, 0, 2*f-1, 1, 1-2*f]
    elif family == "r_state":
        x = [1, f-1, 1-f, 2*f-1, f, -f]
    else:
        v = (4*f-1)/3
        c = 2*v*np.sqrt(q*(1-q))
        x = [1, v*(1-2*q), v*(1-2*q), v, c, -c]
    x = np.asarray(x, dtype=float)
    polish_x_func(x[None, :], atol=1e-12)
    return x


class ModelEvaluation(Evaluation):
    def __init__(self, hardware, numerics, coordinates, model="full"):
        if model not in (*MODELS, "werner_each_stage"):
            raise ValueError(f"unknown prediction model: {model}")
        super().__init__(hardware, numerics)
        self.coordinates, self.model = np.asarray(coordinates, dtype=float), model
        if self.coordinates.shape != (6,) or not np.isclose(self.coordinates[0], 1):
            raise ValueError("source must have six normalized real-X coordinates")
        polish_x_func(self.coordinates[None, :], atol=1e-12)

    def initial_coordinates(self, choice):
        if self.model in ("bell_source", "bell_each_stage", "bell_channel_twirl"):
            return project(self.coordinates, "bell")
        if self.model in ("werner_source", "werner_each_stage"):
            return project(self.coordinates, "werner")
        return self.coordinates.copy()

    def memory_rates(self):
        g, k = super().memory_rates()
        if self.model in ("channel_twirl", "bell_channel_twirl"):
            # Exact same finite-time contractions, with the affine shift removed.
            # Do not merely set k=0: that would also remove its contractions.
            return g + k[:, None]*np.array([.5, .5, 1.]), np.zeros_like(k)
        return g, k

    def operation_output(self, coordinates):
        if self.model == "bell_each_stage":
            return project(coordinates, "bell")
        if self.model == "werner_each_stage":
            return project(coordinates, "werner")
        return coordinates


@lru_cache(None)
def all_trees(lo, hi):
    """All ordered full binary trees, represented by half-open link spans."""
    if hi-lo == 1:
        return ((lo, hi, None, None),)
    return tuple((lo, hi, left, right)
                 for mid in range(lo+1, hi)
                 for left in all_trees(lo, mid) for right in all_trees(mid, hi))


def tree_from_protocol(protocol, links):
    segments = {i: (i, i+1, None, None) for i in range(links)}
    for step in protocol:
        i = int(step[1:])
        j = min(k for k in segments if k > i)
        left, right = segments.pop(i), segments.pop(j)
        segments[j] = (left[0], right[1], left, right)
    return next(iter(segments.values()))


def vertices(tree):
    yield tree
    if tree[2] is not None:
        yield from vertices(tree[2])
        yield from vertices(tree[3])


def compile_tree(tree, counts):
    lo, hi, left, right = tree
    steps = [] if left is None else [*compile_tree(left, counts), *compile_tree(right, counts), f"s{left[1]-1}"]
    return [*steps, *[f"d{hi-1}"]*counts.get((lo, hi), 0)]


def candidate_pool(links, p_gen, *, tree_mode="balanced", cutoff_factors=(None, .5, 2.),
                   max_dists=2, placements="structured", acceptance="level", q=.5):
    """Predeclare a finite family; no model-dependent sampling or search budget.

    Structured recurrence placements: none, 1/2 at root, 1 at each root child,
    all leaves or all height-one links (the latter patterns obey max_dists).
    'vertices' additionally tries 1..max_dists at every individual vertex.
    """
    if links < 2 or max_dists < 0 or placements not in ("structured", "vertices") or not 0 < q <= .5:
        raise ValueError("invalid chain or recurrence family")
    if tree_mode == "all_small":
        trees = all_trees(0, links) if links <= 4 else tuple(
            tree_from_protocol(fixed_protocol(links, shape), links) for shape in ("balanced", "left", "right"))
    elif tree_mode == "balanced":
        trees = (tree_from_protocol(fixed_protocol(links), links),)
    else:
        raise ValueError("tree mode must be balanced or all_small")
    cutoffs = []
    for factor in cutoff_factors:
        if factor is not None and (not np.isfinite(factor) or factor <= 0):
            raise ValueError("cutoff factors must be positive or None")
        cutoff = None if factor is None else max(1, round(factor/min(p_gen)))
        if cutoff not in cutoffs:
            cutoffs.append(cutoff)
    rows = {}
    for tree in trees:
        nodes = list(vertices(tree))
        root = (0, links)
        patterns = [{}] + [{root: k} for k in range(1, max_dists+1)]
        for chosen in ([tree[2], tree[3]], [n for n in nodes if n[2] is None],
                       [n for n in nodes if n[2] is not None and n[2][2] is None and n[3][2] is None]):
            if 0 < len(chosen) <= max_dists:
                patterns.append({(n[0], n[1]): 1 for n in chosen})
        if placements == "vertices":
            patterns += [{(n[0], n[1]): k} for n in nodes for k in range(1, max_dists+1)]
        policies = [[p]*(links-1) for p in ("all_corrected", "phi_pair", "psi_pair")]
        if acceptance == "level":
            # Physical boundary, not operation order. Concentrate elementary pairs only.
            ps = ["all_corrected"]*(links-1)
            for n in nodes:
                if n[2] is not None and n[2][2] is None and n[3][2] is None:
                    ps[n[2][1]-1] = "psi_pair"
            policies.append(ps)
            # Late filtering can help Bell sources after lower-level damping.
            # Include it explicitly; first-level concentration alone misses it.
            for p in ("psi_pair", "phi_pair"):
                root_only = ["all_corrected"]*(links-1)
                root_only[tree[2][1]-1] = p
                policies.append(root_only)
        elif acceptance != "uniform":
            raise ValueError("acceptance must be uniform or level")
        for counts, cutoff, ps in product(patterns, cutoffs, policies):
            c = candidate(compile_tree(tree, counts), q, ps)
            c["cutoff"] = cutoff
            # q is fixed in coordinates, not optimized: it is inert here.
            rows[digest(c)] = c
    return [rows[k] for k in sorted(rows)]


def in_space(choice, space):
    d = any(s.startswith("d") for s in choice["protocol"])
    post = any(p != "all_corrected" for p in choice["policies"])
    return {"swaps": not d and not post, "distillation": not post,
            "postselection": not d, "joint": True}[space]


def compare(rows, model, *, tie_rtol=1e-4, tie_atol=1e-10):
    """Report actual transferred policy plus range over prediction ties.

    A partial common subset is labelled, never silently promoted to a global
    finite-family comparison. Zero-rate ties are not evidence of robust regret.
    """
    paired = [r for r in rows if all(r["results"][m]["status"] == "converged" for m in ("full", model))]
    result = dict(candidates=len(rows), paired=len(paired), complete=len(paired) == len(rows))
    if not paired:
        return dict(result, status="missing")
    top = max(r["results"]["full"]["skr"] for r in paired)
    predicted = max(r["results"][model]["skr"] for r in paired)
    ties = [r for r in paired if abs(r["results"][model]["skr"]-predicted) <= tie_atol+tie_rtol*predicted]
    # Strict predicted-rate argmax. The near-tie interval is separate from the
    # deterministic choice and captures sensitivity at the configured tolerance.
    selected = min(paired, key=lambda r: (-r["results"][model]["skr"], r["candidate_id"]))
    physical_best = min(paired, key=lambda r: (-r["results"]["full"]["skr"], r["candidate_id"]))
    rates = [r["results"]["full"]["skr"] for r in ties]
    actual = selected["results"]["full"]["skr"]
    return dict(result, status="complete" if result["complete"] else "paired_subset",
                physical_best=top, approximate_best_prediction=predicted,
                selected_prediction=selected["results"][model]["skr"], physical_selected=actual,
                absolute_loss=max(0., top-actual), relative_loss=(max(0., top-actual)/top if top > 0 else None),
                tie_count=len(ties), all_zero_prediction=predicted == 0,
                tie_loss_min=max(0., top-max(rates)), tie_loss_max=max(0., top-min(rates)),
                robust_relative_loss=(max(0., top-max(rates))/top if top > 0 else None),
                prediction_error=selected["results"][model]["skr"]-actual,
                physical_winner_id=physical_best["candidate_id"], selected_id=selected["candidate_id"],
                physical_winner=physical_best["candidate"], selected=selected["candidate"],
                physical_winner_metrics=physical_best["results"]["full"],
                selected_physical_metrics=selected["results"]["full"],
                tied_ids=[r["candidate_id"] for r in ties])


class StudyJournal(Journal):
    def export(self):
        # The batch exporter makes tables/plots, avoiding repeated plotting per trial.
        pass


def run(spec, output, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    h = Hardware(**spec["hardware"])
    n = Numerics(**spec["numerics"])
    coordinates = initial_state(**spec["source"])
    models = spec["models"]
    if models[0] != "full" or len(set(models)) != len(models):
        raise ValueError("models must be unique, starting with full")
    choices = candidate_pool(h.links, h.p_gen, **spec["family"])
    root = Path(__file__).resolve().parents[2]
    implementation = [*sorted((root/"src/core").rglob("*.py")),
                      root/"src/optimization/pauli_ad.py", Path(__file__).resolve(),
                      root/"src/scripts/twirled_regret.py"]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation}
    settings = dict(study="twirled_regret", source=spec["source"], family=spec["family"],
                    implementation_hashes=hashes,
                    model_definitions="source-only / stage-boundary / local-channel; v1",
                    candidate_ids=[digest(c) for c in choices], tie_rtol=spec["tie_rtol"], tie_atol=spec["tie_atol"])
    journals = {}
    try:
        for model in models:
            evaluator = ModelEvaluation(h, n, coordinates, model)
            journals[model] = StudyJournal(output/model, evaluator, {**settings, "model": model}, resume)
        rows = []
        for c in choices:
            results = {m: journals[m].evaluate(c, "enumeration")["result"] for m in models}
            row = dict(candidate_id=digest(c), candidate=c, results=results)
            rows.append(row)
            logging.info("candidate %s (%d/%d)", digest(c), len(rows), len(choices))
        comparisons = []
        for space, model in product(SPACES, models[1:]):
            subset = [r for r in rows if in_space(r["candidate"], space)]
            comparisons.append(dict(space=space, model=model, **compare(
                subset, model, tie_rtol=spec["tie_rtol"], tie_atol=spec["tie_atol"])))
        # Fixed-policy errors distinguish prediction error from optimizer decision loss.
        diagnostics = []
        for model in models[1:]:
            for space in SPACES:
                subset = [r for r in rows if in_space(r["candidate"], space)]
                covered = [r for r in subset if all(r["results"][m]["status"] == "converged" for m in ("full", model))]
                gaps = {k: max((abs(r["results"][model][k]-r["results"]["full"][k]) for r in covered), default=None)
                        for k in ("skr", "skf", "mean_wait", "fidelity", "eX", "eZ")}
                diagnostics.append(dict(model=model, space=space, paired=len(covered), candidates=len(subset),
                                        max_absolute_errors=gaps))
        summary = dict(schema=1, spec=spec, source_coordinates=coordinates.tolist(), candidates=len(choices),
                       comparisons=comparisons, diagnostics=diagnostics,
                       numerical_status={m: {s: sum(r["results"][m]["status"] == s for r in rows)
                                             for s in ("converged", "unconverged", "failed")} for m in models})
        write_json(output/"summary.json", summary)
        # Also save a single model-aligned view, while per-model journals remain durable.
        write_json(output/"paired.json", rows)
        return summary
    finally:
        for journal in journals.values():
            journal.close()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(json.loads(args.case.read_text()), args.output, args.resume)


if __name__ == "__main__":
    main()
