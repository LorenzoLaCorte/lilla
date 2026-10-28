"""Compare uniform, level-wise, and per-swap Bell-outcome acceptance."""
from __future__ import annotations

from argparse import ArgumentParser
from itertools import product

from skopt.space import Categorical

from src.optimization.pauli_ad import (
    POLICIES, Journal, add_common_args, bayesian_search, candidate,
    fixed_protocol, protocol_nodes, setup, write_json,
)


def compare(record, baseline):
    if record is None or baseline is None:
        return None
    rate, reference = record["result"]["skr"], baseline["result"]["skr"]
    return dict(skr=rate, baseline_skr=reference, absolute_gain=rate-reference,
                relative_gain=(rate/reference-1) if reference > 0 else None,
                activates_key=rate > 0 and reference == 0)


def run(args):
    evaluator = setup(args)
    links = evaluator.hardware.links
    protocol = tuple(args.protocol) if args.protocol is not None else fixed_protocol(links, args.tree)
    nodes = protocol_nodes(protocol, links)
    if any(step.startswith("d") for step in protocol):
        raise ValueError("this study fixes a swap tree without distillation; use gp_pauliAD to optimize protocols")
    if not 0 < args.q <= .5 or args.max_exact < 1:
        raise ValueError("q must lie in (0,1/2], max-exact must be positive")
    settings = dict(study="postselection", protocol=protocol, q=args.q, method=args.method,
                    max_exact=args.max_exact, gp_calls=args.gp_calls, initial_points=args.initial_points,
                    seed=args.seed, level_convention="output subtree height; elementary links have height zero")
    journal = Journal(args.output, evaluator, settings, args.resume)
    results, stages = {}, {}
    try:
        uniform = [journal.evaluate(candidate(protocol, args.q, [p]*(links-1)), "uniform", p)
                   for p in POLICIES]
        stages["uniform"] = dict(method="exhaustive", space_size=3, records=uniform)
        levels = sorted({n["height"] for n in nodes})
        warm = [r["candidate"]["policies"] for r in uniform]
        for name, groups in (("level", [levels.index(n["height"]) for n in nodes]),
                             ("node", list(range(links-1)))):
            count = max(groups)+1
            size = 3**count
            exact = args.method == "exhaustive" or (args.method == "auto" and size <= args.max_exact)
            def decode(point):
                return candidate(protocol, args.q, [point[g] for g in groups])
            if exact:
                rows = [journal.evaluate(decode(point), name, list(point))
                        for point in product(POLICIES, repeat=count)]
            else:
                anchors = []
                for schedule in warm:
                    point = [schedule[groups.index(g)] for g in range(count)]
                    if all(point[g] == schedule[i] for i, g in enumerate(groups)) and point not in anchors:
                        anchors.append(point)
                # Include the notebook's promising final-only placement explicitly.
                root_only = ["all_corrected"]*count
                root = next(n["boundary"] for n in nodes if n["depth"] == 0)
                root_only[groups[root]] = "psi_pair"
                if root_only not in anchors:
                    anchors.append(root_only)
                rows = bayesian_search(journal,
                    [Categorical(POLICIES, name=f"policy_{i}") for i in range(count)],
                    decode, args.gp_calls, args.initial_points, args.seed, name, anchors)
            best = journal.best(rows)
            if best:
                warm.append(best["candidate"]["policies"])
            stages[name] = dict(method="exhaustive" if exact else "GP best found", space_size=size, records=rows)
        for name, stage in stages.items():
            best = journal.best(stage["records"])
            candidates = {r["candidate_id"] for r in stage["records"]}
            valid = {r["candidate_id"] for r in stage["records"] if r["result"]["status"] == "converged"}
            results[name] = dict(method=stage["method"], space_size=stage["space_size"],
                                 distinct_candidates=len(candidates), converged_candidates=len(valid),
                                 certified=stage["method"] == "exhaustive" and len(valid) == stage["space_size"],
                                 best=best)
        all_outcomes = next(r for r in uniform if r["candidate"]["policies"][0] == "all_corrected")
        summary = dict(study="postselection", swap_nodes=nodes, results=results,
                       uniform_vs_all=compare(results["uniform"]["best"], journal.best([all_outcomes])),
                       level_vs_uniform=compare(results["level"]["best"], results["uniform"]["best"]),
                       node_vs_uniform=compare(results["node"]["best"], results["uniform"]["best"]),
                       failures=sum(r["result"]["status"] != "converged" for r in journal.records))
        write_json(args.output / "summary.json", summary)
        lines = ["Postselection placement study", "SKR = pooled BB84 SKF / mean accepted-delivery time (bits/slot).",
                 "Rejected outcomes consume their preparation time and trigger destructive retries.",
                 "Policies are indexed by physical swap boundary s0, s1, ...; see swap_nodes in summary.json."]
        for name, result in results.items():
            best = result["best"]
            lines.append(f"{name}: {result['method']}; certified={result['certified']}; "
                         f"converged {result['converged_candidates']}/{result['space_size']} candidates")
            if best:
                r = best["result"]
                lines.append(f"  SKR={r['skr']:.10g}, SKF={r['skf']:.7g}, mean wait={r['mean_wait']:.7g}; "
                             f"schedule={best['candidate']['policies']}")
        for key in ("uniform_vs_all", "level_vs_uniform", "node_vs_uniform"):
            lines.append(f"{key}: {summary[key]}")
        lines += [f"Failed/unconverged requests: {summary['failures']}",
                  "A small observed GP gain does not bound the gain of untested nonuniform strategies."]
        (args.output / "summary.txt").write_text("\n".join(lines)+"\n")
        journal.save_arrays(journal.best(), "best")
        return summary
    finally:
        try:
            journal.export()
        finally:
            journal.close()


def main():
    parser = ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--q", type=float, default=.5, help="Global source excitation weight; Bell = 0.5")
    parser.add_argument("--tree", choices=("balanced", "left", "right"), default="balanced")
    parser.add_argument("--protocol", nargs="+", default=None, help="Optional fixed swap sequence, e.g. s0 s2 s1")
    parser.add_argument("--method", choices=("auto", "exhaustive", "gp"), default="auto")
    parser.add_argument("--max-exact", type=int, default=243)
    args = parser.parse_args()
    try:
        summary = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print((args.output / "summary.txt").read_text(), flush=True)
    return 0 if any(v["best"] for v in summary["results"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
