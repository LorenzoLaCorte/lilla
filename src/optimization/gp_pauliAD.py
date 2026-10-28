"""Bayesian search over swap trees, distillation, uniform acceptance and source q."""
from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
from skopt.space import Categorical, Integer, Real

from src.optimization.pauli_ad import (
    POLICIES, Journal, add_common_args, bayesian_search, candidate, digest,
    protocol_nodes, setup, write_json,
)
from src.utils.gp_utils import get_protocol_from_center_spacing_symmetricity


def encoded_protocol(nodes, max_dists, point, seed):
    """Reuse the paper encoding with reproducible, isolated protocol sampling.

    The existing encoding samples distillation placements using NumPy's global
    RNG. Give each encoding its own stable seed and restore the caller's RNG.
    Equal encodings within a replicate then describe exactly the same protocol.
    """
    gamma, rounds, eta, tau = point
    state = np.random.get_state()
    try:
        np.random.seed(int(digest([int(seed), list(point)])[:8], 16))
        return tuple(get_protocol_from_center_spacing_symmetricity(
            nodes, max_dists, float(gamma), int(rounds), float(eta), float(tau)))
    finally:
        np.random.set_state(state)


def run(args):
    if args.max_dists < 0 or not 0 < args.q_min < .5:
        raise ValueError("max-dists must be nonnegative; q-min must lie in (0,1/2)")
    # Catalan enumeration is inherited from gp_asymmetric; avoid accidental enormous spaces.
    if args.nodes > 11:
        raise ValueError("the inherited tree enumeration supports these studies up to 11 nodes")
    evaluator = setup(args)
    links = evaluator.hardware.links
    mode = getattr(args, "search_mode", "full")
    if args.joint_only and mode != "full":
        raise ValueError("--joint-only cannot be combined with a restricted --search-mode")
    q_grid = getattr(args, "q_grid", None) or []
    total_cap = getattr(args, "max_total_dists", None)
    total_cap = min((2*links-1)*args.max_dists,
                    total_cap if total_cap is not None else (2*links-1)*args.max_dists)
    if total_cap < 0 or any(not args.q_min <= q <= .5 for q in q_grid):
        raise ValueError("invalid total distillation cap or q grid")
    settings = dict(study="protocol", max_dists=args.max_dists, q_min=args.q_min,
                    gp_calls=args.gp_calls, initial_points=args.initial_points, seed=args.seed,
                    ablations=not args.joint_only, search_mode=mode, max_total_dists=total_cap,
                    q_grid=q_grid, encoding="existing gamma,K,eta,tau; stable per-encoding RNG")
    journal = Journal(args.output, evaluator, settings, args.resume)
    stages = (("bell_all", False, False), ("bell_policy", False, True),
              ("q_all", True, False), ("joint", True, True))
    if args.joint_only:
        stages = stages[-1:]
    elif mode == "bell":
        stages = (("bell_swaps", False, True), ("bell_policy", False, True))
    elif mode == "source":
        stages = (("bell_policy", False, True), ("joint", True, True))
    results, incumbents = {}, []
    try:
        for name, vary_q, vary_policy in stages:
            space = [Real(0., 1., name="gamma")]
            if total_cap and name != "bell_swaps":
                space += [Integer(0, total_cap, name="rounds"),
                          Real(-1., 1., name="eta"), Real(0., 1., name="tau")]
            if vary_policy:
                space.append(Categorical(POLICIES, name="acceptance"))
            if vary_q:
                space.append(Real(args.q_min, .5, name="q"))
            def decode(point):
                values = dict(zip((d.name for d in space), point))
                encoding = [values["gamma"], values.get("rounds", 0), values.get("eta", 0.), values.get("tau", 0.)]
                protocol = encoded_protocol(args.nodes, args.max_dists, encoding, args.seed)
                return candidate(protocol, values.get("q", .5), [values.get("acceptance", "all_corrected") ]*(links-1))
            def anchor(gamma=1., rounds=0, eta=-1., tau=0., policy="all_corrected", q=.5):
                values = dict(gamma=gamma, rounds=rounds, eta=eta, tau=tau, acceptance=policy, q=q)
                return [values[d.name] for d in space]
            anchors = []
            # Test Bell and PE baselines, both tree extremes, and early distillation.
            for gamma in (1., 0.):
                for policy in (POLICIES if vary_policy else ("all_corrected",)):
                    for q in ((.5, max(args.q_min, .25), max(args.q_min, .4)) if vary_q else (.5,)):
                        anchors.append(anchor(gamma=gamma, policy=policy, q=q))
            if total_cap and name != "bell_swaps":
                for rounds in sorted({min(links, total_cap), total_cap}):
                    anchors.append(anchor(rounds=rounds))
                for rounds in range(1, min(args.max_dists, total_cap)+1):
                    for policy in (POLICIES if vary_policy else ("all_corrected",)):
                        anchors.append(anchor(rounds=rounds, eta=1., policy=policy))
            # Prior ablation winners seed compatible supersets, guaranteeing a matched baseline.
            for _, old_point in incumbents:
                values = dict(old_point)
                if (not vary_q and values.get("q", .5) != .5) or (
                        not vary_policy and values.get("acceptance", "all_corrected") != "all_corrected"):
                    continue
                values.setdefault("q", .5)
                values.setdefault("acceptance", "all_corrected")
                values.setdefault("rounds", 0)
                values.setdefault("eta", 0.)
                values.setdefault("tau", 0.)
                anchors.append([values[d.name] for d in space])
            unique = []
            for point in anchors:
                if point not in unique:
                    unique.append(point)
            rows = bayesian_search(journal, space, decode, args.gp_calls, args.initial_points,
                                   args.seed, name, unique)
            best = journal.best(rows)
            results[name] = dict(best=best, proposals=args.gp_calls, anchors=len(unique),
                                 distinct_candidates=len({r["candidate_id"] for r in rows}),
                                 failed_requests=sum(r["result"]["status"] != "converged" for r in rows))
            if best:
                proposal = best["proposal"]
                point = proposal.get("point", proposal.get("anchor"))
                incumbents.append((name, dict(zip((d.name for d in space), point))))
        # Attribute gains on a matched tree instead of relying only on separate GP searches.
        matched = {}
        winner_stage = "bell_policy" if mode == "bell" else "joint"
        original = results[winner_stage]["best"]
        if original:
            c = original["candidate"]
            controls = {
                "same_protocol_bell": candidate(c["protocol"], .5, c["policies"]),
                "same_protocol_all": candidate(c["protocol"], c["q"], ["all_corrected"]*(links-1)),
                "same_protocol_bell_all": candidate(c["protocol"], .5, ["all_corrected"]*(links-1)),
                "same_tree_no_distillation": candidate([s for s in c["protocol"] if s.startswith("s")], c["q"], c["policies"]),
            }
            for label, control in controls.items():
                matched[label] = journal.evaluate(control, "winner_controls", proposal=label)
            matched["original_joint_winner"] = original
            results[winner_stage]["best"] = journal.best([original, *matched.values()])
            if "bell_swaps" in results:
                results["bell_swaps"]["best"] = journal.best([
                    r for r in (results["bell_swaps"]["best"], matched["same_tree_no_distillation"]) if r])
        # One-dimensional, matched-protocol q profiles distinguish source gains
        # from different trees discovered by two stochastic searches. Always
        # include Bell, on both the Bell winner and original joint protocol.
        profiles = {}
        if q_grid and mode != "bell":
            for label, record in (("bell_winner", results.get("bell_policy", {}).get("best")),
                                  ("joint_winner", original)):
                if record is None:
                    continue
                c = record["candidate"]
                profiles[label] = [journal.evaluate(candidate(c["protocol"], q, c["policies"]),
                                    "q_profile", proposal=dict(reference=label, q=q))
                                   for q in sorted(set([*q_grid, .5, c["q"]]))]
            results[winner_stage]["best"] = journal.best([
                r for r in [results[winner_stage]["best"], *[r for rows in profiles.values() for r in rows]] if r])
            if "bell_policy" in results:
                results["bell_policy"]["best"] = journal.best([
                    r for r in [results["bell_policy"]["best"],
                        *[r for rows in profiles.values() for r in rows if r["candidate"]["q"] == .5]] if r])
        summary = dict(study="protocol", certification="best found; not a proof of global optimality",
                       results=results, matched_controls=matched, q_profiles=profiles,
                       failures=sum(r["result"]["status"] != "converged" for r in journal.records))
        write_json(args.output / "summary.json", summary)
        lines = ["Pauli/AD protocol optimization", "SKR = pooled BB84 SKF / mean accepted-delivery time, in bits/slot.",
                 "All outcomes/retries and distillation resource preparation are included.",
                 "Source q and swap acceptance are global; tree and distillation placement may be asymmetric.",
                 "Source preparation cost is independent of q; any source damping is included at delivery.",
                 "All reported winners passed coverage and doubled-horizon checks; all searches are best found."]
        for name, result in results.items():
            best = result["best"]
            lines.append(f"{name}: {result['distinct_candidates']} distinct candidates, {result['failed_requests']} failed requests")
            if best:
                r, c = best["result"], best["candidate"]
                lines += [f"  SKR={r['skr']:.10g}, SKF={r['skf']:.7g}, mean wait={r['mean_wait']:.7g}, q={c['q']:.7g}",
                          f"  acceptance={c['policies'][0]}; protocol={c['protocol']}"]
                result["swap_nodes"] = protocol_nodes(c["protocol"], links)
        if matched:
            lines.append("Matched controls of the original joint winner (same tree):")
            for label, record in matched.items():
                lines.append(f"  {label}: {record['result']['status']}, SKR={record['result'].get('skr')}")
            lines.append("Improving matched controls are promoted to the reported winner.")
        if profiles:
            lines.append("Matched q profiles are saved in summary.json; q=0.5 controls also update the Bell winner.")
        write_json(args.output / "summary.json", summary)
        (args.output / "summary.txt").write_text("\n".join(lines)+"\n")
        journal.save_arrays(journal.best())
        return summary
    finally:
        try:
            journal.export()
        finally:
            journal.close()


def main():
    parser = ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--max-dists", type=int, default=1, help="Maximum recurrence rounds per tree vertex")
    parser.add_argument("--q-min", type=float, default=.1)
    parser.add_argument("--joint-only", action="store_true", help="Skip three matched ablation searches")
    parser.add_argument("--search-mode", choices=("full", "bell", "source"), default="full",
                        help="Bell: swaps-only vs distillation; source: Bell vs optimized q; full: original four ablations")
    parser.add_argument("--max-total-dists", type=int, default=None,
                        help="Also bound total recurrence rounds across the tree")
    parser.add_argument("--q-grid", type=float, nargs="+", default=None,
                        help="Matched q profiles on the Bell and joint winners, always including q=0.5")
    args = parser.parse_args()
    try:
        summary = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print((args.output / "summary.txt").read_text(), flush=True)
    return 0 if any(v["best"] for v in summary["results"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
