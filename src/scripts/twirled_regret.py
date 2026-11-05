"""Plan, run, resume and plot matched-hardware twirling decision-loss studies."""
from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import sys
import threading

import numpy as np

from src.optimization.pauli_ad import Hardware, Numerics, canonical, digest, write_json
from src.optimization.twirled_regret import MODELS, SOURCES, candidate_pool, initial_state, in_space
from src.scripts.gp_pauliAD import execute
from src.scripts.pauli_ad_hardware import hardware as reference_hardware, write_table


def plan(args):
    smoke = args.preset == "smoke"
    controls = args.preset == "controls"
    threshold = args.preset == "threshold"
    links = args.links or ([2] if smoke else [2, 3, 4] if controls else [2, 3, 4, 8])
    sources = args.sources or (["r_state"] if smoke else ["r_state", "pe"] if controls else
                               ["werner"] if threshold else ["werner", "phase_error", "r_state"])
    fidelities = args.fidelities or ([.97] if smoke or controls else [.97, .99])
    scales = args.memory_scales or ([1.] if smoke or controls else
                                   [.02, .05, .1, .2, .5, 1., 2.] if threshold else [.05, .2, 1., 2.])
    noises = args.noises or (["pauli"] if smoke else ["none", "pauli"] if controls else
                            ["ad"] if threshold else ["pauli", "ad"])
    distances = args.total_km or args.link_km or [5.]
    cutoffs = [None, *args.cutoff_factors] if args.cutoff_factors is not None else [None, .5] if smoke else [None, .5, 2.]
    cap = args.max_dists if args.max_dists is not None else 0 if threshold else 1 if smoke else 2
    if (any(l < 2 or l > 16 for l in links) or any(not .5 <= f <= 1 for f in fidelities)
            or any(not np.isfinite(s) or s <= 0 for s in scales)
            or args.workers < 1 or args.case_timeout <= 0 or args.tie_rtol < 0 or args.tie_atol < 0):
        raise ValueError("invalid grid, workers, time limit or tie tolerance")
    if len(set(args.models)) != len(args.models) or not 0 < args.pe_q <= .5:
        raise ValueError("models must be unique; PE q must lie in (0,1/2]")
    n = Numerics(args.initial_horizon, args.max_horizon, args.coverage, args.rtol, args.atol)
    rows = {}
    for length, family, f, scale, noise, distance_reference in product(links, sources, fidelities, scales, noises, distances):
        distance = distance_reference/length if args.total_km else distance_reference
        ref = reference_hardware("metro97" if f < .98 else "metro99", "network", length+1, scale,
                                 args.geometry, metro_link_km=distance)
        t1, t2 = np.asarray(ref["t1_slots"]), np.asarray(ref["t2_slots"])
        if noise == "ad":
            k = 1/t1
            phi = 1/t2-k/2
            g = np.column_stack([phi, phi, np.zeros(length+1)])
        elif noise == "ad_pure":
            k, g = 1/t1, np.zeros((length+1, 3))
            t2 = 2*t1
        elif noise == "pauli":
            k = np.zeros(length+1)
            g = np.column_stack([1/t2, 1/t2, 1/t1])
        elif noise == "dephasing":
            k = np.zeros(length+1)
            g = np.column_stack([1/t2, 1/t2, np.zeros(length+1)])
        else:
            k, g = np.zeros(length+1), np.zeros((length+1, 3))
        h = Hardware(length+1, tuple(ref["p_gen"]), ref["p_swap"], args.p_distillation,
                     tuple(map(tuple, g)), tuple(k))
        initial_state(family, f, args.pe_q)
        policy_family = dict(tree_mode=args.trees, cutoff_factors=cutoffs, max_dists=cap,
                             placements=args.placements, acceptance=args.acceptance)
        if family == "pe":
            policy_family["q"] = args.pe_q
        pool = candidate_pool(length, h.p_gen, **policy_family)
        metadata = dict(links=length, nodes=length+1, source=family, fidelity_reference=f,
                        memory_scale=scale, noise=noise, geometry=args.geometry,
                        elementary_km=distance, total_km=distance*length,
                        distance_convention="fixed_total" if args.total_km else "fixed_link",
                        distance_reference_km=distance_reference,
                        slot_seconds=ref["slot_seconds"], t1_seconds=(t1*ref["slot_seconds"]).tolist(),
                        t2_seconds=(t2*ref["slot_seconds"]).tolist(), reference_hardware=ref)
        spec = dict(schema=1, hardware=asdict(h), numerics=asdict(n),
                    source=dict(family=family, fidelity=f, q=args.pe_q), family=policy_family,
                    models=["full", *[m for m in args.models if m != "full"]], metadata=metadata,
                    tie_rtol=args.tie_rtol, tie_atol=args.tie_atol)
        name = f"L{length}_{family}_F{f:g}_{noise}_s{scale:g}_{distance:g}km_{digest(spec)[:8]}"
        directory = args.output/name
        command = [sys.executable, "-m", "src.optimization.twirled_regret", "--case",
                   str(directory/"case.json"), "--output", str(directory)]
        rows[name] = dict(name=name, directory=str(directory), command=command,
                          candidates=len(pool), evaluations=len(pool)*len(spec["models"]), spec=spec)
    return list(rows.values())


def summarize(cases, output):
    """Read-only export, including partial journals from interrupted workers."""
    output = Path(output)
    comparisons, diagnostics, candidates, statuses, margins = [], [], [], [], []
    for case in cases:
        spec = case["spec"]
        meta = {k: v for k, v in spec["metadata"].items()
                if k not in ("reference_hardware", "t1_seconds", "t2_seconds")}
        meta["case"] = case["name"]
        slot = meta["slot_seconds"]
        summary_path = Path(case["directory"])/"summary.json"
        if summary_path.exists():
            saved = json.loads(summary_path.read_text())
            for r in saved["comparisons"]:
                flat = {k: v for k, v in r.items() if k not in
                        ("physical_winner_metrics", "selected_physical_metrics")}
                for k in ("physical_best", "physical_selected", "approximate_best_prediction",
                          "absolute_loss", "tie_loss_min", "tie_loss_max", "prediction_error"):
                    if k in r:
                        flat[k+"_bits_per_second"] = r[k]/slot
                comparisons.append({**meta, **flat})
            diagnostics += [{**meta, **{k: v for k, v in r.items() if k != "max_absolute_errors"},
                             **{f"max_error_{k}": v for k, v in r["max_absolute_errors"].items()}}
                            for r in saved["diagnostics"]]
        for model in spec["models"]:
            journal = Path(case["directory"])/model/"trials.jsonl"
            records = {}
            if journal.exists():
                for line in journal.read_text().splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Interrupted final line; worker validates on resume.
                    records[record["candidate_id"]] = record
            for record in records.values():
                r, c = record["result"], record["candidate"]
                row = dict(meta, model=model, candidate_id=record["candidate_id"], status=r["status"],
                           protocol=c["protocol"], policies=c["policies"], cutoff=c["cutoff"],
                           **{k: r.get(k) for k in ("skr", "skf", "mean_wait", "fidelity", "eX", "eZ",
                                                   "bb84_margin", "coverage", "horizon", "error", "convergence_reason")})
                row["skr_bits_per_second"] = r["skr"]/slot if r["status"] == "converged" else None
                row["mean_wait_seconds"] = r["mean_wait"]*slot if r["status"] == "converged" else None
                candidates.append(row)
            # Threshold diagnostic: max signed margin over the swaps-only family.
            eligible = [r for r in records.values() if r["result"]["status"] == "converged"
                        and not any(s.startswith("d") for s in r["candidate"]["protocol"])
                        and all(p == "all_corrected" for p in r["candidate"]["policies"])]
            if eligible:
                best = max(eligible, key=lambda r: r["result"]["bb84_margin"])
                expected = sum(in_space(c, "swaps") for c in candidate_pool(
                    spec["hardware"]["nodes"]-1, spec["hardware"]["p_gen"], **spec["family"]))
                margins.append(dict(meta, model=model, max_margin=best["result"]["bb84_margin"],
                                    candidate_id=best["candidate_id"], covered_candidates=len(eligible),
                                    candidates=expected, complete=len(eligible) == expected))
        status_path = Path(case["directory"])/"runner_status.json"
        if status_path.exists():
            statuses.append(json.loads(status_path.read_text()))
    for name, rows in (("comparisons", comparisons), ("fixed-policy-errors", diagnostics),
                       ("evaluations", candidates), ("statuses", statuses), ("key-margins", margins)):
        # JSON preserves nested policies and nulls; CSV is convenient for plots/tables.
        write_json(output/f"{name}.json", rows)
        write_table(output/f"{name}.csv", [{k: json.dumps(v) if isinstance(v, (dict, list)) else v
                                          for k, v in r.items()} for r in rows])
    positive = [r for r in comparisons if r.get("status") == "complete" and r.get("physical_best", 0) > 0]
    unresolved = sum(r["status"] != "converged" for r in candidates)
    readme = ("# Twirled-model policy transfer\n\n"
              f"Completed case summaries: {sum((Path(c['directory'])/'summary.json').exists() for c in cases)} / {len(cases)}.\n\n"
              f"Saved candidate/model evaluations: {len(candidates)}; unresolved/failed: {unresolved}. "
              f"Complete positive-reference comparisons: {len(positive)} (nested spaces, not independent replicates).\n\n"
              "Read comparisons.csv/json for policy loss, fixed-policy-errors.csv for prediction errors, "
              "evaluations.csv for all scores, key-margins.csv for signed threshold scans, and statuses.csv for runtimes. "
              "Per-model JSONL journals retain convergence histories. Missing results are never zero rates.\n\n"
              "Rates ending in bits_per_second use each case's physical slot duration. Other SKR values are bits/slot. "
              "relative_loss and robust_relative_loss are fractions. Ratios are null when the physical optimum is zero. "
              "The selected policy uses a physical-information-free deterministic tie break; tie_loss_min/max cover "
              "all prediction ties. An all-zero surrogate is a false-negative/tie, not evidence of robust regret. "
              "Incomplete families are labelled paired_subset and omitted from the main plots.\n\n"
              "The threshold plot shows sampled signed margins, not fitted or certified lifetime roots.\n\n"
              "Replot without simulation: python -m src.scripts.twirled_regret --summarize --output "
              f"{output}\n\nDesign and limitations: src/optimization/TWIRLED_REGRET.md\n")
    (output/"README.md").write_text(readme)
    try:
        plot_results(comparisons, diagnostics, margins, output)
        write_json(output/"plot_status.json", dict(status="complete"))
    except (ValueError, RuntimeError, ImportError) as exc:
        write_json(output/"plot_status.json", dict(status="failed", error=str(exc)))
        print(f"Tables saved; plot failed: {exc}", flush=True)
    return comparisons


def plot_results(rows, diagnostics, margins, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.optimization.pauli_ad import draw_protocol

    valid = [r for r in rows if r.get("status") == "complete"]
    models = list(dict.fromkeys(r["model"] for r in valid))
    if not models:
        return
    noises = list(dict.fromkeys(r["noise"] for r in valid))
    sources = list(dict.fromkeys(r["source"] for r in valid))
    colors = dict(zip(models, plt.cm.tab10.colors))
    positive = [r for r in valid if r.get("physical_best", 0) > 0]
    ticks = sorted({r["links"] for r in valid})
    # Keep hardware fixed along each line. Never maximize/average away source or scale.
    for space in ("swaps", "distillation", "postselection", "joint"):
        fig, axes = plt.subplots(len(sources), len(noises), figsize=(5*len(noises), 3.1*len(sources)), squeeze=False, constrained_layout=True)
        for (source, noise), ax in zip(product(sources, noises), axes.flat):
            groups = {}
            for r in positive:
                if r["space"] == space and r["noise"] == noise and r["source"] == source:
                    key = (r["model"], r["source"], r["fidelity_reference"], r["memory_scale"],
                           r.get("distance_convention"), r.get("distance_reference_km", r["elementary_km"]), r["geometry"])
                    groups.setdefault(key, []).append(r)
            for key, group in groups.items():
                group.sort(key=lambda r: r["links"])
                ax.plot([r["links"] for r in group], [100*r["relative_loss"] for r in group],
                        color=colors[key[0]], alpha=.32, linewidth=.8)
            for model in models:
                data = [r for r in positive if r["model"] == model and r["space"] == space
                        and r["noise"] == noise and r["source"] == source]
                unique = [r for r in data if not r["all_zero_prediction"]]
                tied = [r for r in data if r["all_zero_prediction"]]
                ax.scatter([r["links"] for r in unique], [100*r["relative_loss"] for r in unique],
                           color=colors[model], s=22, label=model)
                ax.scatter([r["links"] for r in tied], [100*r["relative_loss"] for r in tied],
                           color=colors[model], s=35, marker="x")
            largest = max((100*r["relative_loss"] for r in positive if r["space"] == space), default=0)
            upper = min(102, max(1, 1.15*largest+1))
            ax.set(title=f"{source} / {noise}", xlabel="Elementary links", ylabel="Physical SKR loss (%)", ylim=(-.02*upper, upper))
            ax.set_xticks(ticks)
            ax.grid(alpha=.2)
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(f"{space}: complete finite families; × = all-zero surrogate tie", fontsize=11)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"loss-{space}.{ext}", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model in models:
        data = [r for r in positive if r["model"] == model and r["space"] == "joint"]
        axes[0].scatter([r["physical_best_bits_per_second"] for r in data],
                        [r["physical_selected_bits_per_second"] for r in data], s=18, label=model)
        nonzero = [r for r in data if not r["all_zero_prediction"]]
        axes[1].scatter([100*r["relative_loss"] for r in nonzero],
                        [100*r["robust_relative_loss"] for r in nonzero], s=18, label=model)
    limit = max((r["physical_best_bits_per_second"] for r in positive), default=1)
    axes[0].plot([0, limit], [0, limit], color=".6", linestyle="--")
    axes[0].set(xlabel="Best physical SKR in family (bits/s)", ylabel="Transferred physical SKR (bits/s)")
    axes[1].plot([0, 100], [0, 100], color=".6", linestyle="--")
    axes[1].set(xlabel="Loss with deterministic tie break (%)", ylabel="Minimum loss over prediction ties (%)")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"transfer-and-ties.{ext}", dpi=170)
    plt.close(fig)
    # Numerical exactness controls: per-candidate discrepancies, not just tied optima.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model in models:
        data = [r for r in diagnostics if r["model"] == model and r["space"] == "distillation"
                and r["paired"] == r["candidates"] and r.get("max_error_skr") is not None]
        for noise, marker in (("pauli", "o"), ("ad", "^"), ("none", "s"), ("dephasing", "D"), ("ad_pure", "v")):
            d = [r for r in data if r["noise"] == noise]
            axes[0].scatter([r["links"] for r in d], [max(r["max_error_skr"], 1e-16) for r in d],
                            s=20, marker=marker, color=colors[model], label=f"{model}/{noise}" if d else None)
    axes[0].set(xlabel="Elementary links", ylabel="Max fixed-policy |SKR error| (bits/slot)", yscale="log")
    axes[0].set_xticks(ticks)
    axes[0].set_title("All-corrected recurrence family; zeros shown at 1e-16", fontsize=9)
    axes[0].legend(fontsize=5, ncol=2)
    margin_groups = {}
    for r in margins:
        if not r["complete"]:
            continue
        key = (r["model"], r["noise"], r["links"], r["source"], r["fidelity_reference"], r["elementary_km"])
        if r["model"] in ("full", "channel_twirl") and r["noise"] in ("ad", "ad_pure"):
            margin_groups.setdefault(key, []).append(r)
    for key, data in margin_groups.items():
        data.sort(key=lambda r: r["memory_scale"])
        axes[1].plot([r["memory_scale"] for r in data], [r["max_margin"] for r in data],
                     linestyle="-" if key[0] == "full" else "--", alpha=.55,
                     marker="o", markersize=3, markerfacecolor="white" if key[0] != "full" else None,
                     label=f"{key[0]}, L={key[2]}, {key[3]}, F={key[4]}" if len(margin_groups) <= 12 else None)
    axes[1].axhline(0, color=".4", linewidth=.7)
    axes[1].set(xlabel="Memory lifetime scale (log)", ylabel="Best signed BB84 margin, swaps only", xscale="log")
    if not margin_groups:
        axes[1].text(.5, .5, "No AD lifetime scan in this batch", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    if margin_groups and len(margin_groups) <= 12:
        axes[1].legend(fontsize=6)
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"prediction-controls.{ext}", dpi=170)
    plt.close(fig)
    if margin_groups:
        msources = list(dict.fromkeys(r["source"] for r in margins))
        mnoises = list(dict.fromkeys(r["noise"] for r in margins))
        fig, axes = plt.subplots(len(msources), len(mnoises), figsize=(5*len(mnoises), 3.4*len(msources)),
                                 squeeze=False, constrained_layout=True)
        size_colors = dict(zip(ticks, plt.cm.tab10.colors))
        for (source, noise), ax in zip(product(msources, mnoises), axes.flat):
            groups = {}
            for r in margins:
                if r["complete"] and r["model"] in ("full", "channel_twirl") and r["source"] == source and r["noise"] == noise:
                    key = (r["model"], r["links"], r["fidelity_reference"], r.get("distance_reference_km", r["elementary_km"]))
                    groups.setdefault(key, []).append(r)
            for key, data in groups.items():
                data.sort(key=lambda r: r["memory_scale"])
                ax.plot([r["memory_scale"] for r in data], [r["max_margin"] for r in data],
                        linestyle="-" if key[0] == "full" else "--", marker="o" if key[2] < .98 else "s",
                        markersize=3, color=size_colors[key[1]], linewidth=1,
                        markerfacecolor=size_colors[key[1]] if key[0] == "full" else "white",
                        label=f"{key[0]}, L={key[1]}, F={key[2]}")
            ax.axhline(0, color=".4", linewidth=.7)
            ax.set(title=f"{source} / {noise}", xlabel="Memory scale", ylabel="Best signed margin, swaps only", xscale="log")
            ax.grid(alpha=.2)
            ax.legend(fontsize=5, ncol=2)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"key-regimes.{ext}", dpi=170)
        plt.close(fig)
    examples = sorted((r for r in positive if r["space"] == "joint" and not r["all_zero_prediction"]),
                      key=lambda r: r["absolute_loss_bits_per_second"], reverse=True)[:4]
    if examples:
        fig, axes = plt.subplots(len(examples), 2, figsize=(10, 3*len(examples)), squeeze=False, constrained_layout=True)
        for r, pair in zip(examples, axes):
            for ax, key, label in zip(pair, ("physical_winner", "selected"), ("Full-model optimum", "Surrogate-selected")):
                draw_protocol(ax, r[key], r["links"])
                ax.set_title(f"{label}; cutoff={r[key]['cutoff']} slots", fontsize=9)
            pair[0].text(0, -.12, f"{r['case']} / {r['model']}: loss {100*r['relative_loss']:.3g}%",
                         transform=pair[0].transAxes, fontsize=7)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"policy-examples.{ext}", dpi=170)
        plt.close(fig)


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "controls", "overnight", "threshold"), default="overnight")
    p.add_argument("--output", type=Path, default=Path("results/twirled-regret/overnight"))
    p.add_argument("--links", nargs="+", type=int)
    p.add_argument("--sources", nargs="+", choices=SOURCES)
    p.add_argument("--fidelities", nargs="+", type=float)
    p.add_argument("--memory-scales", nargs="+", type=float)
    p.add_argument("--noises", nargs="+", choices=("none", "dephasing", "pauli", "ad", "ad_pure"))
    distance = p.add_mutually_exclusive_group()
    distance.add_argument("--link-km", nargs="+", type=float)
    distance.add_argument("--total-km", nargs="+", type=float,
                          help="Hold chain span fixed instead of elementary span when increasing link count")
    p.add_argument("--geometry", choices=("balanced", "slow_link", "weak_memory", "bottleneck"), default="balanced")
    p.add_argument("--models", nargs="+", choices=(*MODELS, "werner_each_stage"), default=list(MODELS[1:]))
    p.add_argument("--pe-q", type=float, default=.25)
    p.add_argument("--trees", choices=("balanced", "all_small"), default="balanced")
    p.add_argument("--placements", choices=("structured", "vertices"), default="structured")
    p.add_argument("--max-dists", type=int)
    p.add_argument("--acceptance", choices=("uniform", "level"), default="level")
    p.add_argument("--cutoff-factors", nargs="*", type=float,
                   help="Cutoff/minimum-generation-mean factors; no-cutoff always included; empty means no-cutoff only")
    p.add_argument("--p-distillation", type=float, default=1.)
    p.add_argument("--initial-horizon", type=int, default=16384)
    p.add_argument("--max-horizon", type=int, default=1048576)
    p.add_argument("--coverage", type=float, default=.999999)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--tie-rtol", type=float, default=1e-4)
    p.add_argument("--tie-atol", type=float, default=2e-9)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--case-timeout", type=float, default=7200.)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--summarize", action="store_true")
    return p


def main():
    args = parser().parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output/".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output/".cache"))
    path = args.output/"plan.json"
    if args.summarize:
        summarize(json.loads(path.read_text())["cases"], args.output)
        return
    cases = plan(args)
    content = dict(schema=1, cases=cases)
    if path.exists():
        if not (args.resume or args.dry_run):
            raise ValueError("output already has a plan; use --resume or a fresh output")
        if canonical(json.loads(path.read_text())) != canonical(content):
            raise ValueError("plan differs from saved plan; use a new output directory")
    else:
        write_json(path, content)
    for c in cases:
        directory = Path(c["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory/"case.json", c["spec"])
    print(f"{len(cases)} cases; {sum(c['candidates'] for c in cases)} policies across cells; "
          f"{sum(c['evaluations'] for c in cases)} candidate/model evaluations (each with horizon checks).", flush=True)
    print(f"Plan: {path}", flush=True)
    if args.dry_run:
        return
    started = datetime.now(timezone.utc).isoformat()
    write_json(args.output/"batch.json", dict(started=started, status="running", command=sys.argv))
    stop = threading.Event()
    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = {}
            for c in cases:
                status = Path(c["directory"])/"runner_status.json"
                if args.resume and status.exists() and json.loads(status.read_text())["status"] == "complete" and (Path(c["directory"])/"summary.json").exists():
                    results.append(json.loads(status.read_text()))
                    continue
                work = dict(c, command=c["command"]+(["--resume"] if args.resume else []))
                tasks[pool.submit(execute, work, args, stop)] = c
            try:
                for task in as_completed(tasks):
                    results.append(task.result())
                    write_json(args.output/"manifest.json", results)
            except BaseException:
                stop.set()
                raise
    finally:
        write_json(args.output/"manifest.json", results)
        write_json(args.output/"batch.json", dict(started=started, ended=datetime.now(timezone.utc).isoformat(),
                   status="complete" if len(results) == len(cases) and all(r["status"] == "complete" for r in results) else "incomplete"))
        summarize(cases, args.output)


if __name__ == "__main__":
    main()
