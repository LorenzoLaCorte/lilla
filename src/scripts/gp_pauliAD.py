"""Run and summarize independent overnight Pauli/AD optimization studies."""
from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def cases(args):
    if args.preset == "hardware":
        from src.scripts.pauli_ad_hardware import cases as hardware_cases
        return hardware_cases(args)
    studies = ("postselection", "protocol") if args.study == "both" else (args.study,)
    nodes = args.nodes or ([3] if args.preset == "smoke" else [3, 5, 9])
    lifetimes = args.t1 or ([75.] if args.preset == "smoke" else [25., 74., 200., 600.])
    seeds = args.seeds or ([7] if args.preset == "smoke" else [7, 42])
    plan = []
    for study in studies:
        for node in nodes:
            for t1 in lifetimes:
                for q in ((.5, .25) if study == "postselection" else (None,)):
                    for seed in seeds:
                        name = f"{study}_N{node}_T1_{t1:g}_q_{q if q is not None else 'opt'}_seed{seed}"
                        directory = args.output / name
                        command = [sys.executable, "-m", f"src.optimization.{'optimize_postselection' if study == 'postselection' else 'gp_pauliAD'}",
                                   "--nodes", str(node), "--t1-slots", str(t1), "--t2-slots", str(t1*args.t2_ratio),
                                   "--p-gen", str(args.p_gen), "--p-swap", str(args.p_swap),
                                   "--p-distillation", str(args.p_distillation),
                                   "--initial-horizon", str(args.initial_horizon), "--max-horizon", str(args.max_horizon),
                                   "--gp-calls", str(min(2, args.gp_calls) if args.preset == "smoke" else args.gp_calls),
                                   "--initial-points", str(min(2, args.initial_points) if args.preset == "smoke" else args.initial_points),
                                   "--seed", str(seed), "--output", str(directory)]
                        if study == "postselection":
                            command += ["--q", str(q), "--method", args.post_method, "--max-exact", str(args.max_exact)]
                        else:
                            command += ["--max-dists", str(0 if args.preset == "smoke" else args.max_dists), "--q-min", str(args.q_min)]
                        if args.resume:
                            command += ["--resume"]
                        plan.append(dict(name=name, study=study, nodes=node, t1=t1, q=q, seed=seed,
                                         directory=str(directory), command=command))
    return plan


def execute(case, args, stop=None):
    stop = stop or threading.Event()
    if stop.is_set():
        return dict(name=case["name"], status="cancelled", returncode=None, elapsed_seconds=0.)
    directory = Path(case["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    print(f"START {case['name']} (log: {directory / 'console.log'})", flush=True)
    started = time.perf_counter()
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    env.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(args.output / ".cache"))
    with (directory / "console.log").open("a") as log:
        process = subprocess.Popen(case["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.perf_counter()+args.case_timeout
            while process.poll() is None and not stop.is_set() and time.perf_counter() < deadline:
                try:
                    process.wait(timeout=min(1., max(.001, deadline-time.perf_counter())))
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code, status = process.returncode, "interrupted" if stop.is_set() else "timeout"
            else:
                code = process.returncode
                status = "complete" if code == 0 else "failed"
        except BaseException:
            process.terminate()
            process.wait()
            raise
    result = dict(name=case["name"], status=status, returncode=code,
                  elapsed_seconds=time.perf_counter()-started)
    write_json(directory / "runner_status.json", result)
    print(f"{status.upper()} {case['name']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    return result


def summarize(plan, output):
    if plan and "section" in plan[0]:
        from src.scripts.pauli_ad_hardware import summarize as hardware_summary
        return hardware_summary(plan, output)
    import csv
    rows, comparisons, matched_rows = [], [], []
    for case in plan:
        path = Path(case["directory"]) / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for label, record in summary.get("matched_controls", {}).items():
            matched_rows.append(dict(name=case["name"], control=label,
                status=record["result"]["status"], skr=record["result"].get("skr"),
                q=record["candidate"]["q"], policies=json.dumps(record["candidate"]["policies"]),
                protocol=json.dumps(record["candidate"]["protocol"])))
        stage_rates = {stage: result["best"]["result"]["skr"] for stage, result in summary["results"].items()
                       if result["best"] is not None}
        for stage, reference in (("level", "uniform"), ("node", "uniform")) if case["study"] == "postselection" else (
                ("joint", "bell_all"), ("joint", "bell_policy"), ("joint", "q_all")):
            if stage in stage_rates and reference in stage_rates:
                rate, baseline = stage_rates[stage], stage_rates[reference]
                comparisons.append(dict(name=case["name"], nodes=case["nodes"], t1=case["t1"],
                    seed=case["seed"], target=stage, baseline=reference, skr=rate, baseline_skr=baseline,
                    absolute_gain=rate-baseline, relative_gain=rate/baseline-1 if baseline > 0 else None,
                    activates_key=rate > 0 and baseline == 0))
        for stage, result in summary["results"].items():
            best = result["best"]
            row = {k: case[k] for k in ("name", "study", "nodes", "t1", "q", "seed")}
            row.update(stage=stage, status="converged" if best else "no_valid_result",
                       certified=result.get("certified", False),
                       failed_requests=result.get("failed_requests", result.get("distinct_candidates", 0)-result.get("converged_candidates", 0)),
                       links=case["nodes"]-1)
            if best:
                row.update({k: best["result"][k] for k in ("skr", "skf", "mean_wait", "coverage", "horizon")})
                row.update(q=best["candidate"]["q"], policies=json.dumps(best["candidate"]["policies"]),
                           protocol=json.dumps(best["candidate"]["protocol"]))
            rows.append(row)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "summary.csv.tmp").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.csv.tmp").replace(output / "summary.csv")
    if comparisons:
        with (output / "comparisons.csv.tmp").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        (output / "comparisons.csv.tmp").replace(output / "comparisons.csv")
    if matched_rows:
        with (output / "matched_controls.csv.tmp").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matched_rows[0]))
            writer.writeheader()
            writer.writerows(matched_rows)
        (output / "matched_controls.csv.tmp").replace(output / "matched_controls.csv")
    lines = ["Pauli/AD batch results", "Nodes = links + 1. Rates are bits/slot and include destructive postselection retries.",
             "See manifest.json for job completion; trials.jsonl includes invalid/unconverged candidates.",
             "Only exhaustive searches with all candidates converged certify optimality.",
             "Repeated seeds measure search variability; they are not independent physical measurements.", ""]
    for row in rows:
        lines.append(f"{row['name']} {row['stage']}: SKR={row.get('skr', 'missing')}, q={row['q']}, "
                     f"wait={row.get('mean_wait', 'missing')}, certified={row['certified']}")
    lines += ["", "Interpretation: compare node/level with uniform at fixed q and hardware;",
              "compare joint with bell_all, bell_policy and q_all at fixed hardware and seed.",
              "A zero baseline makes percentage gain undefined; inspect absolute gain and key activation."]
    for comparison in comparisons:
        relative = comparison["relative_gain"]
        gain = f"{100*relative:.3g}%" if relative is not None else "undefined (zero baseline)"
        lines.append(f"{comparison['name']}: {comparison['target']} vs {comparison['baseline']}: "
                     f"gain={gain}; absolute={comparison['absolute_gain']:.6g}; activates key={comparison['activates_key']}")
    (output / "summary.txt").write_text("\n".join(lines)+"\n")
    try:
        plot_summary(rows, output)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        # Presentation errors must not interrupt collection of completed jobs.
        write_json(output / "plot_status.json", dict(status="failed", error=str(exc)))
        print(f"WARNING: batch tables saved, but plotting failed: {exc}", file=sys.stderr, flush=True)
    else:
        write_json(output / "plot_status.json", dict(status="complete"))


def recover_manifest(plan, output):
    """Recover job completion after a parent/plotting failure, without evaluations."""
    previous = output / "manifest.json"
    saved = {r["name"]: r for r in json.loads(previous.read_text())} if previous.exists() else {}
    for case in plan:
        status = Path(case["directory"]) / "runner_status.json"
        if status.exists():
            saved[case["name"]] = json.loads(status.read_text())
    records = [saved[case["name"]] for case in plan if case["name"] in saved]
    write_json(previous, records)
    return records


def plot_summary(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    table = pd.DataFrame(rows)
    table.to_json(output / "summary-data.json", orient="records", indent=2)
    for study in table.study.unique():
        data = table[(table.study == study) & (table.status == "converged")]
        if data.empty:
            continue
        nodes = sorted(data.nodes.unique())
        fig, axes = plt.subplots(2, len(nodes), figsize=(4*len(nodes), 6), squeeze=False,
                                 constrained_layout=True)
        for column, node in enumerate(nodes):
            sub = data[data.nodes == node]
            # A positive T1 domain exists even if every SKF is zero and the q
            # panel has no observations. Empty pandas plots can otherwise leave
            # invalid (zero/negative) limits for Matplotlib's log locator.
            for ax in axes[:, column]:
                ax.set_xlim(float(sub.t1.min())*.8, float(sub.t1.max())*1.25)
            group_keys = ["stage", "q"] if study == "postselection" else ["stage"]
            for key, group in sub.groupby(group_keys):
                label = ", ".join(map(str, key if isinstance(key, tuple) else (key,)))
                # Median and seed range of best-found rates, not a statistical confidence interval.
                grouped = group.groupby("t1").skr.agg(["median", "min", "max"])
                axes[0, column].plot(grouped.index, grouped["median"], ".-", label=label)
                axes[0, column].fill_between(grouped.index, grouped["min"], grouped["max"], alpha=.12)
            axes[0, column].set(title=f"{node-1} links ({node} nodes)", xscale="log", ylabel="Best SKR (bits/slot)")
            axes[0, column].legend(fontsize=7)
            if study == "protocol":
                for stage, group in sub.groupby("stage"):
                    q = group[group.skf > 1e-9].groupby("t1").q.median()
                    if not q.empty:
                        axes[1, column].plot(q.index, q, ".-", label=stage)
                axes[1, column].set(ylabel="Winning q (positive key only)", ylim=(0, .52))
                if not (sub.skf > 1e-9).any():
                    axes[1, column].text(.5, .5, "No positive-key result", ha="center",
                                         transform=axes[1, column].transAxes)
            else:
                for q, group in sub.groupby("q"):
                    pivot = group.pivot_table(index=["t1", "seed"], columns="stage", values="skr")
                    if {"uniform", "node"}.issubset(pivot.columns):
                        gain = (pivot["node"]-pivot["uniform"]).groupby("t1").median()
                        axes[1, column].plot(gain.index, gain, ".-", label=f"q={q:g}")
                axes[1, column].set(ylabel="Per-swap minus uniform SKR")
            axes[1, column].set(xlabel="T1 (slots)", xscale="log")
            if axes[1, column].get_legend_handles_labels()[0]:
                axes[1, column].legend(fontsize=7)
        fig.savefig(output / f"{study}.png", dpi=170)
        fig.savefig(output / f"{study}.pdf")
        plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "overnight", "hardware"), default="smoke")
    parser.add_argument("--study", choices=("postselection", "protocol", "both"), default="both")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--nodes", type=int, nargs="+", default=None)
    parser.add_argument("--t1", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--gp-calls", type=int, default=100)
    parser.add_argument("--initial-points", type=int, default=12)
    parser.add_argument("--max-dists", type=int, default=None)
    parser.add_argument("--q-min", type=float, default=.1)
    parser.add_argument("--post-method", choices=("auto", "exhaustive", "gp"), default="auto")
    parser.add_argument("--max-exact", type=int, default=243)
    parser.add_argument("--p-gen", type=float, default=.2)
    parser.add_argument("--p-swap", type=float, default=1.)
    parser.add_argument("--p-distillation", type=float, default=1.)
    parser.add_argument("--t2-ratio", type=float, default=2., help="T2/T1; 2 means pure AD")
    parser.add_argument("--initial-horizon", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--sections", nargs="+", choices=("placement", "distillation", "asymmetry", "source"),
                        default=["placement", "distillation", "asymmetry", "source"], help="Hardware follow-up questions")
    parser.add_argument("--hardware-regimes", nargs="+", choices=("C", "D"), default=["C", "D"])
    parser.add_argument("--memory-profiles", nargs="+", choices=("network", "benchmark"), default=["network"])
    parser.add_argument("--relaxation-models", nargs="+", choices=("ad", "unital"), default=["ad"],
                        help="Hardware preset: ground-state AD assumption or unbiased T1 relaxation control")
    parser.add_argument("--asymmetry-geometries", nargs="+", choices=("slow_link", "weak_memory", "bottleneck", "left_comb"),
                        default=["slow_link", "weak_memory"])
    parser.add_argument("--memory-scales", nargs="+", type=float, default=[.25, .5, 1., 2., 4.])
    parser.add_argument("--source-dampings", nargs="+", type=float, default=[.05, .15],
                        help="Per-qubit AD probabilities at source delivery; fresh and benchmark-mixed controls are also tested")
    parser.add_argument("--max-total-dists", type=int, default=2, help="Hardware preset: cap total rounds over all tree vertices")
    parser.add_argument("--q-grid", nargs="+", type=float, default=[.1, .2, .3, .4, .45, .475, .49, .5])
    parser.add_argument("--workers", type=int, default=1, help="Independent subprocesses, each using one BLAS/Numba thread")
    parser.add_argument("--case-timeout", type=float, default=3600., help="Seconds per job; timed-out journals remain resumable")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.max_dists = args.max_dists if args.max_dists is not None else (2 if args.preset == "hardware" else 1)
    args.initial_horizon = args.initial_horizon or (16384 if args.preset == "hardware" else 256)
    args.max_horizon = args.max_horizon or (1048576 if args.preset == "hardware" else 262144)
    if args.workers < 1 or args.case_timeout <= 0 or not 0 < args.t2_ratio <= 2:
        parser.error("workers and timeout must be positive; T2/T1 must lie in (0,2]")
    if (any(not 0 < v < float("inf") for v in args.memory_scales)
            or any(not 0 < d < 1 for d in args.source_dampings)
            or any(not args.q_min <= q <= .5 for q in args.q_grid)
            or args.max_total_dists < 0):
        parser.error("invalid hardware memory scales, source damping, q grid or distillation cap")
    if args.output is None:
        args.output = ROOT / "results" / "pauliAD" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output / ".cache"))
    plan = cases(args)
    if args.summarize_only:
        saved = args.output / "plan.json"
        if not saved.exists():
            parser.error("--summarize-only needs an existing batch directory with plan.json")
        saved_plan = json.loads(saved.read_text())
        recover_manifest(saved_plan, args.output)
        summarize(saved_plan, args.output)
        return 0
    if args.dry_run:
        print(f"{len(plan)} jobs; GP budget is per ablation, plus baseline anchors. Each job timeout: {args.case_timeout:g}s.")
        for case in plan:
            print(shlex.join(case["command"]))
        return 0
    if (args.output / "plan.json").exists() and not args.resume:
        parser.error("batch already exists; use --resume or a new output directory")
    if args.resume and (args.output / "plan.json").exists():
        saved = json.loads((args.output / "plan.json").read_text())
        strip_resume = lambda p: [{**c, "command": [a for a in c["command"] if a != "--resume"]} for c in p]
        if strip_resume(saved) != strip_resume(plan):
            parser.error("resume plan differs; repeat the original study parameters")
    write_json(args.output / "plan.json", plan)
    manifest = []
    print(f"Running {len(plan)} jobs; results in {args.output}", flush=True)
    stop, interrupted = threading.Event(), False
    def collect(future, case):
        if future.cancelled():
            manifest.append(dict(name=case["name"], status="cancelled"))
        else:
            try:
                manifest.append(future.result())
            except Exception as exc:
                manifest.append(dict(name=case["name"], status="runner_failed", error=str(exc)))
        write_json(args.output / "manifest.json", manifest)
        summarize(plan, args.output)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, case, args, stop): case for case in plan}
        try:
            for future in as_completed(futures):
                collect(future, futures[future])
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
            for future in futures:
                future.cancel()
            print("Stopping workers; flushed trial journals can be resumed.", flush=True)
    if interrupted:
        recorded = {r["name"] for r in manifest}
        for future, case in futures.items():
            if case["name"] not in recorded:
                collect(future, case)
        return 130
    print(f"Read {args.output / 'summary.txt'} and the two study plots.", flush=True)
    return 0 if all(c["status"] == "complete" for c in manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
