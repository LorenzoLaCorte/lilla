"""Resumable physical-performance comparison of four real-X distillation maps.

    python -m src.scripts.compare_distillation_strategies --preset extensive --time-budget-hours 4
    python -m src.scripts.compare_distillation_strategies --plot-only

Each strategy searches the same finite family of balanced nested protocols
(all placements of up to K distillations) and shared memory cutoffs. This is
an exhaustive finite-family comparison, not a global protocol optimum.
Checkpoints are written after each evaluation. A time budget stops between
evaluations; repeating the command resumes. Horizons double on the original
integer grid until both coverage and metric convergence pass. Failed or
under-covered candidates never become zero-rate physical observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import itertools
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import time
import traceback

import numpy as np

from src.core.pauli_adc import DISTILLATION_STRATEGIES, RealXState
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.utils.utility_functions import get_mean_waiting_time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "distillation_strategies"
LABELS = dict(bbpssw="BBPSSW", bell_diagonal="Bell-diagonal twirl",
              none="No twirl", dejmps="DEJMPS")
PRESETS = {
    "smoke": dict(levels=[1], visibilities=[.95], source_q=[.3],
                  source_families=["biased_x"], noise_models=["pauli_adc"],
                  noise_strengths=[.02], p_gen=[.2], p_swap=[.8],
                  communication=[True], cutoff_factors=[2., None], max_distillations=1),
    "extensive": dict(levels=[1, 2, 3], visibilities=[.85, .95], source_q=[.2, .35, .5],
                      source_families=["werner", "bell_diagonal", "biased_x"],
                      noise_models=["isotropic_pauli", "anisotropic_pauli", "pauli_adc"],
                      noise_strengths=[.005, .02, .08], p_gen=[.05, .2], p_swap=[.5, .9],
                      communication=[False, True], cutoff_factors=[.5, 2., 8., None],
                      max_distillations=2),
    "focused": dict(levels=[1, 2, 3], visibilities=[.85, .95], source_q=[.2, .35],
                    source_families=["werner", "bell_diagonal", "biased_x"],
                    noise_models=["isotropic_pauli", "anisotropic_pauli", "pauli_adc"],
                    noise_strengths=[.005, .02, .08], p_gen=[.05, .2], p_swap=[.5, .9],
                    communication=[False, True], cutoff_factors=[2., 8., None],
                    max_distillations=2, scenario_design="focused-v1"),
    "pure_pe": dict(levels=[1, 2], visibilities=[1.], source_q=[.25, .5],
                    source_families=["biased_x"],
                    noise_models=["isotropic_pauli", "anisotropic_pauli", "pauli_adc"],
                    noise_strengths=[0., .02], p_gen=[.2], p_swap=[.9],
                    communication=[False, True], cutoff_factors=[2., None],
                    max_distillations=2, scenario_design="pure-pe-v1"),
}


def stable_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def protocol_candidates(levels, max_distillations):
    """Every ordering with exactly levels swaps and at most K distillations."""
    for count in range(max_distillations + 1):
        for positions in itertools.combinations(range(levels + count), count):
            selected = set(positions)
            yield tuple(int(index in selected) for index in range(levels + count))


def source_coordinates(family, visibility, q):
    """Werner, anisotropic Bell, or white-noise-mixed Schmidt-biased source."""
    if family == "werner":
        return [1., 0., 0., visibility, visibility, -visibility]
    if family == "bell_diagonal":
        # Same Phi+ fidelity as the Werner source at this visibility, but
        # distribute the residual weight unequally over Psi+,Psi-,Phi-.
        a = (1 + 3*visibility) / 4
        b, c, d = (1-a)*np.array([.7, .2, .1])
        return [1., 0., 0., float(a-b-c+d), float(a+b-c-d), float(-a+b-c+d)]
    coherence = visibility * 2 * math.sqrt(q*(1-q))
    bias = visibility * (1-2*q)
    return [1., bias, bias, visibility, coherence, -coherence]


def scenarios(settings):
    sources = []
    for family in settings["source_families"]:
        for visibility in settings["visibilities"]:
            for q in settings["source_q"] if family == "biased_x" else [None]:
                sources.append(dict(source_family=family, visibility=visibility, source_q=q))
    cases = []
    for source, level, noise, strength, p_gen, p_swap, communication in itertools.product(
        sources, settings["levels"], settings["noise_models"], settings["noise_strengths"],
        settings["p_gen"], settings["p_swap"], settings["communication"],
    ):
        case = dict(source, levels=level, noise_model=noise, noise_strength=strength,
                    p_gen=p_gen, p_swap=p_swap, communication=communication)
        if settings.get("scenario_design") == "focused-v1" and not focused_scenario(case):
            continue
        if settings.get("scenario_design") == "pure-pe-v1" and strength == 0. and noise != "isotropic_pauli":
            # At zero strength the three channel labels are the same channel.
            continue
        cases.append(case)
    np.random.default_rng(settings["seed"]).shuffle(cases)
    return cases


def focused_scenario(case):
    """Fixed mechanism/control design, independent of observed key rates.

    72 main five-node cases, 24 chain-size controls, 24 communication controls,
    and 24 hardware controls. q=1/2 biased-X sources duplicate Werner sources
    at equal white-noise visibility and are therefore omitted.
    """
    main_hardware = case["p_gen"] == .2 and case["p_swap"] == .9
    central_noise = case["noise_strength"] == .02
    central_visibility = case["visibility"] == .95
    if main_hardware and case["communication"]:
        return case["levels"] == 2 or (
            case["levels"] in (1, 3) and central_noise and central_visibility
        )
    if case["levels"] != 2 or not central_noise:
        return False
    if main_hardware and not case["communication"]:
        return True
    return central_visibility and case["communication"] and (
        (case["p_gen"] == .05 and case["p_swap"] == .9)
        or (case["p_gen"] == .2 and case["p_swap"] == .5)
    )


def candidate_tasks(case, settings):
    """Keep each four-strategy comparison together, with no-distillation first."""
    case_id = stable_id(case)
    for protocol in protocol_candidates(case["levels"], settings["max_distillations"]):
        for factor in settings["cutoff_factors"]:
            cutoff = (np.iinfo(np.int32).max if factor is None
                      else max(0, math.ceil(factor/case["p_gen"])))
            group = dict(scenario_id=case_id, protocol=list(protocol), cutoff=int(cutoff))
            for strategy in DISTILLATION_STRATEGIES:
                task = dict(group, strategy=strategy, scenario=case)
                task["task_id"] = stable_id(dict(group, strategy=strategy))
                task["group_id"] = stable_id(group)
                yield task


def simulation_parameters(task):
    case = task["scenario"]
    scale = case["noise_strength"] * case["p_gen"]
    rates = [scale]*3 if case["noise_model"] == "isotropic_pauli" else [
        scale, 1.4*scale, .8*scale,
    ]
    return dict(
        protocol=task["protocol"], p_gen=case["p_gen"], p_swap=case["p_swap"],
        p_distillation=1., cut_type="memory_time", mt_cut=task["cutoff"],
        real_x_coordinates=source_coordinates(
            case["source_family"], case["visibility"], case["source_q"],
        ),
        pauli_mode_decay_rates=rates,
        amplitude_damping_rate=2*scale if case["noise_model"] == "pauli_adc" else 0.,
        bell_outcomes="all_corrected", distillation_strategy=task["strategy"],
    )


def physical_metrics(pmf, states):
    """Pooled BB84, with actual XX/ZZ errors rather than a fidelity surrogate."""
    coverage = float(np.sum(pmf))
    if coverage <= 0:
        return dict(coverage=coverage, mean_time=None, generation_rate=None, fidelity=None,
                    e_x=None, e_z=None, secret_fraction=None, key_rate=None, pooled_state=None)
    average = np.sum(pmf[:, None]*states, axis=0) / coverage
    e_x, e_z = np.clip((1-average[[4, 3]])/2, 0., 1.)
    def h2(value):
        return 0. if value in (0., 1.) else -value*math.log2(value)-(1-value)*math.log2(1-value)
    fraction = max(0., 1-h2(e_x)-h2(e_z))
    mean_time = float(get_mean_waiting_time(pmf, extrapolation=False))
    return dict(coverage=coverage, mean_time=mean_time, generation_rate=1/mean_time,
                fidelity=float((1+average[3]+average[4]-average[5])/4),
                e_x=float(e_x), e_z=float(e_z), secret_fraction=float(fraction),
                key_rate=float(fraction/mean_time), pooled_state=average.tolist())


def metrics_converged(previous, current, settings):
    if previous is None or previous["mean_time"] is None or current["mean_time"] is None:
        return False
    if abs(previous["mean_time"]/current["mean_time"]-1) > settings["metric_rtol"]:
        return False
    return all(abs(previous[name]-current[name]) <= settings["metric_atol"]
               for name in ("fidelity", "e_x", "e_z", "secret_fraction"))


def evaluate_task(task, settings, max_horizon, previous=None):
    params = simulation_parameters(task)
    evaluator = RepeaterChainEvaluation(
        state_type=RealXState, efficient=True, use_fft=True,
        classical_communication=task["scenario"]["communication"],
    )
    attempts = list(previous.get("attempts", [])) if previous else []
    prior_metrics = attempts[-1]["metrics"] if attempts else None
    horizon = attempts[-1]["horizon"]*2 if attempts else settings["initial_horizon"]
    started = time.perf_counter()
    result = dict(task, parameters=params, attempts=attempts, status="horizon_limited")
    while horizon <= max_horizon:
        pmf, states = evaluator.nested_protocol(dict(params, t_trunc=horizon))
        metrics = physical_metrics(pmf, states)
        attempts.append(dict(horizon=horizon, metrics=metrics))
        result.update(horizon=horizon, metrics=metrics)
        if metrics["coverage"] >= settings["coverage"] and metrics_converged(
            prior_metrics, metrics, settings
        ):
            result["status"] = "complete"
            break
        prior_metrics = metrics
        horizon *= 2
    result["seconds"] = (previous.get("seconds", 0.) if previous else 0.) + time.perf_counter()-started
    return result


def physics_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "core").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    for function in (source_coordinates, simulation_parameters, physical_metrics,
                     metrics_converged, evaluate_task):
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def load_records(path, *, repair=True):
    """Read a snapshot, optionally repairing an interrupted final write.

    Plotting a live sweep uses repair=False: ignore an unfinished final line
    without changing the checkpoint that the evaluator is still appending to.
    """
    records = {}
    if not path.exists():
        return records
    lines = path.read_bytes().splitlines(keepends=True)
    offset = 0
    truncated = False
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines)-1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt checkpoint at line {index+1} of {path}")
            # Only a truncated, unterminated final line can be discarded.
            if repair:
                with path.open("r+b") as stream:
                    stream.truncate(offset)
            truncated = True
            break
        records[record["task_id"]] = record
        offset += len(line)
    if repair and lines and not truncated and not lines[-1].endswith(b"\n"):
        # A complete last JSON object can lose only its newline on interruption.
        with path.open("ab") as stream:
            stream.write(b"\n")
    return records


def append_record(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def import_records(sources, tasks, records, path, metadata):
    """Copy compatible selected tasks; leave every source checkpoint untouched."""
    expected = {task["task_id"]: task for task in tasks}
    source_rows = []
    imported = 0
    numerical_keys = ("initial_horizon", "coverage", "metric_rtol", "metric_atol")
    for source in sources:
        source = source.resolve()
        if source == path.parent:
            raise ValueError("--reuse-from must differ from --output-dir")
        saved = json.loads((source / "settings.json").read_text())
        if saved.get("schema_version") != metadata["schema_version"] or saved.get(
            "physics_digest"
        ) != metadata["physics_digest"]:
            raise ValueError(f"Cannot reuse {source}: different physics implementation")
        if any(saved["settings"].get(key) != metadata["settings"][key] for key in numerical_keys):
            raise ValueError(f"Cannot reuse {source}: different numerical tolerances/time grid")
        # A live source may be partway through its final write. Never repair it.
        saved_rows = load_records(source / "evaluations.jsonl", repair=False)
        source_rows.extend(saved_rows.values())
        for task_id, row in saved_rows.items():
            task = expected.get(task_id)
            if task is None:
                continue
            if row["scenario"] != task["scenario"] or row["parameters"] != simulation_parameters(task):
                raise ValueError(f"Cannot reuse {source}: parameter mismatch for {task_id}")
            previous = records.get(task_id)
            if previous and (previous["status"] == "complete" or (
                row["status"] != "complete" and previous.get("horizon", 0) >= row.get("horizon", 0)
            )):
                continue
            copied = dict(row, imported_from=str(source))
            append_record(path, copied)
            records[task_id] = copied
            imported += 1
    return imported, source_rows


def timing_bucket(row):
    case = row["scenario"]
    cutoff = None if row["cutoff"] == np.iinfo(np.int32).max else row["cutoff"]
    return (case["levels"], case["p_gen"], case["p_swap"], case["communication"],
            row["protocol"].count(1), cutoff)


def plan_summary(tasks, records, reference_rows, max_horizon):
    costs = {}
    seen = set()
    for row in list(reference_rows) + list(records.values()):
        if row["task_id"] in seen:
            continue
        seen.add(row["task_id"])
        if row.get("seconds", 0.) >= 0:
            costs.setdefault(timing_bucket(row), []).append(row.get("seconds", 0.))
    all_costs = [v for values in costs.values() for v in values]
    fallback = statistics.mean(all_costs) if all_costs else None
    remaining = 0
    seconds = 0.
    complete = limited = errors = 0
    for task in tasks:
        row = records.get(task["task_id"])
        if row and row["status"] == "complete":
            complete += 1
        elif row and row["status"] == "error":
            errors += 1
        elif row and row.get("horizon", 0) >= max_horizon:
            limited += 1
        else:
            remaining += 1
            values = costs.get(timing_bucket(task))
            seconds += statistics.mean(values) if values else (fallback or 0.)
    return dict(scenarios=len({t["scenario_id"] for t in tasks}), evaluations=len(tasks),
                cached_evaluations=sum(t["task_id"] in records for t in tasks),
                complete=complete, horizon_limited=limited, errors=errors,
                remaining_evaluations=remaining,
                estimated_remaining_seconds=seconds if all_costs else None,
                estimate_note="Heuristic from saved geometry/hardware/cutoff timings; not a deadline guarantee.")


def matched_best(records):
    """Optimize each strategy only over groups completed by all four strategies."""
    groups = {}
    for record in records.values():
        groups.setdefault(record["group_id"], {})[record["strategy"]] = record
    best, counts = {}, {}
    for group in groups.values():
        if set(group) != set(DISTILLATION_STRATEGIES) or any(
            row["status"] != "complete" for row in group.values()
        ):
            continue
        case_id = next(iter(group.values()))["scenario_id"]
        counts[case_id] = counts.get(case_id, 0)+1
        for strategy, row in group.items():
            key = case_id, strategy
            if key not in best or row["metrics"]["key_rate"] > best[key]["metrics"]["key_rate"]:
                best[key] = row
    return best, counts


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_results(output, records, metadata):
    rows = []
    for row in records.values():
        metrics = row.get("metrics", {})
        rows.append(dict(task_id=row["task_id"], scenario_id=row["scenario_id"],
                         strategy=row["strategy"], status=row["status"],
                         protocol=json.dumps(row["protocol"]), cutoff=row["cutoff"],
                         horizon=row.get("horizon"), seconds=row["seconds"],
                         **{k: metrics.get(k) for k in ("coverage", "mean_time", "generation_rate",
                            "fidelity", "e_x", "e_z", "secret_fraction", "key_rate")}))
    write_csv(output / "evaluations.csv", rows)
    best, counts = matched_best(records)
    best_rows = []
    settings = metadata["settings"]
    for (case_id, strategy), row in sorted(best.items()):
        total_candidates = len(list(protocol_candidates(row["scenario"]["levels"],
            settings["max_distillations"])))*len(settings["cutoff_factors"])
        best_rows.append(dict(scenario_id=case_id, strategy=strategy,
            matched_candidates=counts[case_id], requested_candidates=total_candidates,
            search_complete=counts[case_id] == total_candidates,
            protocol=json.dumps(row["protocol"]), cutoff=row["cutoff"],
            horizon=row["horizon"], **row["scenario"],
            **{k: row["metrics"][k] for k in ("key_rate", "mean_time", "generation_rate",
                 "fidelity", "e_x", "e_z", "secret_fraction", "coverage")}))
    write_csv(output / "best_policies.csv", best_rows)
    complete = sum(r["status"] == "complete" for r in records.values())
    fully_searched = sum(r["strategy"] == "bbpssw" and r["search_complete"]
                         for r in best_rows)
    statuses = {}
    for row in records.values():
        statuses[row["status"]] = statuses.get(row["status"], 0)+1
    atomic_json(output / "progress.json", dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        evaluations=len(records), statuses=statuses, matched_scenarios=len(counts),
        fully_searched_scenarios=fully_searched, matched_groups=sum(counts.values()),
        positive_key_scenarios={s: sum(r["strategy"] == s and r["key_rate"] > 0
                                      for r in best_rows) for s in DISTILLATION_STRATEGIES},
    ))
    report = ["# Distillation strategy comparison", "",
              f"Numerically complete evaluations: {complete}/{len(records)}.",
              f"Scenarios with matched candidate groups: {len(counts)}.", "",
              f"Scenarios with every requested candidate group numerically complete: {fully_searched}.", "",
              "Each strategy's best pooled asymptotic BB84 key rate is selected over the same",
              "completed protocol/cutoff candidates. Incomplete groups are excluded for all four.",
              "`search_complete` indicates whether every requested candidate was compared.",
              "Zero key rates are retained; horizon-limited/error cases are unresolved.", "",
              "Partial results and exclusions caused by numerical limits need not represent the",
              "whole requested grid. The summary bar plot counts positive best-found rates,",
              "highlighting cases where the matched BBPSSW search gives zero.", "",
              "The figure compares best rates; circles are positive/positive pairs, upward",
              "triangles mark positive strategy rates with zero BBPSSW rate, and downward",
              "triangles mark zero strategy rates with positive BBPSSW rate. Zero is displayed",
              "on a labelled plotting floor, not converted to a positive physical rate.", "",
              "## Model and numerical policy", "",
              "All gates, rotations, and twirls are ideal and instantaneous. All four maps",
              "accept 00/11 and discard failures; copies prepare independently in parallel.",
              "Additional distillation copies and full-reset retries are included in latency;",
              "no memory-count/cost constraint is optimized. No adaptive Bell-weight sorting",
              "or additional basis optimization is included. All-corrected BSM is fixed.",
              "Storage uses amplitude damping followed by Pauli noise, per endpoint.",
              "The sweep includes matched Pauli-only controls and communication on/off.",
              "Source visibility is white-noise visibility for biased-X states; Werner and",
              "anisotropic Bell sources have matched Phi+ fidelity at equal visibility.", "",
              "Rates use pooled XX/ZZ errors and the repository's finite-horizon mean-time",
              "estimate, without fitting/extrapolating a tail. Coverage and successive-horizon",
              "metric checks limit truncation sensitivity; they are not rigorous error bounds.", "",
              f"Coverage target: {settings['coverage']}; mean-time relative tolerance:",
              f"{settings['metric_rtol']}; absolute fidelity/error/secret-fraction tolerance:",
              f"{settings['metric_atol']}. Integer time bins are never coarsened.", "",
              "BBPSSW: https://arxiv.org/abs/quant-ph/9511027 (Eq. 4, A1–A3).",
              "DEJMPS: https://arxiv.org/abs/quant-ph/9604039 (Eqs. 2–8).", ""]
    (output / "summary.md").write_text("\n".join(report))
    plot_results(output, best)


def plot_results(output, best):
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import seaborn as sns
    palette = sns.color_palette("colorblind")
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(180/25.4, 68/25.4))
    all_positive = [r["metrics"]["key_rate"] for r in best.values() if r["metrics"]["key_rate"] > 0]
    floor = min(all_positive)/3 if all_positive else 1e-8
    ceiling = max(all_positive)*2 if all_positive else 1e-3
    for axis, strategy, color in zip(axes, DISTILLATION_STRATEGIES[1:], palette[1:4]):
        positive, gained, lost, zero = [], [], [], 0
        for (case, label), row in best.items():
            if label != strategy:
                continue
            x, y = best[case, "bbpssw"]["metrics"]["key_rate"], row["metrics"]["key_rate"]
            if x > 0 and y > 0:
                positive.append((x, y))
            elif y > 0:
                gained.append((floor, y))
            elif x > 0:
                lost.append((x, floor))
            else:
                zero += 1
        for pairs, marker, label in ((positive, "o", "Both positive"),
                                    (gained, "^", "BBPSSW zero"), (lost, "v", "Strategy zero")):
            if pairs:
                x, y = np.array(pairs).T
                axis.scatter(x, y, s=9, marker=marker, alpha=.6, color=color, label=label)
        axis.plot([floor, ceiling], [floor, ceiling], color=".5", lw=.7)
        axis.set(xscale="log", yscale="log", xlim=(floor/1.4, ceiling),
                 ylim=(floor/1.4, ceiling), xlabel="BBPSSW key rate", ylabel=LABELS[strategy]+" key rate")
        axis.text(.04, .96, f"Both zero: {zero}", transform=axis.transAxes, va="top", fontsize=6)
        axis.text(.04, .03, f"Zero display floor: {floor:.1e}",
                  transform=axis.transAxes, va="bottom", fontsize=5)
        if axis.collections:
            axis.legend(fontsize=5, loc="lower right")
    fig.tight_layout(pad=.7)
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("strategy_comparison."+suffix), dpi=220)
    plt.close(fig)

    # Complement the rate scatter with an explicit count of zero-to-positive
    # cases, which cannot be represented on logarithmic axes at their true zero.
    cases = {case for case, _ in best}
    shared, additional = [], []
    for strategy in DISTILLATION_STRATEGIES:
        common = extra = 0
        for case in cases:
            baseline = best[case, "bbpssw"]["metrics"]["key_rate"]
            candidate = best[case, strategy]["metrics"]["key_rate"]
            if candidate > 0:
                common += int(baseline > 0)
                extra += int(baseline == 0)
        shared.append(common)
        additional.append(extra)
    fig, axis = plt.subplots(figsize=(88/25.4, 68/25.4))
    positions = np.arange(len(DISTILLATION_STRATEGIES))
    axis.barh(positions, shared, color=palette[0], label="BBPSSW also positive")
    axis.barh(positions, additional, left=shared, color=palette[1], label="BBPSSW zero")
    for position, common, extra in zip(positions, shared, additional):
        axis.text(common+extra+max(1, len(cases)*.015), position, str(common+extra),
                  va="center", fontsize=7)
    axis.set(yticks=positions, yticklabels=[LABELS[s] for s in DISTILLATION_STRATEGIES],
             xlim=(0, max(1, len(cases))*1.15),
             xlabel=f"Scenarios with positive best-found key rate\n({len(cases)} matched scenarios so far)")
    axis.invert_yaxis()
    axis.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.01),
                frameon=False, borderaxespad=0.)
    fig.tight_layout(pad=.7)
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("strategy_summary."+suffix), dpi=220)
    plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--preset", choices=PRESETS, default="extensive")
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--plot-only", action="store_true")
    result.add_argument("--report-dir", type=Path,
                        help="Separate destination for --plot-only reports from a running sweep")
    result.add_argument("--time-budget-hours", type=float, default=None)
    result.add_argument("--max-evaluations", type=int, default=None)
    result.add_argument("--max-horizon", type=int, default=262144)
    result.add_argument("--initial-horizon", type=int, default=256)
    result.add_argument("--coverage", type=float, default=.999)
    result.add_argument("--metric-rtol", type=float, default=.002)
    result.add_argument("--metric-atol", type=float, default=.0002)
    result.add_argument("--seed", type=int, default=20261114)
    result.add_argument("--report-interval-seconds", type=float, default=300.)
    result.add_argument("--levels", nargs="+", type=int)
    result.add_argument("--max-distillations", type=int)
    result.add_argument("--reuse-from", nargs="+", type=Path, default=[],
                        help="Import matching evaluations from compatible existing study directories")
    result.add_argument("--prepare-only", action="store_true",
                        help="Prepare the study/import results and report costs without new evaluations")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    output = args.output_dir.resolve()
    if args.plot_only:
        # A purely read-only checkpoint load: do not repair or rewrite it.
        metadata = json.loads((output / "settings.json").read_text())
        records = load_records(output / "evaluations.jsonl", repair=False)
        report_dir = args.report_dir.resolve() if args.report_dir else output
        report_dir.mkdir(parents=True, exist_ok=True)
        export_results(report_dir, records, metadata)
        print(f"Regenerated reports and plots from {len(records)} saved evaluations in {report_dir}")
        return
    if args.report_dir is not None:
        raise ValueError("--report-dir requires --plot-only")
    if args.max_horizon < args.initial_horizon or args.initial_horizon < 2:
        raise ValueError("Require max-horizon >= initial-horizon >= 2")
    horizon_ratio = args.max_horizon // args.initial_horizon
    if horizon_ratio & (horizon_ratio-1) or args.max_horizon % args.initial_horizon:
        raise ValueError("max-horizon must equal initial-horizon times a power of two")
    if not 0 < args.coverage < 1 or args.metric_rtol <= 0 or args.metric_atol <= 0:
        raise ValueError("Require 0 < coverage < 1 and positive convergence tolerances")
    for name in ("time_budget_hours", "max_evaluations", "report_interval_seconds"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    settings = dict(PRESETS[args.preset], seed=args.seed, initial_horizon=args.initial_horizon,
                    coverage=args.coverage, metric_rtol=args.metric_rtol, metric_atol=args.metric_atol)
    if args.levels is not None:
        settings["levels"] = sorted(set(args.levels))
    if args.max_distillations is not None:
        settings["max_distillations"] = args.max_distillations
    if min(settings["levels"]) < 0 or settings["max_distillations"] < 0:
        raise ValueError("Levels and max-distillations must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema_version=1, physics_digest=physics_digest(), settings=settings,
                    environment=dict(python=platform.python_version(), numpy=np.__version__,
                                     platform=platform.platform()))
    settings_path = output / "settings.json"
    if settings_path.exists():
        saved = json.loads(settings_path.read_text())
        if any(saved.get(key) != metadata[key] for key in ("schema_version", "physics_digest", "settings")):
            raise ValueError("Checkpoint physics/settings differ; select another --output-dir. --plot-only still works.")
        metadata = saved
    else:
        atomic_json(settings_path, metadata)
    path = output / "evaluations.jsonl"
    records = load_records(path)
    cases = scenarios(settings)
    total = sum(len(list(protocol_candidates(c["levels"], settings["max_distillations"])))
                *len(settings["cutoff_factors"])*4 for c in cases)
    tasks = ([task for case in cases for task in candidate_tasks(case, settings)]
             if args.reuse_from or args.prepare_only or args.preset == "focused" else None)
    reference_rows = []
    if args.reuse_from:
        imported, reference_rows = import_records(args.reuse_from, tasks, records, path, metadata)
        print(f"Imported {imported} matching saved evaluations; source checkpoints unchanged.", flush=True)
    if tasks is not None:
        plan = plan_summary(tasks, records, reference_rows, args.max_horizon)
        atomic_json(output / "plan.json", plan)
        estimate = plan["estimated_remaining_seconds"]
        print(f"Remaining evaluations: {plan['remaining_evaluations']}; "
              f"cached complete: {plan['complete']}; unresolved at horizon: {plan['horizon_limited']}.", flush=True)
        if estimate is not None:
            print(f"Estimated remaining evaluation time: {estimate/3600:.2f} hours "
                  "(heuristic; horizons and machine load can change this).", flush=True)
    if args.prepare_only:
        export_results(output, records, metadata)
        print(f"Prepared {len(cases)} scenarios and {total} evaluations in {output}; no new simulations run.")
        return
    print(f"{len(cases)} scenarios; {total} strategy/candidate evaluations; {len(records)} cached.", flush=True)
    logging.getLogger().setLevel(logging.ERROR)
    started, completed = time.monotonic(), 0
    last_report = started
    deadline = started + args.time_budget_hours*3600 if args.time_budget_hours else math.inf
    stop = False
    baseline_cache = {r["group_id"]: r for r in records.values()
                      if 1 not in r["protocol"] and r["status"] == "complete"}
    try:
        for case in cases:
            for task in candidate_tasks(case, settings):
                previous = records.get(task["task_id"])
                if previous and (previous["status"] in ("complete", "error")
                                 or previous.get("horizon", 0) >= args.max_horizon):
                    continue
                if time.monotonic() >= deadline or (args.max_evaluations is not None
                                                    and completed >= args.max_evaluations):
                    stop = True
                    break
                try:
                    cached = baseline_cache.get(task["group_id"])
                    if 1 not in task["protocol"] and cached:
                        row = dict(cached, **task, parameters=simulation_parameters(task),
                                   seconds=0., reused_from=cached["task_id"])
                    else:
                        row = evaluate_task(task, settings, args.max_horizon, previous)
                except Exception as error:
                    row = dict(task, parameters=simulation_parameters(task), status="error", seconds=0.,
                               error=repr(error), traceback=traceback.format_exc())
                    print(f"ERROR {task['task_id']}: {error}", flush=True)
                append_record(path, row)
                records[task["task_id"]] = row
                if 1 not in task["protocol"] and row["status"] == "complete":
                    baseline_cache[task["group_id"]] = row
                completed += 1
                m = row.get("metrics", {})
                print(f"{len(records)}/{total} {task['strategy']:13} protocol={task['protocol']} "
                      f"C={task['cutoff']} {row['status']} T={row.get('horizon')} "
                      f"coverage={m.get('coverage')} SKR={m.get('key_rate')}", flush=True)
            # Raw progress is durable after every candidate. Avoid repeatedly
            # plotting/rewriting the whole study for cheap scenarios.
            if time.monotonic()-last_report >= args.report_interval_seconds:
                export_results(output, records, metadata)
                last_report = time.monotonic()
            if stop:
                break
    except KeyboardInterrupt:
        print("Interrupted; completed evaluations are saved. Repeat the command to resume.")
    export_results(output, records, metadata)
    if tasks is not None:
        atomic_json(output / "plan.json", plan_summary(tasks, records, reference_rows, args.max_horizon))
    print(f"Saved {len(records)} evaluations in {output}. Repeat the command to resume; --plot-only redraws.")


if __name__ == "__main__":
    main()
