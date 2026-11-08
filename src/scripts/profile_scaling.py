"""Profile completed scaling cases without recalibrating their saved horizons.

One case:
    python -m src.scripts.profile_scaling --p-swap 0.9 --min-level 9 --max-level 9
All completed cases in computation_time.json (completed profiles are reused):
    python -m src.scripts.profile_scaling --repeats 3

The original timing checkpoint is read only. Each case gets fresh uninstrumented
reference timings, then one cProfile run per backend with disjoint phase timers.
Phase times describe instrumented runs, not the plotted benchmark times.
Raw .prof files, JSON, CSV, and a Markdown summary are written beside the input.
Only saved, completed cases are profiled; missing larger cases are not launched.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
import cProfile
import csv
from functools import wraps
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import scipy

from src.core import repeater_algorithm as algorithm
from src.core.pauli_adc import swap_join_sequential_efficient as real_x
from src.core.werner import renewal as werner_renewal
from src.scripts import benchmark_scaling as benchmark


PHASES = {
    "probability_join": "Scalar cutoff acceptance/rejection kernels (Werner) or rejection kernel (X-state).",
    "legacy_scalar_renewals": "Werner's four scalar renewals per level: two PMF and two state-weighted; includes repeated FFT denominators.",
    "state_join": "Werner probability-weighted state join, including waiting noise.",
    "waiting_noise_aggregation": "X-state finite-window filtering and channel contractions, including the trace coordinate.",
    "quantum_map": "X-state vectorized bilinear swap map, including its trace/success mass.",
    "attempt_validation": "X-state input validation and physicality checks.",
    "attempt_assembly": "Remaining X-state attempt assembly, including communication aging and success/failure accounting.",
    "shared_retry_resolvent": "One scalar inverse series (1-F)^(-1) per level, shared by all weighted output columns.",
    "batched_renewal_application": "Apply the shared resolvent to two Werner or six X-state columns, including completion PMF.",
    "renewal_validation": "Weighted-renewal assembly, probability/state checks, and roundoff handling.",
    "state_normalization": "X-state final conditioning on completion time and physicality checks.",
    "orchestration": "Remaining evaluator work, generation, communication shifts, and allocation/deallocation.",
}


class PhaseTimer:
    """Subtract nested instrumented intervals so phase totals never overlap."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = []
        self.renewal_widths = []

    def wrap(self, function, phase):
        @wraps(function)
        def timed(*args, **kwargs):
            name = phase(*args, **kwargs) if callable(phase) else phase
            if name == "batched_renewal_application":
                first = np.asarray(args[0])
                self.renewal_widths.append(1 if first.ndim == 1 else first.shape[1])
            frame = [0.0]
            self.stack.append(frame)
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self.stack.pop()
                self.seconds[name] += elapsed - frame[0]
                self.calls[name] += 1
                if self.stack:
                    self.stack[-1][0] += elapsed
        return timed

    @contextmanager
    def instrument(self):
        def werner_phase(*args, **kwargs):
            return "probability_join" if kwargs["evaluate_func"] == "1" else "state_join"

        targets = [
            (algorithm, "werner_join_efficient", werner_phase),
            (algorithm.RepeaterChainEvaluation, "iterative_convolution", "legacy_scalar_renewals"),
            (algorithm, "real_x_attempt_join_sequential_efficient", "attempt_assembly"),
            (real_x, "_validate_attempt_inputs", "attempt_validation"),
            (real_x, "_cutoff_failure_kernel", "probability_join"),
            (real_x, "_aggregate_sequential_state_sum", "waiting_noise_aggregation"),
            (real_x, "_bilinear_operation", "quantum_map"),
            (algorithm, "renewal_sum_sequential_efficient_weighted", "renewal_validation"),
            (real_x, "_prepare_renewal", "shared_retry_resolvent"),
            (real_x, "_convolve_with_resolvent", "batched_renewal_application"),
            (algorithm, "renew_werner", "renewal_validation"),
            (werner_renewal, "_prepare_renewal", "shared_retry_resolvent"),
            (werner_renewal, "_convolve_with_resolvent", "batched_renewal_application"),
            (algorithm, "normalize_weighted_x_func", "state_normalization"),
        ]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for owner, name, phase in targets:
                stack.enter_context(patch.object(owner, name, self.wrap(getattr(owner, name), phase)))
            yield


def implementation_digest():
    """Fingerprint timed code; report formatting and file paths do not affect it."""
    digest = hashlib.sha256()
    paths = sorted((benchmark.REPO_ROOT / "src/core").rglob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(benchmark.REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    # Increment if the evaluation/warmup protocol in profile_case changes.
    digest.update(b"profile-timing-v1:tiny-warmup:separate-reference:cprofile-disjoint-phases")
    for function in (PhaseTimer.wrap, PhaseTimer.instrument, benchmark.evaluate, benchmark.measure):
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()


def profile_case(record, settings, repeats, output_dir):
    parameters = record["parameters"]
    horizon, padding = record["horizon"], settings["padding_factor"]
    simulators = {
        label: algorithm.RepeaterChainEvaluation(
            state_type=state, classical_communication=settings["classical_communication"])
        for label, state in zip(benchmark.BACKENDS, (benchmark.WernerState, benchmark.RealXState))
    }
    # Compile the same signatures before measuring; never repeat calibration.
    for label in benchmark.BACKENDS:
        benchmark.evaluate(simulators[label], parameters[label], min(64, horizon), padding)
    samples = benchmark.measure(simulators, parameters, horizon, repeats, padding)
    result = {key: record[key] for key in ("p_swap", "nesting_level", "nodes", "horizon", "parameters")}
    result["backends"] = {}
    pmfs = {}
    raw_dir = output_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    for label in benchmark.BACKENDS:
        timer, profiler = PhaseTimer(), cProfile.Profile()
        with timer.instrument():
            output = profiler.runcall(timer.wrap(benchmark.evaluate, "orchestration"),
                                     simulators[label], parameters[label], horizon, padding)
        pmfs[label], states = output
        del output, states
        coverage = float(pmfs[label].sum())
        if not np.isfinite(coverage) or coverage < settings["coverage_target"]:
            raise ValueError(f"{label} profiling case no longer meets saved coverage target: {coverage}")
        level = record["nesting_level"]
        if label == "X-state":
            for phase in ("probability_join", "shared_retry_resolvent", "batched_renewal_application"):
                if timer.calls[phase] != level:
                    raise AssertionError(f"Expected one {phase} per level; got {timer.calls[phase]} for {level} levels")
            if timer.renewal_widths != [6] * level:
                raise AssertionError(f"Expected six columns per shared renewal: {timer.renewal_widths}")
        else:
            if (timer.calls["shared_retry_resolvent"] != level
                    or timer.calls["batched_renewal_application"] != level
                    or timer.renewal_widths != [2] * level):
                raise AssertionError("Expected one shared two-column Werner renewal per level")
        slug = f"p_swap_{record['p_swap']:g}_n_{level}_{label.lower().replace('-', '_')}"
        raw_path = raw_dir / f"{slug}.prof"
        profiler.dump_stats(str(raw_path))
        stats = pstats.Stats(profiler)
        top = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:60]
        total = sum(timer.seconds.values())
        result["backends"][label] = {
            "reference_samples_seconds": samples[label],
            "reference_median_seconds": statistics.median(samples[label]),
            "original_benchmark_median_seconds": record["median_seconds"][label],
            "profiled_seconds": total, "coverage": coverage,
            "phases": {name: {"seconds": timer.seconds[name], "fraction": timer.seconds[name] / total,
                               "calls": timer.calls[name]} for name in timer.seconds},
            "renewal_column_counts": timer.renewal_widths,
            "cprofile_file": str(raw_path.relative_to(output_dir)),
            "top_functions_by_cumulative_time": [
                {"file": key[0], "line": key[1], "name": key[2], "primitive_calls": value[0],
                 "calls": value[1], "self_seconds": value[2], "cumulative_seconds": value[3]}
                for key, value in top
            ],
        }
        print(f"  {label}: reference {statistics.median(samples[label]):.6f} s, "
              f"profiled {total:.6f} s, coverage {coverage:.9f}", flush=True)
    np.testing.assert_allclose(pmfs["Werner"], pmfs["X-state"], atol=1e-9, rtol=1e-7)
    result["max_pmf_discrepancy"] = float(np.max(np.abs(pmfs["Werner"] - pmfs["X-state"])))
    return result


def save_report(folder, document):
    temporary = folder / "profiling.json.tmp"
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(folder / "profiling.json")
    with (folder / "profiling.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["p_swap", "nodes", "horizon", "backend", "phase", "calls", "seconds", "reference_median_seconds"])
        for case in document["records"]:
            for backend, data in case["backends"].items():
                for phase, values in data["phases"].items():
                    writer.writerow([case["p_swap"], case["nodes"], case["horizon"], backend,
                                     phase, values["calls"], values["seconds"],
                                     data["reference_median_seconds"]])
    lines = ["# Scaling evaluator profiles", "",
             "Fresh reference times exclude instrumentation and JIT warmup. Phase times come from one instrumented "
             "run per backend and sum without overlap; they are not the plotted benchmark times. "
             "Nested phase intervals are subtracted from their parents. Raw cProfile cumulative times overlap and must not be summed.", "",
             "These are swap-only, memory-cutoff, all-corrected Bell measurements. Their timing kernels are state-independent; "
             "this conclusion does not extend to state-dependent distillation/postselection or fidelity cutoffs. "
             "The X-state trace (completion mass) is propagated jointly with five state coordinates. "
             "A shared scalar retry resolvent avoids rebuilding it per coordinate, but applying it to six columns still costs work. "
             "The optimized Werner backend uses the same causal FFT retry arithmetic for two columns. "
             "The two backends retain different state joins and noise models, so this is not a controlled test of coefficient count alone.", ""]
    for case in document["records"]:
        lines += [f"## p_swap={case['p_swap']:g}, {case['nodes']} nodes, T={case['horizon']}", "",
                  f"Maximum absolute PMF difference between backends: {case['max_pmf_discrepancy']:.3e}.", ""]
        for backend, data in case["backends"].items():
            lines += [f"### {backend}", "",
                      f"Uninstrumented median: {data['reference_median_seconds']:.6f} s; "
                      f"instrumented total: {data['profiled_seconds']:.6f} s.", "",
                      "| Phase | Calls | Seconds |", "|---|---:|---:|"]
            for name, values in sorted(data["phases"].items(), key=lambda item: item[1]["seconds"], reverse=True):
                lines.append(f"| {name} | {values['calls']} | {values['seconds']:.6f} |")
            lines.append("")
    lines += ["## Phase definitions", ""] + [f"- `{name}`: {text}" for name, text in PHASES.items()]
    (folder / "profiling.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=benchmark.REPO_ROOT / "results/benchmarks/chain_scaling/computation_time.json")
    parser.add_argument("--output-dir", type=Path, help="Default: the input checkpoint's directory.")
    parser.add_argument("--p-swap", type=float, nargs="+")
    parser.add_argument("--min-level", type=int)
    parser.add_argument("--max-level", type=int)
    parser.add_argument("--repeats", type=int, default=1, help="Fresh uninstrumented samples per backend (default: 1).")
    parser.add_argument("--overwrite", action="store_true", help="Reprofile selected cases even if already saved.")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    source = json.loads(args.input.read_text())
    records = [r for r in source["records"]
               if (args.p_swap is None or r["p_swap"] in args.p_swap)
               and (args.min_level is None or r["nesting_level"] >= args.min_level)
               and (args.max_level is None or r["nesting_level"] <= args.max_level)]
    if not records:
        parser.error("No completed benchmark points match the selection")
    folder = args.output_dir or args.input.parent
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "profiling.json"
    document = json.loads(path.read_text()) if path.exists() else {"records": []}
    document.update({"source_checkpoint": str(args.input.resolve()), "phase_definitions": PHASES,
                     "method": "Tiny untimed warmup; separate uninstrumented samples; one cProfile plus disjoint phase-timer run per backend.",
                     "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                                     "platform": platform.platform(), "logical_cpu_count": os.cpu_count()}})
    settings = source["settings"]
    digest = implementation_digest()
    logging.basicConfig(level=logging.ERROR)
    for record in records:
        key = benchmark.point_key(record)
        fingerprint_data = {"implementation": digest, "parameters": record["parameters"],
                            "horizon": record["horizon"], "padding_factor": settings["padding_factor"],
                            "classical_communication": settings["classical_communication"],
                            "coverage_target": settings["coverage_target"]}
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
        previous = next((r for r in document["records"] if benchmark.point_key(r) == key), None)
        if (not args.overwrite and previous is not None and previous["fingerprint"] == fingerprint
                and all(len(previous["backends"][b]["reference_samples_seconds"]) >= args.repeats for b in benchmark.BACKENDS)):
            print(f"Reusing profile p_swap={key[0]:g}, n={key[1]}", flush=True)
            continue
        print(f"Profiling p_swap={key[0]:g}, n={key[1]}, nodes={record['nodes']}, T={record['horizon']}", flush=True)
        result = profile_case(record, settings, args.repeats, folder)
        result.update({"fingerprint": fingerprint, "settings": settings, "environment": document["environment"]})
        document["records"] = [r for r in document["records"] if benchmark.point_key(r) != key] + [result]
        document["records"].sort(key=benchmark.point_key)
        save_report(folder, document)
    print(f"Saved profiling.json, profiling.csv, profiling.md and *.prof in {folder}", flush=True)


if __name__ == "__main__":
    main()
