"""Recreate the Li et al. li10 scaling plot for Werner and real-X backends.

Run from the repository root:
    python -m src.scripts.benchmark_scaling

The original caption in improve_complexity.tex specifies swap-only nesting,
N=2**n+1 nodes, p_gen=0.1, w0=1, t_coh=500/p_swap**(n-1), and
cutoff=floor(42/p_swap**(n-1)). The default p_swap values are 0.2, 0.5, and
0.9, with a Werner and an X-state curve for each probability. Color identifies
p_swap; marker and line style identify the backend. For a custom sweep use,
for example, --p-swap 0.3 0.7. A single --p-swap 0.5 still gives two curves.
The original generator survives in git at 989d0e8:plot_paper_new.py (the
__main__ block); plot_paper_new.py was later removed in commit e6f287a.

Each point uses one common horizon with >=99% coverage in BOTH backends.
Horizon calibration warms the implementations and is excluded from timing.
The real-X run includes per-node Pauli noise and, by default, nonzero ADC.
Swap-only all-corrected protocols have state-independent success probability,
so their PMFs must agree even though their state/noise models differ.
Communication, including cutoff-abort heralding, is enabled by default.
Use --no-classical-communication for the original paper's timing assumption.
If a probability reaches --max-horizon before the coverage target, its sweep
stops at the last covered point and the remaining probabilities continue.
Existing checkpoints are resumed by default. Completed points and individual
timing samples are reused; --no-resume starts a new measurement set. To extend
a stopped sweep without changing the time grid, increase --max-horizon, e.g.
    python -m src.scripts.benchmark_scaling --max-horizon 16777216
To redraw existing measurements without any simulations:
    python -m src.scripts.benchmark_scaling --plot-only
After upgrading the Werner retry engine, retain saved X-state samples and
retime only Werner (the old Werner samples are kept as provenance):
    python -m src.scripts.benchmark_scaling --refresh-werner
"""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
import csv
import gc
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import scipy

from src.core.pauli_adc.state import RealXState
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.core.werner.state import WernerState


BACKENDS = ("Werner", "X-state")
WERNER_RETRY_ENGINE = "shared-full-reset-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


class HorizonCoverageError(RuntimeError):
    """The configured horizon limit cannot cover the next chain size."""


def point_key(record):
    return float(record["p_swap"]), int(record["nesting_level"])


def load_checkpoint(output_dir, settings):
    """Reuse compatible measurements, allowing larger sweeps/horizon limits."""
    path = output_dir / "computation_time.json"
    if not settings["resume"] or not path.exists():
        return [], [], []
    document = json.loads(path.read_text())
    previous = document["settings"]
    for name in ("p_gen", "adc_factor", "classical_communication", "padding_factor", "coverage_target"):
        if previous.get(name) != settings[name]:
            raise ValueError(
                f"Checkpoint {name}={previous.get(name)!r} differs from requested {settings[name]!r}. "
                "Use a different --output-dir or --no-resume to start a new measurement set."
            )
    records = document["records"]
    pending = document.get("pending_records", [])
    for collection in (records, pending):
        seen = set()
        for record in collection:
            # Also accept the earlier single-probability checkpoint format.
            record.setdefault("p_swap", record["parameters"]["Werner"]["p_swap"])
            key = point_key(record)
            if key in seen:
                raise ValueError(f"Duplicate checkpoint point {key}")
            seen.add(key)
            expected = case_parameters(key[1], settings["p_gen"], key[0], settings["adc_factor"])
            if record["parameters"] != json.loads(json.dumps(expected)):
                raise ValueError(f"Checkpoint parameters do not match point {key}")
            for label in BACKENDS:
                samples = record["samples_seconds"][label]
                if any(not math.isfinite(value) or value < 0 for value in samples):
                    raise ValueError(f"Invalid checkpoint timing samples for {key}, {label}")
                if record["coverage"][label] < settings["coverage_target"]:
                    raise ValueError(f"Checkpoint point {key} does not meet the coverage target")
    stopped = previous.get("stopped_sweeps", [])
    for record in stopped:
        record.setdefault("max_horizon", previous["max_horizon"])
    if previous.get("werner_retry_engine", "legacy-nested") != settings.get("werner_retry_engine", WERNER_RETRY_ENGINE):
        if not settings.get("refresh_werner", False):
            raise ValueError(
                "Checkpoint Werner timings predate the shared-retry optimization. "
                "Use --refresh-werner to repeat only Werner timings and reuse X-state samples, "
                "or --plot-only to draw the historical results."
            )
        # Keep old Werner samples as provenance. Cases become pending until
        # their new Werner samples are measured; X-state samples and horizons
        # remain available even if the refresh is interrupted or run in parts.
        work = {point_key(r): r for r in records}
        work.update({point_key(r): r for r in pending})
        refreshed = []
        for record in work.values():
            updated = dict(record)
            updated["previous_werner_timing"] = {
                "engine": previous.get("werner_retry_engine", "legacy-nested"),
                "samples_seconds": list(record["samples_seconds"]["Werner"]),
                "median_seconds": record.get("median_seconds", {}).get("Werner"),
            }
            updated["reused_x_state_environment"] = record.get("reused_x_state_environment", document.get("environment"))
            updated["samples_seconds"] = {"Werner": [], "X-state": list(record["samples_seconds"]["X-state"])}
            updated.pop("median_seconds", None)
            refreshed.append(updated)
        return [], refreshed, stopped
    return records, pending, stopped


def case_parameters(level, p_gen, p_swap, adc_factor):
    """Use the paper's level-dependent hardware and cutoff parameters."""
    scale = p_swap ** (level - 1)
    t_coh = 500.0 / scale
    shared = {
        "protocol": (0,) * level,
        "p_gen": p_gen,
        "p_swap": p_swap,
        "cut_type": "memory_time",
        "mt_cut": int(42.0 / scale),
    }
    return {
        "Werner": dict(shared, w0=1.0, t_coh=t_coh),
        "X-state": dict(
            shared,
            lambdas=[1.0, 0.0, 0.0, 0.0],
            # Werner's scalar coherence is for the entire pair. Each of the
            # two endpoints contributes half that Pauli mode-decay rate.
            pauli_mode_decay_rates=1.0 / (2.0 * t_coh),
            amplitude_damping_rate=adc_factor / t_coh,
            bell_outcomes="all_corrected",
        ),
    }


def evaluate(simulator, parameters, horizon, padding_factor):
    # The legacy Werner FFT quotient needs padding to suppress wraparound.
    # The real-X formal-series solver manages its own causal FFT sizes.
    simulator.zero_padding_size = 1 << (padding_factor * horizon - 1).bit_length()
    return simulator.nested_protocol(dict(parameters, t_trunc=horizon))


def calibrate_horizon(simulators, parameters, initial, maximum, target, padding_factor):
    """Double the shared horizon until both PMFs cover the requested mass."""
    horizon = initial
    history = []
    while True:
        outputs = {
            label: evaluate(simulators[label], parameters[label], horizon, padding_factor)
            for label in BACKENDS
        }
        coverage = {label: float(outputs[label][0].sum()) for label in BACKENDS}
        for label in BACKENDS:
            pmf = outputs[label][0]
            if not np.all(np.isfinite(pmf)) or np.min(pmf) < -1e-10 or coverage[label] > 1.0 + 1e-8:
                raise RuntimeError(f"{label} produced an invalid PMF at T={horizon}")
        history.append({"horizon": horizon, "coverage": coverage})
        if min(coverage.values()) >= target:
            discrepancy = float(np.max(np.abs(outputs["Werner"][0] - outputs["X-state"][0])))
            # This compares the scheduling calculation, not state fidelity:
            # ADC changes real-X fidelity but cannot affect corrected swap PMFs.
            np.testing.assert_allclose(
                outputs["Werner"][0], outputs["X-state"][0], atol=1e-9, rtol=1e-7,
                err_msg="Backend PMFs disagree; increase --padding-factor if FFT wraparound is material",
            )
            return horizon, coverage, discrepancy, history
        if horizon == maximum:
            raise HorizonCoverageError(
                f"Coverage target {target:.6f} not reached at --max-horizon={maximum}: {coverage}. "
                "Increase --max-horizon or reduce --max-level."
            )
        # Release large intermediate arrays before evaluating the next horizon.
        del outputs
        horizon = min(2 * horizon, maximum)


def measure(simulators, parameters, horizon, repeats, padding_factor, samples=None, on_sample=None):
    """Append missing samples only, checkpointing each completed evaluation."""
    samples = {label: list(samples[label]) if samples else [] for label in BACKENDS}
    for repeat in range(repeats):
        order = BACKENDS if repeat % 2 == 0 else BACKENDS[::-1]
        for label in order:
            if len(samples[label]) > repeat:
                continue
            gc.collect()
            start = time.perf_counter()
            result = evaluate(simulators[label], parameters[label], horizon, padding_factor)
            samples[label].append(time.perf_counter() - start)
            del result
            if on_sample is not None:
                on_sample(samples)
    return samples


def write_checkpoint(output_dir, records, settings, pending_records=()):
    """Atomically save completed points and samples from unfinished points."""
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "settings": settings,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "method": {
            "werner_retry_engine": settings.get("werner_retry_engine", WERNER_RETRY_ENGINE),
            "reference": "Li et al. (2021), improve_complexity.tex, li10.pdf",
            "original_generator": "989d0e8:plot_paper_new.py, __main__ block; removed in e6f287a",
            "x_axis": "number of nodes = 2**nesting_level + 1",
            "y_axis": "median wall time of a complete nested_protocol evaluation in seconds",
            "calibration_timed": False,
            "common_horizon": True,
            "plot_encoding": "color=p_swap; marker and line style=backend",
            "pauli_adc_order": "pauli_after_adc",
            "noise_comparison": "Pauli link decay matches Werner when adc_factor=0; nonzero ADC changes states, not corrected swap PMFs",
        },
        "records": records,
        "pending_records": list(pending_records),
    }
    temporary = output_dir / "computation_time.json.tmp"
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_dir / "computation_time.json")


def save_results(output_dir, records, settings, pending_records=()):
    """Save samples and two backend curves for every measured p_swap."""
    write_checkpoint(output_dir, records, settings, pending_records)
    render_results(output_dir, records, settings)


def render_results(output_dir, records, settings):
    """Regenerate figures and CSV without changing the measurement checkpoint."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import cycler
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    import seaborn as sns

    with (output_dir / "computation_time.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["p_swap", "backend", "nesting_level", "nodes", "horizon", "coverage", "median_seconds", "samples_seconds"])
        for record in records:
            for label in BACKENDS:
                writer.writerow([
                    record["p_swap"], label, record["nesting_level"], record["nodes"], record["horizon"],
                    record["coverage"][label], record["median_seconds"][label],
                    json.dumps(record["samples_seconds"][label]),
                ])

    nodes = sorted({record["nodes"] for record in records})
    # Preserve other cached probability curves when resuming a subset.
    probabilities = list(dict.fromkeys(settings["p_swap"] + [r["p_swap"] for r in records]))
    probabilities = [p for p in probabilities if any(r["p_swap"] == p for r in records)]
    palette = sns.color_palette("colorblind")
    backend_styles = {"Werner": ("o", "-"), "X-state": ("s", "--")}
    # Nature research-journal artwork guide: 88 mm single column, 5–7 pt text.
    # https://www.nature.com/documents/NRJs-guide-to-preparing-final-artwork.pdf
    with plt.rc_context({
        "axes.prop_cycle": cycler(color=palette),
        "font.family": "sans-serif", "font.size": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": None,
    }):
        fig, ax = plt.subplots(figsize=(88 / 25.4, 66 / 25.4), constrained_layout=True)
        probability_handles = []
        for index, p_swap in enumerate(probabilities):
            color = palette[index % len(palette)]
            group = sorted((r for r in records if r["p_swap"] == p_swap), key=lambda r: r["nodes"])
            for label in BACKENDS:
                marker, linestyle = backend_styles[label]
                ax.loglog([r["nodes"] for r in group], [r["median_seconds"][label] for r in group],
                          marker=marker, linestyle=linestyle, linewidth=1.0, markersize=3,
                          color=color, label=f"{label}, p_swap={p_swap:g}")
            probability_handles.append(Line2D(
                [0], [0], color=color, linewidth=1.0,
                label=rf"$p_{{\mathrm{{swap}}}}={p_swap:g}$",
            ))
        ax.set_xticks(nodes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r"Number of nodes $2^n+1$")
        ax.set_ylabel("Median computation time (s)")
        probability_legend = ax.legend(handles=probability_handles, loc="upper left", frameon=False)
        ax.add_artist(probability_legend)
        backend_handles = [
            Line2D([0], [0], color="0.25", marker=backend_styles[label][0],
                   linestyle=backend_styles[label][1], linewidth=1.0, markersize=3, label=label)
            for label in BACKENDS
        ]
        ax.legend(handles=backend_handles, loc="lower right", frameon=False)
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"computation_time.{suffix}", dpi=300)
        plt.close(fig)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--min-level", type=int, default=2)
    parser.add_argument("--max-level", type=int, default=9)
    parser.add_argument("--p-gen", type=float, default=0.1)
    parser.add_argument("--p-swap", type=float, nargs="+", default=[0.2, 0.5, 0.9],
                        help="Swap probabilities to compare (default: 0.2 0.5 0.9).")
    parser.add_argument("--adc-factor", type=float, default=1.0,
                        help="Real-X per-node ADC rate times t_coh; use 0 for the matched Werner physical submodel.")
    parser.add_argument("--coverage", type=float, default=0.99)
    parser.add_argument("--initial-horizon", type=int, default=256)
    parser.add_argument("--max-horizon", type=int, default=4_194_304)
    parser.add_argument("--padding-factor", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True,
                        help="Reuse saved points and samples in --output-dir (default: enabled).")
    parser.add_argument("--refresh-werner", action="store_true",
                        help="Upgrade legacy checkpoints by retiming only Werner; retain X-state samples and old Werner provenance.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate PNG/PDF/CSV from the saved JSON using its settings; run no simulations.")
    parser.add_argument("--classical-communication", action=BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/benchmarks/chain_scaling")
    args = parser.parse_args(argv)
    if args.plot_only:
        path = args.output_dir / "computation_time.json"
        try:
            document = json.loads(path.read_text())
            records, settings = document["records"], document["settings"]
            if not records:
                raise ValueError("checkpoint contains no completed measurements")
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"Cannot plot checkpoint {path}: {error}")
        render_results(args.output_dir, records, settings)
        print(f"Regenerated PNG, PDF, and CSV from {len(records)} saved points in {args.output_dir}; no simulations run.")
        return
    if not 1 <= args.min_level <= args.max_level:
        parser.error("require 1 <= --min-level <= --max-level")
    if not 0 < args.p_gen <= 1 or any(not 0 < p <= 1 for p in args.p_swap):
        parser.error("--p-gen and every --p-swap value must lie in (0,1]")
    if len(set(args.p_swap)) != len(args.p_swap):
        parser.error("--p-swap values must be distinct")
    if not 0 < args.coverage < 1:
        parser.error("--coverage must lie in (0,1)")
    if not math.isfinite(args.adc_factor) or args.adc_factor < 0:
        parser.error("--adc-factor must be finite and nonnegative")
    if args.repeats < 1 or args.padding_factor < 2:
        parser.error("--repeats must be positive and --padding-factor at least 2")
    if not 2 <= args.initial_horizon <= args.max_horizon:
        parser.error("require 2 <= --initial-horizon <= --max-horizon")

    # Low-coverage calibration is expected and reported in the saved history.
    logging.basicConfig(level=logging.ERROR)
    simulators = {
        "Werner": RepeaterChainEvaluation(state_type=WernerState, classical_communication=args.classical_communication),
        "X-state": RepeaterChainEvaluation(state_type=RealXState, classical_communication=args.classical_communication),
    }
    settings = vars(args).copy()
    settings["output_dir"] = str(args.output_dir)
    settings["coverage_target"] = settings.pop("coverage")
    settings["werner_retry_engine"] = WERNER_RETRY_ENGINE
    try:
        records, pending_records, stopped = load_checkpoint(args.output_dir, settings)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(f"Cannot resume checkpoint: {error}")
    settings["stopped_sweeps"] = stopped
    completed = {point_key(record): record for record in records}
    pending = {point_key(record): record for record in pending_records}
    print(f"Loaded {len(records)} completed points and {len(pending)} unfinished points.", flush=True)
    print(f"{'p_swap':>7} {'n':>2} {'nodes':>7} {'horizon':>10} {'coverage W/X':>19} {'Werner (s)':>13} {'X-state (s)':>13}", flush=True)
    for p_swap in args.p_swap:
        horizon_hint = args.initial_horizon
        for level in range(args.min_level, args.max_level + 1):
            key = (p_swap, level)
            cached = completed.get(key)
            if cached is not None and key not in pending and min(len(cached["samples_seconds"][label]) for label in BACKENDS) >= args.repeats:
                print(f"Reusing p_swap={p_swap:g}, n={level}, nodes={cached['nodes']}, horizon={cached['horizon']}", flush=True)
                horizon_hint = min(cached["horizon"], args.max_horizon)
                continue
            parameters = case_parameters(level, args.p_gen, p_swap, args.adc_factor)
            work = pending.get(key, cached)
            if work is None:
                previous_stop = next((r for r in settings["stopped_sweeps"] if point_key(r) == key), None)
                if previous_stop is not None:
                    limit = previous_stop["max_horizon"]
                    if args.max_horizon <= limit:
                        print(f"p_swap={p_swap:g}, n={level} already reached horizon {limit}; increase --max-horizon to continue.", flush=True)
                        break
                    horizon_hint = max(horizon_hint, min(2 * limit, args.max_horizon))
                try:
                    horizon, coverage, discrepancy, history = calibrate_horizon(
                        simulators, parameters, horizon_hint, args.max_horizon, args.coverage, args.padding_factor
                    )
                except HorizonCoverageError as error:
                    settings["stopped_sweeps"] = [r for r in settings["stopped_sweeps"] if point_key(r) != key]
                    settings["stopped_sweeps"].append({
                        "p_swap": p_swap, "nesting_level": level,
                        "max_horizon": args.max_horizon, "reason": str(error),
                    })
                    print(f"Stopping p_swap={p_swap:g} before n={level}: {error}", flush=True)
                    write_checkpoint(args.output_dir, records, settings, pending.values())
                    break
                work = {
                    "p_swap": p_swap, "nesting_level": level, "nodes": 2**level + 1,
                    "horizon": horizon, "parameters": parameters, "coverage": coverage,
                    "max_pmf_discrepancy": discrepancy, "calibration": history,
                    "samples_seconds": {label: [] for label in BACKENDS},
                }
            else:
                # Warm compiled code on a tiny grid; do not repeat the saved
                # large-horizon calibration or its successful timing samples.
                if work["horizon"] > args.max_horizon:
                    raise ValueError(f"Resuming {key} requires --max-horizon >= {work['horizon']}")
                if min(len(work["samples_seconds"][label]) for label in BACKENDS) < args.repeats:
                    for label in BACKENDS:
                        if len(work["samples_seconds"][label]) < args.repeats:
                            evaluate(simulators[label], parameters[label], min(64, work["horizon"]), args.padding_factor)
            pending[key] = work
            settings["stopped_sweeps"] = [r for r in settings["stopped_sweeps"] if point_key(r) != key]
            write_checkpoint(args.output_dir, records, settings, pending.values())

            def checkpoint_samples(samples):
                work["samples_seconds"] = samples
                write_checkpoint(args.output_dir, records, settings, pending.values())

            horizon = work["horizon"]
            coverage = work["coverage"]
            samples = measure(simulators, parameters, horizon, args.repeats, args.padding_factor,
                              samples=work["samples_seconds"], on_sample=checkpoint_samples)
            medians = {label: statistics.median(samples[label]) for label in BACKENDS}
            result = dict(work, samples_seconds=samples, median_seconds=medians)
            if cached is not None:
                records[records.index(cached)] = result
            else:
                records.append(result)
            completed[key] = result
            del pending[key]
            print(f"{p_swap:7g} {level:2d} {2**level+1:7d} {horizon:10d} "
                  f"{coverage['Werner']:.6f}/{coverage['X-state']:.6f} "
                  f"{medians['Werner']:13.6f} {medians['X-state']:13.6f}", flush=True)
            # Keep completed measurements and the plot if a later, larger case
            # reaches the configured horizon/resource limit.
            save_results(args.output_dir, records, settings, pending.values())
            horizon_hint = horizon
    if not records:
        raise RuntimeError("No case reached the coverage target; increase --max-horizon or reduce --min-level.")
    save_results(args.output_dir, records, settings, pending.values())
    missing = [
        (p, n) for p in args.p_swap for n in range(args.min_level, args.max_level + 1)
        if (p, n) not in completed
        or min(len(completed[p, n]["samples_seconds"][label]) for label in BACKENDS) < args.repeats
    ]
    if missing:
        print(f"{len(missing)} requested points remain unfinished: {missing}. Increase --max-horizon to extend stopped sweeps.", flush=True)
    print(f"Saved PNG, PDF, CSV, and JSON to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
