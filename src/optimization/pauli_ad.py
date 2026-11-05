"""Shared, UI-independent evaluation and durable output for Pauli/AD searches."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
from skopt import Optimizer
from skopt.space import Space
from skopt.utils import cook_estimator

from src.core.pauli_adc.state import RealXState, polish_x_func
from src.core.pauli_adc.protocol_units import apply_local_channels, sequential_pauli_adc_channel
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.types.repeater_types import checkAsymProtocol
from src.utils.utility_functions import entropy, get_mean_waiting_time, secret_key_rate

POLICIES = ("all_corrected", "psi_pair", "phi_pair")
SCHEMA = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=lambda x: x.item() if isinstance(x, np.generic) else list(x))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:20]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json.loads(canonical(value)), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def broadcast(value, count, name):
    a = np.asarray(value, dtype=float).reshape(-1)
    if len(a) == 1:
        a = np.repeat(a, count)
    if len(a) != count:
        raise ValueError(f"{name} requires one value or {count} values")
    return a


@dataclass(frozen=True)
class Hardware:
    nodes: int
    p_gen: tuple
    p_swap: float
    p_distillation: float
    pauli_rates: tuple
    ad_rates: tuple
    visibility: float = 1.0
    cutoff: int | None = None
    source_damping: float = 0.0

    def __post_init__(self):
        if self.nodes < 3:
            raise ValueError("at least three nodes (two links) are required")
        p = np.asarray(self.p_gen)
        g, k = np.asarray(self.pauli_rates), np.asarray(self.ad_rates)
        if p.shape != (self.nodes-1,) or np.any(~np.isfinite(p)) or np.any((p <= 0) | (p > 1)):
            raise ValueError("p_gen must have one probability in (0,1] per link")
        if g.shape != (self.nodes, 3) or k.shape != (self.nodes,):
            raise ValueError("memory rates require one triple and one AD rate per node")
        if np.any(~np.isfinite(g)) or np.any(~np.isfinite(k)) or np.any(g < 0) or np.any(k < 0):
            raise ValueError("memory rates must be finite and nonnegative")
        if np.any(g.sum(axis=1)[:, None] - 2*g < -1e-14):
            raise ValueError("Pauli mode rates violate the Lindblad triangle inequalities")
        if not (0 < self.p_swap <= 1 and 0 < self.p_distillation <= 1 and 0 <= self.visibility <= 1):
            raise ValueError("hardware probabilities must be in (0,1], visibility in [0,1]")
        if self.cutoff is not None and self.cutoff < 1:
            raise ValueError("cutoff must be a positive integer")
        if not np.isfinite(self.source_damping) or not 0 <= self.source_damping <= 1:
            raise ValueError("source damping must lie in [0,1]")

    @property
    def links(self):
        return self.nodes-1


@dataclass(frozen=True)
class Numerics:
    initial_horizon: int = 256
    max_horizon: int = 262144
    coverage: float = .999999
    rtol: float = 1e-5
    atol: float = 1e-9

    def __post_init__(self):
        if self.initial_horizon < 8 or self.max_horizon < 2*self.initial_horizon:
            raise ValueError("max horizon must allow at least one doubled-horizon check")
        if not 0 < self.coverage < 1 or self.rtol <= 0 or self.atol <= 0:
            raise ValueError("invalid convergence tolerances")


def source(q, visibility=1., damping=0.):
    if not 0 < q <= .5:
        raise ValueError("source q must lie in (0,1/2]")
    if not np.isfinite(damping) or not 0 <= damping <= 1:
        raise ValueError("source damping must lie in [0,1]")
    coordinates = np.array([1, visibility*(1-2*q), visibility*(1-2*q), visibility,
                            visibility*2*np.sqrt(q*(1-q)), -visibility*2*np.sqrt(q*(1-q))])
    channel = sequential_pauli_adc_channel([1., 1., 1., 1.], 1-damping)
    return apply_local_channels(coordinates, channel, channel)


def fixed_protocol(links, shape="balanced"):
    def tree(lo, hi):
        if hi-lo == 1:
            return []
        middle = (lo+hi)//2 if shape == "balanced" else (hi-1 if shape == "left" else lo+1)
        return tree(lo, middle) + tree(middle, hi) + [f"s{middle-1}"]
    if shape not in ("balanced", "left", "right"):
        raise ValueError("unknown tree shape")
    return tuple(tree(0, links))


def protocol_nodes(protocol, links):
    """Map physical swap boundary to its output span, height, and tree depth."""
    checkAsymProtocol(tuple(protocol), links)
    segments = {i: {"span": [i, i+1], "height": 0, "children": []} for i in range(links)}
    swaps = {}
    for step in protocol:
        i = int(step[1:])
        if step[0] == "d":
            continue
        j = min(x for x in segments if x > i)
        left, right = segments.pop(i), segments.pop(j)
        node = {"span": [left["span"][0], right["span"][1]],
                "height": max(left["height"], right["height"])+1,
                "children": [left, right], "boundary": i}
        swaps[i] = node
        segments[j] = node
    def visit(node, depth):
        node["depth"] = depth
        for child in node["children"]:
            visit(child, depth+1)
    visit(next(iter(segments.values())), 0)
    return [{k: n[k] for k in ("boundary", "span", "height", "depth")} for _, n in sorted(swaps.items())]


def candidate(protocol, q, policies):
    return {"protocol": list(protocol), "q": float(q), "policies": list(policies)}


def objective(result):
    """Optimize SKR; guide zero-key searches by their unclipped BB84 margin."""
    if result["status"] != "converged":
        return 1.0
    if result["skr"] > 0:
        return -result["skr"]
    return 1e-6*(1-np.clip(result["bb84_margin"], -1, 0))


class Evaluation:
    def __init__(self, hardware, numerics):
        self.hardware, self.numerics = hardware, numerics
        self.backend = RepeaterChainEvaluation(state_type=RealXState, efficient=True,
                                               use_fft=True, use_gpu=False, w_twirling=False)

    def initial_coordinates(self, choice):
        return source(choice["q"], self.hardware.visibility, self.hardware.source_damping)

    def memory_rates(self):
        return np.asarray(self.hardware.pauli_rates), np.asarray(self.hardware.ad_rates)

    def operation_output(self, coordinates):
        """Prediction-model hook; the physical evaluator preserves all coordinates."""
        return coordinates

    def at_horizon(self, choice, horizon):
        h = self.hardware
        protocol, policies = choice["protocol"], choice["policies"]
        checkAsymProtocol(tuple(protocol), h.links)
        if len(policies) != h.links-1 or any(p not in POLICIES for p in policies):
            raise ValueError("one acceptance policy per physical swap boundary is required")
        initial = self.initial_coordinates(choice)
        cutoff = choice.get("cutoff", h.cutoff)
        if cutoff is not None and (isinstance(cutoff, bool) or int(cutoff) != cutoff or cutoff < 1):
            raise ValueError("candidate cutoff must be a positive integer or None")
        times = np.arange(horizon)
        segments = {}
        for i, probability in enumerate(h.p_gen):
            pmf = np.zeros(horizon)
            pmf[1:] = probability*(1-probability)**(times[1:]-1)
            segments[i] = (pmf, np.tile(initial, (horizon, 1)), i, i+1)
        rates, ad = self.memory_rates()
        for step in protocol:
            i = int(step[1:])
            left = segments[i]
            params = dict(p_gen=h.p_gen, p_swap=h.p_swap, p_distillation=h.p_distillation,
                          cutoff=cutoff if cutoff is not None else horizon,
                          cut_type="memory_time", real_x_werner_twirling=False)
            if step[0] == "s":
                j = min(x for x in segments if x > i)
                right = segments[j]
                endpoints = [left[2], right[2], right[3]]
                params.update(bell_outcomes=policies[i],
                              _real_x_node_pauli_rates=rates[endpoints],
                              _real_x_node_adc_rates=ad[endpoints])
                p, x = self.backend.compute_unit(params, *left[:2], *right[:2], unit_kind="swap")
                x = self.operation_output(x)
                del segments[i]
                segments[j] = (p, x, left[2], right[3])
            else:
                endpoints = [left[2], left[3]]
                params.update(_real_x_node_pauli_rates=rates[endpoints],
                              _real_x_node_adc_rates=ad[endpoints])
                p, x = self.backend.compute_unit(params, *left[:2], unit_kind="dist")
                x = self.operation_output(x)
                segments[i] = (p, x, left[2], left[3])
        p, x, _, _ = next(iter(segments.values()))
        return self.metrics(p, x), p, x

    @staticmethod
    def metrics(p, x):
        p, x = np.array(p, copy=True), np.asarray(x)
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(x)) or np.min(p) < -1e-10:
            raise ValueError("invalid/nonfinite backend output")
        if abs(p[0]) > 1e-10 or p.sum() > 1+1e-10:
            raise ValueError("invalid completion probability")
        p[0] = 0
        p[p < 0] = 0
        if p.sum() > 1:
            p /= p.sum()
        coverage = float(p.sum())
        if coverage <= 0:
            raise ValueError("no completion within horizon")
        populated = p > 0
        if np.any(abs(x[populated, 0]-1) > 1e-10):
            raise ValueError("unnormalized output states")
        polish_x_func(x[populated], atol=1e-10)  # Retain backend physicality validation.
        pooled = np.sum(p[:, None]*x, axis=0)/coverage
        ex, ez = (1-pooled[4])/2, (1-pooled[3])/2
        if not (-1e-10 <= ex <= 1+1e-10 and -1e-10 <= ez <= 1+1e-10):
            raise ValueError("invalid BB84 errors")
        ex, ez = float(np.clip(ex, 0, 1)), float(np.clip(ez, 0, 1))
        margin = float(1-entropy(ex)-entropy(ez))
        skf = max(0., margin)
        wait = float(get_mean_waiting_time(p, extrapolation=False))
        skr = float(secret_key_rate(p, x, state_type=RealXState, extrapolation=False))
        if not np.isclose(skr, skf/wait, rtol=1e-9, atol=1e-12):
            raise ValueError("independent SKR cross-check failed")
        return dict(skr=skr, skf=skf, mean_wait=wait, bb84_margin=margin,
                    coverage=coverage, eX=ex, eZ=ez,
                    fidelity=float((1+pooled[3]+pooled[4]-pooled[5])/4),
                    pooled_coordinates=pooled.tolist())

    def run(self, choice):
        n, horizon, previous, history = self.numerics, self.numerics.initial_horizon, None, []
        try:
            while True:
                result, _, _ = self.at_horizon(choice, horizon)
                history.append({"horizon": horizon, **result})
                keys = ("skr", "skf", "mean_wait", "eX", "eZ")
                if (previous is not None and min(previous["coverage"], result["coverage"]) >= n.coverage
                        and np.allclose([previous[k] for k in keys], [result[k] for k in keys],
                                        rtol=n.rtol, atol=n.atol)):
                    return dict(status="converged", horizon=horizon, history=history, **result)
                if horizon >= n.max_horizon:
                    reason = ("coverage" if result["coverage"] < n.coverage else
                              "confirmation" if previous is None or previous["coverage"] < n.coverage else
                              "metrics")
                    messages = dict(coverage="coverage target not reached",
                                    confirmation="only the last horizon reached coverage; another doubling is needed",
                                    metrics="successive-horizon metrics are not stable")
                    return dict(status="unconverged", convergence_reason=reason,
                                error=f"{messages[reason]} at horizon {horizon} (coverage={result['coverage']:.12g})",
                                horizon=horizon, history=history, diagnostics=result)
                previous = result
                horizon = min(2*horizon, n.max_horizon)
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            return dict(status="failed", horizon=horizon, history=history, error=str(exc),
                        traceback=traceback.format_exc())


class Journal:
    """Append-and-fsync each trial; replay requests and cache physical candidates."""
    def __init__(self, directory, evaluator, settings, resume=False):
        self.directory, self.evaluator = Path(directory), evaluator
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = (self.directory / ".run.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise ValueError(f"another process is already writing {directory}") from None
        self.path = self.directory / "trials.jsonl"
        config = dict(schema=SCHEMA, hardware=asdict(evaluator.hardware),
                      numerics=asdict(evaluator.numerics), settings=settings)
        config_path = self.directory / "config.json"
        self.records, self.requests, self.cache = [], {}, {}
        if config_path.exists():
            if not resume:
                raise ValueError(f"{directory} already contains a run; use --resume or another --output")
            if canonical(json.loads(config_path.read_text())) != canonical(config):
                raise ValueError("resume configuration differs from saved configuration")
        else:
            write_json(config_path, config)
            try:
                git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                git = None
            write_json(self.directory / "environment.json", dict(
                started=datetime.now(timezone.utc).isoformat(), python=sys.version,
                platform=platform.platform(), git_revision=git, command=sys.argv,
                packages={name: version(name) for name in ("numpy", "scipy", "numba", "scikit-optimize", "scikit-learn")},
                source_hashes={str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__).resolve(),
                                         Path(__file__).with_name("gp_pauliAD.py"),
                                         Path(__file__).with_name("optimize_postselection.py"),
                                         Path(__file__).resolve().parents[1] / "scripts" / "gp_pauliAD.py",
                                         Path(__file__).resolve().parents[1] / "scripts" / "pauli_ad_hardware.py",
                                         Path(__file__).resolve().parents[1] / "scripts" / "reproduce_results.py",
                                         Path(__file__).resolve().parents[1] / "core" / "repeater_algorithm.py",
                                         Path(__file__).resolve().parents[1] / "core" / "pauli_adc" / "state.py",
                                         Path(__file__).resolve().parents[1] / "core" / "pauli_adc" / "swap_join_sequential_efficient.py")}))
        if self.path.exists():
            # A killed write can leave only the final JSONL line incomplete.
            with self.path.open("rb+") as handle:
                while line := handle.readline():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if handle.read():
                            raise ValueError("corrupt journal before its final line")
                        handle.truncate(handle.tell()-len(line))
                        break
                    if not line.endswith(b"\n"):
                        handle.write(b"\n")
                    self._remember(record)

    def close(self):
        if getattr(self, "_lock", None) is not None:
            self._lock.close()

    def __del__(self):
        self.close()

    def _remember(self, record):
        self.records.append(record)
        self.requests[record["request_id"]] = record
        self.cache[record["candidate_id"]] = record

    def evaluate(self, choice, stage, proposal=None):
        request = digest(dict(candidate=choice, stage=stage, proposal=proposal))
        if request in self.requests:
            return self.requests[request]
        key, started = digest(choice), time.perf_counter()
        cached = key in self.cache
        result = self.cache[key]["result"] if cached else self.evaluator.run(choice)
        record = dict(index=len(self.records), request_id=request, candidate_id=key,
                      stage=stage, proposal=proposal, candidate=choice, cached=cached,
                      elapsed_seconds=time.perf_counter()-started, result=result)
        with self.path.open("a") as handle:
            handle.write(canonical(record)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._remember(record)
        logging.info("trial=%d stage=%s status=%s SKR=%s q=%.5g cached=%s", record["index"], stage,
                     result["status"], result.get("skr"), choice["q"], cached)
        if len(self.records) % 10 == 0 or not (self.directory / "trials.csv").exists():
            self.export()
        return record

    def best(self, records=None):
        valid = [r for r in (self.records if records is None else records) if r["result"]["status"] == "converged"]
        return max(valid, key=lambda r: (r["result"]["skr"], r["result"]["bb84_margin"]), default=None)

    def export(self):
        import csv
        fields = ("index", "stage", "cached", "status", "skr", "skf", "bb84_margin", "mean_wait", "fidelity",
                  "coverage", "horizon", "q", "protocol", "policies", "elapsed_seconds", "error")
        temporary = self.directory / "trials.csv.tmp"
        with temporary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in self.records:
                row = {**r, **r["candidate"], **r["result"]}
                writer.writerow({f: canonical(row[f]) if f in ("protocol", "policies") else row.get(f)
                                 for f in fields})
        temporary.replace(self.directory / "trials.csv")
        best = self.best()
        if best:
            write_json(self.directory / "best.json", best)
        try:
            plot_search(self)
        except (ImportError, RuntimeError) as exc:
            logging.warning("plot unavailable: %s", exc)

    def save_arrays(self, record, label="best"):
        if record:
            _, p, x = self.evaluator.at_horizon(record["candidate"], record["result"]["horizon"])
            np.savez_compressed(self.directory / f"{label}.npz", pmf=p, coordinates=x)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fidelity = (1+x[:, 3]+x[:, 4]-x[:, 5])/4
            fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
            times = np.arange(len(p))
            visible = p > max(1e-12, p.max()*1e-6)
            axes[0].plot(times[visible], p[visible])
            axes[0].set(xlabel="Completion time (slots)", ylabel="Probability per slot")
            axes[1].hist(fidelity[p > 0], bins=30, weights=p[p > 0])
            axes[1].set(xlabel="Conditional Phi+ fidelity", ylabel="Completion probability")
            fig.savefig(self.directory / f"{label}_delivery.png", dpi=160)
            plt.close(fig)


def bayesian_search(journal, dimensions, decode, calls, initial_points, seed, stage, seeds=()):
    if calls < 1 or initial_points < 1:
        raise ValueError("GP calls and initial points must be positive")
    estimator = cook_estimator("GP", space=Space(dimensions), random_state=seed, noise=1e-10)
    optimizer = Optimizer(dimensions, base_estimator=estimator, n_initial_points=initial_points,
                          acq_func="LCB", acq_func_kwargs={"kappa": 1.96},
                          acq_optimizer="sampling", random_state=seed)
    records = []
    for point in seeds:
        # Anchors are outside the GP proposal budget, and are observations too.
        point = list(point)
        record = journal.evaluate(decode(point), stage, proposal={"anchor": point})
        optimizer.tell(point, objective(record["result"]))
        records.append(record)
    for i in range(calls):
        point = optimizer.ask()
        record = journal.evaluate(decode(point), stage, proposal={"iteration": i, "point": point})
        optimizer.tell(point, objective(record["result"]))
        records.append(record)
    return records


def plot_search(journal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = journal.records
    valid = [r for r in records if r["result"]["status"] == "converged"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), constrained_layout=True)
    for stage in dict.fromkeys(r["stage"] for r in records):
        rows = [r for r in valid if r["stage"] == stage]
        if not rows:
            continue
        indices = [r["index"] for r in rows]
        rates = [r["result"]["skr"] for r in rows]
        axes[0].plot(indices, np.maximum.accumulate(rates), label=stage)
        axes[1].scatter([r["candidate"]["q"] for r in rows], rates, s=12, alpha=.55)
        axes[2].scatter([r["result"]["mean_wait"] for r in rows],
                        [r["result"]["skf"] for r in rows], s=12, alpha=.55)
    axes[0].set(xlabel="Trial (includes anchors/cache)", ylabel="Best SKR (bits/slot)")
    axes[1].set(xlabel="Source q", ylabel="SKR (bits/slot)")
    axes[2].set(xlabel="Mean waiting time (slots)", ylabel="SKF", xscale="log")
    if valid:
        axes[0].legend(fontsize=7)
    fig.savefig(journal.directory / "search.png", dpi=160)
    plt.close(fig)
    best = journal.best()
    if not best:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    draw_protocol(ax, best["candidate"], journal.evaluator.hardware.links)
    ax.set_title(f"Best found: SKR={best['result']['skr']:.5g}, q={best['candidate']['q']:.4g}")
    fig.savefig(journal.directory / "best_protocol.png", dpi=160)
    plt.close(fig)


def draw_protocol(ax, choice, links):
    segments = {i: (float(i), 0., 0) for i in range(links)}
    colors = {"all_corrected": "#0072B2", "psi_pair": "#D55E00", "phi_pair": "#CC79A7"}
    for step in choice["protocol"]:
        i = int(step[1:])
        if step[0] == "d":
            x, y, count = segments[i]
            segments[i] = (x, y, count+1)
        else:
            j = min(k for k in segments if k > i)
            left, right = segments.pop(i), segments.pop(j)
            x, y = (left[0]+right[0])/2, max(left[1], right[1])+1
            for child in (left, right):
                ax.plot([child[0], x], [child[1], y], color=".6", zorder=0)
                ax.text(child[0], child[1], str(child[2]), ha="center", va="center",
                        bbox=dict(boxstyle="circle", facecolor="white", edgecolor=".4"))
            ax.text(x, y+.17, choice["policies"][i].replace("_pair", "").replace("all_corrected", "all"),
                    ha="center", color=colors[choice["policies"][i]], fontsize=8)
            segments[j] = (x, y, 0)
    x, y, count = next(iter(segments.values()))
    ax.text(x, y, str(count), ha="center", va="center",
            bbox=dict(boxstyle="circle", facecolor="white", edgecolor=".4"))
    ax.set(xlim=(-.5, links-.5), ylim=(-.5, y+.7))
    ax.set_axis_off()
    ax.text(.01, .98, "Counters: distillation rounds", transform=ax.transAxes, fontsize=8)


def add_common_args(parser):
    parser.add_argument("--nodes", type=int, default=5, help="Nodes = elementary links + 1")
    parser.add_argument("--p-gen", type=float, nargs="+", default=[.2])
    parser.add_argument("--p-swap", type=float, default=1.)
    parser.add_argument("--p-distillation", type=float, default=1.)
    parser.add_argument("--t1-slots", type=float, nargs="+", default=None)
    parser.add_argument("--t2-slots", type=float, nargs="+", default=None,
                        help="Default 2*T1; use inf for infinite lifetime")
    parser.add_argument("--pauli-rates", type=float, nargs="+", default=None,
                        help="Raw mode rates: one X,Y,Z triple or a flattened triple per node")
    parser.add_argument("--ad-rates", type=float, nargs="+", default=None)
    parser.add_argument("--visibility", type=float, default=1., help="Source v|psi_q><psi_q|+(1-v)I/4")
    parser.add_argument("--source-damping", type=float, default=0.,
                        help="Independent AD probability per source qubit at delivery; included in initial state, not an extra timed operation")
    parser.add_argument("--memory-cutoff-slots", type=int, default=None)
    parser.add_argument("--initial-horizon", type=int, default=256)
    parser.add_argument("--max-horizon", type=int, default=262144)
    parser.add_argument("--coverage", type=float, default=.999999)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--gp-calls", type=int, default=100, help="GP proposals per search, in addition to anchors")
    parser.add_argument("--initial-points", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")


def setup(args):
    nodes = args.nodes
    if args.pauli_rates is not None or args.ad_rates is not None:
        if args.t1_slots is not None or args.t2_slots is not None:
            raise ValueError("choose raw rates or T1/T2, not both")
        g = np.asarray(args.pauli_rates if args.pauli_rates is not None else [0., 0., 0.]).reshape(-1, 3)
        if len(g) == 1:
            g = np.repeat(g, nodes, axis=0)
        k = broadcast(args.ad_rates if args.ad_rates is not None else [0.], nodes, "AD rates")
    else:
        t1 = broadcast(args.t1_slots if args.t1_slots is not None else [100.], nodes, "T1")
        t2 = broadcast(args.t2_slots, nodes, "T2") if args.t2_slots is not None else 2*t1
        if np.any(np.isnan(t1)) or np.any(np.isnan(t2)) or np.any(t1 <= 0) or np.any(t2 <= 0):
            raise ValueError("lifetimes must be positive or inf")
        k, phi = 1/t1, 1/t2 - .5/t1
        if np.any(phi < -1e-14):
            raise ValueError("physical T1/T2 require T2 <= 2*T1")
        g = np.column_stack([np.maximum(phi, 0), np.maximum(phi, 0), np.zeros(nodes)])
    hardware = Hardware(nodes, tuple(broadcast(args.p_gen, nodes-1, "p_gen")),
                        args.p_swap, args.p_distillation, tuple(map(tuple, g)), tuple(k),
                        args.visibility, args.memory_cutoff_slots, args.source_damping)
    numerics = Numerics(args.initial_horizon, args.max_horizon, args.coverage, args.rtol, args.atol)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output / ".cache"))
    handler = logging.FileHandler(args.output / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return Evaluation(hardware, numerics)
