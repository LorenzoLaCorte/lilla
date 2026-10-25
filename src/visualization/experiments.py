"""Bounded sweeps and protocol searches shared by Streamlit and the CLI."""
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import prod
from time import perf_counter
from uuid import uuid4

import numpy as np

from src.visualization.config import canonical, expand_config
from src.visualization.evaluate import source_fingerprint
from src.visualization.protocols import locations, ordered_trees, parse_tree

PARAMETERS = {
    "Initial fidelity": ("initial_state", "fidelity"),
    "Swap hardware success": ("hardware", "p_swap"),
    "Distillation hardware success": ("hardware", "p_distillation"),
    "Generation success": ("hardware", "p_gen"),
    "T1": ("memory", "t1_slots"), "T2": ("memory", "t2_slots"),
    "Shared memory cutoff": ("protocol", "memory_cutoff_slots"),
    "Horizon": ("numerics", "t_trunc"),
}
METRICS = {"SKR": "rate_bits_per_slot", "SKF": "pooled_secret_fraction",
           "Average waiting time": "mean_wait_repository_slots",
           "Average Phi+ fidelity": "average_fidelity", "Covered probability": "coverage"}


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def axis_values(axis):
    name = axis["parameter"]
    if name not in PARAMETERS:
        raise ValueError(f"unknown sweep parameter: {name}")
    count = integer(axis.get("points", 15), "points", 1)
    if count > 1000:
        raise ValueError("use at most 1000 points per axis")
    low, high = float(axis["min"]), float(axis["max"])
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("axis bounds must be finite and increasing")
    spacing = axis.get("spacing", "linear")
    if spacing not in ("linear", "log") or (spacing == "log" and low <= 0):
        raise ValueError("log spacing needs positive bounds; choose linear or log")
    values = (np.geomspace if spacing == "log" else np.linspace)(low, high, count)
    if name in ("Horizon", "Shared memory cutoff"):
        values = np.rint(values).astype(int)
    return list(dict.fromkeys(values.tolist()))


@dataclass
class Trial:
    index: int
    point: dict
    config: dict
    run: object = None
    status: str = "invalid"
    error: str = ""
    cached: bool = False
    bundle: str = None

    @property
    def eligible(self):
        return self.run is not None and self.run.summary["status"] == "covered"

    def row(self):
        summary = self.run.summary if self.run else {}
        return {"Trial": self.index, **self.point, "Status": self.status,
                **{label: summary.get(field) for label, field in METRICS.items()},
                "Cached": self.cached, "Error": self.error}

    def record(self):
        return {**self.row(), "config": self.config, "bundle": self.bundle,
                "summary": self.run.summary if self.run else None}


class Experiment:
    """One explicit evaluation per step, allowing cancellation between trials."""

    def __init__(self, config, request):
        self.base, _, _ = expand_config(config)
        self.request = deepcopy(request)
        self.mode = request["mode"]
        self.id = uuid4().hex
        self.entries = []
        self.cancelled = False
        self.started = perf_counter()
        self.optimizer = None
        self._evaluated = {}
        if self.mode == "sweep":
            self.axes = deepcopy(request["axes"])
            names = [a["parameter"] for a in self.axes]
            if not 1 <= len(names) <= 3 or len(set(names)) != len(names):
                raise ValueError("choose one to three distinct sweep parameters")
            if request.get("pure_relaxation") and ("T1" not in names or "T2" in names):
                raise ValueError("T2=2T1 requires a T1 axis and no independent T2 axis")
            if "Initial fidelity" in names and config["initial_state"]["kind"] != "werner":
                raise ValueError("fidelity sweeps require Werner inputs")
            if ({"T1", "T2"} & set(names)) and config["memory"]["kind"] != "t1_t2":
                raise ValueError("lifetime sweeps require T1/T2 memory controls")
            values = [axis_values(a) for a in self.axes]
            self.planned = prod(map(len, values))
            if self.planned > integer(request.get("max_evaluations", 500), "maximum evaluations", 1):
                raise ValueError(f"grid has {self.planned} points; reduce points or raise the explicit budget")
            self._points = iter(product(*values))
        elif self.mode == "optimize":
            self._prepare_search()
        else:
            raise ValueError("experiment mode must be sweep or optimize")

    def _prepare_search(self):
        from skopt import Optimizer
        from skopt.space import Categorical, Integer

        r = self.request
        selected = r.get("dimensions", ["tree", "cutoff"])
        if not selected or len(set(selected)) != len(selected) or set(selected)-{
                "tree", "cutoff", "distillation", "bell_outcomes"}:
            raise ValueError("choose distinct protocol search dimensions")
        if self.base["protocol"]["cutoff_overrides"] and {"tree", "distillation"} & set(selected):
            raise ValueError("clear per-operation overrides before searching trees or distillation")
        self.domains = []
        self.trees = ordered_trees(self.base["links"])
        if "tree" in selected:
            fixed = {s for s, n in self.base["protocol"]["distillation_rounds"].items()
                     if n and "distillation" not in selected}
            valid = [i for i, tree in enumerate(self.trees)
                     if fixed <= locations(parse_tree(tree, self.base["links"])).keys()]
            self.domains.append(("tree", valid))
        if "cutoff" in selected:
            low = integer(r.get("cutoff_min", 0), "minimum cutoff")
            high = integer(r.get("cutoff_max", 20), "maximum cutoff")
            if low > high or high-low > 10000:
                raise ValueError("choose increasing cutoff bounds spanning at most 10000 slots")
            values = ([None] if r.get("include_no_cutoff", True) else []) + list(range(low, high+1))
            self.domains.append(("cutoff", values))
        if "bell_outcomes" in selected:
            values = r.get("outcomes", ["all_corrected", "phi_pair", "psi_pair"])
            if not values or set(values)-{"all_corrected", "phi_pair", "psi_pair", "phi_plus"}:
                raise ValueError("choose supported Bell acceptance policies")
            self.domains.append(("bell_outcomes", list(dict.fromkeys(values))))
        if "distillation" in selected:
            maximum = integer(r.get("distillation_max", 1), "maximum distillation rounds")
            for index in range(2*self.base["links"]-1):
                self.domains.append((f"distillation:{index}", list(range(maximum+1))))
        size = prod(len(values) for _, values in self.domains)
        self.planned = min(integer(r.get("budget", 20), "trial budget", 1), size)
        self.space_size = size
        self.method = r.get("method", "bayesian")
        if self.method not in ("bayesian", "enumeration"):
            raise ValueError("search method must be bayesian or enumeration")
        if self.method == "bayesian" and size > 1:
            initial = min(integer(r.get("initial", 5), "initial trials", 1), self.planned)
            dims = [Integer(0, len(v)-1, name=k) if len(v)>1 else Categorical([0], name=k)
                    for k, v in self.domains]
            self.optimizer = Optimizer(dims, base_estimator="GP", n_initial_points=initial,
                                       random_state=integer(r.get("seed", 0), "seed"),
                                       acq_func="LCB", acq_optimizer="sampling",
                                       acq_func_kwargs={"kappa": 1.96})
        else:
            self._points = iter(product(*(range(len(v)) for _, v in self.domains)))

    @property
    def finished(self):
        return self.cancelled or len(self.entries) >= self.planned

    @property
    def best(self):
        eligible = [e for e in self.entries if e.eligible]
        return max(eligible, key=lambda e: e.run.summary["rate_bits_per_slot"], default=None)

    def _candidate(self, values):
        c = deepcopy(self.base)
        if self.mode == "sweep":
            point = {a["parameter"]: value for a, value in zip(self.axes, values)}
            for name, value in point.items():
                group, field = PARAMETERS[name]
                c[group][field] = value
            if self.request.get("pure_relaxation"):
                c["memory"]["t2_slots"] = 2*point["T1"]
            return c, point
        point = {key: choices[int(value)] for (key, choices), value in zip(self.domains, values)}
        p = c["protocol"]
        if "tree" in point:
            p["tree"] = deepcopy(self.trees[point["tree"]])
        spans = sorted(locations(parse_tree(p["tree"], c["links"])),
                       key=lambda s: tuple(map(int, s.split(":"))))
        if "distillation" in self.request.get("dimensions", ["tree", "cutoff"]):
            p["distillation_rounds"] = {span: point[f"distillation:{i}"]
                                         for i, span in enumerate(spans) if point[f"distillation:{i}"]}
        else:
            p["distillation_rounds"] = {s: n for s, n in p["distillation_rounds"].items() if s in spans}
        if "cutoff" in point:
            p["memory_cutoff_slots"] = point["cutoff"]
            # A shared-cutoff search must actually vary every operation's cutoff.
            p["cutoff_overrides"] = {}
        if "bell_outcomes" in point:
            p["bell_outcomes"] = point["bell_outcomes"]
        return c, point

    def step(self, service):
        if self.finished:
            raise StopIteration
        values = self.optimizer.ask() if self.optimizer else next(self._points)
        c, point = self._candidate(values)
        entry = Trial(len(self.entries)+1, point, c)
        try:
            expand_config(c)
            entry.status = "failed"
            key = canonical(c) + source_fingerprint()
            entry.cached = key in self._evaluated
            entry.run = self._evaluated[key] if entry.cached else service.evaluate(c)
            self._evaluated[key] = entry.run
            entry.status = entry.run.summary["status"]
        except (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError) as error:
            entry.error = str(error)
        self.entries.append(entry)
        if self.optimizer:
            self.optimizer.tell(list(map(int, values)),
                                -entry.run.summary["rate_bits_per_slot"] if entry.eligible else 1.0)
        return entry

    def history(self):
        return {"schema_version": 1, "experiment_id": self.id, "mode": self.mode,
                "base_config": self.base, "request": self.request, "planned": self.planned,
                "cancelled": self.cancelled, "finished": self.finished,
                "wall_seconds": perf_counter()-self.started,
                "best_trial": self.best.index if self.best else None,
                "trials": [e.record() for e in self.entries]}
