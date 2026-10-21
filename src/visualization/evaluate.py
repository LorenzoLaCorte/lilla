"""Shared synchronous evaluation service and bounded, source-aware cache."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import subprocess
from time import perf_counter
from uuid import uuid4

import numpy as np

from src.core.pauli_adc.state import RealXState
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.visualization.config import canonical, expand_config
from src.visualization.metrics import summarize, validate_outputs

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = dict(efficient=True, use_fft=True, use_gpu=False, w_twirling=False)


@dataclass
class Run:
    metadata: dict
    pmf: np.ndarray
    x: np.ndarray

    @property
    def config(self):
        return self.metadata["config"]

    @property
    def summary(self):
        return self.metadata["summary"]


def source_fingerprint():
    digest = sha256(b"visualization-schema-1")
    paths = sorted(p for folder in ("core", "types", "utils", "visualization")
                   for p in (ROOT/"src"/folder).rglob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_identity(config):
    c, p, _ = expand_config(config)
    return sha256(canonical({"parameters": p, "links": c["links"],
                            "model": c["prediction_model"], "options": OPTIONS,
                            "source": source_fingerprint()}).encode()).hexdigest()


def git_context():
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                            stderr=subprocess.DEVNULL, text=True).strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


class EvaluationService:
    def __init__(self, cache_size=8):
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.backend_calls = 0

    def is_cached(self, config):
        return cache_identity(config) in self.cache

    def evaluate(self, config):
        c, p, operations = expand_config(config)
        key = cache_identity(c)
        start = perf_counter()
        hit = key in self.cache
        if hit:
            pmf, x, cleanup, compute_seconds = deepcopy(self.cache.pop(key))
        else:
            evaluator = RepeaterChainEvaluation(state_type=RealXState, **OPTIONS)
            self.backend_calls += 1
            pmf, x = evaluator.asymmetric_heterogeneous_protocol(deepcopy(p), c["links"])
            pmf, x, cleanup = validate_outputs(pmf, x, p["t_trunc"])
            compute_seconds = perf_counter()-start
        self.cache[key] = deepcopy((pmf, x, cleanup, compute_seconds))
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        summary, diagnostics = summarize(pmf, x, c["numerics"]["coverage_target"],
                                         c["slot_duration_seconds"])
        diagnostics["cleanup"] = cleanup
        metadata = {"schema_version": 1, "run_id": uuid4().hex,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "config": c, "backend_parameters": p, "operations": operations,
                    "evaluator_options": OPTIONS.copy(), "source": git_context(),
                    "source_fingerprint": source_fingerprint(), "cache_identity": key,
                    "cache_hit": hit, "compute_seconds": compute_seconds,
                    "wall_seconds": perf_counter()-start, "horizon": p["t_trunc"],
                    "diagnostics": diagnostics, "summary": summary}
        return Run(metadata, pmf, x)
