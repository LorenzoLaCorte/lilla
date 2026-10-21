"""Atomic JSON/NPZ run bundles and full-resolution offline exports."""
import csv
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import numpy as np

from src.visualization.config import canonical, expand_config
from src.visualization.evaluate import ROOT, Run
from src.visualization.metrics import bell_populations, time_quantities, validate_outputs

DEFAULT_DIRECTORY = ROOT / "results" / "visualization"


def config_json(config):
    return json.dumps(config, indent=2, sort_keys=True, allow_nan=False)


def save_run(run, directory=DEFAULT_DIRECTORY):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"run-{run.metadata['run_id']}-{uuid4().hex[:8]}"
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    try:
        np.savez_compressed(staging/"arrays.npz", pmf=run.pmf, x=run.x)
        (staging/"run.json").write_text(config_json(run.metadata), encoding="utf-8")
        (staging/"config.json").write_text(config_json(run.config), encoding="utf-8")
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_run(path):
    path = Path(path)
    metadata = json.loads((path/"run.json").read_text(encoding="utf-8"))
    canonical(metadata)  # refuse NaN/Infinity in persisted summaries too
    if metadata["schema_version"] != 1:
        raise ValueError("unsupported saved-run schema")
    expand_config(metadata["config"])
    with np.load(path/"arrays.npz", allow_pickle=False) as arrays:
        pmf, x = arrays["pmf"].copy(), arrays["x"].copy()
    validate_outputs(pmf, x, metadata["config"]["numerics"]["t_trunc"])
    return Run(metadata, pmf, x)  # retain original parameters, status and arrays


def csv_text(rows):
    stream = StringIO()
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue()


def per_time_csv(run):
    quantities = time_quantities(run.x)
    populations = bell_populations(run.x)
    names = ("II", "IZ", "ZI", "ZZ", "XX", "YY")
    bell_names = ("phi+", "psi+", "psi-", "phi-")
    cdf = run.pmf.cumsum()
    rows = []
    for t, mass in enumerate(run.pmf):
        row = {"slot": t, "time_seconds": t*run.config["slot_duration_seconds"]
               if run.config["slot_duration_seconds"] else None,
               "pmf": mass, "cdf": cdf[t], "populated": bool(mass > 0)}
        row.update({name: run.x[t, j] if mass > 0 else None for j, name in enumerate(names)})
        row.update({"weighted_"+name: mass*run.x[t, j] for j, name in enumerate(names)})
        row.update({name: arr[t] if mass > 0 else None for name, arr in quantities.items()})
        row.update({"bell_"+name: populations[t, j] if mass > 0 else None
                    for j, name in enumerate(bell_names)})
        rows.append(row)
    return csv_text(rows)


def summary_csv(runs):
    from src.visualization.comparisons import comparison_rows
    rows = comparison_rows(runs)
    for row, run in zip(rows, runs):
        row.update({k: canonical(v) if isinstance(v, list) else v for k, v in run.summary.items()})
        row.update(timestamp=run.metadata["timestamp"], wall_seconds=run.metadata["wall_seconds"],
                   provisional=run.metadata["diagnostics"]["provisional"],
                   source_fingerprint=run.metadata["source_fingerprint"])
    return csv_text(rows)


def figures_html(figures):
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Pauli–ADC run</title></head><body>"]
    for i, fig in enumerate(figures):
        parts.append(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))
    parts.append("</body></html>")
    return "\n".join(parts)
