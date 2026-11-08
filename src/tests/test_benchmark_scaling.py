"""Checkpoint tests use fake evaluations; no scaling benchmark is run."""

import json
from pathlib import Path

import pytest

from src.scripts import benchmark_scaling as benchmark


def settings():
    return {
        "resume": True, "p_gen": 0.1, "adc_factor": 1.0,
        "classical_communication": True, "padding_factor": 4,
        "coverage_target": 0.99, "p_swap": [0.2, 0.5, 0.9],
        "max_horizon": 4096, "stopped_sweeps": [],
        "werner_retry_engine": benchmark.WERNER_RETRY_ENGINE,
    }


def record(p_swap=0.5, level=2, horizon=512):
    return {
        "p_swap": p_swap, "nesting_level": level, "nodes": 2**level + 1,
        "horizon": horizon,
        "parameters": benchmark.case_parameters(level, 0.1, p_swap, 1.0),
        "coverage": {label: 0.995 for label in benchmark.BACKENDS},
        "calibration": [], "max_pmf_discrepancy": 0.0,
        "samples_seconds": {label: [0.01, 0.02, 0.03] for label in benchmark.BACKENDS},
        "median_seconds": {label: 0.02 for label in benchmark.BACKENDS},
    }


def save_fixture(folder, records, previous=None, pending=()):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "computation_time.json"
    path.write_text(json.dumps({"settings": previous or settings(), "records": records,
                               "pending_records": list(pending)}))
    return path


def no_evaluation(*args, **kwargs):
    pytest.fail("a completed point must not be recalibrated or reevaluated")


def test_resume_reuses_complete_points_and_preserves_other_probability_curves(tmp_path, monkeypatch):
    records = [record(), record(level=3, horizon=2048), record(p_swap=0.9)]
    path = save_fixture(tmp_path, records)
    expected = json.loads(path.read_text())["records"]
    monkeypatch.setattr(benchmark, "evaluate", no_evaluation)
    monkeypatch.setattr(benchmark, "calibrate_horizon", no_evaluation)
    saved = []
    monkeypatch.setattr(benchmark, "save_results", lambda folder, rows, *_: saved.append(rows))
    benchmark.main(["--p-swap", "0.5", "--max-level", "3", "--output-dir", str(tmp_path)])
    assert saved[-1] == expected


@pytest.mark.parametrize("option,value", [("--p-gen", "0.2"), ("--adc-factor", "0"), ("--padding-factor", "8")])
def test_resume_rejects_changed_model_before_touching_checkpoint(tmp_path, monkeypatch, option, value):
    path = save_fixture(tmp_path, [record()])
    original = path.read_bytes()
    monkeypatch.setattr(benchmark, "evaluate", no_evaluation)
    with pytest.raises(SystemExit) as error:
        benchmark.main([option, value, "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert path.read_bytes() == original


def test_interrupted_point_keeps_samples_and_resumes_without_recalibration(tmp_path, monkeypatch):
    old = record()
    save_fixture(tmp_path, [old])
    monkeypatch.setattr(benchmark, "calibrate_horizon", lambda *args: (
        256, {label: 0.995 for label in benchmark.BACKENDS}, 0.0, []
    ))
    calls = []

    def interrupted(simulator, parameters, horizon, padding):
        label = "Werner" if simulator.state_type == benchmark.WernerState else "X-state"
        calls.append((label, horizon))
        if label == "X-state":
            raise KeyboardInterrupt
        return None

    monkeypatch.setattr(benchmark, "evaluate", interrupted)
    argv = ["--p-swap", "0.9", "--max-level", "2", "--output-dir", str(tmp_path)]
    with pytest.raises(KeyboardInterrupt):
        benchmark.main(argv)
    checkpoint = json.loads((tmp_path / "computation_time.json").read_text())
    pending = checkpoint["pending_records"][0]
    assert len(pending["samples_seconds"]["Werner"]) == 1
    assert pending["samples_seconds"]["X-state"] == []
    first_sample = pending["samples_seconds"]["Werner"][0]
    assert checkpoint["records"][0] == json.loads(json.dumps(old))

    calls.clear()
    monkeypatch.setattr(benchmark, "calibrate_horizon", no_evaluation)

    def completed(simulator, parameters, horizon, padding):
        calls.append(horizon)
        return None

    monkeypatch.setattr(benchmark, "evaluate", completed)
    monkeypatch.setattr(benchmark, "save_results", benchmark.write_checkpoint)
    benchmark.main(argv)
    checkpoint = json.loads((tmp_path / "computation_time.json").read_text())
    assert checkpoint["pending_records"] == []
    new = next(r for r in checkpoint["records"] if r["p_swap"] == 0.9)
    assert new["samples_seconds"]["Werner"][0] == first_sample
    assert all(len(new["samples_seconds"][label]) == 3 for label in benchmark.BACKENDS)
    assert calls.count(256) == 5  # Only the five missing timed evaluations.
    assert calls.count(64) == 2   # Tiny untimed warmups, no large calibration rerun.


def test_larger_limit_starts_after_previously_failed_horizon(tmp_path, monkeypatch):
    previous = settings()
    previous["stopped_sweeps"] = [{"p_swap": 0.2, "nesting_level": 3, "reason": "coverage limit"}]
    old = record(p_swap=0.2, horizon=2048)
    save_fixture(tmp_path, [old], previous)
    starts = []

    def calibrate(simulators, parameters, initial, *args):
        starts.append(initial)
        return initial, {label: 0.995 for label in benchmark.BACKENDS}, 0.0, []

    monkeypatch.setattr(benchmark, "calibrate_horizon", calibrate)
    monkeypatch.setattr(benchmark, "measure", lambda *args, **kwargs: {label: [0.04]*3 for label in benchmark.BACKENDS})
    monkeypatch.setattr(benchmark, "save_results", benchmark.write_checkpoint)
    benchmark.main(["--p-swap", "0.2", "--max-level", "3", "--max-horizon", "16384", "--output-dir", str(tmp_path)])
    assert starts == [8192]
    checkpoint = json.loads((tmp_path / "computation_time.json").read_text())
    assert checkpoint["records"][0] == json.loads(json.dumps(old))
    assert checkpoint["settings"]["stopped_sweeps"] == []


def test_known_horizon_limit_is_not_repeated(tmp_path, monkeypatch):
    previous = settings()
    previous["stopped_sweeps"] = [{"p_swap": 0.2, "nesting_level": 3, "reason": "coverage limit"}]
    save_fixture(tmp_path, [record(p_swap=0.2)], previous)
    monkeypatch.setattr(benchmark, "calibrate_horizon", no_evaluation)
    monkeypatch.setattr(benchmark, "save_results", benchmark.write_checkpoint)
    benchmark.main(["--p-swap", "0.2", "--max-level", "3", "--max-horizon", "4096", "--output-dir", str(tmp_path)])


def test_failed_atomic_replace_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = save_fixture(tmp_path, [record()])
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated interrupted replacement")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        benchmark.write_checkpoint(tmp_path, [record(level=3)], settings())
    assert path.read_bytes() == original


def test_plot_only_regenerates_outputs_without_simulations_or_checkpoint_changes(tmp_path, monkeypatch):
    path = save_fixture(tmp_path, [record(p_swap=p) for p in (0.2, 0.5, 0.9)])
    original = path.read_bytes()
    monkeypatch.setattr(benchmark, "evaluate", no_evaluation)
    monkeypatch.setattr(benchmark, "calibrate_horizon", no_evaluation)
    monkeypatch.setattr(benchmark, "RepeaterChainEvaluation", no_evaluation)
    monkeypatch.setattr(benchmark, "write_checkpoint", no_evaluation)
    benchmark.main(["--plot-only", "--output-dir", str(tmp_path)])
    assert path.read_bytes() == original
    for suffix in ("png", "pdf", "csv"):
        assert (tmp_path / f"computation_time.{suffix}").stat().st_size > 0


def test_plot_only_missing_checkpoint_fails_without_simulations(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "RepeaterChainEvaluation", no_evaluation)
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--plot-only", "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert not (tmp_path / "computation_time.json").exists()


def test_legacy_timings_require_explicit_refresh_without_modifying_checkpoint(tmp_path, monkeypatch):
    previous = settings()
    del previous["werner_retry_engine"]
    path = save_fixture(tmp_path, [record()], previous)
    original = path.read_bytes()
    monkeypatch.setattr(benchmark, "evaluate", no_evaluation)
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert path.read_bytes() == original


def test_refresh_only_times_werner_and_keeps_x_samples_and_legacy_provenance(tmp_path, monkeypatch):
    previous = settings()
    del previous["werner_retry_engine"]
    old = record()
    other = record(p_swap=0.9)
    save_fixture(tmp_path, [old, other], previous)
    calls = []
    monkeypatch.setattr(benchmark, "calibrate_horizon", no_evaluation)

    def evaluate(simulator, parameters, horizon, padding):
        if horizon > 64:
            calls.append(simulator.state_type)
        return None

    monkeypatch.setattr(benchmark, "evaluate", evaluate)
    monkeypatch.setattr(benchmark, "save_results", benchmark.write_checkpoint)
    argv = ["--refresh-werner", "--p-swap", "0.5", "--max-level", "2", "--output-dir", str(tmp_path)]
    benchmark.main(argv)
    assert calls == [benchmark.WernerState] * 3
    checkpoint = json.loads((tmp_path / "computation_time.json").read_text())
    new = checkpoint["records"][0]
    assert new["samples_seconds"]["X-state"] == old["samples_seconds"]["X-state"]
    assert new["previous_werner_timing"]["samples_seconds"] == old["samples_seconds"]["Werner"]
    assert checkpoint["settings"]["werner_retry_engine"] == benchmark.WERNER_RETRY_ENGINE
    assert len(checkpoint["pending_records"]) == 1
    assert checkpoint["pending_records"][0]["p_swap"] == 0.9
    assert checkpoint["pending_records"][0]["samples_seconds"]["X-state"] == other["samples_seconds"]["X-state"]
    monkeypatch.setattr(benchmark, "evaluate", no_evaluation)
    benchmark.main(argv)  # The flag does not refresh already-upgraded timings again.
