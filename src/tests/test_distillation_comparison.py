"""Checks for fair comparisons, numerical convergence, and durable resuming."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.scripts import compare_distillation_strategies as study


def test_all_distillation_placements_and_source_families_are_valid():
    protocols = list(study.protocol_candidates(2, 2))
    assert len(protocols) == len(set(protocols)) == 10
    assert (1, 1, 0, 0) in protocols and (0, 0, 1, 1) in protocols
    assert all(p.count(0) == 2 and p.count(1) <= 2 for p in protocols)
    for family in study.PRESETS["extensive"]["source_families"]:
        for q in [.2, .5]:
            state = study.RealXState(coordinates=study.source_coordinates(family, .95, q))
            assert state is not None  # Constructor enforces trace/positivity.


def test_key_metric_uses_errors_rather_than_only_fidelity():
    pmf = np.array([0., 1., 0., 0.])
    # Both inputs have Phi+ fidelity .9 but distinct X/Z errors.
    werner = np.tile(study.source_coordinates("werner", 13/15, None), (4, 1))
    anisotropic = np.tile(study.source_coordinates("bell_diagonal", 13/15, None), (4, 1))
    first, second = study.physical_metrics(pmf, werner), study.physical_metrics(pmf, anisotropic)
    assert_allclose(first["fidelity"], second["fidelity"])
    assert first["key_rate"] != second["key_rate"]
    assert first["mean_time"] == second["mean_time"] == 1.
    assert first["coverage"] == second["coverage"] == 1.


def test_incomplete_groups_never_bias_strategy_search():
    records = {}
    for strategy in study.DISTILLATION_STRATEGIES:
        records[strategy] = dict(group_id="common", strategy=strategy, scenario_id="case",
            status="complete", metrics=dict(key_rate=1.))
    records["extra"] = dict(group_id="partial", strategy="dejmps", scenario_id="case",
                             status="complete", metrics=dict(key_rate=100.))
    best, counts = study.matched_best(records)
    assert counts == {"case": 1}
    assert all(row["metrics"]["key_rate"] == 1. for row in best.values())
    records["dejmps"]["status"] = "horizon_limited"
    assert study.matched_best(records) == ({}, {})


def test_extend_horizon_resumes_without_repeating_previous_grid(monkeypatch):
    horizons = []
    def simulate(self, params):
        horizon = params["t_trunc"]
        horizons.append(horizon)
        pmf = np.zeros(horizon)
        pmf[1] = 1. if horizon >= 16 else .9
        states = np.tile([1., 0., 0., 1., 1., -1.], (horizon, 1))
        return pmf, states
    monkeypatch.setattr(study.RepeaterChainEvaluation, "nested_protocol", simulate)
    settings = dict(study.PRESETS["smoke"], seed=1, initial_horizon=4,
                    coverage=.999, metric_rtol=.002, metric_atol=.0002)
    task = next(study.candidate_tasks(study.scenarios(settings)[0], settings))
    first = study.evaluate_task(task, settings, 8)
    assert first["status"] == "horizon_limited"
    second = study.evaluate_task(task, settings, 32, previous=first)
    assert horizons == [4, 8, 16, 32]
    assert second["status"] == "complete"
    assert [a["horizon"] for a in second["attempts"]] == horizons


@pytest.mark.parametrize("tail", [b'{"task_id":', b'{"task_id":"last"}'])
def test_partial_last_checkpoint_line_can_be_resumed(tmp_path, tail):
    path = tmp_path / "evaluations.jsonl"
    path.write_bytes(b'{"task_id":"first"}\n'+tail)
    records = study.load_records(path)
    assert "first" in records
    study.append_record(path, dict(task_id="new"))
    assert "new" in study.load_records(path)
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_full_resume_and_plot_only_do_not_evaluate_or_rewrite_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "plot_results", lambda *args: None)
    argv = ["--preset", "smoke", "--max-distillations", "0", "--output-dir", str(tmp_path),
            "--max-horizon", "2048"]
    study.main(argv)
    path = tmp_path / "evaluations.jsonl"
    original = path.read_bytes()
    assert len(study.load_records(path)) == 8
    def forbidden(*args, **kwargs):
        raise AssertionError("An already complete study must not evaluate again")
    monkeypatch.setattr(study.RepeaterChainEvaluation, "nested_protocol", forbidden)
    study.main(argv)
    assert path.read_bytes() == original
    study.main(["--plot-only", "--output-dir", str(tmp_path)])
    assert path.read_bytes() == original
    assert (tmp_path / "best_policies.csv").exists()
    # A plot request while the evaluator is appending must not repair/truncate
    # its unfinished write, nor evaluate additional candidates.
    path.write_bytes(original+b'{"task_id":')
    live_snapshot = path.read_bytes()
    report_dir = tmp_path / "partial"
    study.main(["--plot-only", "--output-dir", str(tmp_path),
                "--report-dir", str(report_dir)])
    assert path.read_bytes() == live_snapshot
    assert (report_dir / "best_policies.csv").exists()


def test_focused_design_is_balanced_subset_of_extensive_grid():
    extensive = dict(study.PRESETS["extensive"], seed=20261114)
    focused = dict(study.PRESETS["focused"], seed=20261114)
    cases = study.scenarios(focused)
    assert len(cases) == 144
    assert len({study.stable_id(c) for c in cases}) == 144
    assert {study.stable_id(c) for c in cases} <= {
        study.stable_id(c) for c in study.scenarios(extensive)
    }
    assert sum(len(list(study.candidate_tasks(c, focused))) for c in cases) == 17424
    assert {c["source_family"] for c in cases} == {"werner", "bell_diagonal", "biased_x"}
    assert {c["noise_model"] for c in cases} == set(focused["noise_models"])
    assert all(c["source_q"] != .5 for c in cases if c["source_family"] == "biased_x")
    assert sum(c["levels"] == 2 and c["p_gen"] == .2 and c["p_swap"] == .9
               and c["communication"] for c in cases) == 72
    assert sum(c["levels"] in (1, 3) for c in cases) == 24
    assert sum(not c["communication"] for c in cases) == 24
    assert sum(c["p_gen"] == .05 or c["p_swap"] == .5 for c in cases) == 24


def test_pure_pe_design_contains_exact_slide_source_and_bell_control():
    settings = dict(study.PRESETS["pure_pe"], seed=20261114)
    cases = study.scenarios(settings)
    assert len(cases) == 32
    assert sum(len(list(study.candidate_tasks(c, settings))) for c in cases) == 2048
    assert {c["source_q"] for c in cases} == {.25, .5}
    assert all(c["visibility"] == 1. for c in cases)
    assert sum(c["noise_strength"] == 0. for c in cases) == 8
    for q in (.25, .5):
        rho = study.source_coordinates("biased_x", 1., q)
        from src.core.pauli_adc import coordinates_to_matrix
        matrix = coordinates_to_matrix(rho)
        assert_allclose(np.trace(matrix @ matrix), 1.)


def test_import_reuses_candidates_across_grids_without_new_simulations(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "plot_results", lambda *args: None)
    source, destination = tmp_path / "source", tmp_path / "destination"
    study.main(["--preset", "smoke", "--max-distillations", "0", "--output-dir", str(source)])
    original = (source / "evaluations.jsonl").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparing/importing cannot run simulations")
    monkeypatch.setattr(study.RepeaterChainEvaluation, "nested_protocol", forbidden)
    argv = ["--preset", "smoke", "--output-dir", str(destination),
            "--reuse-from", str(source), "--prepare-only"]
    study.main(argv)
    assert (source / "evaluations.jsonl").read_bytes() == original
    records = study.load_records(destination / "evaluations.jsonl")
    assert len(records) == 8
    assert all(row["status"] == "complete" and row["imported_from"] == str(source)
               for row in records.values())
    plan = json.loads((destination / "plan.json").read_text())
    assert plan["evaluations"] == 24 and plan["complete"] == 8
    assert plan["remaining_evaluations"] == 16
    checkpoint = (destination / "evaluations.jsonl").read_bytes()
    study.main(argv)
    assert (destination / "evaluations.jsonl").read_bytes() == checkpoint
    # Incompatible numerical policies cannot be silently reused.
    metadata = json.loads((source / "settings.json").read_text())
    metadata["settings"]["coverage"] = .95
    (source / "settings.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="numerical tolerances"):
        study.main(argv)
