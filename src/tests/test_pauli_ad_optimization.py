"""Scientific cross-checks and restart checks for the standalone searches."""
from argparse import ArgumentParser, Namespace
import ast
from dataclasses import replace
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from skopt.space import Real

from src.core.pauli_adc.state import RealXState
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.optimization.gp_pauliAD import encoded_protocol, run as protocol_run
from src.optimization.optimize_postselection import run as placement_run
from src.optimization.pauli_ad import (
    Evaluation, Hardware, Journal, Numerics, POLICIES, add_common_args,
    bayesian_search, candidate, fixed_protocol, objective, protocol_nodes, setup, source,
)
from src.scripts.gp_pauliAD import execute, plot_summary, recover_manifest, summarize


@pytest.fixture
def hardware():
    return Hardware(3, (.2, .2), 1., 1., ((0., 0., 0.),)*3, (0.,)*3)


@pytest.fixture(autouse=True)
def plot_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_pe_concentration_includes_destructive_retries(hardware):
    evaluator = Evaluation(replace(hardware, p_gen=(1., 1.)), Numerics(32, 1024))
    all_result = evaluator.run(candidate(["s0"], .25, ["all_corrected"]))
    psi_result = evaluator.run(candidate(["s0"], .25, ["psi_pair"]))
    assert all_result["status"] == psi_result["status"] == "converged"
    assert all_result["mean_wait"] == pytest.approx(1.)
    assert psi_result["mean_wait"] == pytest.approx(1/(2*.25*.75))
    assert psi_result["skf"] == pytest.approx(1.)
    assert psi_result["skr"] == pytest.approx(.375)
    assert all_result["skr"] > psi_result["skr"]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("protocol", [("s1", "s0"), ("d0", "s0", "d1", "s1", "d2")])
def test_uniform_matches_existing_heterogeneous_protocol(policy, protocol):
    h = Hardware(4, (.35, .2, .4), .9, .85,
                 ((.002, .002, 0.), (.004, .004, 0.), (.001, .001, 0.), (.003, .003, 0.)),
                 (.01, .015, .005, .02))
    evaluator = Evaluation(h, Numerics(128, 1024))
    choice = candidate(protocol, .3, [policy]*2)
    _, p, x = evaluator.at_horizon(choice, 512)
    parameters = dict(protocol=protocol, t_trunc=512, p_gen=list(h.p_gen), p_swap=h.p_swap,
                      p_distillation=h.p_distillation, real_x_coordinates=[source(.3).tolist()]*3,
                      pauli_mode_decay_rates=list(h.pauli_rates), amplitude_damping_rate=list(h.ad_rates),
                      bell_outcomes=policy, real_x_werner_twirling=False,
                      cutoff=[512]*len(protocol), cut_type="memory_time")
    p_ref, x_ref = evaluator.backend.asymmetric_heterogeneous_protocol(parameters, 3)
    np.testing.assert_allclose(p, p_ref, atol=1e-12)
    np.testing.assert_allclose(p[:, None]*x, p_ref[:, None]*x_ref, atol=1e-11)


def test_final_only_selection_reproduces_notebook_gain():
    h = Hardware(5, (.2,)*4, 1., 1., ((0., 0., 0.),)*5, (1/74,)*5)
    evaluator = Evaluation(h, Numerics(128, 4096, rtol=1e-6))
    protocol = fixed_protocol(4)
    all_result = evaluator.run(candidate(protocol, .5, ["all_corrected"]*3))
    # Physical boundary s1 is the root, even though it runs after s2.
    root_result = evaluator.run(candidate(protocol, .5, ["all_corrected", "psi_pair", "all_corrected"]))
    assert all_result["status"] == root_result["status"] == "converged"
    assert root_result["skr"]/all_result["skr"] == pytest.approx(1.45, rel=.01)
    assert root_result["mean_wait"]/all_result["mean_wait"] == pytest.approx(2., rel=.01)


def test_small_exhaustive_placement_supersets_and_resume(tmp_path):
    parser = ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--nodes", "5", "--t1-slots", "74", "--initial-horizon", "128",
                              "--max-horizon", "4096", "--output", str(tmp_path)])
    args = Namespace(**vars(args), q=.5, tree="balanced", protocol=None, method="auto", max_exact=243)
    result = placement_run(args)
    assert result["results"]["node"]["certified"]
    assert result["results"]["node"]["space_size"] == 27
    assert result["node_vs_uniform"]["relative_gain"] > .4
    assert result["results"]["node"]["best"]["result"]["skr"] >= result["results"]["level"]["best"]["result"]["skr"]
    before = (tmp_path / "trials.jsonl").read_bytes()
    args.resume = True
    assert placement_run(args) == result
    assert (tmp_path / "trials.jsonl").read_bytes() == before


def test_unconverged_is_missing_not_zero(hardware):
    evaluator = Evaluation(replace(hardware, p_gen=(.01, .01)), Numerics(8, 16))
    result = evaluator.run(candidate(["s0"], .25, ["psi_pair"]))
    assert result["status"] == "unconverged"
    assert "skr" not in result
    assert objective(result) > objective(dict(status="converged", skr=0., bb84_margin=-.5))


def test_truncation_requires_a_second_covered_horizon(hardware, monkeypatch):
    evaluator = Evaluation(hardware, Numerics(64, 128))
    def sampled(choice, horizon):
        return dict(coverage=.999 if horizon == 64 else .9999999, skr=.01, skf=.1,
                    mean_wait=10., eX=.1, eZ=.1), None, None
    monkeypatch.setattr(evaluator, "at_horizon", sampled)
    choice = candidate(["s0"], .25, ["psi_pair"])
    result = evaluator.run(choice)
    assert result["status"] == "unconverged"
    assert result["convergence_reason"] == "confirmation"
    evaluator.numerics = Numerics(64, 256)
    result = evaluator.run(choice)
    assert result["status"] == "converged"
    assert result["horizon"] == 256


def test_truncation_reason_distinguishes_coverage_from_unstable_metrics(hardware, monkeypatch):
    evaluator = Evaluation(hardware, Numerics(64, 128))
    def sampled(choice, horizon):
        return dict(coverage=.99, skr=.01, skf=.1, mean_wait=horizon, eX=.1, eZ=.1), None, None
    monkeypatch.setattr(evaluator, "at_horizon", sampled)
    choice = candidate(["s0"], .25, ["psi_pair"])
    assert evaluator.run(choice)["convergence_reason"] == "coverage"
    evaluator.numerics = Numerics(64, 128, coverage=.98)
    assert evaluator.run(choice)["convergence_reason"] == "metrics"


def test_gp_replay_and_physical_cache(hardware, tmp_path, monkeypatch):
    evaluator = Evaluation(hardware, Numerics(64, 2048))
    monkeypatch.setattr(Journal, "export", lambda self: None)
    settings = {"seed": 7}
    journal = Journal(tmp_path, evaluator, settings)
    decode = lambda point: candidate(["s0"], point[0], ["psi_pair"])
    rows = bayesian_search(journal, [Real(.2, .5)], decode, 3, 2, 7, "joint", [[.5]])
    before = (tmp_path / "trials.jsonl").read_bytes()
    # Recover from a killed final write, retaining all complete trial records.
    with (tmp_path / "trials.jsonl").open("ab") as handle:
        handle.write(b'{"partial":')
    journal.close()
    replay = Journal(tmp_path, evaluator, settings, resume=True)
    monkeypatch.setattr(evaluator, "run", lambda c: pytest.fail("replay should use saved evaluations"))
    again = bayesian_search(replay, [Real(.2, .5)], decode, 3, 2, 7, "joint", [[.5]])
    assert again == rows
    assert (tmp_path / "trials.jsonl").read_bytes() == before
    extra = replay.evaluate(decode([.5]), "another-stage")
    assert extra["cached"]
    replay.close()
    with pytest.raises(ValueError, match="differs"):
        Journal(tmp_path, evaluator, {"seed": 99}, resume=True)


def test_encoding_reproducible_and_does_not_touch_global_rng():
    np.random.seed(2718)
    state = np.random.get_state()
    point = [.7, 4, -.2, .4]
    protocol = encoded_protocol(5, 1, point, 42)
    assert protocol == encoded_protocol(5, 1, point, 42)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_joint_ablations_matched_controls_and_resume(tmp_path):
    parser = ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--nodes", "3", "--t1-slots", "75", "--initial-horizon", "64",
                              "--max-horizon", "2048", "--gp-calls", "2", "--initial-points", "2",
                              "--output", str(tmp_path)])
    args = Namespace(**vars(args), max_dists=0, q_min=.1, joint_only=False)
    result = protocol_run(args)
    assert set(result["results"]) == {"bell_all", "bell_policy", "q_all", "joint"}
    rate = lambda stage: result["results"][stage]["best"]["result"]["skr"]
    assert rate("joint") >= max(rate("bell_all"), rate("bell_policy"), rate("q_all"))
    assert len(result["matched_controls"]) == 5
    original = result["matched_controls"]["original_joint_winner"]["candidate"]
    for label, record in result["matched_controls"].items():
        assert record["candidate"]["protocol"] == original["protocol"]  # No distillation in this test.
    before = (tmp_path / "trials.jsonl").read_bytes()
    args.resume = True
    assert protocol_run(args) == result
    assert (tmp_path / "trials.jsonl").read_bytes() == before


def test_concurrent_writers_are_rejected(hardware, tmp_path):
    evaluator = Evaluation(hardware, Numerics(64, 2048))
    journal = Journal(tmp_path, evaluator, {})
    with pytest.raises(ValueError, match="another process"):
        Journal(tmp_path, evaluator, {}, resume=True)
    journal.close()


def test_runner_timeout_preserves_logs_and_cancel_does_not_start(tmp_path):
    args = Namespace(output=tmp_path, case_timeout=.1)
    case = dict(name="slow", directory=str(tmp_path / "slow"),
                command=[sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"])
    result = execute(case, args)
    assert result["status"] == "timeout"
    assert (tmp_path / "slow" / "runner_status.json").exists()
    assert "started" in (tmp_path / "slow" / "console.log").read_text()
    stop = threading.Event()
    stop.set()
    case["directory"] = str(tmp_path / "cancelled")
    assert execute(case, args, stop)["status"] == "cancelled"
    assert not (tmp_path / "cancelled").exists()


def test_batch_plot_handles_an_entire_zero_key_panel(tmp_path):
    rows = []
    for nodes in (5, 9):
        for stage in ("bell_all", "bell_policy", "q_all", "joint"):
            for t1 in (25., 74.):
                positive = nodes == 9 and t1 == 74.
                rows.append(dict(study="protocol", status="converged", nodes=nodes, stage=stage,
                                 t1=t1, q=.5, skf=.2 if positive else 0., skr=.01 if positive else 0.))
    plot_summary(rows, tmp_path)
    assert (tmp_path / "protocol.png").stat().st_size > 1000
    assert (tmp_path / "protocol.pdf").stat().st_size > 1000


def test_batch_plot_failure_keeps_tables_and_manifest_recovery(tmp_path, monkeypatch):
    directory = tmp_path / "case"
    directory.mkdir()
    best = dict(candidate=candidate(["s0"], .5, ["all_corrected"]),
                result=dict(skr=0., skf=0., mean_wait=10., coverage=1., horizon=512))
    (directory / "summary.json").write_text(json.dumps(dict(results={"bell_all": dict(best=best)})))
    status = dict(name="case", status="complete", returncode=0, elapsed_seconds=10.)
    (directory / "runner_status.json").write_text(json.dumps(status))
    plan = [dict(name="case", study="protocol", nodes=3, t1=25., q=None, seed=7, directory=str(directory))]
    def broken_plot(*args):
        raise ValueError("simulated plot failure")
    monkeypatch.setattr("src.scripts.gp_pauliAD.plot_summary", broken_plot)
    summarize(plan, tmp_path)
    assert (tmp_path / "summary.csv").exists()
    assert json.loads((tmp_path / "plot_status.json").read_text())["status"] == "failed"
    assert recover_manifest(plan, tmp_path) == [status]
    assert json.loads((tmp_path / "manifest.json").read_text()) == [status]


def test_lifetimes_raw_rates_and_validation(tmp_path):
    parser = ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--nodes", "3", "--t1-slots", "100", "--t2-slots", "100", "--output", str(tmp_path)])
    evaluation = setup(args)
    np.testing.assert_allclose(evaluation.hardware.ad_rates, .01)
    np.testing.assert_allclose(evaluation.hardware.pauli_rates, [[.005, .005, 0]]*3)
    args.t2_slots = [201]
    with pytest.raises(ValueError, match="T2"):
        setup(args)
    with pytest.raises(ValueError, match="Lindblad"):
        replace(evaluation.hardware, pauli_rates=((1., 0., 0.),)*3)


def test_tree_boundaries_and_no_visualization_dependency():
    assert fixed_protocol(4) == ("s0", "s2", "s1")
    nodes = protocol_nodes(fixed_protocol(4), 4)
    assert nodes[1] == dict(boundary=1, span=[0, 4], height=2, depth=0)
    root = Path(__file__).resolve().parents[1]
    for relative in ("optimization/pauli_ad.py", "optimization/optimize_postselection.py",
                     "optimization/gp_pauliAD.py", "scripts/gp_pauliAD.py"):
        tree = ast.parse((root / relative).read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(m and m.startswith("src.visualization") for m in imports)
