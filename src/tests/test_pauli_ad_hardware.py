"""Hardware mapping, source preparation and matched follow-up comparisons."""
from argparse import ArgumentParser, Namespace
import json

import numpy as np
import pandas as pd
import pytest

from src.core.pauli_adc.state import coordinates_to_matrix
from src.optimization.gp_pauliAD import run as protocol_run
from src.optimization.pauli_ad import add_common_args, candidate, fixed_protocol, source
from src.scripts.pauli_ad_hardware import cases, hardware, summarize


def options(tmp_path, **kwargs):
    values = dict(nodes=[5], seeds=[7], hardware_regimes=["C"], memory_profiles=["network"],
                  memory_scales=[1.], sections=["placement", "distillation", "asymmetry", "source"],
                  source_dampings=[.15], asymmetry_geometries=["slow_link", "weak_memory"],
                  relaxation_models=["ad", "unital"], output=tmp_path, p_distillation=1.,
                  initial_horizon=16384, max_horizon=1048576, gp_calls=1, initial_points=1,
                  post_method="auto", max_exact=243, max_dists=2, max_total_dists=2,
                  q_min=.1, q_grid=[.25, .4, .49, .5], resume=False)
    return Namespace(**(values | kwargs))


def test_reference_units_visibility_and_joint_to_local_mapping():
    h = hardware("C", "network", 5, 1.)
    assert h["p_gen"] == [.00092]*4
    assert h["p_swap"] == .85
    assert h["visibility"] == .96  # F0=.97, not .96.
    assert h["slot_seconds"] == pytest.approx(.00025)
    np.testing.assert_allclose(h["t1_slots"], 9600.)
    np.testing.assert_allclose(h["t2_slots"], 6840.)
    legacy = hardware("C", "benchmark", 5, 1.)
    t = 50000.
    # Match transverse two-qubit decay, not a false equivalence of full channels.
    assert np.exp(-2*t/legacy["t2_slots"][0]) == pytest.approx(np.exp(-t/1_400_000))
    assert legacy["t1_slots"][0] == 10*legacy["t2_slots"][0]


@pytest.mark.parametrize("q,visibility,damping", [(.5, 1., .15), (.25, .96, .4), (.4, 1., 0.), (.5, .96, 1.)])
def test_delivered_source_damping_matches_dense_kraus_oracle(q, visibility, damping):
    rho = coordinates_to_matrix(source(q, visibility))
    kraus = (np.diag([1., np.sqrt(1-damping)]), np.array([[0., np.sqrt(damping)], [0., 0.]]))
    expected = sum(np.kron(a, b) @ rho @ np.kron(a, b).conj().T for a in kraus for b in kraus)
    np.testing.assert_allclose(coordinates_to_matrix(source(q, visibility, damping)), expected, atol=1e-14)


def test_hardware_plan_asymmetry_and_unbiased_control(tmp_path):
    plan = cases(options(tmp_path))
    assert len(plan) == len({c["name"] for c in plan}) == 20
    for case in plan:
        command = case["command"]
        assert command[command.index("--output")+1] == case["directory"]
        if case["relaxation_model"] == "unital":
            assert "--t1-slots" not in command
            parser = ArgumentParser()
            add_common_args(parser)
            args, _ = parser.parse_known_args(command[3:])
            rates = np.array(args.pauli_rates).reshape(-1, 3)
            np.testing.assert_allclose(rates[:, 0], 1/np.array(case["t2_slots"]))
            np.testing.assert_allclose(rates[:, 2], 1/np.array(case["t1_slots"]))
            assert args.ad_rates == [0.]
        if case["geometry"] == "slow_link":
            assert case["p_gen"][-1] == pytest.approx(case["p_gen"][0]/5)
            assert len(set(case["t1_slots"])) == 1
        if case["geometry"] == "weak_memory":
            assert len(set(case["p_gen"])) == 1
            assert case["t1_slots"][-2] == case["t1_slots"][0]/4


@pytest.mark.parametrize("mode", ["bell", "source"])
def test_restricted_searches_include_matched_controls_and_total_cap(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path/"mpl"))
    # Exercise actual evaluator/search, omitting only redundant per-trial figures.
    monkeypatch.setattr("src.optimization.pauli_ad.Journal.export", lambda self: None)
    monkeypatch.setattr("src.optimization.pauli_ad.Journal.save_arrays", lambda *args: None)
    parser = ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--nodes", "3", "--t1-slots", "75", "--source-damping", ".1",
                              "--initial-horizon", "128", "--max-horizon", "4096",
                              "--gp-calls", "1", "--initial-points", "1", "--output", str(tmp_path)])
    args = Namespace(**vars(args), max_dists=2, max_total_dists=1, q_min=.1,
                     joint_only=False, search_mode=mode, q_grid=[.25, .4, .49, .5] if mode == "source" else [])
    result = protocol_run(args)
    records = [json.loads(line) for line in (tmp_path/"trials.jsonl").read_text().splitlines()]
    assert all(sum(s.startswith("d") for s in r["candidate"]["protocol"]) <= 1 for r in records)
    if mode == "bell":
        assert all(r["candidate"]["q"] == .5 for r in records)
        assert result["results"]["bell_policy"]["best"]["result"]["skr"] >= result["results"]["bell_swaps"]["best"]["result"]["skr"]
    else:
        assert result["results"]["joint"]["best"]["result"]["skr"] >= result["results"]["bell_policy"]["best"]["result"]["skr"]
        for profile in result["q_profiles"].values():
            assert any(r["candidate"]["q"] == .5 for r in profile)
            assert len({tuple(r["candidate"]["protocol"]) for r in profile}) == 1
            assert len({tuple(r["candidate"]["policies"]) for r in profile}) == 1
        bell_rates = [r["result"]["skr"] for r in records if r["candidate"]["q"] == .5 and r["result"]["status"] == "converged"]
        assert result["results"]["bell_policy"]["best"]["result"]["skr"] == max(bell_rates)


def test_all_four_summary_figures_keep_units_missing_values_and_q_attribution(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path/"mpl"))
    plan = cases(options(tmp_path, relaxation_models=["ad"]))
    def record(q, rate=.001):
        return dict(candidate=candidate(fixed_protocol(4), q, ["all_corrected"]*3),
                    result=dict(status="converged", skr=rate, skf=.1, mean_wait=100., fidelity=.95, coverage=1., horizon=65536))
    for case in plan:
        path = tmp_path/case["name"]
        path.mkdir()
        stages = ("uniform", "level", "node") if case["study"] == "postselection" else (
            ("bell_swaps", "bell_policy") if case["section"] == "distillation" else ("bell_policy", "joint"))
        result = dict(results={s: dict(best=record(case["q"] or .5)) for s in stages})
        if case["section"] == "source":
            result["q_profiles"] = dict(bell_winner=[record(.4, .002), record(.5)])
        (path/"summary.json").write_text(json.dumps(result))
    summarize(plan, tmp_path)
    table = pd.read_csv(tmp_path/"summary.csv")
    np.testing.assert_allclose(table.skr_bits_per_second, table.skr/.00025)
    matched = pd.read_csv(tmp_path/"comparisons.csv")
    assert np.allclose(matched[matched.target == "matched_q_bell_winner"].relative_gain, 1.)
    for section in ("1-placement", "2-distillation", "3-asymmetry", "4-source"):
        assert next(tmp_path.glob(section+"*.png")).stat().st_size > 1000
    assert json.loads((tmp_path/"plot_status.json").read_text())["status"] == "complete"


def test_partial_zero_key_summary_has_a_valid_log_domain(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path/"mpl"))
    plan = cases(options(tmp_path, sections=["placement"], relaxation_models=["ad"]))
    case = plan[0]
    path = tmp_path/case["name"]
    path.mkdir()
    record = dict(candidate=candidate(fixed_protocol(4), .5, ["all_corrected"]*3),
                  result=dict(status="converged", skr=0., skf=0., mean_wait=100., fidelity=.7, coverage=1., horizon=65536))
    (path/"summary.json").write_text(json.dumps(dict(results=dict(
        uniform=dict(best=record), level=dict(best=None), node=dict(best=None)))))
    summarize(plan, tmp_path)
    assert json.loads((tmp_path/"plot_status.json").read_text())["status"] == "complete"
    table = pd.read_csv(tmp_path/"summary.csv")
    assert len(table[table.status == "no_valid_result"]) == 2
    assert table.loc[table.status == "no_valid_result", "skr"].isna().all()


def test_unresolved_trials_from_an_interrupted_job_are_retained(tmp_path):
    plan = cases(options(tmp_path, sections=["source"], relaxation_models=["ad"]))
    case = plan[0]
    path = tmp_path/case["name"]
    path.mkdir()
    trial = dict(candidate_id="example", result=dict(status="unconverged", convergence_reason="confirmation",
                 error="needs another horizon", horizon=1048576))
    (path/"trials.jsonl").write_text(json.dumps(trial)+"\n"+json.dumps(trial)+"\n{" )
    summarize(plan, tmp_path)
    unresolved = pd.read_csv(tmp_path/"unresolved.csv")
    assert len(unresolved) == 1  # Distinct evaluations, not cached requests.
    assert unresolved.iloc[0].reason == "confirmation"
