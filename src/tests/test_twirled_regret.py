"""Exactness boundaries, actual policy transfer, and durable matched studies."""
from dataclasses import asdict, replace
import json
import sys

import numpy as np
import pytest

from src.core.pauli_adc.protocol_units import (
    accepted_bell_swap, apply_local_channels, recurrence_distillation,
    sequential_pauli_adc_channel,
)
from src.optimization.pauli_ad import Evaluation, Hardware, Numerics, candidate, digest, fixed_protocol
from src.optimization.twirled_regret import (
    MODELS, ModelEvaluation, candidate_pool, compare, in_space, initial_state, project, run,
)
from src.scripts.twirled_regret import main, parser, plan, summarize


@pytest.fixture
def hardware():
    return Hardware(4, (.3, .2, .4), .9, .95,
                    ((.01, .012, .008),)*4, (0.,)*4)


def test_source_fidelity_and_projection_preserve_unnormalized_mass():
    for family in ("werner", "phase_error", "bit_error", "r_state"):
        x = initial_state(family, .95)
        assert (1+x[3]+x[4]-x[5])/4 == pytest.approx(.95)
        for kind in ("bell", "werner"):
            y = project(.37*x, kind)
            assert y[0] == pytest.approx(.37)
            assert (y[0]+y[3]+y[4]-y[5])/4 == pytest.approx(.37*.95)
    assert np.all(project(initial_state("r_state", .95), "bell")[1:3] == 0)


def test_bell_projection_commutes_with_swap_and_recurrence_including_success():
    # Different biases and unnormalized masses: test the actual bilinear map.
    a = .4*initial_state("pe", .99, .2)
    b = .7*initial_state("r_state", .93)
    for operation in (accepted_bell_swap, recurrence_distillation):
        actual, probability = operation(a, b)
        reduced, probability_b = operation(project(a, "bell"), project(b, "bell"))
        np.testing.assert_allclose(project(actual, "bell"), reduced, atol=1e-15)
        assert probability == pytest.approx(probability_b)


@pytest.mark.parametrize("protocol", [("s0", "s1"), ("d0", "s0", "d1", "s1", "d2")])
@pytest.mark.parametrize("model", ["bell_source", "bell_each_stage", "channel_twirl", "bell_channel_twirl"])
def test_pauli_all_corrected_exact_time_distributions_with_recurrence(hardware, protocol, model):
    x = initial_state("r_state", .97)
    c = {**candidate(protocol, .5, ["all_corrected"]*2), "cutoff": 4}
    _, p, state = ModelEvaluation(hardware, Numerics(64, 512), x).at_horizon(c, 256)
    _, p_b, state_b = ModelEvaluation(hardware, Numerics(64, 512), x, model).at_horizon(c, 256)
    np.testing.assert_allclose(p, p_b, atol=2e-12)
    indices = [0, 3, 4, 5]
    np.testing.assert_allclose(p[:, None]*state[:, indices], p_b[:, None]*state_b[:, indices], atol=2e-12)


def test_postselection_exposes_bias_even_without_memory_noise():
    h = Hardware(3, (1., 1.), 1., 1., ((0., 0., 0.),)*3, (0.,)*3)
    x = initial_state("pe", 1., .25)
    c = candidate(["s0"], .5, ["psi_pair"])
    physical = ModelEvaluation(h, Numerics(32, 512), x).run(c)
    bell = ModelEvaluation(h, Numerics(32, 512), x, "bell_source").run(c)
    assert physical["status"] == bell["status"] == "converged"
    assert physical["mean_wait"] == pytest.approx(1/.375)
    assert bell["mean_wait"] == pytest.approx(2.)
    assert physical["skf"] == pytest.approx(1.)
    assert bell["skf"] < .7


def test_ad_channel_twirl_preserves_contractions_not_translation(hardware):
    h = replace(hardware, ad_rates=(.02, .03, .04, .05))
    evaluator = ModelEvaluation(h, Numerics(), initial_state("werner", .99), "channel_twirl")
    g, k = evaluator.memory_rates()
    np.testing.assert_array_equal(k, 0)
    np.testing.assert_allclose(g, np.asarray(h.pauli_rates)+np.asarray(h.ad_rates)[:, None]*[.5, .5, 1.])
    # Independently compare one finite-time channel acting on |0><0| endpoints.
    t, rate = 7, h.ad_rates[0]
    mu = np.r_[1, np.exp(-np.asarray(h.pauli_rates[0])*t)]
    physical = sequential_pauli_adc_channel(mu, np.exp(-rate*t))
    reduced = sequential_pauli_adc_channel(np.r_[1, np.exp(-g[0]*t)], 1.)
    np.testing.assert_allclose([physical.scale_x, physical.scale_y, physical.scale_z],
                               [reduced.scale_x, reduced.scale_y, reduced.scale_z])
    assert physical.shift_z > 0 and reduced.shift_z == 0


def test_nonunital_waiting_breaks_bell_reduction(hardware):
    h = replace(hardware, ad_rates=(.04,)*4)
    c = candidate(fixed_protocol(3), .5, ["all_corrected"]*2)
    x = initial_state("pe", .99, .25)
    results = {m: ModelEvaluation(h, Numerics(128, 1024), x, m).run(c) for m in MODELS}
    assert all(r["status"] == "converged" for r in results.values())
    assert abs(results["full"]["eZ"]-results["bell_source"]["eZ"]) > .001
    assert abs(results["full"]["eZ"]-results["bell_each_stage"]["eZ"]) > .001
    assert abs(results["full"]["eZ"]-results["channel_twirl"]["eZ"]) > .001


def test_werner_loses_error_structure_under_noiseless_recurrence():
    x = initial_state("phase_error", .95)
    a, _ = recurrence_distillation(x, x, conditioned=True)
    w = project(x, "werner")
    b, _ = recurrence_distillation(w, w, conditioned=True)
    fidelity = lambda c: (1+c[3]+c[4]-c[5])/4
    assert fidelity(a) == pytest.approx(.905)
    assert fidelity(b) == pytest.approx(.9649643705463184)


def test_candidate_family_nested_and_cutoff_physical_semantics(hardware):
    pool = candidate_pool(3, hardware.p_gen, tree_mode="all_small", placements="vertices")
    assert len({digest(c) for c in pool}) == len(pool)
    assert {c["cutoff"] for c in pool} == {None, 2, 10}
    for c in pool:
        if in_space(c, "swaps"):
            assert in_space(c, "distillation") and in_space(c, "postselection")
        assert in_space(c, "joint")
        assert sum(s.startswith("d") for s in c["protocol"]) <= 2
    c = candidate(["s0", "s1"], .5, ["all_corrected"]*2)
    regular = Evaluation(replace(hardware, cutoff=4), Numerics())
    override = Evaluation(hardware, Numerics())
    _, a, x = regular.at_horizon(c, 128)
    _, b, y = override.at_horizon({**c, "cutoff": 4}, 128)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError, match="cutoff"):
        override.at_horizon({**c, "cutoff": 0}, 128)
    four = candidate_pool(4, [.2]*4)
    assert any(c["policies"] == ["all_corrected", "psi_pair", "all_corrected"] for c in four)
    assert any(c["policies"] == ["all_corrected", "phi_pair", "all_corrected"] for c in four)


def fake_row(i, physical, approximate, status="converged"):
    return dict(candidate_id=str(i), candidate={"id": i}, results={
        "full": dict(status="converged", skr=physical),
        "proxy": dict(status=status, skr=approximate)})


def test_regret_measures_execution_and_retains_zero_ties_and_missingness():
    rows = [fake_row(0, .1, .05), fake_row(1, .04, .2)]
    r = compare(rows, "proxy")
    assert r["selected_id"] == "1"
    assert r["absolute_loss"] == pytest.approx(.06)
    assert r["relative_loss"] == pytest.approx(.6)
    assert r["prediction_error"] == pytest.approx(.16)
    r = compare([fake_row(0, .1, 0), fake_row(1, 0, 0)], "proxy")
    assert r["all_zero_prediction"] and r["tie_count"] == 2
    assert r["tie_loss_min"] == 0 and r["tie_loss_max"] == .1
    assert r["robust_relative_loss"] == 0
    r = compare([fake_row(0, 0, 0), fake_row(1, 0, 0)], "proxy")
    assert r["relative_loss"] is None
    r = compare([*rows, fake_row(2, .3, .5, "unconverged")], "proxy")
    assert r["status"] == "paired_subset" and not r["complete"]
    assert r["paired"] == 2 and r["candidates"] == 3


def test_plan_uses_identical_hardware_and_physical_units(tmp_path):
    args = parser().parse_args(["--preset", "overnight", "--links", "2", "--sources", "r_state",
                               "--fidelities", ".97", "--memory-scales", "1", "--output", str(tmp_path)])
    cases = plan(args)
    assert len(cases) == 2
    p, a = [c["spec"] for c in cases]
    np.testing.assert_array_equal(p["hardware"]["p_gen"], a["hardware"]["p_gen"])
    rate = np.asarray(a["hardware"]["pauli_rates"])+np.asarray(a["hardware"]["ad_rates"])[:, None]*[.5, .5, 1.]
    np.testing.assert_allclose(rate, p["hardware"]["pauli_rates"])
    assert a["metadata"]["slot_seconds"] == pytest.approx(35e-6)
    assert a["metadata"]["total_km"] == 10


def test_fixed_total_distance_recomputes_links_and_pe_q(tmp_path):
    args = parser().parse_args(["--preset", "smoke", "--links", "2", "4", "--sources", "pe",
                               "--total-km", "20", "--pe-q", ".3", "--output", str(tmp_path)])
    cases = plan(args)
    a, b = [c["spec"] for c in cases]
    assert a["metadata"]["total_km"] == b["metadata"]["total_km"] == 20
    assert a["metadata"]["elementary_km"] == 10 and b["metadata"]["elementary_km"] == 5
    assert a["hardware"]["p_gen"][0] < b["hardware"]["p_gen"][0]
    assert all(c["q"] == .3 for c in candidate_pool(2, a["hardware"]["p_gen"], **a["family"]))


def test_dry_plan_round_trip_can_be_resumed(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["twirled_regret", "--preset", "smoke", "--dry-run", "--output", str(tmp_path)])
    main()
    saved = (tmp_path/"plan.json").read_bytes()
    main()
    assert (tmp_path/"plan.json").read_bytes() == saved


def test_small_durable_worker_resume_and_read_only_exports(hardware, tmp_path, monkeypatch):
    spec = dict(hardware=asdict(hardware), numerics=asdict(Numerics(128, 2048)),
                source=dict(family="r_state", fidelity=.99), models=list(MODELS), tie_rtol=1e-4, tie_atol=1e-10,
                family=dict(tree_mode="balanced", cutoff_factors=[None], max_dists=1, acceptance="uniform"),
                metadata=dict(links=3, nodes=4, source="r_state", fidelity_reference=.99, memory_scale=1.,
                              noise="pauli", geometry="balanced", elementary_km=5., total_km=15., slot_seconds=35e-6))
    output = tmp_path/"case"
    first = run(spec, output)
    assert first["candidates"] > 0
    before = (output/"full/trials.jsonl").read_bytes()
    monkeypatch.setattr(ModelEvaluation, "run", lambda *a: pytest.fail("resume must not evaluate"))
    assert run(spec, output, resume=True) == first
    assert (output/"full/trials.jsonl").read_bytes() == before
    # Summarization must work without invoking a backend and includes all spaces.
    monkeypatch.setattr("src.scripts.twirled_regret.plot_results", lambda *a: None)
    rows = summarize([dict(name="case", directory=str(output), spec=spec)], tmp_path)
    assert {r["space"] for r in rows} == {"swaps", "distillation", "postselection", "joint"}
    assert all(r["relative_loss"] == 0 for r in rows if r["space"] == "distillation" and r["model"] == "bell_source")
    saved = json.loads((tmp_path/"evaluations.json").read_text())
    assert all(r["skr_bits_per_second"] == pytest.approx(r["skr"]/35e-6) for r in saved if r["status"] == "converged")
