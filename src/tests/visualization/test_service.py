from copy import deepcopy
import csv
from io import StringIO
import json

import numpy as np
import pytest

from src.core.pauli_adc.protocol_units import (accepted_bell_swap, apply_local_channels,
                                             sequential_pauli_adc_channel)
from src.core.pauli_adc.state import RealXState, phi_plus_fidelity
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.visualization.comparisons import (candidate_key, comparison_plan, near_tie,
                                          scenario_key, selection_loss)
from src.visualization.config import DEFAULT, converted, expand_config, load_config
from src.visualization.evaluate import OPTIONS, EvaluationService, cache_identity
from src.visualization.metrics import bell_populations, summarize, validate_outputs
from src.visualization.plots import state_figure, time_figures
from src.visualization.protocols import compile_protocol, ordered_trees
from src.visualization.storage import (config_json, figures_html, load_run, per_time_csv,
                                       save_run, summary_csv)


@pytest.fixture
def config():
    c = deepcopy(DEFAULT)
    c["numerics"]["t_trunc"] = 32
    return c


def test_default_mapping_and_heterogeneous_endpoints(config):
    _, p, _ = expand_config(config)
    assert np.shape(p["lambdas"]) == (4, 4)
    assert np.shape(p["pauli_mode_decay_rates"]) == (5, 3)
    np.testing.assert_allclose(p["lambdas"][0], [0.98, 0.02/3, 0.02/3, 0.02/3])
    np.testing.assert_allclose(p["pauli_mode_decay_rates"], [[0.00075, 0.00075, 0]]*5)
    np.testing.assert_allclose(p["amplitude_damping_rate"], [0.0005]*5)
    config["hardware"]["p_gen"] = [0.2, 0.05, 0.3, 0.4]
    config["memory"] = {"kind": "t1_t2", "t1_slots": [None, 100, 200, 300, 400],
                        "t2_slots": [None, 50, 100, 150, 200]}
    _, p, _ = expand_config(config)
    np.testing.assert_allclose(p["p_gen"], [0.2, 0.05, 0.3, 0.4])
    np.testing.assert_allclose(p["amplitude_damping_rate"], [0, 0.01, 0.005, 1/300, 0.0025])
    np.testing.assert_allclose(np.asarray(p["pauli_mode_decay_rates"])[:, 0],
                               [0, 0.015, 0.0075, 0.005, 0.00375])


@pytest.mark.parametrize("path,value", [
    (("memory", "t1_slots"), 0), (("memory", "t2_slots"), 5000),
    (("memory", "t1_slots"), [100]*4), (("hardware", "p_gen"), [0.1]*3),
    (("hardware", "p_swap"), -0.1), (("hardware", "p_distillation"), float("nan")),
    (("initial_state", "fidelity"), 1.1), (("numerics", "t_trunc"), 2.5),
    (("protocol", "bell_outcomes"), "per_swap"), (("prediction_model",), "werner_projection"),
    (("slot_duration_seconds",), 0), (("hardware", "p_gen"), True),
])
def test_invalid_config_is_rejected_before_backend(config, path, value):
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        expand_config(config)


@pytest.mark.parametrize("initial", [
    {"kind": "bell_weights", "values": [0.8, 0.1, 0.1, 0.1]},
    {"kind": "bell_weights", "values": [1.1, -0.1, 0, 0]},
    {"kind": "real_x", "values": [1, 1, 1, -1, 1, 1]},
    {"kind": "real_x", "values": [0.9, 0, 0, 0, 0, 0]},
    {"kind": "matrix", "values": np.eye(4).tolist()},
])
def test_invalid_states(config, initial):
    config["initial_state"] = initial
    with pytest.raises(ValueError):
        expand_config(config)


@pytest.mark.parametrize("rates", [[1, 0, 0], [-1, 1, 1], [float("inf"), 1, 1], [1, 1]])
def test_invalid_pauli_rates(config, rates):
    config["memory"] = {"kind": "pauli_adc", "pauli_mode_decay_rates": rates,
                        "amplitude_damping_rate": 0}
    with pytest.raises(ValueError):
        expand_config(config)


def test_tree_operations_and_cutoff_identity(config):
    assert len(ordered_trees(4)) == 5
    protocol = config["protocol"]
    sequence, ops = compile_protocol(protocol, 4, 32)
    assert sequence == ("s0", "s2", "s1")
    assert [o["id"] for o in ops] == ["swap:0:1", "swap:2:3", "swap:0:3"]
    protocol["distillation_rounds"] = {"0:3": 1, "0:0": 1, "2:3": 1}
    protocol["cutoff_overrides"] = {"swap:0:3": 7, "dist:2:3": 2, "dist:0:3": None}
    sequence, ops = compile_protocol(protocol, 4, 32)
    assert sequence == ("d0", "s0", "s2", "d3", "s1", "d3")
    assert [o["backend_cutoff"] for o in ops] == [32, 32, 32, 2, 7, 32]
    assert [o["endpoint_nodes"] for o in ops] == [[0, 1], [0, 2], [2, 4], [2, 4], [0, 4], [0, 4]]
    protocol["distillation_rounds"] = {"0:3": 1}
    protocol["cutoff_overrides"] = {}
    assert compile_protocol(protocol, 4, 32)[0] == ("s0", "s2", "s1", "d3")


@pytest.mark.parametrize("tree", [[[1, 0], [2, 3]], [[0, 0], [2, 3]], [0, 1], [0, 1, 2], "s0"])
def test_invalid_tree(config, tree):
    config["protocol"]["tree"] = tree
    with pytest.raises(ValueError):
        expand_config(config)


def test_stale_overrides_and_unsupported_rounds(config):
    config["protocol"]["cutoff_overrides"] = {"dist:0:3": 10}
    with pytest.raises(ValueError, match="nonexistent"):
        expand_config(config)
    config["protocol"]["cutoff_overrides"] = {}
    config["protocol"]["distillation_rounds"] = {"0:3": 2}
    with pytest.raises(ValueError, match="0 or 1"):
        expand_config(config)


def test_service_direct_and_cli_agree(config, tmp_path, capsys):
    from src.visualization.cli import main
    config["hardware"]["p_gen"] = [0.2, 0.05, 0.3, 0.4]
    config["memory"] = {"kind": "t1_t2", "t1_slots": [None, 100, 200, 300, 400],
                        "t2_slots": [None, 50, 100, 150, 200]}
    config["protocol"].update(memory_cutoff_slots=3, bell_outcomes="psi_pair",
                              distillation_rounds={"0:3": 1})
    _, parameters, _ = expand_config(config)
    pmf, x = RepeaterChainEvaluation(state_type=RealXState, **OPTIONS).asymmetric_heterogeneous_protocol(
        deepcopy(parameters), 4)
    service_run = EvaluationService().evaluate(config)
    np.testing.assert_allclose(service_run.pmf, pmf, atol=1e-12)
    np.testing.assert_allclose(service_run.pmf[:, None]*service_run.x, pmf[:, None]*x, atol=1e-12)
    config_file = tmp_path/"config.json"
    config_file.write_text(config_json(config))
    assert main(["--config", str(config_file), "--output", str(tmp_path/"runs")]) == 0
    cli_run = load_run(next((tmp_path/"runs").glob("run-*")))
    np.testing.assert_allclose(cli_run.pmf, service_run.pmf, atol=1e-12)
    np.testing.assert_allclose(cli_run.pmf[:, None]*cli_run.x,
                               service_run.pmf[:, None]*service_run.x, atol=1e-12)
    assert cli_run.summary == service_run.summary
    assert "bundle" in capsys.readouterr().out


def test_channel_twirl_control_and_physical_config_preserved(config):
    config["memory"] = {"kind": "t1_t2", "t1_slots": None, "t2_slots": 100}
    original = deepcopy(config)
    service = EvaluationService()
    full = service.evaluate(config)
    twirl = service.evaluate(converted(config, model="local_channel_twirl"))
    np.testing.assert_allclose(full.pmf, twirl.pmf, atol=1e-12)
    np.testing.assert_allclose(full.pmf[:, None]*full.x, twirl.pmf[:, None]*twirl.x, atol=1e-12)
    assert config == original
    config["memory"] = {"kind": "t1_t2", "t1_slots": 200, "t2_slots": 100}
    twirled = converted(config, model="local_channel_twirl")
    _, p, _ = expand_config(twirled)
    np.testing.assert_allclose(p["pauli_mode_decay_rates"], [[0.01, 0.01, 0.005]]*5)
    assert twirled["memory"] == config["memory"]
    _, restored, _ = expand_config(converted(twirled, model="full_real_x"))
    np.testing.assert_allclose(restored["amplitude_damping_rate"], [0.005]*5)
    channel = sequential_pauli_adc_channel([1, 1, 1, 1], 0.8)
    full_x = apply_local_channels([1, 0, 0, 1, 1, -1], channel, channel)
    bell_x = full_x.copy()
    bell_x[1:3] = 0
    twirled_channel = sequential_pauli_adc_channel([1, 0.8**0.5, 0.8**0.5, 0.8], 1)
    approx_x = apply_local_channels([1, 0, 0, 1, 1, -1], twirled_channel, twirled_channel)
    assert phi_plus_fidelity(full_x) == pytest.approx(0.82)
    assert phi_plus_fidelity(bell_x) == pytest.approx(0.82)
    assert phi_plus_fidelity(approx_x) == pytest.approx(0.81)


def test_corrected_swap_preserves_bell_populations_of_biased_inputs():
    channel = sequential_pauli_adc_channel([1, 1, 1, 1], 0.8)
    a = apply_local_channels([1, 0, 0, 1, 1, -1], channel)
    b = apply_local_channels([1, 0, 0, 1, 1, -1], channel, channel)
    ap, bp = a.copy(), b.copy()
    ap[1:3] = bp[1:3] = 0
    full, _ = accepted_bell_swap(a, b)
    projected, _ = accepted_bell_swap(ap, bp)
    np.testing.assert_allclose(bell_populations(full), bell_populations(projected), atol=1e-12)
    full_q = summarize(np.array([0, 1.0]), np.tile(full, (2, 1)), 0.99)[0]
    projected_q = summarize(np.array([0, 1.0]), np.tile(projected, (2, 1)), 0.99)[0]
    assert full_q["pooled_secret_fraction"] == pytest.approx(projected_q["pooled_secret_fraction"])


@pytest.mark.parametrize("zero_field", [None, "p_gen", "p_swap", "p_distillation"])
def test_deterministic_and_no_completion_have_safe_exports(config, zero_field, tmp_path):
    config["hardware"] = {"p_gen": 1, "p_swap": 1, "p_distillation": 1}
    config["initial_state"]["fidelity"] = 1
    config["memory"].update(t1_slots=None, t2_slots=None)
    if zero_field:
        config["hardware"][zero_field] = 0
        if zero_field == "p_distillation":
            config["protocol"]["distillation_rounds"] = {"0:3": 1}
    run = EvaluationService().evaluate(config)
    if zero_field:
        assert run.summary["status"] == "no_completion"
        assert run.summary["coverage"] == run.summary["rate_bits_per_slot"] == 0
        assert run.summary["pooled_coordinates"] is None
    else:
        assert run.pmf[1] == pytest.approx(1)
        assert run.summary["average_fidelity"] == run.summary["pooled_secret_fraction"] == pytest.approx(1)
    serialized = config_json(run.metadata)
    assert "NaN" not in serialized and "Infinity" not in serialized
    loaded = load_run(save_run(run, tmp_path))
    np.testing.assert_array_equal(loaded.pmf, run.pmf)
    assert "nan" not in per_time_csv(run).lower()
    assert len(time_figures([run])) == 2


def test_horizon_cache_metadata_and_roundtrip(config, tmp_path):
    service = EvaluationService(cache_size=2)
    config["numerics"]["t_trunc"] = 8
    run = service.evaluate(config)
    assert run.metadata["diagnostics"]["provisional"]
    doubled = service.evaluate(converted(config, double_horizon=True))
    assert doubled.metadata["backend_parameters"]["cutoff"] == [16]*3
    assert run.config["protocol"]["memory_cutoff_slots"] is None
    assert doubled.metadata["run_id"] != run.metadata["run_id"]
    assert doubled.summary["coverage"] > run.summary["coverage"]
    rename = deepcopy(config)
    rename["name"] = "Renamed without changing dynamics"
    rename["slot_duration_seconds"] = 0.5
    rename["numerics"]["coverage_target"] = 0.1
    assert cache_identity(rename) == cache_identity(config)
    renamed = service.evaluate(rename)
    assert service.backend_calls == 2
    assert renamed.config == rename
    assert renamed.summary["rate_bits_per_second"] == pytest.approx(2*renamed.summary["rate_bits_per_slot"])
    bundle = save_run(run, tmp_path)
    another = save_run(run, tmp_path)
    assert bundle != another
    loaded = load_run(bundle)
    np.testing.assert_array_equal(loaded.pmf, run.pmf)
    np.testing.assert_array_equal(loaded.x, run.x)
    assert loaded.metadata == json.loads(config_json(run.metadata))
    assert load_config(config_json(loaded.config)) == config
    rows = list(csv.DictReader(StringIO(per_time_csv(loaded))))
    assert len(rows) == 8
    assert rows[0]["II"] == ""
    assert float(rows[1]["weighted_II"]) == pytest.approx(loaded.pmf[1])
    assert "rate_bits_per_slot" in summary_csv([loaded])
    html = figures_html(time_figures([loaded])+[state_figure(loaded)])
    assert "plotly.js" in html and "cdn.plot.ly" not in html[:200]


def test_source_change_invalidates_cache(config, monkeypatch):
    import src.visualization.evaluate as module
    service = EvaluationService()
    service.evaluate(config)
    monkeypatch.setattr(module, "source_fingerprint", lambda: "changed-code")
    assert not service.is_cached(config)
    service.evaluate(config)
    assert service.backend_calls == 2


def test_cache_bound(config):
    service = EvaluationService(cache_size=2)
    for horizon in (8, 16, 32):
        config["numerics"]["t_trunc"] = horizon
        service.evaluate(config)
    assert len(service.cache) == 2


@pytest.mark.parametrize("defect", ["negative", "mass", "zero_bin", "nan", "state"])
def test_invalid_outputs_raise(defect):
    p = np.array([0.0, 0.3, 0.2])
    x = np.tile([1, 0, 0, 1, 1, -1], (3, 1)).astype(float)
    if defect == "negative":
        p[2] = -0.01
    elif defect == "mass":
        p[1] = 1
    elif defect == "zero_bin":
        p[0] = 0.01
    elif defect == "nan":
        x[1, 0] = float("nan")
    else:
        x[1] = [1, 1, 1, -1, 1, 1]
    with pytest.raises(ValueError):
        validate_outputs(p, x, 3)


def test_documented_roundoff_is_recorded():
    p, x, cleanup = validate_outputs([1e-14, 0.5, -1e-14],
                                     np.tile([1, 0, 0, 0, 0, 0], (3, 1)), 3)
    np.testing.assert_array_equal(p, [0, 0.5, 0])
    assert {entry["kind"] for entry in cleanup} == {"bin_zero_residue", "negative_mass_roundoff"}


def test_controlled_comparison_errors_loss_ties_and_zero(config):
    service = EvaluationService()
    variant = deepcopy(config)
    variant["protocol"]["memory_cutoff_slots"] = 10
    plan = comparison_plan([config, variant])
    runs = [service.evaluate(c) for c in plan]
    expected = {candidate_key(c) for c in plan}
    # Synthetic headline rates isolate ranking logic from scientific integration tests above.
    for r, rate in zip(runs, [0.10, 0.06, 0.08, 0.09]):
        r.summary["rate_bits_per_slot"] = rate
    loss = selection_loss(runs, expected)
    assert loss["absolute_selection_loss_range"] == pytest.approx([0.02, 0.02])
    assert loss["relative_selection_loss_range"] == pytest.approx([0.2, 0.2])
    assert loss["prediction_errors_bits_per_slot"][candidate_key(config)] == pytest.approx(-0.04)
    with pytest.raises(ValueError, match="incomplete"):
        selection_loss(runs[:-1], expected)
    bad = deepcopy(config)
    bad["hardware"]["p_swap"] = 0.8
    with pytest.raises(ValueError, match="scenario"):
        comparison_plan([config, bad])
    bad = converted(config, double_horizon=True)
    with pytest.raises(ValueError, match="horizons"):
        comparison_plan([config, bad])
    for r in runs:
        r.summary["rate_bits_per_slot"] = 0
    loss = selection_loss(runs, expected)
    assert loss["relative_selection_loss_range"] is None
    assert len(loss["full_winners"]) == len(loss["twirl_winners"]) == 2
    assert near_tie(0.1, 0.10001) and not near_tie(0.1, 0.101)
    for r, rate in zip(runs, [0.1, 0.10001, 0.08, 0.1]):
        r.summary["rate_bits_per_slot"] = rate
    tied = selection_loss(list(reversed(runs)), expected)
    assert tied["absolute_selection_loss_range"] == pytest.approx([0, 0.02])
    assert tied["selected_full_rate_range"] == pytest.approx([0.08, 0.1])


def test_equivalent_scenario_spellings(config):
    alternate = deepcopy(config)
    alternate["hardware"]["p_gen"] = [0.2]*4
    alternate["memory"]["t1_slots"] = [2000]*5
    alternate["memory"]["t2_slots"] = [1000]*5
    assert scenario_key(alternate) == scenario_key(config)


def test_twelve_candidates_and_general_real_x_channel_baseline(config):
    config["links"] = 1
    config["protocol"]["tree"] = 0
    config["initial_state"] = {"kind": "real_x", "values": [1, 0.2, 0.2, 0.68, 0.8, -0.8]}
    service = EvaluationService()
    twirl = service.evaluate(converted(config, model="local_channel_twirl"))
    assert twirl.summary["pooled_coordinates"][1] == pytest.approx(0.2)
    assert twirl.summary["pooled_coordinates"][2] == pytest.approx(0.2)
    candidates = []
    for cutoff in range(12):
        c = deepcopy(config)
        c["protocol"]["distillation_rounds"] = {"0:0": 1}
        c["protocol"]["memory_cutoff_slots"] = cutoff
        candidates.append(c)
    plan = comparison_plan(candidates)
    matrix = [service.evaluate(c) for c in plan]
    assert len(matrix) == 24
    assert selection_loss(matrix)["candidates"] == 12
