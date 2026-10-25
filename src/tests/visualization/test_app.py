"""Exercise the real app: explicit runs, compact layout, studies and persistence."""
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.visualization.config import DEFAULT
from src.visualization.storage import load_run


def widget(items, label):
    return next(item for item in items if item.label == label)


def tree_args(at):
    return next(args for item in at.get("component_instance")
                if (args := json.loads(item.proto.json_args))["editable"])


def edit_node(at, span, count):
    args = tree_args(at)
    counts = {node["span"]: node["rounds"] for node in args["nodes"]}
    counts[span] = count
    at.session_state[args["key"]] = dict(tree_id=args["tree_id"], event_id=str(uuid4()), rounds=counts)
    at.run()


@pytest.fixture
def app(tmp_path, monkeypatch):
    import src.visualization.storage as storage
    monkeypatch.setattr(storage, "DEFAULT_DIRECTORY", tmp_path)
    at = AppTest.from_file(str(Path(__file__).parents[2]/"visualization"/"app.py"), default_timeout=60).run()
    assert not at.exception
    return at, tmp_path


def fast_controls(at):
    widget(at.number_input, "Horizon").set_value(128)
    widget(at.number_input, "Generation success").set_value(1.)
    widget(at.number_input, "Swap hardware success").set_value(1.)


def test_compact_defaults_explicit_run_exports_edits_reload_and_horizon(app):
    at, directory = app
    service = at.session_state["service"]
    assert service.backend_calls == 0
    assert widget(at.number_input, "Horizon").value == 10000
    assert widget(at.number_input, "Generation success").value == .02
    assert widget(at.number_input, "Swap hardware success").value == .5
    assert widget(at.selectbox, "State mode").value == "Werner fidelity"
    assert {e.label for e in at.expander} >= {"Load a scenario or configuration", "Advanced settings",
                                             "More visualizations", "Saved runs / Compare"}
    fast_controls(at)
    widget(at.button, "Run").click().run()
    assert not at.exception and not at.error
    assert service.backend_calls == 1
    result = at.session_state["last"]
    assert result.config["numerics"]["t_trunc"] == 128
    assert {m.label for m in at.metric} == {"SKR (bits/slot)", "SKF", "Average waiting time",
                                          "Average Φ+ fidelity", "Covered probability"}
    assert len(list(directory.glob("run-*"))) == 1
    assert {d.proto.label for d in at.get("download_button")} >= {"Configuration JSON", "Arrays NPZ", "Per-time CSV", "HTML figures"}
    widget(at.checkbox, "Logarithmic PMF axis").check().run()
    widget(at.selectbox, "Fidelity view").set_value("Probability distribution").run()
    widget(at.number_input, "Initial fidelity F0").set_value(.95).run()
    assert not at.exception and service.backend_calls == 1
    assert at.session_state["last"].config == result.config
    widget(at.number_input, "T2 (slots)").set_value(10000.).run()
    assert at.error and service.backend_calls == 1
    assert at.session_state["last"].metadata["run_id"] == result.metadata["run_id"]
    widget(at.selectbox, "Saved run").set_value(next(directory.glob("run-*")))
    widget(at.button, "Load saved run").click().run()
    assert not at.exception and not at.error
    assert at.session_state["last"].metadata == load_run(next(directory.glob("run-*"))).metadata
    widget(at.button, "Run at 2× horizon").click().run()
    assert not at.exception and service.backend_calls == 2
    assert at.session_state["last"].metadata["horizon"] == 256
    assert len(list(directory.glob("run-*"))) == 2


def test_tree_slider_counters_and_advanced_controls_do_not_evaluate(app):
    at, _ = app
    service = at.session_state["service"]
    assert widget(at.slider, "Tree shape").value == 0
    widget(at.slider, "Tree shape").set_value(1.).run()
    assert at.session_state["draft"]["protocol"]["tree"] == [0, [1, [2, 3]]]
    widget(at.slider, "Tree shape").set_value(0.).run()
    assert at.session_state["draft"]["protocol"]["tree"] == [[0, 1], [2, 3]]
    widget(at.slider, "Tree shape").set_value(-1.).run()
    assert at.session_state["draft"]["protocol"]["tree"] == [[[0, 1], 2], 3]
    edit_node(at, "0:3", 2)
    assert at.session_state["draft"]["protocol"]["distillation_rounds"] == {"0:3": 2}
    assert next(node for node in tree_args(at)["nodes"] if node["span"] == "0:3")["rounds"] == 2
    assert not at.get("plotly_chart")
    assert not any(item.label.startswith(("Link ", "Span ")) for item in at.number_input)
    widget(at.toggle, "Per-link / per-node settings").set_value(True).run()
    assert not at.exception and not at.error
    assert len(at.session_state["draft"]["memory"]["t1_slots"]) == 5
    widget(at.selectbox, "Memory controls").set_value("pauli_adc").run()
    assert not at.exception and not at.error and service.backend_calls == 0
    assert len(at.session_state["draft"]["memory"]["pauli_mode_decay_rates"]) == 5
    c = deepcopy(DEFAULT)
    c["links"] = 2
    c["protocol"]["tree"] = [0, 1]
    c["initial_state"] = {"kind": "real_x", "values": [1, .1, .1, .82, .9, -.9]}
    c["numerics"]["t_trunc"] = 64
    c["prediction_model"] = "local_channel_twirl"
    widget(at.text_area, "Paste configuration JSON").set_value(json.dumps(c))
    widget(at.button, "Load pasted configuration").click().run()
    assert not at.exception and not at.error and service.backend_calls == 0
    assert at.session_state["draft"] == c
    widget(at.button, "Run").click().run()
    assert not at.exception and at.session_state["last"].config == c


def test_sweep_runs_only_on_submission_and_inspects_each_result(app):
    at, directory = app
    service = at.session_state["service"]
    fast_controls(at)
    widget(at.radio, "Mode").set_value("Parameter sweep").run()
    widget(at.number_input, "Points").set_value(2).run()
    assert service.backend_calls == 0
    widget(at.button, "Run sweep").click().run()
    assert not at.exception and not at.error
    study = at.session_state["experiment"]
    assert study.finished and len(study.entries) == 2
    assert service.backend_calls == 2
    assert len(list(directory.glob("study-*/history.json"))) == 1
    widget(at.selectbox, "Inspect trial").set_value(2).run()
    assert at.session_state["last"].config["memory"]["t1_slots"] == 4000
    widget(at.checkbox, "Logarithmic PMF axis").check().run()
    assert not at.exception and service.backend_calls == 2


def test_tikz_example_counts_import_edit_and_repeated_balanced_positions(app):
    at, _ = app
    c = deepcopy(DEFAULT)
    c["links"] = 3
    c["protocol"].update(tree=[[0, 1], 2],
        distillation_rounds={"0:2": 2, "0:0": 1, "1:1": 1, "2:2": 1})
    widget(at.text_area, "Paste configuration JSON").set_value(json.dumps(c))
    widget(at.button, "Load pasted configuration").click().run()
    assert not at.exception and not at.error
    assert widget(at.slider, "Tree shape").value == 0
    assert {n["span"]: n["rounds"] for n in tree_args(at)["nodes"]} == {
        "0:2": 2, "0:1": 0, "0:0": 1, "1:1": 1, "2:2": 1}
    for rounds in (3, 0, 2):
        edit_node(at, "0:2", rounds)
        assert next(n for n in tree_args(at)["nodes"] if n["span"] == "0:2")["rounds"] == rounds
    for position in (-1., 1., 0.):
        widget(at.slider, "Tree shape").set_value(position).run()
        assert not at.exception
        assert next(n for n in tree_args(at)["nodes"] if n["span"] == "0:2")["rounds"] == 2
    assert at.session_state["draft"]["protocol"]["tree"] == [[0, 1], 2]
    assert at.session_state["service"].backend_calls == 0


def test_node_edit_is_used_by_run_and_saved_configuration(app):
    at, directory = app
    c = deepcopy(DEFAULT)
    c["links"] = 2
    c["protocol"]["tree"] = [0, 1]
    c["hardware"].update(p_gen=.5, p_swap=1.)
    c["numerics"]["t_trunc"] = 32
    widget(at.text_area, "Paste configuration JSON").set_value(json.dumps(c))
    widget(at.button, "Load pasted configuration").click().run()
    edit_node(at, "0:1", 2)
    assert not at.exception and at.session_state["service"].backend_calls == 0
    widget(at.button, "Run").click().run()
    assert not at.exception and not at.error
    assert at.session_state["last"].config["protocol"]["distillation_rounds"] == {"0:1": 2}
    assert load_run(next(directory.glob("run-*"))).config["protocol"]["distillation_rounds"] == {"0:1": 2}


def test_tree_change_does_not_replay_edits_from_previous_shape(app):
    at, _ = app
    edit_node(at, "0:1", 1)
    old_args = tree_args(at)
    old_event = deepcopy(at.session_state[old_args["key"]])
    widget(at.slider, "Tree shape").set_value(1.).run()
    assert at.session_state["draft"]["protocol"]["distillation_rounds"] == {}
    # A delayed event for the previous topology is consumed, but never applied.
    old_event["event_id"] = str(uuid4())
    at.session_state[old_args["key"]] = old_event
    at.run()
    widget(at.slider, "Tree shape").set_value(0.).run()
    assert not at.exception and not at.error
    assert at.session_state["draft"]["protocol"]["distillation_rounds"] == {}
    assert at.session_state["service"].backend_calls == 0


def test_bayesian_budget_persists_history_and_selects_best_eligible(app):
    at, directory = app
    fast_controls(at)
    widget(at.radio, "Mode").set_value("Bayesian optimization").run()
    widget(at.multiselect, "Search protocol controls").set_value(["cutoff"])
    widget(at.number_input, "Trial budget").set_value(3)
    widget(at.number_input, "Initial trials").set_value(2)
    widget(at.number_input, "Maximum searched cutoff").set_value(1)
    at.run()
    assert at.session_state["service"].backend_calls == 0
    widget(at.button, "Optimize").click().run()
    assert not at.exception and not at.error
    study = at.session_state["experiment"]
    assert study.finished and len(study.entries) == 3
    assert study.best is not None
    trees = [json.loads(item.proto.json_args) for item in at.get("component_instance")]
    assert [item["editable"] for item in trees] == [True, False]
    assert at.session_state["last"].metadata["run_id"] == study.best.run.metadata["run_id"]
    history = json.loads(next(directory.glob("study-*/history.json")).read_text())
    assert len(history["trials"]) == 3 and history["best_trial"] == study.best.index
    calls = at.session_state["service"].backend_calls
    at.run()
    assert not at.exception and at.session_state["service"].backend_calls == calls


def test_failed_run_preserves_last_success(app, monkeypatch):
    at, _ = app
    fast_controls(at)
    widget(at.button, "Run").click().run()
    result = at.session_state["last"]
    service = at.session_state["service"]
    monkeypatch.setattr(service, "evaluate", lambda config: (_ for _ in ()).throw(ValueError("deliberate failure")))
    widget(at.button, "Run").click().run()
    assert not at.exception and any("deliberate failure" in e.value for e in at.error)
    assert at.session_state["last"].metadata["run_id"] == result.metadata["run_id"]
