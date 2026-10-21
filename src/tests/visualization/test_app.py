"""Exercise the real Streamlit script: form batching, failures, exports and disk reload."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.visualization.config import DEFAULT
from src.visualization.storage import load_run


def widget(items, label):
    return next(item for item in items if item.label == label)


@pytest.fixture
def app(tmp_path, monkeypatch):
    import src.visualization.storage as storage
    save = storage.save_run
    monkeypatch.setattr(storage, "DEFAULT_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage, "save_run", lambda run: save(run, tmp_path))
    at = AppTest.from_file(str(Path(__file__).parents[2]/"visualization"/"app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at, tmp_path


def test_form_display_exports_failure_reload_and_horizon(app):
    at, directory = app
    service = at.session_state["service"]
    assert service.backend_calls == 0
    widget(at.number_input, "Horizon T (bins 0,...,T-1)").set_value(32)
    widget(at.button, "Preview configuration").click().run()
    assert not at.exception and service.backend_calls == 0
    widget(at.button, "Evaluate").click().run()
    assert not at.exception and not at.error
    assert service.backend_calls == 1
    result = at.session_state["last"]
    assert result.config["numerics"]["t_trunc"] == 32
    assert len(list(directory.glob("run-*"))) == 1
    downloads = at.get("download_button")
    assert {d.proto.label for d in downloads} >= {
        "Export submitted config JSON", "Download per-time CSV", "Download self-contained HTML figures"}
    widget(at.checkbox, "Logarithmic PMF axis").check().run()
    widget(at.selectbox, "Inspect state").set_value(1).run()
    widget(at.text_input, "Run name").set_value("Edited form, original result retained").run()
    assert service.backend_calls == 1
    assert at.session_state["last"].config == result.config
    widget(at.text_area, "T1 in slots (null = infinite)").set_value("-1")
    widget(at.button, "Evaluate").click().run()
    assert not at.exception and at.error
    assert at.session_state["last"].metadata["run_id"] == result.metadata["run_id"]
    assert service.backend_calls == 1
    widget(at.text_area, "T1 in slots (null = infinite)").set_value("2000")
    widget(at.button, "Reload result").click().run()
    assert not at.exception
    assert at.session_state["last"].metadata == load_run(next(directory.glob("run-*"))).metadata
    widget(at.button, "Re-evaluate at 2× horizon").click().run()
    assert not at.exception
    assert service.backend_calls == 2
    assert at.session_state["last"].metadata["horizon"] == 64
    assert len(list(directory.glob("run-*"))) == 2
    assert "horizon_change" in at.session_state


def test_heterogeneous_controls_and_import_dont_evaluate(app):
    at, _ = app
    service = at.session_state["service"]
    widget(at.toggle, "Per-link / per-node tables").set_value(True).run()
    assert not at.exception and not at.error
    widget(at.button, "Preview configuration").click().run()
    assert not at.exception and not at.error and service.backend_calls == 0
    assert len(at.session_state["draft"]["memory"]["t1_slots"]) == 5
    widget(at.selectbox, "Memory controls").set_value("pauli_adc").run()
    widget(at.button, "Preview configuration").click().run()
    assert not at.exception and not at.error and service.backend_calls == 0
    assert len(at.session_state["draft"]["memory"]["pauli_mode_decay_rates"]) == 5
    config = deepcopy(DEFAULT)
    config["links"] = 2
    config["protocol"]["tree"] = [0, 1]
    config["prediction_model"] = "local_channel_twirl"
    config["name"] = "Imported scenario"
    config["numerics"]["t_trunc"] = 16
    widget(at.text_area, "Or paste a complete version-1 config").set_value(json.dumps(config))
    widget(at.button, "Load pasted configuration").click().run()
    assert not at.exception and not at.error
    assert service.backend_calls == 0
    assert at.session_state["draft"] == config
    widget(at.button, "Evaluate").click().run()
    assert not at.exception and not at.error
    assert at.session_state["last"].config == config
    widget(at.button, "Re-evaluate in full model").click().run()
    assert not at.exception and not at.error
    assert at.session_state["last"].config["memory"] == config["memory"]
    assert at.session_state["last"].config["prediction_model"] == "full_real_x"


def test_manual_candidate_comparison_progress_and_display_cache(app, monkeypatch):
    at, directory = app
    service = at.session_state["service"]
    widget(at.number_input, "Horizon T (bins 0,...,T-1)").set_value(32)
    widget(at.button, "Evaluate").click().run()
    widget(at.text_input, "Shared memory cutoff in slots (null = No cutoff)").set_value("10")
    widget(at.button, "Evaluate").click().run()
    assert not at.exception and service.backend_calls == 2
    selector = widget(at.multiselect, "Saved runs to overlay / compare (up to 12)")
    selector.set_value([str(p) for p in directory.glob("run-*")]).run()
    widget(at.button, "Evaluate candidates in both models").click().run()
    assert not at.exception and not at.error
    assert len(at.session_state["recent_comparison"]) == 4
    assert service.backend_calls == 4  # full-model rows were cached
    at.run()
    assert not at.exception and service.backend_calls == 4
    evaluate = service.evaluate

    def fail_one(config):
        if config["prediction_model"] == "local_channel_twirl" and config["protocol"]["memory_cutoff_slots"] == 10:
            raise ValueError("deliberate partial-matrix failure")
        return evaluate(config)

    monkeypatch.setattr(service, "evaluate", fail_one)
    widget(at.button, "Evaluate candidates in both models").click().run()
    assert not at.exception
    assert len(at.session_state["recent_comparison"]) == 3
    assert any("incomplete" in warning.value for warning in at.warning)
