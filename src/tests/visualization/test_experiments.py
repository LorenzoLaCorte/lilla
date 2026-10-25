from copy import deepcopy
import json

import numpy as np
import pytest

from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService, Run
from src.visualization.experiments import Experiment
from src.visualization.plots import main_figures, sweep_figures, tree_figure
from src.visualization.protocols import (balanced_tree, ordered_trees, symmetry_trees,
                                         tree_slider_choices, tree_slider_position)
from src.visualization.storage import load_run, save_experiment


@pytest.fixture
def config():
    c = deepcopy(DEFAULT)
    c["hardware"].update(p_gen=1, p_swap=1)
    c["memory"].update(t1_slots=None, t2_slots=None)
    c["numerics"]["t_trunc"] = 128
    return c


@pytest.mark.parametrize("links", [1, 2, 4, 5, 8])
def test_symmetry_order_keeps_mirrors_and_library_trees_immutable(links):
    entries = symmetry_trees(links)
    assert [s for s, _ in entries] == sorted(s for s, _ in entries)
    trees = ordered_trees(links)
    left_comb = 0
    for index in range(1, links):
        left_comb = [left_comb, index]
    assert trees[0] == left_comb
    assert len({json.dumps(t) for t in trees}) == len(trees)
    if links > 2:
        assert [0, [1, 2]] in ordered_trees(3)
        assert [[0, 1], 2] in ordered_trees(3)
    if isinstance(trees[0], list):
        trees[0][0] = "mutated"
        assert ordered_trees(links)[0] == left_comb


@pytest.mark.parametrize("links", range(1, 9))
def test_tree_slider_endpoints_center_mirror_arms_and_complete_search_space(links):
    from src.types.protocol_types import generate_swap_space
    choices = tree_slider_choices(links)
    middle = len(choices)//2
    assert len(choices) % 2 == 1
    assert choices[middle] == balanced_tree(links)
    assert tree_slider_position(balanced_tree(links), links) == 0
    left, right = 0, links-1
    for i in range(1, links):
        left = [left, i]
        right = [links-1-i, right]
    assert choices[0] == left and choices[-1] == right
    catalogue = {json.dumps(tree) for tree in ordered_trees(links)}
    assert len(catalogue) == len(list(generate_swap_space(links)))
    assert {json.dumps(tree) for tree in choices} == catalogue
    assert max(_leaf_depths(choices[middle])) == int(np.ceil(np.log2(links)))
    for index in range(middle):
        assert _mirror_tree(choices[index], links-1) == choices[-1-index]
    if middle:
        assert tree_slider_position(left, links) == (0 if left == balanced_tree(links) else -1)
        assert tree_slider_position(right, links) == (0 if right == balanced_tree(links) else 1)
        choices[0][0] = "mutated"
        assert tree_slider_choices(links)[0] == left


def _mirror_tree(tree, last):
    return last-tree if isinstance(tree, int) else [_mirror_tree(tree[1], last), _mirror_tree(tree[0], last)]


def _leaf_depths(tree, depth=0):
    if isinstance(tree, int):
        return [depth]
    return _leaf_depths(tree[0], depth+1) + _leaf_depths(tree[1], depth+1)


def test_tree_node_labels_follow_distillation_notation():
    rounds = {"0:2": 2, "0:0": 1, "1:1": 1, "2:2": 1}
    fig = tree_figure([[0, 1], 2], 3, rounds)
    nodes = fig.data[-1]
    assert list(nodes.text) == ["2", "0", "1", "1", "1"]
    assert list(nodes.customdata) == ["0:2", "0:1", "0:0", "1:1", "2:2"]
    assert nodes.textposition == "middle center"
    assert "Link span 0:2" in nodes.hovertext[0]
    assert "Distillation rounds: 2" in nodes.hovertext[0]
    assert len(fig.data)-1 == 4  # One edge per non-root link.
    assert rounds == {"0:2": 2, "0:0": 1, "1:1": 1, "2:2": 1}


@pytest.mark.parametrize("dimensions,trace", [(1, "scatter"), (2, "heatmap"), (3, "scatter3d")])
def test_sweep_fixed_tree_isolation_backend_parity_and_plot_shapes(config, dimensions, trace):
    original = deepcopy(config)
    axes = [{"parameter": name, "min": low, "max": high, "points": 2}
            for name, low, high in [("Generation success", .5, 1),
                                     ("Swap hardware success", .5, 1),
                                     ("Initial fidelity", .95, 1)][:dimensions]]
    study = Experiment(config, {"mode": "sweep", "axes": axes})
    service = EvaluationService()
    while not study.finished:
        entry = study.step(service)
        assert entry.config["protocol"]["tree"] == config["protocol"]["tree"]
        assert not entry.error
    assert len(study.entries) == 2**dimensions
    assert study.best is not None
    assert config == original
    direct = service.evaluate(study.entries[-1].config)
    np.testing.assert_allclose(direct.pmf, study.entries[-1].run.pmf)
    assert study.entries[-1].run.summary == direct.summary
    figures = sweep_figures(study)
    assert len(figures) == 4
    assert figures[0].data[0].type == trace


def test_invalid_lifetimes_retained_and_cancel_checkpoint(config, tmp_path):
    config["memory"].update(t1_slots=20, t2_slots=30)
    study = Experiment(config, {"mode": "sweep", "axes": [
        {"parameter": "T1", "min": 5, "max": 20, "points": 3}]})
    service = EvaluationService()
    while not study.finished:
        study.step(service)
    assert [e.status for e in study.entries] == ["invalid", "invalid", "covered"]
    assert service.backend_calls == 1
    path = save_experiment(study, tmp_path)
    history = json.loads((path/"history.json").read_text())
    assert len(history["trials"]) == 3
    assert history["best_trial"] == 3
    saved = load_run(history["trials"][-1]["bundle"])
    assert saved.config == study.entries[-1].config
    second = Experiment(config, study.request)
    second.step(service)
    second.cancelled = True
    path = save_experiment(second, tmp_path)
    history = json.loads((path/"history.json").read_text())
    assert history["cancelled"] and len(history["trials"]) == 1
    with pytest.raises(StopIteration):
        second.step(service)


def test_sweep_coupling_validation_and_explicit_budget(config):
    request = {"mode": "sweep", "axes": [{"parameter": "T1", "min": 10, "max": 20, "points": 2}],
               "pure_relaxation": True}
    study = Experiment(config, request)
    entry = study.step(EvaluationService())
    assert entry.config["memory"]["t2_slots"] == 2*entry.config["memory"]["t1_slots"]
    with pytest.raises(ValueError, match="budget"):
        Experiment(config, {**request, "max_evaluations": 1})
    with pytest.raises(ValueError, match="positive"):
        Experiment(config, {"mode": "sweep", "axes": [
            {"parameter": "T1", "min": 0, "max": 20, "points": 2, "spacing": "log"}]})
    with pytest.raises(ValueError, match="distinct"):
        Experiment(config, {"mode": "sweep", "axes": request["axes"]*2})


def test_seeded_bayesian_search_fixed_scenario_and_budget(config):
    request = {"mode": "optimize", "dimensions": ["tree", "cutoff"],
               "budget": 4, "initial": 2, "cutoff_max": 1, "seed": 17}
    studies = [Experiment(config, request), Experiment(config, request)]
    for study in studies:
        service = EvaluationService()
        while not study.finished:
            entry = study.step(service)
            assert entry.config["hardware"] == config["hardware"]
            assert entry.config["memory"] == config["memory"]
            assert entry.config["initial_state"] == config["initial_state"]
            assert entry.config["numerics"] == config["numerics"]
        assert len(study.entries) == 4 and service.backend_calls <= 4
        assert study.best.eligible
    assert [e.point for e in studies[0].entries] == [e.point for e in studies[1].entries]


def test_repeated_gp_candidate_reuses_full_result_and_no_completion_has_no_winner(config, monkeypatch):
    study = Experiment(config, {"mode": "optimize", "dimensions": ["cutoff"], "budget": 3, "cutoff_max": 2})
    monkeypatch.setattr(study.optimizer, "ask", lambda: [0])
    monkeypatch.setattr(study.optimizer, "tell", lambda *args: None)
    service = EvaluationService()
    while not study.finished:
        study.step(service)
    assert service.backend_calls == 1
    assert [e.cached for e in study.entries] == [False, True, True]
    config["hardware"]["p_swap"] = 0
    study = Experiment(config, {"mode": "optimize", "method": "enumeration",
                               "dimensions": ["cutoff"], "cutoff_max": 0, "budget": 20})
    while not study.finished:
        study.step(service)
    assert len(study.entries) == 2
    assert study.best is None


def test_distillation_search_multiple_rounds_and_partial_backend_failure(config, monkeypatch):
    config.update(links=2)
    config["hardware"]["p_gen"] = .5
    config["numerics"]["t_trunc"] = 32
    config["protocol"]["tree"] = [0, 1]
    study = Experiment(config, {"mode": "optimize", "method": "enumeration", "dimensions": ["distillation"],
                               "distillation_max": 2, "budget": 3})
    service = EvaluationService()
    study.step(service)
    study.step(service)
    entry = study.step(service)
    assert entry.config["protocol"]["distillation_rounds"] == {"1:1": 2}
    assert [o["id"] for o in entry.run.metadata["operations"]][:2] == ["dist:1:1", "dist:1:1:2"]
    study = Experiment(config, {"mode": "optimize", "method": "enumeration", "dimensions": ["cutoff"], "budget": 3})
    study.step(service)
    monkeypatch.setattr(service, "evaluate", lambda c: (_ for _ in ()).throw(RuntimeError("deliberate failure")))
    entry = study.step(service)
    assert entry.status == "failed" and entry.error == "deliberate failure"
    assert study.best.index == 1


def test_undercovered_trials_cannot_win(config):
    config["hardware"]["p_gen"] = .1
    config["numerics"]["t_trunc"] = 2
    study = Experiment(config, {"mode": "sweep", "axes": [
        {"parameter": "Swap hardware success", "min": .5, "max": 1, "points": 2}]})
    while not study.finished:
        study.step(EvaluationService())
    assert all(e.status == "provisional" for e in study.entries)
    assert study.best is None


def test_fidelity_histogram_is_probability_weighted(config):
    x = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 1, -1], [1, 0, 0, 0, 0, 0]])
    run = Run({"config": config, "horizon": 3, "run_id": "histogram"}, np.array([0, .9, .1]), x)
    fig = main_figures([run], histogram=True)[1]
    assert sum(fig.data[0].y) == pytest.approx(1)
    assert fig.data[0].y[-1] == pytest.approx(.9)
    assert max(fig.data[0].y[:-1]) == pytest.approx(.1)


@pytest.mark.parametrize("request_name", ["sweep", "sweep-2d", "sweep-3d", "optimize", "enumerate"])
def test_documented_cli_studies(request_name, tmp_path, capsys):
    from pathlib import Path
    from src.visualization.cli import main
    examples = Path(__file__).parents[2]/"visualization"/"examples"
    assert main(["--config", str(examples/"smoke.json"), "--study", str(examples/f"{request_name}.json"),
                 "--output", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    history = json.loads((Path(result["study"])/"history.json").read_text())
    assert history["finished"] and not history["cancelled"]
    assert len(history["trials"]) == result["trials"]
    assert result["best_trial"] is not None
    assert (Path(result["study"])/"figures.html").exists()
