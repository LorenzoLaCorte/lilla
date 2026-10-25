"""Edits cross a component boundary; validate counts and reject stale trees."""
from copy import deepcopy

import pytest

from src.visualization.tree_editor import apply_tree_event, tree_payload


def test_tikz_layout_counts_and_tree_identity():
    counts = {"0:2": 2, "0:0": 1, "1:1": 1, "2:2": 1}
    payload = tree_payload([[0, 1], 2], 3, counts, "editor")
    nodes = {n["span"]: n for n in payload["nodes"]}
    assert len(nodes) == 5 and len(payload["edges"]) == 4
    assert nodes["0:2"]["rounds"] == 2 and nodes["0:1"]["rounds"] == 0
    assert nodes["0:2"]["x"] == (nodes["0:1"]["x"] + nodes["2:2"]["x"])/2
    assert nodes["0:2"]["depth"] == 0 and nodes["0:0"]["depth"] == 2
    assert payload["tree_id"] == tree_payload([[0, 1], 2], 3, {}, "editor")["tree_id"]
    assert payload["tree_id"] != tree_payload([0, [1, 2]], 3, counts, "editor")["tree_id"]
    assert payload["tree_id"] != tree_payload([[0, 1], 2], 3, counts, "imported")["tree_id"]


def test_valid_edit_filters_zero_and_stale_edit_preserves_counts():
    payload = tree_payload([0, 1], 2, {}, "editor")
    event = dict(tree_id=payload["tree_id"], rounds={"0:0": 0, "1:1": 1, "0:1": 2})
    assert apply_tree_event(payload, event, {}) == {"1:1": 1, "0:1": 2}
    previous = {"0:1": 3}
    event["tree_id"] = "old-topology"
    assert apply_tree_event(payload, event, previous) == previous
    assert previous == {"0:1": 3}


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2", None])
def test_invalid_count_is_rejected_without_mutation(bad):
    payload = tree_payload([0, 1], 2, {}, "editor")
    event = dict(tree_id=payload["tree_id"], rounds={"0:0": 0, "1:1": 0, "0:1": bad})
    previous = {"0:1": 2}
    snapshot = deepcopy(event)
    with pytest.raises(ValueError, match="nonnegative integer"):
        apply_tree_event(payload, event, previous)
    assert event == snapshot and previous == {"0:1": 2}


@pytest.mark.parametrize("counts", [{}, {"0:0": 0, "1:1": 0, "0:1": 1, "9:9": 1}])
def test_edit_must_cover_exactly_the_visible_nodes(counts):
    payload = tree_payload([0, 1], 2, {}, "editor")
    with pytest.raises(ValueError, match="Each tree node"):
        apply_tree_event(payload, dict(tree_id=payload["tree_id"], rounds=counts), {})
