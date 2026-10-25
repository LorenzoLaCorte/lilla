"""Static protocol tree with editable nodes, served locally by Streamlit."""
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from src.visualization.protocols import locations, parse_tree


def tree_payload(tree, links, rounds, key):
    root = parse_tree(tree, links)
    nodes, edges = [], []

    def visit(node, depth=0):
        span = f"{node.segments[0]}:{node.segments[1]}"
        x = node.segments[0]
        if node.left is not None:
            children = [visit(child, depth+1) for child in (node.left, node.right)]
            x = sum(child[0] for child in children)/2
            edges.extend({"parent": span, "child": child[1]} for child in children)
        label = (f"Elementary link {node.segments[0]}" if node.left is None else
                 f"Link span {node.segments[0]}–{node.segments[1]}")
        count = rounds.get(span, 0)
        if type(count) is not int or count < 0:
            raise ValueError("distillation rounds must be nonnegative integers")
        nodes.append(dict(span=span, x=x, depth=depth, rounds=count, label=label))
        return x, span

    visit(root)
    identity = json.dumps([key, links, tree], separators=(",", ":"))
    return {"tree_id": sha256(identity.encode()).hexdigest(), "links": links,
            "nodes": nodes, "edges": edges}


def apply_tree_event(payload, event, previous):
    """Validate full count updates; stale events from a different tree are ignored."""
    if not isinstance(event, dict) or event.get("tree_id") != payload["tree_id"]:
        return deepcopy(previous)
    counts = event.get("rounds")
    spans = {node["span"] for node in payload["nodes"]}
    if (not isinstance(counts, dict) or set(counts) != spans
            or any(type(n) is not int or n < 0 for n in counts.values())):
        raise ValueError("Each tree node needs a nonnegative integer distillation count.")
    return {span: n for span, n in counts.items() if n}


@lru_cache(maxsize=1)
def _component():
    from streamlit.components.v1 import declare_component
    return declare_component("protocol_tree", path=str(Path(__file__).with_name("tree_component")))


def protocol_tree(tree, links, rounds, *, key, editable=True):
    """Apply a pending edit before rendering, so the next Run sees current counts."""
    import streamlit as st

    # Filter obsolete spans when the topology changes, matching the visible tree.
    valid = locations(parse_tree(tree, links))
    counts = {span: n for span, n in rounds.items() if span in valid and n}
    payload = tree_payload(tree, links, counts, key)
    event = st.session_state.get(key)
    ack_key = f"{key}-ack"
    if editable and isinstance(event, dict) and isinstance(event.get("event_id"), str):
        if event["event_id"] != st.session_state.get(ack_key):
            # Consume even stale events: returning to that topology must not replay them.
            st.session_state[ack_key] = event["event_id"]
            try:
                counts = apply_tree_event(payload, event, counts)
            except ValueError as error:
                st.error(str(error))
            payload = tree_payload(tree, links, counts, key)
    _component()(**payload, editable=editable, ack_id=st.session_state.get(ack_key), key=key, default=None)
    return counts
