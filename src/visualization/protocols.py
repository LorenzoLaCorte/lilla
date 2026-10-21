"""Compile ordered trees using the backend serializer and active-slot semantics."""
from copy import deepcopy

from src.types.protocol_types import SwapTreeVertex, generate_swap_space


def tree_json(node):
    return (node.segments[0] if node.left is None else
            [tree_json(node.left), tree_json(node.right)])


def ordered_trees(links):
    return [tree_json(deepcopy(t)) for t, _ in generate_swap_space(links)]


def parse_tree(value, links):
    leaves = []

    def build(v):
        if type(v) is int:
            leaves.append(v)
            return SwapTreeVertex(segments=(v, v))
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("tree must be an integer leaf or a two-child list")
        left, right = build(v[0]), build(v[1])
        return SwapTreeVertex(segments=(left.segments[0], right.segments[1]),
                              left=left, right=right)

    root = build(value)
    if leaves != list(range(links)):
        raise ValueError("tree leaves must be exactly 0,...,links-1 in order")
    return root


def locations(root):
    result = {f"{root.segments[0]}:{root.segments[1]}": root}
    if root.left is not None:
        result.update(locations(root.left))
        result.update(locations(root.right))
    return result


def cutoff_value(value, horizon):
    if value is None:
        return horizon
    if type(value) is not int or value < 0:
        raise ValueError("memory cutoffs must be nonnegative integers or null")
    return value


def compile_protocol(protocol, links, horizon):
    root = deepcopy(parse_tree(protocol["tree"], links))
    nodes = locations(root)
    rounds = protocol["distillation_rounds"]
    if not isinstance(rounds, dict) or set(rounds) - set(nodes):
        raise ValueError("distillation location does not belong to this tree")
    for span, count in rounds.items():
        if type(count) is not int or count not in (0, 1):
            raise ValueError("only 0 or 1 recurrence rounds per location are supported")
        nodes[span].dists = count
    sequence = root.get_sequence()
    slots = [(i, i) for i in range(links)]
    operations = []
    # Swapping removes the left slot and retains the right slot in the backend.
    for code in sequence:
        kind, index = code[0], int(code[1:])
        if kind not in ("s", "d") or index >= links or slots[index] is None:
            raise ValueError(f"invalid operation {code}")
        start, end = slots[index]
        if kind == "s":
            other = next((j for j in range(index + 1, links)
                          if slots[j] is not None), None)
            if other is None or end + 1 != slots[other][0]:
                raise ValueError(f"invalid adjacent swap {code}")
            end = slots[other][1]
            slots[index] = None
            slots[other] = (start, end)
        span = f"{start}:{end}"
        if span not in nodes or (kind == "d" and nodes[span].dists != 1):
            raise ValueError(f"operation {code} does not match tree location")
        operations.append({"id": f"{'swap' if kind == 's' else 'dist'}:{span}",
                           "code": code, "endpoint_nodes": [start, end + 1]})
    expected = {f"swap:{s}" for s, n in nodes.items() if n.left is not None}
    expected |= {f"dist:{s}" for s, n in nodes.items() if n.dists}
    if {o["id"] for o in operations} != expected:
        raise ValueError("serialized operations do not match requested tree")
    overrides = protocol["cutoff_overrides"]
    if not isinstance(overrides, dict) or set(overrides) - expected:
        raise ValueError("cutoff override refers to a nonexistent operation ID")
    shared = protocol["memory_cutoff_slots"]
    cutoff_value(shared, horizon)
    for op in operations:
        value = overrides.get(op["id"], shared)
        op["cutoff_slots"] = value
        op["backend_cutoff"] = cutoff_value(value, horizon)
    return sequence, operations
