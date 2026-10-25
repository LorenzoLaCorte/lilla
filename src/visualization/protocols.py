"""Compile ordered trees using the backend serializer and active-slot semantics."""
from copy import deepcopy
from functools import lru_cache

from src.types.protocol_types import SwapTreeVertex, generate_swap_space


def tree_json(node):
    return (node.segments[0] if node.left is None else
            [tree_json(node.left), tree_json(node.right)])


def ordered_trees(links):
    """Unique physical trees in the editor's left-balanced-right order."""
    trees = []
    seen = set()
    for tree in _tree_slider_choices(links):
        identity = repr(tree)
        if identity not in seen:
            trees.append(deepcopy(tree))
            seen.add(identity)
    return trees


def _mirror(tree, last):
    return last-tree if type(tree) is int else [_mirror(tree[1], last), _mirror(tree[0], last)]


def balanced_tree(links):
    """Split leaf counts evenly; choose mirrored children for even-sized subtrees."""
    def build(start, size):
        if size == 1:
            return start
        half = (size+1)//2
        left = build(start, half)
        right = _mirror(left, 2*start+size-1) if size % 2 == 0 else build(start+half, size//2)
        return [left, right]

    if type(links) is not int or not 1 <= links <= 8:
        raise ValueError("tree editor supports 1 to 8 elementary links")
    return build(0, links)


@lru_cache(maxsize=8)
def _tree_slider_choices(links):
    """Mirror-related arms meet at a balanced representative, including all trees.

    A self-mirrored alternative appears on both arms. For odd leaf counts the
    central representative also occurs on one arm. These few repetitions give
    an exact selectable center without dropping a physical ordered tree.
    """
    center = balanced_tree(links)
    left = []

    def imbalance(node):
        if node.left is None:
            return 0
        l, r = node.left.segments, node.right.segments
        return abs((l[1]-l[0]+1)-(r[1]-r[0]+1)) + imbalance(node.left) + imbalance(node.right)

    for node, _ in generate_swap_space(links):
        tree = tree_json(node)
        depths = tuple(node.get_leaf_depths())
        if depths > depths[::-1] or (depths == depths[::-1] and tree != center):
            left.append((imbalance(node), max(depths), tuple(-d for d in depths), tree))
    # Most structurally unbalanced first, with the fully left-leaning comb first.
    left.sort(key=lambda item: (-item[0], -item[1], item[2]))
    arm = [item[3] for item in left]
    return tuple(arm + [center] + [_mirror(tree, links-1) for tree in reversed(arm)])


def tree_slider_choices(links):
    return deepcopy(list(_tree_slider_choices(links)))


def tree_slider_position(tree, links):
    """Normalized position; import repeated balanced shapes at the nearest center."""
    choices = _tree_slider_choices(links)
    middle = len(choices)//2
    indices = [i for i, candidate in enumerate(choices) if candidate == tree]
    if not indices:
        raise ValueError("tree does not belong to the selected link count")
    index = min(indices, key=lambda i: (abs(i-middle), i))
    return (index-middle)/middle if middle else 0.0


@lru_cache(maxsize=8)
def symmetry_trees(links):
    """Ascending library symmetry, with a deterministic left-comb-first tie break."""
    from src.utils.gp_utils import get_swap_space
    # Leaf depths decrease from left to right for the left-leaning comb.
    entries = sorted(get_swap_space(links),
                     key=lambda entry: (entry[0], tuple(-d for d in entry[1].get_leaf_depths()),
                                        entry[2]))
    return tuple((score, tree_json(tree)) for score, tree, _ in entries)


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
        if type(count) is not int or count < 0:
            raise ValueError("distillation rounds must be nonnegative integers")
        nodes[span].dists = count
    sequence = root.get_sequence()
    slots = [(i, i) for i in range(links)]
    operations = []
    dist_counts = {}
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
        if span not in nodes or (kind == "d" and nodes[span].dists == 0):
            raise ValueError(f"operation {code} does not match tree location")
        op_id = f"swap:{span}"
        if kind == "d":
            dist_counts[span] = dist_counts.get(span, 0) + 1
            op_id = f"dist:{span}" + (f":{dist_counts[span]}" if dist_counts[span] > 1 else "")
        operations.append({"id": op_id,
                           "code": code, "endpoint_nodes": [start, end + 1]})
    expected = {f"swap:{s}" for s, n in nodes.items() if n.left is not None}
    expected |= {f"dist:{s}" + (f":{r}" if r > 1 else "")
                 for s, n in nodes.items() for r in range(1, n.dists+1)}
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
