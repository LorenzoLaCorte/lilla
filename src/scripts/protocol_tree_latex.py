"""Render asymmetric repeater protocols as TikZ binary trees.

Examples
--------
python -m src.scripts.protocol_tree_latex "('d0', 'd1', 's0')"

python -m src.scripts.protocol_tree_latex \
  --from-output results/pf_gp-1/example.txt --top 1 --label "(b)"
"""

from __future__ import annotations

from argparse import ArgumentParser
import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable


@dataclass
class ProtocolTree:
    left: "ProtocolTree | None" = None
    right: "ProtocolTree | None" = None
    start: int = 0
    end: int = 0
    dists: int = 0
    leaf: bool = True
    history: list[str] = field(default_factory=list)


def parse_protocol(protocol: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(protocol, str):
        parsed = ast.literal_eval(protocol)
    else:
        parsed = tuple(protocol)

    if isinstance(parsed, str):
        parsed = (parsed,)
    if not isinstance(parsed, tuple) or not all(isinstance(step, str) for step in parsed):
        raise ValueError(f"Expected a tuple of protocol steps, got: {parsed!r}")
    return parsed


def protocol_to_tree(protocol: tuple[str, ...]) -> ProtocolTree:
    number_of_segments = _infer_number_of_segments(protocol)
    active = {
        idx: ProtocolTree(start=idx, end=idx, leaf=True)
        for idx in range(number_of_segments)
    }

    for step in protocol:
        if len(step) < 2 or step[0] not in {"d", "s"}:
            raise ValueError(f"Invalid protocol step: {step!r}")
        operation, idx = step[0], int(step[1:])

        if idx not in active:
            raise ValueError(f"Step {step!r} targets no active segment in {protocol!r}")

        if operation == "d":
            active[idx].dists += 1
            active[idx].history.append(step)
            continue

        right_idx = min((candidate for candidate in active if candidate > idx), default=None)
        if right_idx is None:
            raise ValueError(f"Swap step {step!r} has no active right neighbor in {protocol!r}")

        left = active.pop(idx)
        right = active[right_idx]
        active[right_idx] = ProtocolTree(
            left=left,
            right=right,
            start=left.start,
            end=right.end,
            leaf=False,
        )

    if len(active) != 1:
        raise ValueError(f"Protocol does not reduce to one end-to-end link: {protocol!r}")
    return next(iter(active.values()))


def tree_to_tikz(
    tree: ProtocolTree,
    *,
    protocol: tuple[str, ...] | None = None,
    label: str | None = None,
    root_at: str = "2.5, 0",
    include_comment: bool = True,
    indent: str = "  ",
) -> str:
    lines: list[str] = []
    if include_comment and protocol is not None:
        lines.append(f"{indent}% Binary tree: {protocol}")

    if label is not None:
        lines.append(f"{indent}\\node at ({root_at}) {{{label}}}")
        _append_child(lines, tree, level=1, base_indent=indent)
        lines[-1] += ";"
    else:
        lines.append(f"{indent}\\node[treenode] {{{tree.dists}}}")
        _append_children(lines, tree, level=1, base_indent=indent)
        lines[-1] += ";"

    return "\n".join(lines)


def output_file_protocols(path: Path, top: int) -> list[tuple[tuple[str, ...], float]]:
    text = path.read_text(encoding="utf-8")
    json_text = text.split("\n\nUnique values:", 1)[0]
    results = json.loads(json_text)
    protocols: list[tuple[tuple[str, ...], float]] = []
    for protocol_s, rate in list(results.items())[:top]:
        protocols.append((parse_protocol(protocol_s), float(rate)))
    return protocols


def _infer_number_of_segments(protocol: tuple[str, ...]) -> int:
    max_dist = max((int(step[1:]) for step in protocol if step.startswith("d")), default=-1)
    max_swap = max((int(step[1:]) for step in protocol if step.startswith("s")), default=-1)
    return max(max_dist + 1, max_swap + 2, 1)


def _append_child(lines: list[str], tree: ProtocolTree, *, level: int, base_indent: str) -> None:
    pad = base_indent + "  " * level
    lines.append(f"{pad}child {{node[treenode] {{{tree.dists}}}")
    _append_children(lines, tree, level=level + 1, base_indent=base_indent)
    lines[-1] += "}"


def _append_children(lines: list[str], tree: ProtocolTree, *, level: int, base_indent: str) -> None:
    if tree.left is not None:
        _append_child(lines, tree.left, level=level, base_indent=base_indent)
    if tree.right is not None:
        _append_child(lines, tree.right, level=level, base_indent=base_indent)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("protocol", nargs="?", help="Protocol tuple, e.g. \"('d0', 'd1', 's0')\".")
    parser.add_argument("--from-output", type=Path, help="Read protocol(s) from an optimization output file.")
    parser.add_argument("--top", type=int, default=1, help="Number of top protocols to render from --from-output.")
    parser.add_argument("--label", default=None, help="Optional label placed above the tree, e.g. '(b)'.")
    parser.add_argument("--root-at", default="2.5, 0", help="TikZ coordinate for the optional label node.")
    parser.add_argument("--no-comment", action="store_true", help="Do not include a protocol comment.")
    args = parser.parse_args()

    if args.from_output is None and args.protocol is None:
        parser.error("provide a protocol tuple or --from-output")

    if args.from_output is not None:
        for idx, (protocol, rate) in enumerate(output_file_protocols(args.from_output, args.top), start=1):
            tree = protocol_to_tree(protocol)
            label = args.label if args.label is not None else f"({idx})"
            print(f"% SKR: {rate:.6g}")
            print(tree_to_tikz(tree, protocol=protocol, label=label, root_at=args.root_at, include_comment=not args.no_comment))
            if idx < args.top:
                print()
    else:
        protocol = parse_protocol(args.protocol)
        tree = protocol_to_tree(protocol)
        print(tree_to_tikz(tree, protocol=protocol, label=args.label, root_at=args.root_at, include_comment=not args.no_comment))


if __name__ == "__main__":
    main()
