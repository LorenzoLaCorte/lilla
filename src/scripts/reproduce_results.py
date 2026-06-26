"""Reproduce asymmetric Werner runs with concise Python case definitions."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.scripts.batch_runner import REPO_ROOT, add_arg, python_bin, run_module, slug


NumberOrList = float | int | tuple[float | int, ...]


@dataclass(frozen=True)
class WernerCase:
    label: str
    t_coh: NumberOrList
    p_gen: NumberOrList
    p_swap: float
    w0: NumberOrList
    nodes: int
    max_dists: int
    optimizer: str
    seed: Optional[int] = None


DEFAULT_HOMOGENEOUS_CASES = (
    WernerCase("D_beta2_gp", 3_600_000, 0.0026, 0.85, 0.958, 11, 2, "gp", 40),
    WernerCase("C_w096_beta3_gp", 1_400_000, 0.00092, 0.85, 0.96, 5, 3, "gp", 40),
)

HETEROGENEOUS_CASES = (
    WernerCase(
        "heterogeneous_C_bf",
        (1_080_000, 950_000, 720_000, 560_000),
        (0.002588, 0.0009187, 0.0009082),
        0.85,
        (0.9577, 0.9524, 0.9523),
        4,
        2,
        "bf",
    ),
    WernerCase(
        "faulty_tcoh_bf",
        (100_000, 100_000, 100_000, 10_000),
        (0.0025, 0.0025, 0.0025),
        0.85,
        (0.95, 0.95, 0.95),
        4,
        2,
        "bf",
    ),
    WernerCase(
        "faulty_pgen_bf",
        (100_000, 100_000, 100_000, 100_000),
        (0.0025, 0.0025, 0.00025),
        0.85,
        (0.95, 0.95, 0.95),
        4,
        2,
        "bf",
    ),
    WernerCase(
        "faulty_w0_bf",
        (100_000, 100_000, 100_000, 100_000),
        (0.0025, 0.0025, 0.0025),
        0.85,
        (0.95, 0.95, 0.90),
        4,
        2,
        "bf",
    ),
)

SMOKE_CASES = (
    WernerCase("smoke_werner_gp", 400, 0.2, 0.9, 0.95, 3, 1, "gp", 11),
)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny GP check instead of the paper cases.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--include-heterogeneous", action="store_true", help="Also run the heterogeneous/faulty-chain cases.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results_asymmetric")
    parser.add_argument("--gp-shots", type=int, default=100)
    parser.add_argument("--gp-initial-points", type=int, default=10)
    parser.add_argument("--cdf-threshold", type=float, default=0.99)
    parser.add_argument("--python", default=python_bin())
    args = parser.parse_args()

    cases = list(SMOKE_CASES if args.smoke else DEFAULT_HOMOGENEOUS_CASES)
    if not args.smoke and args.include_heterogeneous:
        cases.extend(HETEROGENEOUS_CASES)

    out_dir = _resolve(args.out_dir)
    for case in cases:
        output_file = out_dir / f"{_case_name(case)}.txt"
        log_file = out_dir / "logs" / f"{_case_name(case)}.log"
        run_module(
            "src.optimization.gp_asymmetric",
            _module_args(case, output_file, args),
            output_file=output_file,
            log_file=log_file,
            python=args.python,
            dry_run=args.dry_run,
        )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _case_name(case: WernerCase) -> str:
    return slug(
        f"werner_{case.label}_N{case.nodes}_beta{case.max_dists}"
        f"_pgen{case.p_gen}_pswap{case.p_swap}_w0{case.w0}_{case.optimizer}"
    )


def _module_args(case: WernerCase, output_file: Path, args) -> list[str]:
    module_args: list[str] = []
    for flag, value in (
        ("--state", "werner"),
        ("--nodes", case.nodes),
        ("--max_dists", case.max_dists),
        ("--optimizer", case.optimizer),
        ("--cdf_threshold", args.cdf_threshold),
        ("--p_swap", case.p_swap),
        ("--p_gen", _as_tuple(case.p_gen)),
        ("--w0", _as_tuple(case.w0)),
        ("--t_coh", _as_tuple(case.t_coh)),
        ("--filename", output_file),
    ):
        add_arg(module_args, flag, value)

    if case.optimizer == "gp":
        add_arg(module_args, "--gp_shots", args.gp_shots if not args.smoke else min(args.gp_shots, 6))
        add_arg(module_args, "--gp_initial_points", args.gp_initial_points if not args.smoke else min(args.gp_initial_points, 2))
    if case.seed is not None:
        add_arg(module_args, "--seed", case.seed)
    return module_args


def _as_tuple(value: NumberOrList) -> tuple[float | int, ...]:
    return value if isinstance(value, tuple) else (value,)


if __name__ == "__main__":
    main()
