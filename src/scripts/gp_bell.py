"""Batch runner for Pauli-Fourier Bell-diagonal asymmetric searches."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.scripts.batch_runner import REPO_ROOT, add_arg, run_module, slug, python_bin
RATE_A = 1 / 360_000
RATE_B = 1 / 720_000
RATE_C = 1 / 1_400_000
RATE_D = 1 / 3_600_000

@dataclass(frozen=True)
class PfCase:
    label: str
    depolarizing_rate: float
    dephasing_rate: float
    p_gen: float
    p_swap: float
    lambdas: tuple[float, float, float, float]
    nodes: int
    max_dists: int
    optimizer: str
    seed: Optional[int] = None


QCNC_CASES = (
    PfCase("qcnc_A_beta2_bf", RATE_A, RATE_A, 0.000000096, 0.85, (0.52, 0.16, 0.16, 0.16), 2, 2, "bf"),
    PfCase("qcnc_B_beta2_bf", RATE_B, RATE_B, 0.000015, 0.85, (0.925, 0.025, 0.025, 0.025), 3, 2, "bf"),
    PfCase("qcnc_C_beta1_bf", RATE_C, RATE_C, 0.00092, 0.85, (0.964, 0.012, 0.012, 0.012), 5, 1, "bf"),
    PfCase("qcnc_C_beta1_gp", RATE_C, RATE_C, 0.00092, 0.85, (0.964, 0.012, 0.012, 0.012), 5, 1, "gp", 42),
    PfCase("qcnc_C_beta2_gp", RATE_C, RATE_C, 0.00092, 0.85, (0.964, 0.012, 0.012, 0.012), 5, 2, "gp", 42),
    PfCase("qcnc_D_beta2_gp", RATE_D, RATE_D, 0.0026, 0.85, (0.964, 0.012, 0.012, 0.012), 11, 2, "gp", 42),
    PfCase("qcnc_w094_beta0_bf", RATE_C, RATE_C, 0.00092, 0.85, (0.94, 0.02, 0.02, 0.02), 5, 0, "bf"),
    PfCase("qcnc_w0955_beta0_bf", RATE_C, RATE_C, 0.00092, 0.85, (0.955, 0.015, 0.015, 0.015), 5, 0, "bf"),
    PfCase("qcnc_w097_beta0_bf", RATE_C, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf"),
    PfCase("qcnc_w0985_beta0_bf", RATE_C, RATE_C, 0.00092, 0.85, (0.985, 0.005, 0.005, 0.005), 5, 0, "bf"),
    PfCase("qcnc_w094_beta3_gp", RATE_C, RATE_C, 0.00092, 0.85, (0.94, 0.02, 0.02, 0.02), 5, 3, "gp", 42),
    PfCase("qcnc_w0955_beta3_gp", RATE_C, RATE_C, 0.00092, 0.85, (0.955, 0.015, 0.015, 0.015), 5, 3, "gp", 42),
    PfCase("qcnc_w097_beta3_gp", RATE_C, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_w0985_beta3_gp", RATE_C, RATE_C, 0.00092, 0.85, (0.985, 0.005, 0.005, 0.005), 5, 3, "gp", 42),
    # new analysis fixing w_0 but moving the noise bidimensionally to do a countour plot of the optimal beta 
    PfCase("qcnc_dAzA_beta0_bf", RATE_A, RATE_A, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dAzA_beta3_gp", RATE_A, RATE_A, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dAzB_beta0_bf", RATE_A, RATE_B, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dAzB_beta3_gp", RATE_A, RATE_B, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dAzC_beta0_bf", RATE_A, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dAzC_beta3_gp", RATE_A, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dAzD_beta0_bf", RATE_A, RATE_D, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dAzD_beta3_gp", RATE_A, RATE_D, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dBzA_beta0_bf", RATE_B, RATE_A, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dBzA_beta3_gp", RATE_B, RATE_A, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dBzB_beta0_bf", RATE_B, RATE_B, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dBzB_beta3_gp", RATE_B, RATE_B, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dBzC_beta0_bf", RATE_B, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dBzC_beta3_gp", RATE_B, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dBzD_beta0_bf", RATE_B, RATE_D, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dBzD_beta3_gp", RATE_B, RATE_D, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dCzA_beta0_bf", RATE_C, RATE_A, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dCzA_beta3_gp", RATE_C, RATE_A, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dCzB_beta0_bf", RATE_C, RATE_B, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dCzB_beta3_gp", RATE_C, RATE_B, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dCzC_beta0_bf", RATE_C, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dCzC_beta3_gp", RATE_C, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dCzD_beta0_bf", RATE_C, RATE_D, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dCzD_beta3_gp", RATE_C, RATE_D, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dDzA_beta0_bf", RATE_D, RATE_A, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dDzA_beta3_gp", RATE_D, RATE_A, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dDzB_beta0_bf", RATE_D, RATE_B, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dDzB_beta3_gp", RATE_D, RATE_B, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dDzC_beta0_bf", RATE_D, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dDzC_beta3_gp", RATE_D, RATE_C, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
    PfCase("qcnc_dDzD_beta0_bf", RATE_D, RATE_D, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 0, "bf", 42),
    PfCase("qcnc_dDzD_beta3_gp", RATE_D, RATE_D, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 3, "gp", 42),
)

LARGE_NOISE_CASES = (
    PfCase("paper_C_high_depolarized", 0.00000357, 0.0, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 2, "gp", 42),
    PfCase("paper_C_high_dephased", 0.0, 0.00000357, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 2, "gp", 42),
    PfCase("paper_C_high_depol_and_deph", 0.00000357, 0.00000357, 0.00092, 0.85, (0.97, 0.01, 0.01, 0.01), 5, 2, "gp", 42),
)

SMOKE_CASES = (
    PfCase("smoke_pf_dephasing_gp", 0.0, 0.01, 0.2, 0.9, (0.9475, 0.0175, 0.0175, 0.0175), 3, 1, "gp", 7),
)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one tiny GP/PF dephasing check.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--run-qcnc", action="store_true", help="Include the qcnc paper-scale cases.")
    parser.add_argument("--run-large-noise", action="store_true", help="Include the larger GP noise comparison cases.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "pf_gp")
    parser.add_argument("--gp-shots", type=int, default=200)
    parser.add_argument("--gp-initial-points", type=int, default=20)
    parser.add_argument("--cdf-threshold", type=float, default=0.99)
    parser.add_argument("--python", default=python_bin())
    parser.add_argument("--adaptive-t-trunc-attempts", type=int, default=3)
    parser.add_argument("--t-trunc-growth", type=float, default=2.0)
    parser.add_argument("--max-t-trunc", type=int, default=None)
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    cases: list[tuple[str, PfCase]] = []
    if args.smoke:
        cases.extend(("smoke", case) for case in SMOKE_CASES)
    else:
        if args.run_large_noise:
            cases.extend(("large_noise", case) for case in LARGE_NOISE_CASES)
        if args.run_qcnc:
            cases.extend(("qcnc", case) for case in QCNC_CASES)

    for group, case in cases:
        output_file = out_dir / f"{_case_name(group, case)}.txt"
        log_file = out_dir / "logs" / f"{_case_name(group, case)}.log"
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


def _case_name(group: str, case: PfCase) -> str:
    return slug(
        f"pf_{group}_{case.label}_N{case.nodes}_beta{case.max_dists}"
        f"_pgen{case.p_gen}_depol{case.depolarizing_rate}_deph{case.dephasing_rate}_{case.optimizer}"
    )


def _module_args(case: PfCase, output_file: Path, args) -> list[str]:
    module_args: list[str] = []
    for flag, value in (
        ("--state", "pf"),
        ("--nodes", case.nodes),
        ("--max_dists", case.max_dists),
        ("--optimizer", case.optimizer),
        ("--cdf_threshold", args.cdf_threshold),
        ("--p_swap", case.p_swap),
        ("--p_gen", case.p_gen),
        ("--lambdas", case.lambdas),
        ("--depolarizing_rate", case.depolarizing_rate),
        ("--dephasing_rate", case.dephasing_rate),
        ("--filename", output_file),
        ("--adaptive-t-trunc-attempts", args.adaptive_t_trunc_attempts),
        ("--t-trunc-growth", args.t_trunc_growth),
        ("--max-t-trunc", args.max_t_trunc),
    ):
        add_arg(module_args, flag, value)

    if case.optimizer == "gp":
        add_arg(module_args, "--gp_shots", args.gp_shots if not args.smoke else min(args.gp_shots, 6))
        add_arg(module_args, "--gp_initial_points", args.gp_initial_points if not args.smoke else min(args.gp_initial_points, 2))
    if case.seed is not None:
        add_arg(module_args, "--seed", case.seed)
    return module_args


if __name__ == "__main__":
    main()
