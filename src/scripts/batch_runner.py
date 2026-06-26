"""Small helpers for script-style optimization batches."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWN_PLOTS = ("gp.pdf", "bf.pdf", "skopt_gp.pdf")


@dataclass(frozen=True)
class RunResult:
    command: Sequence[str]
    output_file: Path
    log_file: Path
    elapsed_seconds: float
    returncode: int


def python_bin() -> str:
    return os.environ.get("PYTHON_BIN", sys.executable)


def slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("_").lower() or "case"


def add_arg(args: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            args.append(flag)
        return
    if isinstance(value, (list, tuple)):
        args.append(flag)
        args.extend(str(item) for item in value)
        return
    args.extend([flag, str(value)])


def run_module(
    module: str,
    module_args: Sequence[str],
    *,
    output_file: Path,
    log_file: Path,
    python: str | None = None,
    dry_run: bool = False,
    move_plots: bool = True,
) -> RunResult:
    command = [python or python_bin(), "-m", module, *module_args]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"OUT: {output_file}")
    print(f"LOG: {log_file}")
    print(f"CMD: {shlex.join(command)}")
    print("=" * 80)

    if dry_run:
        return RunResult(command, output_file, log_file, 0.0, 0)

    before = {
        name: (REPO_ROOT / name).stat().st_mtime_ns
        for name in KNOWN_PLOTS
        if (REPO_ROOT / name).exists()
    }

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/lilla_mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp/lilla_cache")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start

    if completed.stdout:
        print(completed.stdout, end="")
    log_file.write_text(completed.stdout or "", encoding="utf-8")

    with output_file.open("a", encoding="utf-8") as handle:
        handle.write("\nTime taken:\n")
        handle.write(f"Elapsed seconds: {elapsed:.3f}\n")

    if move_plots:
        _move_plots(output_file, before)

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
        )

    return RunResult(command, output_file, log_file, elapsed, completed.returncode)


def _move_plots(output_file: Path, before: dict[str, int]) -> None:
    stem = output_file.stem
    for name in KNOWN_PLOTS:
        source = REPO_ROOT / name
        if not source.exists():
            continue
        if before.get(name) == source.stat().st_mtime_ns:
            continue
        destination = output_file.with_name(f"{stem}_{name}")
        source.replace(destination)


def run_many(
    calls: Iterable[tuple[str, Sequence[str], Path, Path]],
    *,
    python: str | None = None,
    dry_run: bool = False,
) -> list[RunResult]:
    return [
        run_module(module, args, output_file=out, log_file=log, python=python, dry_run=dry_run)
        for module, args, out, log in calls
    ]
