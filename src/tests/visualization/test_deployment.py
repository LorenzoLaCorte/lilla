"""Exercise hosted results storage through the real CLI process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.visualization.storage import load_run

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("explicit_output", [False, True])
def test_cli_respects_hosted_results_directory(tmp_path, explicit_output):
    hosted = tmp_path/"mounted-disk"/"runs"
    override = tmp_path/"explicit-output"
    command = [sys.executable, "-m", "src.visualization.cli", "--config",
               "src/visualization/examples/smoke.json"]
    if explicit_output:
        command += ["--output", str(override)]
    process = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True,
                             env={**os.environ, "LILLA_RESULTS_DIR": str(hosted)})
    bundle = Path(json.loads(process.stdout)["bundle"])
    assert bundle.parent == (override if explicit_output else hosted)
    assert load_run(bundle).summary["rate_bits_per_slot"] == pytest.approx(.215706, abs=1e-6)
    assert (bundle/"figures.html").is_file()
    if explicit_output:
        assert not hosted.exists()
