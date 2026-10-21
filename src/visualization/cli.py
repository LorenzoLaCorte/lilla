"""Evaluate the same versioned config as the UI, without importing Streamlit."""
import argparse
from pathlib import Path

from src.visualization.config import load_config
from src.visualization.evaluate import EvaluationService
from src.visualization.plots import time_figures, state_figure
from src.visualization.storage import (DEFAULT_DIRECTORY, config_json, figures_html,
                                       per_time_csv, save_run, summary_csv)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args(argv)
    try:
        run = EvaluationService().evaluate(load_config(args.config.read_text(encoding="utf-8")))
        destination = save_run(run, args.output)
        (destination/"per-time.csv").write_text(per_time_csv(run), encoding="utf-8")
        (destination/"summary.csv").write_text(summary_csv([run]), encoding="utf-8")
        (destination/"figures.html").write_text(figures_html(
            time_figures([run])+[state_figure(run)]), encoding="utf-8")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(config_json({"bundle": str(destination), "summary": run.summary,
                       "wall_seconds": run.metadata["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
