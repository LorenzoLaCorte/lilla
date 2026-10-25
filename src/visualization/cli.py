"""Run one configuration or a bounded study without importing Streamlit."""
import argparse
import json
from pathlib import Path

from src.visualization.config import load_config
from src.visualization.evaluate import EvaluationService
from src.visualization.experiments import Experiment
from src.visualization.plots import main_figures, search_figure, sweep_figures
from src.visualization.storage import (DEFAULT_DIRECTORY, config_json, figures_html,
                                       per_time_csv, save_experiment, save_run, summary_csv)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study", type=Path, help="Sweep/search request JSON; omit for a single run")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args(argv)
    experiment = None
    try:
        config = load_config(args.config.read_text(encoding="utf-8"))
        service = EvaluationService()
        if args.study is None:
            run = service.evaluate(config)
            destination = save_run(run, args.output)
            (destination/"per-time.csv").write_text(per_time_csv(run), encoding="utf-8")
            (destination/"summary.csv").write_text(summary_csv([run]), encoding="utf-8")
            (destination/"figures.html").write_text(figures_html(main_figures([run])), encoding="utf-8")
            result = {"bundle": str(destination), "summary": run.summary,
                      "wall_seconds": run.metadata["wall_seconds"]}
        else:
            request = json.loads(args.study.read_text(encoding="utf-8"))
            experiment = Experiment(config, request)
            while not experiment.finished:
                experiment.step(service)
                destination = save_experiment(experiment, args.output)
            figures = sweep_figures(experiment) if experiment.mode == "sweep" else [search_figure(experiment)]
            (destination/"figures.html").write_text(figures_html(figures), encoding="utf-8")
            result = {"study": str(destination), "trials": len(experiment.entries),
                      "backend_calls": service.backend_calls,
                      "best_trial": experiment.best.index if experiment.best else None,
                      "best_summary": experiment.best.run.summary if experiment.best else None}
    except KeyboardInterrupt:
        if experiment is not None:
            experiment.cancelled = True
            destination = save_experiment(experiment, args.output)
            print(config_json({"cancelled": True, "study": str(destination)}))
        return 130
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(config_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
