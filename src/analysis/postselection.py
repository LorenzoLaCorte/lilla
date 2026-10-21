"""Pure-relaxation study. Run: python -m src.analysis.postselection.

Edit the parameters in main() to change the experiment. Results go to a fresh
results/analysis directory. The evaluator already includes rejection and retries.
"""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

import numpy as np

from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService, ROOT
from src.visualization.metrics import binary_entropy
from src.visualization.storage import config_json, csv_text

POLICIES = ("all_corrected", "phi_pair", "psi_pair")


def bb84_score(summary):
    return 1 - float(binary_entropy(summary["eX"])) - float(binary_entropy(summary["eZ"]))


def evaluate(service, config):
    """Check T against 2T, then return the longer-horizon summary."""
    short = service.evaluate(config).summary
    longer = deepcopy(config)
    longer["numerics"]["t_trunc"] *= 2
    long = service.evaluate(longer).summary
    fields = ("rate_bits_per_slot", "mean_wait_repository_slots", "eX", "eZ")
    if (min(short["coverage"], long["coverage"]) < 0.999999
            or not np.allclose([short[k] for k in fields], [long[k] for k in fields],
                               rtol=1e-6, atol=1e-9)):
        raise ValueError("horizon check failed; increase numerics.t_trunc")
    return {**long, "bb84_score": bb84_score(long), "horizon": longer["numerics"]["t_trunc"]}


def save_plot(rows, config, xfield, title, label):
    """Save a CSV, configuration, and three-panel PNG/PDF figure."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT/"results"/"analysis"/f"{label}-{stamp}"
    output.mkdir(parents=True)
    columns = (xfield, "policy", "rate_bits_per_slot", "bb84_score", "pooled_secret_fraction",
               "mean_wait_repository_slots", "eX", "eZ", "coverage", "horizon")
    (output/"sweep.csv").write_text(csv_text([{k: r[k] for k in columns} for r in rows]))
    (output/"config.json").write_text(config_json(config)+"\n")
    plot_results(rows, xfield, title, output)
    print(output)
    return output


def plot_results(rows, xfield, title, output):
    """Render saved results; eta is displayed as T1 in units of prior storage time."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"lilla-analysis-mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir())/"lilla-analysis-cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (("rate_bits_per_slot", "SKR (bits/slot)"),
              ("bb84_score", "1 - h2(eX) - h2(eZ)"),
              ("mean_wait_repository_slots", "Mean waiting time (slots)"))
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, layout="constrained")
    for policy, color, style in zip(POLICIES, ("#0072B2", "#CC79A7", "#D55E00"), ("-", "--", ":")):
        selected = [r for r in rows if r["policy"] == policy]
        x = np.array([r[xfield] for r in selected])
        if xfield == "eta":
            x = -1/np.log(x)  # eta = exp(-tau/T1), hence T1/tau = -1/log(eta).
        for ax, (field, ylabel) in zip(axes, panels):
            ax.plot(x, [r[field] for r in selected],
                    style, color=color, label=policy, linewidth=2)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[0].legend()
    if xfield == "t1_slots":
        axes[-1].set_xscale("log")
    axes[-1].set_xlabel(r"$T_1$ (slots)" if xfield == "t1_slots"
                        else r"$T_1$")
    note = r"$T_2=2T_1$"
    axes[-1].text(0.5, -0.20, note, transform=axes[-1].transAxes,
                  ha="center", va="top", fontsize=8)
    fig.suptitle(title+"\nEvery point checked at T and 2T", fontsize=11)
    fig.savefig(output/"postselection.png", dpi=200, bbox_inches="tight")
    fig.savefig(output/"postselection.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    config = deepcopy(DEFAULT)
    config["name"] = "Pure relaxation: four balanced links"
    config["initial_state"]["fidelity"] = 0.97
    config["hardware"]["p_swap"] = 1.0
    config["numerics"]["t_trunc"] = 1024
    service, rows = EvaluationService(), []
    for t1 in np.geomspace(5, 4000, 61):
        config["memory"] = {"kind": "t1_t2", "t1_slots": float(t1), "t2_slots": float(2*t1)}
        for policy in POLICIES:
            config["protocol"]["bell_outcomes"] = policy
            rows.append({"t1_slots": float(t1), "policy": policy, **evaluate(service, config)})
    save_plot(rows, config, "t1_slots",
              "Pure relaxation: T2=2T1 · four balanced links · F0=0.97\n"
              "p_gen=0.2 · p_swap=1 · no cutoff or distillation", "pure-relaxation")


if __name__ == "__main__":
    main()
